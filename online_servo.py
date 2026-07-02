#!/usr/bin/env python3
"""online_servo :: low-level 4 ms stream-following controller for the MyCobot Pro 630.

Run on the Pi (loaded by LinuxCNC as the `ctrl` component) INSTEAD of robot_hal when the
arm is driven by a streaming planner node.

ARCHITECTURE -- a planner node emits a short trajectory CHUNK at a fixed rate; this
controller WELDS the chunks into one continuous wall-clock reference q_ref(t) and SERVOS
it at the HAL command rate (controller_params period_ms=4 -> 250 Hz):

    planner node (10 Hz):  every 0.1 s -> 0.4 s chunk (40 pts @ dt=0.01), anchored at
                           an absolute wall-time t_anchor
          --TCP JSON lines--> [:9994] --weld--> q_ref(t)
    servo @250 Hz:  target = q_ref(now + LEAD)  ->  pid -> HAL joint pos/vel cmd
    feedback streamed on :9999 (same as robot_hal).

DEAD-TIME COMPENSATION -- the actuator has a transport dead-time (the motion lags the
command). Because q_ref(t) is known ahead of time (the welder holds the future chunk),
we sample it with a feed-forward LEAD: target = q_ref(now + lead). The delayed motion then
lands on q_ref(now). `lead` should be set to the MEASURED residual dead-time (0 = pure
follower; tune up after a step-response measurement). The lead can never exceed the welder
horizon (~0.4 s of chunk), which is why the planner streams 0.4 s chunks.

NO INTEGRAL WINDUP -- the welded reference IS the feed-forward, so the loop runs as pure PD
(integral zeroed each step). A wound-up integral through the dead-time is what caused the
terminal overshoot; dropping it + the lead is the fix.

Chunk JSON (LinuxCNC deg, exactly what the planner builds):
    {"trajectory": [[6 deg], ...], "traj_dt": float, "t_anchor": abs_wall_seconds}
    {"hold": true}                 # freeze the reference at the current position

REQUIRES elerob_online.hal to run pid + pro_socketcan on a FAST (~4 ms) thread and
pro600.motor_time_interval ~4 -- otherwise the HAL downsamples this 4 ms loop back to the
20 ms slow-thread and the gain is lost. See mycobot-command-rate-throttle.

Loaded by LinuxCNC (robot_hal NOT running):
    loadusr -Wn ctrl python /home/pi/Desktop/mpc/online_servo.py --lead 0.0
"""
import argparse
import json
import socket
import threading
import time

import numpy as np

import hal
import linuxcnc
import robot_hal as rh                      # HAL primitives + pid_solve + stream server
from traj_weld import TrajectoryWelder

MAX_JOINTS = rh.MAX_JOINTS


class StreamFollower:
    def __init__(self, a):
        self.a = a
        # welder fine grid = the chunk dt; sample() interpolates to the 4 ms servo instants
        self.welder = TrajectoryWelder(dof=MAX_JOINTS, fine_dt=a.weld_fine_dt)
        self.lock = threading.Lock()
        self.hold = True
        self.h = hal.component("ctrl")        # name wired to the joints in elerob_online.hal
        for i in range(MAX_JOINTS):
            self.h.newpin(f"joint{i}_pos_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
            self.h.newpin(f"joint{i}_vel_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
        self.h.newpin("enable", hal.HAL_BIT, hal.HAL_OUT)
        self.h.ready()
        for i in range(MAX_JOINTS):
            self.h[f"joint{i}_pos_cmd"] = 0.0
            self.h[f"joint{i}_vel_cmd"] = 0.0
        self.h["enable"] = False
        self.s = linuxcnc.stat()

    # ---------- chunk intake ----------
    def _chunk_server(self):
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.a.chunk_port)); srv.listen(5)
        print(f"[chunks] listening on 0.0.0.0:{self.a.chunk_port}")
        while True:
            conn, _ = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._chunk_client, args=(conn,), daemon=True).start()

    def _chunk_client(self, conn):
        conn.settimeout(2.0); buf = ""
        try:
            while True:
                try:
                    d = conn.recv(65536)
                except socket.timeout:
                    continue
                if not d:
                    break
                buf += d.decode("utf-8", "replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._ingest(line)
        finally:
            conn.close()

    def _ingest(self, line):
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            return
        if c.get("hold"):
            with self.lock:
                self.hold = True
            return
        if "set_max_step" in c:                                # live-tune the target rate-limit
            with self.lock:
                self.a.max_step = float(c["set_max_step"])
            print(f"[cfg] max_step -> {self.a.max_step:.1f} deg")
            return
        if "set_lead" in c:                                    # live-tune the feed-forward lead
            with self.lock:
                self.a.lead = float(c["set_lead"])
            print(f"[cfg] lead -> {self.a.lead*1000:.0f} ms")
            return
        if "set_ff_scale" in c:                                # live-tune the velocity feed-forward
            with self.lock:
                self.a.ff_scale = float(c["set_ff_scale"])
            print(f"[cfg] ff_scale -> {self.a.ff_scale:.3f}")
            return
        if "set_vel_cmd_max" in c:                             # live-tune the overspeed clamp
            with self.lock:
                self.a.vel_cmd_max = float(c["set_vel_cmd_max"])
            print(f"[cfg] vel_cmd_max -> {self.a.vel_cmd_max:.0f}")
            return
        traj = c.get("trajectory")
        if not traj:
            return
        arr = np.array(traj, float)
        dt = float(c.get("traj_dt", 0.01))
        if arr.shape[0] > 1:                                   # SAFETY: reject too-fast chunks
            vmax = float(np.abs(np.diff(arr, axis=0)).max()) / max(dt, 1e-4)
            if vmax > self.a.max_chunk_vel:
                print(f"[chunks] REJECT vmax={vmax:.0f} > {self.a.max_chunk_vel} deg/s")
                return
        ta = float(c.get("t_anchor", time.time() + 0.02))
        with self.lock:
            if self.welder.t is None:
                self.welder.seed(arr[0], time.time() - 0.05)
            self.welder.weld(arr, dt, ta, blend=self.a.blend)
            self.hold = False

    # ---------- servo loop @ HAL command rate (period_ms=4) ----------
    def _servo_loop(self):
        self.h["enable"] = True
        zero_integ = [0.0] * MAX_JOINTS                       # pure PD: never accumulate
        prev_q = None
        t_prev = None
        hold_target = None
        period = rh.PERIOD_SEC                                 # 0.004 s from controller_params
        dv = max(period, 0.004)                               # finite-difference span for vel ff
        print(f"[servo] running @ {1.0/period:.0f} Hz, lead={self.a.lead*1000:.0f} ms")
        while True:
            t0 = time.time()
            dt = (t0 - t_prev) if t_prev is not None else period
            t_prev = t0
            q, q_vel, _, _, _ = rh._poll_feedback(self.s)
            with self.lock:
                # feed-forward LEAD: command where the reference will be `lead` ahead, so the
                # delayed motion lands on q_ref(now). Welder clamps past its horizon -> holds goal.
                ref = self.welder.sample(t0 + self.a.lead) if not self.hold else None
                v_ref = (self.welder.velocity(t0 + self.a.lead)
                         if (ref is not None and self.a.ff_scale) else None)
            if ref is None:
                if hold_target is None:
                    hold_target = list(q)
                target = hold_target
            else:
                hold_target = None
                target = [float(np.clip(ref[i], q[i] - self.a.max_step, q[i] + self.a.max_step))
                          for i in range(MAX_JOINTS)]           # rate-limit backstop
            next_pos, vel_cmd, _ = rh.pid_solve(q, target, zero_integ, q_vel=q_vel,
                                                prev_q=prev_q, dt=dt)
            # Phase 1: reference-VELOCITY feed-forward. The reactive-only loop is too slow to
            # build up the velocity to track a moving reference (measured: 10 deg/s ramp -> only
            # 3.5 deg/s achieved, ~12 deg lag). Adding q_dot_ref directly drives the velocity so
            # the arm tracks the reference from the start; pid stays only as a small trim.
            # ff_scale maps deg/s -> ctrl.vel_cmd drive units (CALIBRATE live; 0 = old behavior).
            if v_ref is not None:
                # bound the FF input to the legit chunk-velocity range: weld-seam blend transients
                # can spike q_dot_ref far past the trajectory speed and, x ff_scale, trip the drive
                # overspeed fault (seen: poscmd=2771 -> "speed over max limit").
                vr = np.clip(np.asarray(v_ref, float), -self.a.max_chunk_vel, self.a.max_chunk_vel)
                vel_cmd = [vel_cmd[i] + self.a.ff_scale * float(vr[i]) for i in range(MAX_JOINTS)]
            # hard drive-safety clamp on total commanded velocity (prevents ANY overspeed fault)
            vc = self.a.vel_cmd_max
            vel_cmd = [(-vc if v < -vc else vc if v > vc else v) for v in vel_cmd]
            rh._write_hal_cmd(self.h, next_pos, vel_cmd)
            prev_q = q
            slp = period - (time.time() - t0)
            if slp > 0:
                time.sleep(slp)

    def start(self):
        threading.Thread(target=rh._stream_server_thread,
                         args=(self.a.stream_port, self.a.stream_rate), daemon=True).start()
        threading.Thread(target=self._chunk_server, daemon=True).start()
        print("Enabling machine..."); time.sleep(2)
        if not rh.enable_machine(self.h):
            print("Failed to enable machine."); return
        self.s.poll()
        q0 = [round(self.s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
        with self.lock:
            self.welder.seed(np.array(q0, float), time.time() - 0.05)
        print(f"online_servo ready at {q0}; holding until chunks arrive on :{self.a.chunk_port}")
        self._servo_loop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-port", type=int, default=9994)
    ap.add_argument("--stream-port", type=int, default=9999)
    ap.add_argument("--stream-rate", type=float, default=50.0)
    ap.add_argument("--lead", type=float, default=0.0,
                    help="feed-forward dead-time lead (s): sample q_ref(now+lead). Set to the "
                         "MEASURED residual dead-time; 0 = pure follower. Must be < chunk horizon.")
    ap.add_argument("--blend", type=float, default=0.04, help="weld cross-fade window (s)")
    ap.add_argument("--weld-fine-dt", type=float, default=0.01, help="welder reference grid (s)")
    ap.add_argument("--max-step", type=float, default=40.0,
                    help="max target lead over feedback (deg); THIS caps slew: measured peak "
                         "vel ~= 0.33*max_step deg/s (8->3.3, 40->13.6, 60->19.6, linear, no "
                         "drive saturation). Keep <=150 to stay under the ~60 deg/s fault ceiling. "
                         "Was 8 (=>~3 deg/s crawl). Live-tune via {\"set_max_step\":N}.")
    ap.add_argument("--ff-scale", dest="ff_scale", type=float, default=0.0,
                    help="reference-velocity feed-forward scale (deg/s -> ctrl.vel_cmd units); "
                         "0 = reactive-only (old). CALIBRATE live via {\"set_ff_scale\":X}.")
    ap.add_argument("--vel-cmd-max", dest="vel_cmd_max", type=float, default=700.0,
                    help="hard clamp on |ctrl.vel_cmd| (drive units) -- overspeed backstop; the drive "
                         "faulted at ~2771. Live-tune via {\"set_vel_cmd_max\":X}.")
    ap.add_argument("--max-chunk-vel", type=float, default=80.0,
                    help="reject chunks implying more than this per-joint speed (deg/s)")
    a = ap.parse_args()
    srv = StreamFollower(a)
    try:
        srv.start()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            srv.h["enable"] = False
        except Exception:
            pass
        rh.power_off_robot()


if __name__ == "__main__":
    main()
