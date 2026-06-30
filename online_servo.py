#!/usr/bin/env python3
"""online_servo :: streaming SERVO controller for the MyCobot Pro 630 -- run on the Pi
INSTEAD of robot_hal when driving the arm from the ONLINE / streaming planner.

robot_hal converges-to-target per command (~600 ms / 29 loops) so it cannot track a
high-rate stream. This instead WELDS incoming trajectory chunks into a continuous
reference q_ref(t) and SERVOS it at the LinuxCNC servo rate (PID), so the planner can
drive the arm smoothly. The welding happens HERE (on the Pi), so the desktop just
streams chunks -- no 12 Hz window re-targeting through robot_hal's per-cmd loop.

  desktop planner --weld chunks (TCP, JSON lines)--> [:9994] --> q_ref(t)
        servo loop @ servo rate:  pid(q, q_ref(now)) -> HAL joint pos/vel cmd
  feedback streamed on :9999 (same as robot_hal).

Chunk JSON (LinuxCNC deg, the same the desktop already builds):
    {"trajectory": [[6 deg], ...], "traj_dt": float, "t_anchor": abs_wall_seconds}
Safety: chunks implying > --max-chunk-vel are REJECTED; if no chunk arrives for
--watchdog seconds the reference HOLDS the last position; per-step target motion is
clamped to --max-step deg. Clocks are NTP-synced (bridge logs ~1 ms offset).

Run on the Pi (LinuxCNC up, robot_hal NOT running):
    python3 online_servo.py
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


class OnlineServo:
    def __init__(self, a):
        self.a = a
        self.welder = TrajectoryWelder(dof=MAX_JOINTS, fine_dt=0.01)
        self.lock = threading.Lock()
        self.last_chunk_t = 0.0
        self.hold = True
        self.h = hal.component("ctrl")        # same name -> wired to the joints in the HAL config
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

    # ---- chunk intake ----
    def _chunk_server(self):
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.a.chunk_port)); srv.listen(5)
        print(f"[chunks] listening on 0.0.0.0:{self.a.chunk_port}")
        while True:
            conn, addr = srv.accept()
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
                    line, buf = buf.split("\n", 1); line = line.strip()
                    if not line:
                        continue
                    try:
                        c = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if c.get("hold"):
                        with self.lock:
                            self.hold = True
                        continue
                    traj = c.get("trajectory")
                    if not traj or len(traj) < 1:
                        continue
                    arr = np.array(traj, float)
                    dt = float(c.get("traj_dt", 0.01))
                    if arr.shape[0] > 1:               # SAFETY: reject too-fast chunks
                        vmax = float(np.abs(np.diff(arr, axis=0)).max()) / max(dt, 1e-4)
                        if vmax > self.a.max_chunk_vel:
                            print(f"[chunks] REJECT chunk vmax={vmax:.0f} deg/s > {self.a.max_chunk_vel}")
                            continue
                    ta = float(c.get("t_anchor", time.time() + 0.05))
                    with self.lock:
                        if self.welder.t is None:
                            self.welder.seed(arr[0], time.time() - 0.2)
                        self.welder.weld(arr, dt, ta, blend=0.06)
                        self.last_chunk_t = time.time(); self.hold = False
        finally:
            conn.close()

    # ---- servo loop @ servo rate ----
    def _servo_loop(self):
        self.h["enable"] = True
        integral = [0.0] * MAX_JOINTS
        prev_q = None; t_prev = None; hold_target = None
        period = rh.PERIOD_SEC
        print(f"[servo] running @ {1.0/period:.0f} Hz")
        while True:
            t0 = time.time()
            dt = (t0 - t_prev) if t_prev is not None else period
            t_prev = t0
            q, q_vel, _, _, _ = rh._poll_feedback(self.s)
            with self.lock:
                stale = (time.time() - self.last_chunk_t) > self.a.watchdog
                ref = self.welder.sample(time.time()) if (not self.hold and not stale) else None
            if ref is None:                                   # hold last position
                if hold_target is None:
                    hold_target = list(q)
                target = hold_target
            else:
                hold_target = None
                target = [float(np.clip(ref[i], q[i] - self.a.max_step, q[i] + self.a.max_step))
                          for i in range(MAX_JOINTS)]          # backstop clamp
            next_pos, vel_cmd, integral = rh.pid_solve(q, target, integral, q_vel=q_vel,
                                                       prev_q=prev_q, dt=dt)
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
            self.welder.seed(np.array(q0, float), time.time() - 0.2)
        print(f"online_servo ready at {q0}; hold until chunks arrive on :{self.a.chunk_port}")
        self._servo_loop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-port", type=int, default=9994)
    ap.add_argument("--stream-port", type=int, default=9999)
    ap.add_argument("--stream-rate", type=float, default=50.0)
    ap.add_argument("--watchdog", type=float, default=0.6, help="hold if no chunk for this long (s)")
    ap.add_argument("--max-step", type=float, default=8.0, help="max target dev from current per servo step (deg)")
    ap.add_argument("--max-chunk-vel", type=float, default=80.0, help="reject chunks faster than this (deg/s)")
    a = ap.parse_args()
    srv = OnlineServo(a)
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
