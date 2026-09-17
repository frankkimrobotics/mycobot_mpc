#!/usr/bin/env python3
"""ctrl_tuner :: local web GUI to step-test robot_hal's controllers with per-move
parameter overrides and see the response plot live.

    desktop browser --HTTP/JSON--> ctrl_tuner.py --TCP :9998/:9999--> Pi robot_hal.py

Needs the override build of robot_hal.py/controller_solvers.py on the Pi (accepts a
"gains" dict + "period_ms" per command; deployed 2026-09-16). Stdlib only.

    python3 ctrl_tuner.py --robot-host 192.168.50.2   # direct cable; 10.0.0.27 is WiFi --port 8765
    # from a laptop over ssh:  ssh -L 8765:localhost:8765 <desktop>  -> http://localhost:8765

Safety: every step is clamped to the LinuxCNC soft limits and to +-MAX_STEP_DEG,
duration to [0.5, 15] s. "E-STOP" sends SIGINT to robot_hal on the Pi (power off +
brakes) -- after that the stack needs a full clean LinuxCNC restart.
"""
import argparse
import base64
import csv
import io
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from joint_conventions import LINUXCNC_SOFT_LIMITS_DEG, HOME_LINUXCNC_DEG, MAX_JOINTS  # noqa: E402

MAX_STEP_DEG = 30.0
PI_UTC_OFFSET_S = 8 * 3600   # Pi runs in CST (+0800); overridable with --pi-utc-offset-h
RUN_DIR = os.path.join(HERE, "logs", "tuner_runs")
ALLOWED_GAIN_KEYS = {
    "mpc": {"k0", "k1", "vmax", "vel_scale"},
    "pid": {"kp", "kd", "ki", "u_max", "integral_clamp"},
    "pd_velff": {"kp", "kd"},
    "invdyn": set(),
}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Robot:
    """Background TCP clients for the Pi's :9999 stream and :9998 command/status."""

    def __init__(self, host, stream_port, cmd_port):
        self.host, self.stream_port, self.cmd_port = host, stream_port, cmd_port
        self.samples = deque(maxlen=120000)   # (t_recv, t_robot, joints_deg[6], torque[6])
        self.lock = threading.Lock()
        self.stream_ok = False
        self.cmd_ok = False
        self.stream_hz = 0.0
        self.clock_offset = 0.0               # t_recv - t_robot (min over window ~ latency)
        self.status = {}
        self.status_t = 0.0
        self.status_seq = 0
        self.cmd_sock = None
        self.cmd_lock = threading.Lock()
        self.log_reply = None
        self.log_event = threading.Event()
        self.acks = {}                         # tag -> {"t_desktop", "t_recv"}
        threading.Thread(target=self._stream_loop, daemon=True).start()
        threading.Thread(target=self._cmd_loop, daemon=True).start()

    # ---- stream (:9999) ----
    def _stream_loop(self):
        while True:
            try:
                s = socket.create_connection((self.host, self.stream_port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(3.0)
                self.stream_ok = True
                buf = b""
                n, t_win = 0, time.time()
                offs = deque(maxlen=200)
                while True:
                    data = s.recv(65536)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tr = time.time()
                        tj = float(m.get("timestamp", tr))
                        offs.append(tr - tj)
                        with self.lock:
                            self.samples.append((tr, tj, m.get("joints_deg", [0.0] * MAX_JOINTS),
                                                 m.get("torque", [0.0] * MAX_JOINTS)))
                            self.clock_offset = min(offs)
                        n += 1
                        if tr - t_win >= 1.0:
                            self.stream_hz = n / (tr - t_win)
                            n, t_win = 0, tr
            except (OSError, socket.timeout):
                pass
            self.stream_ok = False
            self.stream_hz = 0.0
            time.sleep(1.0)

    # ---- command/status (:9998) ----
    def _cmd_loop(self):
        while True:
            try:
                s = socket.create_connection((self.host, self.cmd_port), timeout=5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(5.0)
                with self.cmd_lock:
                    self.cmd_sock = s
                self.cmd_ok = True
                buf = b""
                while True:
                    data = s.recv(65536)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            m = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        st = m.get("state")
                        if st in ("log", "log_error"):
                            self.log_reply = m
                            self.log_event.set()
                        elif st in ("ack", "ack_chunk"):
                            self.acks[m.get("tag")] = {"t_desktop": time.time(), "t_recv": m.get("t_recv"), "n_ref": m.get("n_ref")}
                        else:
                            self.status = m
                            self.status_t = time.time()
                            self.status_seq += 1
            except (OSError, socket.timeout):
                pass
            with self.cmd_lock:
                self.cmd_sock = None
            self.cmd_ok = False
            time.sleep(1.0)

    def send(self, cmd):
        with self.cmd_lock:
            if self.cmd_sock is None:
                raise RuntimeError("command socket not connected")
            self.cmd_sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))

    def get_log(self, name, timeout=15.0):
        self.log_event.clear()
        self.log_reply = None
        self.send({"get_log": name})
        if not self.log_event.wait(timeout):
            return None
        r = self.log_reply
        if not r or r.get("state") != "log":
            return None
        raw = base64.b64decode(r["log_content_base64"]).decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(raw)))
        if not rows:
            return None
        return {"filename": name, "header": rows[0], "rows": rows[1:]}

    def latest(self):
        with self.lock:
            if not self.samples:
                return None
            return self.samples[-1]

    def since(self, t_desktop, limit=20000):
        """Samples whose desktop-aligned robot time > t_desktop."""
        out = []
        with self.lock:
            off = self.clock_offset
            for tr, tj, q, tq in reversed(self.samples):
                ta = tj + off
                if ta <= t_desktop:
                    break
                out.append((ta, q, tq, tr))
                if len(out) >= limit:
                    break
        out.reverse()
        return out


class Tuner:
    def __init__(self, robot, ssh_host):
        self.robot = robot
        self.ssh_host = ssh_host
        self.run = None
        self.run_lock = threading.Lock()
        os.makedirs(RUN_DIR, exist_ok=True)

    # ---- moves ----
    def _validated_target(self, joint, delta, target):
        cur = self.robot.latest()
        if cur is None:
            raise ValueError("no joint stream yet")
        q = list(cur[2])
        if target is not None:
            tgt = [float(x) for x in target]
            if len(tgt) != MAX_JOINTS:
                raise ValueError("target needs 6 values")
        else:
            j = int(joint)
            if not 0 <= j < MAX_JOINTS:
                raise ValueError("joint out of range")
            d = clamp(float(delta), -MAX_STEP_DEG, MAX_STEP_DEG)
            tgt = list(q)
            tgt[j] = q[j] + d
        for i in range(MAX_JOINTS):
            lo, hi = LINUXCNC_SOFT_LIMITS_DEG[i]
            tgt[i] = clamp(tgt[i], lo, hi)
            if abs(tgt[i] - q[i]) > MAX_STEP_DEG:
                raise ValueError(f"joint {i}: move of {tgt[i]-q[i]:.1f} deg exceeds {MAX_STEP_DEG} deg")
        return q, tgt

    def step(self, body):
        controller = body.get("controller", "mpc")
        if controller not in ALLOWED_GAIN_KEYS:
            raise ValueError("unknown controller")
        gains = body.get("gains") or {}
        gains = {k: v for k, v in gains.items() if k in ALLOWED_GAIN_KEYS[controller]}
        duration = clamp(float(body.get("duration", 4.0)), 0.5, 15.0)
        period_ms = body.get("period_ms")
        period_ms = clamp(float(period_ms), 4.0, 50.0) if period_ms else None
        vel_cmd_max = body.get("vel_cmd_max")
        vel_cmd_max = clamp(float(vel_cmd_max), 10.0, 2000.0) if vel_cmd_max else None
        q0, tgt = self._validated_target(body.get("joint", 0), body.get("delta", 0.0), body.get("target"))
        joint = int(body.get("joint", 0))
        with self.run_lock:
            if self.run and not self.run.get("done"):
                raise ValueError("a run is still in progress")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = {
                "id": stamp, "label": str(body.get("label", ""))[:40], "controller": controller,
                "gains": gains, "period_ms": period_ms, "vel_cmd_max": vel_cmd_max, "joint": joint, "q0": q0, "target": tgt,
                "delta": tgt[joint] - q0[joint], "duration": duration, "t0": time.time(),
                "done": False, "status": None, "csv": None, "stream": [], "clock_offset": self.robot.clock_offset,
            }
            self.run = run
        cmd = {"target_deg": tgt, "duration": duration, "controller": controller, "log_stamp": "tuner_" + stamp, "tag": stamp}
        if gains:
            cmd["gains"] = gains
        if period_ms:
            cmd["period_ms"] = period_ms
        if vel_cmd_max:
            cmd["vel_cmd_max"] = vel_cmd_max
        seq0 = self.robot.status_seq
        run["t0"] = time.time()               # t_send: just before the TCP write
        try:
            self.robot.send(cmd)
        except Exception as e:                # robot down: do not leave a stuck "in progress" run
            run["done"] = True
            run["error"] = str(e)
            raise
        threading.Thread(target=self._watch, args=(run, seq0), daemon=True).start()
        return run["id"]

    def _watch(self, run, seq0):
        t_end = run["t0"] + run["duration"] + 4.0
        done_status = None
        while time.time() < t_end:
            st = self.robot.status
            if run.get("t_moving") is None and self.robot.status_seq > seq0 and st.get("state") == "moving":
                run["t_moving"] = round(self.robot.status_t - run["t0"], 4)   # robot accepted the command
            if self.robot.status_seq > seq0 and st.get("state") == "done" and self.robot.status_t > run["t0"] + 0.2:
                done_status = dict(st)
                break
            time.sleep(0.05)
        t_done_seen = self.robot.status_t if done_status else None
        time.sleep(0.6)   # tail so the settle is visible
        run["status"] = done_status
        raw = self.robot.since(run["t0"] - 0.5)
        run["stream"] = [(round(t - run["t0"], 4), q, tq, round(tr - run["t0"], 4)) for t, q, tq, tr in raw]
        run["latency_ms"] = self._latency(run, raw, done_status, t_done_seen)
        name = (done_status or {}).get("last_log_name")
        if name and str(name).endswith(".csv"):
            try:
                log = self.robot.get_log(name)
                if log:
                    run["csv"] = self._csv_to_series(log, run)
            except Exception as e:  # noqa: BLE001
                run["csv_error"] = str(e)
        run["done"] = True
        path = os.path.join(RUN_DIR, f"{run['id']}_{run['controller']}_j{run['joint']}.json")
        with open(path, "w") as f:
            json.dump(run, f)

    def _latency(self, run, raw, st, t_done_seen):
        """Per-hop latency (ms) for one waypoint: desktop send -> Pi -> drive -> stream -> desktop."""
        off = self.robot.clock_offset            # add to a Pi epoch to get desktop epoch
        t0 = run["t0"]
        j = run["joint"]
        q0 = run["q0"][j]
        ack = self.robot.acks.get(run["id"]) or {}
        st = st or {}
        def pi(key):
            v = st.get(key)
            return (v + off) if isinstance(v, (int, float)) else None
        t_recv, t_deq, t_loop0, t_write0, t_done = pi("t_recv"), pi("t_dequeue"), pi("t_loop0"), pi("t_write0"), pi("t_done")
        if t_recv is None and ack.get("t_recv"):
            t_recv = ack["t_recv"] + off
        motion = next(((ta, tr) for ta, q, tq, tr in raw if abs(q[j] - q0) > 0.2), None)
        ms = lambda a, b: round((a - b) * 1000, 1) if (a is not None and b is not None) else None
        L = {
            "cmd_net_desktop_to_pi": ms(t_recv, t0),
            "ack_round_trip": ms(ack.get("t_desktop"), t0),
            "queue_wait_on_pi": ms(t_deq, t_recv),
            "dequeue_to_first_loop": ms(t_loop0, t_deq),
            "first_loop_to_hal_write": ms(t_write0, t_loop0),
            "hal_write_to_motion_sampled": ms(motion[0] if motion else None, t_write0),
            "stream_net_pi_to_desktop": ms(motion[1] if motion else None, motion[0] if motion else None),
            "total_send_to_motion_seen": ms(motion[1] if motion else None, t0),
            "done_status_lag": ms(t_done_seen, t_done),
            "clock_offset_used": round(off * 1000, 1),
        }
        return L

    def _csv_to_series(self, log, run):
        h = log["header"]
        j = run["joint"]
        idx = {k: h.index(k) for k in ("timestamp", f"q{j}", f"qvel{j}", f"target{j}", f"cmd_pos{j}", f"cmd_vel{j}", "solve_ms") if k in h}
        # robot_hal stamps rows with Pi wall-clock "HH:MM:SS.mmm" (local time, not epoch).
        # Rebuild epoch from the Pi's UTC offset + the day of the run, then align with the
        # stream's measured clock offset. (status "moving" lags up to 1 s: cmd socket recv timeout.)
        def _hms(sv):
            try:
                hh, mm, ss = sv.split(":")
                return int(hh) * 3600 + int(mm) * 60 + float(ss)
            except (ValueError, AttributeError):
                return float(sv)
        tz = PI_UTC_OFFSET_S
        off = run.get("clock_offset", 0.0)                          # desktop - Pi epoch
        day0 = ((run["t0"] - off + tz) // 86400) * 86400          # Pi-local midnight, in Pi-local seconds
        t_first = None
        anchor = 0.0
        out = {"t": [], "q": [], "qvel": [], "target": [], "cmd_pos": [], "cmd_vel": [], "solve_ms": []}
        for r in log["rows"]:
            try:
                ts = _hms(r[idx["timestamp"]])
            except (KeyError, ValueError, IndexError):
                continue
            t_epoch = day0 + ts - tz + off                          # desktop-aligned epoch
            t = t_epoch - run["t0"]
            if t < -3600:                                           # midnight wrap
                t += 86400
            if t_first is None:
                t_first = t
            out["t"].append(round(t, 4))
            for k in ("q", "qvel", "target", "cmd_pos", "cmd_vel"):
                key = f"{k}{j}"
                out[k].append(float(r[idx[key]]) if key in idx else None)
            out["solve_ms"].append(float(r[idx["solve_ms"]]) if "solve_ms" in idx else None)
        out["filename"] = log["filename"]
        return out

    # ---- streamed sinusoid ----
    def stream_sine(self, body):
        import math
        cur = self.robot.latest()
        if cur is None:
            raise ValueError("no joint stream yet")
        with self.run_lock:
            if self.run and not self.run.get("done"):
                raise ValueError("a run is still in progress")
            j = int(body.get("joint", 0))
            amp = clamp(float(body.get("amp", 5.0)), -MAX_STEP_DEG, MAX_STEP_DEG)
            freq = clamp(float(body.get("freq", 0.25)), 0.02, 1.0)
            cycles = clamp(float(body.get("cycles", 2)), 0.5, 10)
            chunk_dt = clamp(float(body.get("chunk_dt", 0.1)), 0.02, 1.0)
            traj_dt = clamp(float(body.get("traj_dt", 0.01)), 0.002, 0.05)
            start_delay = clamp(float(body.get("start_delay", 0.3)), 0.05, 2.0)
            gains = {k: float(v) for k, v in (body.get("gains") or {}).items() if k in ("k0", "k1", "vmax", "vel_scale", "vff", "lead")}
            period_ms = body.get("period_ms")
            vel_cmd_max = clamp(float(body.get("vel_cmd_max", 900)), 10, 2000)
            q0 = list(cur[2])
            lo, hi = LINUXCNC_SOFT_LIMITS_DEG[j]
            if not (lo <= q0[j] <= hi and lo <= q0[j] + amp <= hi):
                raise ValueError("sine exceeds soft limits")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            T = cycles / freq
            run = {"id": stamp, "type": "stream", "label": str(body.get("label", ""))[:40], "controller": "stream",
                   "gains": gains, "period_ms": period_ms, "vel_cmd_max": vel_cmd_max, "joint": j, "q0": q0, "target": q0,
                   "delta": amp, "duration": T, "amp": amp, "freq": freq, "cycles": cycles, "chunk_dt": chunk_dt,
                   "traj_dt": traj_dt, "t0": time.time(), "done": False, "status": None, "csv": None, "stream": [],
                   "ref": [], "chunks": [], "clock_offset": self.robot.clock_offset}
            self.run = run
        def qref6(tau):
            tau = min(max(tau, 0.0), T)
            q = list(q0); q[j] = q0[j] + amp * (1.0 - math.cos(2 * math.pi * f * tau)) / 2.0
            return q
        threading.Thread(target=self._stream_thread, args=(run, start_delay, qref6), daemon=True).start()
        return run["id"]

    def stream_traj(self, body):
        """Stream an arbitrary joint trajectory (LinuxCNC deg, uniform dt), e.g. a cuRobo plan."""
        import math
        cur = self.robot.latest()
        if cur is None:
            raise ValueError("no joint stream yet")
        traj = [[float(x) for x in w] for w in body["traj_deg"]]
        dt_in = float(body["dt"])
        if len(traj) < 2 or any(len(w) != MAX_JOINTS for w in traj):
            raise ValueError("traj_deg must be N x 6")
        q_now = list(cur[2])
        if max(abs(traj[0][i] - q_now[i]) for i in range(MAX_JOINTS)) > 2.0:
            raise ValueError("trajectory does not start at the current pose (>2 deg off)")
        for w in traj:
            for i in range(MAX_JOINTS):
                lo, hi = LINUXCNC_SOFT_LIMITS_DEG[i]
                if not lo <= w[i] <= hi:
                    raise ValueError(f"waypoint outside soft limits on joint {i}")
        max_vel = clamp(float(body.get("max_vel_deg", 40.0)), 1.0, 60.0)
        peak = max(abs(traj[k + 1][i] - traj[k][i]) / dt_in for k in range(len(traj) - 1) for i in range(MAX_JOINTS))
        scale = max(1.0, peak / max_vel)                        # only ever slows down
        dt = dt_in * scale
        T = dt * (len(traj) - 1)
        with self.run_lock:
            if self.run and not self.run.get("done"):
                raise ValueError("a run is still in progress")
            gains = {k: float(v) for k, v in (body.get("gains") or {}).items() if k in ("k0", "k1", "vmax", "vel_scale", "vff", "lead")}
            chunk_dt = clamp(float(body.get("chunk_dt", 0.1)), 0.02, 1.0)
            traj_dt = clamp(float(body.get("traj_dt", 0.01)), 0.002, 0.05)
            start_delay = clamp(float(body.get("start_delay", 0.3)), 0.05, 2.0)
            # plot joint = the one that moves most
            j = max(range(MAX_JOINTS), key=lambda i: abs(traj[-1][i] - traj[0][i]))
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run = {"id": stamp, "type": "stream", "label": str(body.get("label", "traj"))[:40], "controller": "stream",
                   "gains": gains, "period_ms": body.get("period_ms"), "vel_cmd_max": clamp(float(body.get("vel_cmd_max", 900)), 10, 2000),
                   "joint": j, "q0": q_now, "target": traj[-1], "delta": traj[-1][j] - traj[0][j], "duration": T,
                   "chunk_dt": chunk_dt, "traj_dt": traj_dt, "time_scale": scale, "peak_vel_planned": peak, "peak_vel_scaled": peak / scale,
                   "t0": time.time(), "done": False, "status": None, "csv": None, "stream": [], "ref": [], "chunks": [],
                   "clock_offset": self.robot.clock_offset, "traj_in": traj, "dt_in": dt_in}
            self.run = run
        def qref6(tau):
            tau = min(max(tau, 0.0), T)
            x = tau / dt; k = min(int(x), len(traj) - 2); a = x - k
            return [traj[k][i] + a * (traj[k + 1][i] - traj[k][i]) for i in range(MAX_JOINTS)]
        threading.Thread(target=self._stream_thread, args=(run, start_delay, qref6), daemon=True).start()
        return run["id"]

    def _stream_thread(self, run, start_delay, qref6):
        j, T = run["joint"], run["duration"]
        chunk_dt, traj_dt, q0 = run["chunk_dt"], run["traj_dt"], run["q0"]
        n_chunks = int(math.ceil(T / chunk_dt)) if T > 0 else 1
        n_pts = int(round(chunk_dt / traj_dt)) + 1               # one point overlap with the next chunk
        T0 = time.time() + start_delay                            # absolute start of the reference (desktop epoch ≈ Pi epoch)
        run["t0"] = T0
        off = self.robot.clock_offset
        qref = lambda tau: qref6(tau)[j]
        seq0 = self.robot.status_seq
        for k in range(n_chunks):
            t_anchor = T0 + k * chunk_dt
            pts = [[round(x, 4) for x in qref6(k * chunk_dt + i * traj_dt)] for i in range(n_pts)]
            tag = f"{run['id']}:{k}"
            cmd = {"chunk": pts, "traj_dt": traj_dt, "t_anchor": t_anchor - off, "seq": k, "tag": tag,
                   "gains": run["gains"], "period_ms": run["period_ms"], "vel_cmd_max": run["vel_cmd_max"],
                   "log_stamp": "tuner_" + run["id"]}
            # send chunk k one chunk-period ahead of its anchor
            t_send_target = t_anchor - chunk_dt - 0.05
            d = t_send_target - time.time()
            if d > 0:
                time.sleep(d)
            t_send = time.time()
            try:
                self.robot.send(cmd)
            except Exception as e:  # noqa: BLE001
                run["error"] = str(e); break
            run["chunks"].append({"seq": k, "t_send": round(t_send - T0, 4), "t_anchor": round(t_anchor - T0, 4), "tag": tag})
        # wait for the Pi to report done (state done + done_reason end) or timeout
        t_end = T0 + T + 4.0
        done_status = None
        while time.time() < t_end:
            st = self.robot.status
            if self.robot.status_seq > seq0 and st.get("state") == "done" and st.get("controller") == "stream" and self.robot.status_t > T0:
                done_status = dict(st); break
            time.sleep(0.05)
        t_done_seen = self.robot.status_t if done_status else None
        time.sleep(0.4)
        run["status"] = done_status
        raw = self.robot.since(T0 - 0.5)
        run["stream"] = [(round(t - T0, 4), q, tq, round(tr - T0, 4)) for t, q, tq, tr in raw]
        run["ref"] = [(round(tau, 3), round(qref(tau), 4)) for tau in [i * 0.01 for i in range(int(T / 0.01) + 1)]]
        run["ref6"] = [(round(tau, 3), [round(v, 4) for v in qref6(tau)]) for tau in [i * 0.02 for i in range(int(T / 0.02) + 1)]]
        # per-chunk latency from acks
        for c in run["chunks"]:
            a = self.robot.acks.get(c["tag"]) or {}
            if a.get("t_recv") is not None:
                c["net_ms"] = round((a["t_recv"] + off - (c["t_send"] + T0)) * 1000, 1)
                c["ack_rtt_ms"] = round((a["t_desktop"] - (c["t_send"] + T0)) * 1000, 1)
                c["margin_ms"] = round((c["t_anchor"] - (a["t_recv"] + off - T0)) * 1000, 1)   # how early the chunk arrived
        name = (done_status or {}).get("last_log_name")
        if name and str(name).endswith(".csv"):
            try:
                log = self.robot.get_log(name)
                if log:
                    run["csv"] = self._csv_to_series(log, run)
            except Exception as e:  # noqa: BLE001
                run["csv_error"] = str(e)
        run["tracking"] = self._tracking(run)
        run["latency_ms"] = self._latency(run, raw, done_status, t_done_seen)
        run["done"] = True
        path = os.path.join(RUN_DIR, f"{run['id']}_stream_j{run['joint']}.json")
        with open(path, "w") as fh:
            json.dump(run, fh)

    def _tracking(self, run):
        """Lag (ms) that best aligns measured q with the reference, RMS/max error raw and lag-compensated."""
        j = run["joint"]
        S = [(t, q[j]) for t, q, tq, tr in run["stream"] if 0 <= t <= run["duration"]]
        if len(S) < 10:
            return None
        import bisect as bs
        rt = [r[0] for r in run["ref"]]; rq = [r[1] for r in run["ref"]]
        def ref_at(t):
            if t <= rt[0]: return rq[0]
            if t >= rt[-1]: return rq[-1]
            i = bs.bisect_right(rt, t) - 1; a = (t - rt[i]) / (rt[i + 1] - rt[i]); return rq[i] + a * (rq[i + 1] - rq[i])
        def rms(lag):
            e = [(q - ref_at(t - lag)) for t, q in S]
            return (sum(x * x for x in e) / len(e)) ** 0.5, max(abs(x) for x in e)
        best = min(((rms(l)[0], l) for l in [i * 0.005 for i in range(0, 121)]), key=lambda x: x[0])
        r0, m0 = rms(0.0); rb, mb = rms(best[1])
        chunks = [c for c in run["chunks"] if "net_ms" in c]
        return {"lag_ms": round(best[1] * 1000, 1), "rms_err_deg": round(r0, 3), "max_err_deg": round(m0, 3),
                "rms_err_lag_comp_deg": round(rb, 3), "max_err_lag_comp_deg": round(mb, 3),
                "chunks_sent": len(run["chunks"]), "chunks_acked": len(chunks),
                "chunk_net_ms_mean": round(sum(c["net_ms"] for c in chunks) / len(chunks), 1) if chunks else None,
                "chunk_net_ms_max": max(c["net_ms"] for c in chunks) if chunks else None,
                "chunk_margin_ms_min": min(c["margin_ms"] for c in chunks) if chunks else None}

    def hold(self):
        cur = self.robot.latest()
        if cur is None:
            raise ValueError("no joint stream")
        self.robot.send({"target_deg": list(cur[2]), "duration": 0.3, "controller": "pid"})
        return list(cur[2])

    def estop(self):
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", self.ssh_host,
                            "pkill -INT -f robot_hal.py; echo sent"], capture_output=True, text=True, timeout=15)
        return (r.stdout + r.stderr).strip()

    def list_runs(self):
        out = []
        for fn in sorted(os.listdir(RUN_DIR), reverse=True)[:200]:
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(RUN_DIR, fn)) as f:
                    r = json.load(f)
                out.append({k: r.get(k) for k in ("id", "label", "controller", "gains", "period_ms", "joint", "delta", "duration", "done")})
            except Exception:  # noqa: BLE001
                continue
        return out

    def load_run(self, rid):
        rid = os.path.basename(rid)
        for fn in os.listdir(RUN_DIR):
            if fn.startswith(rid) and fn.endswith(".json"):
                with open(os.path.join(RUN_DIR, fn)) as f:
                    return json.load(f)
        return None


class Handler(BaseHTTPRequestHandler):
    tuner = None
    html_path = os.path.join(HERE, "ctrl_tuner.html")

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        t, r = self.tuner, self.tuner.robot
        path, _, qs = self.path.partition("?")
        params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
        if path in ("/", "/index.html"):
            with open(self.html_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/status":
            cur = r.latest()
            run = t.run
            self._json({
                "live": True, "stream_ok": r.stream_ok, "cmd_ok": r.cmd_ok, "stream_hz": round(r.stream_hz, 1),
                "clock_offset_ms": round(r.clock_offset * 1000, 1),
                "q": cur[2] if cur else None, "torque": cur[3] if cur else None,
                "age_s": round(time.time() - cur[0], 2) if cur else None,
                "status": r.status, "status_age_s": round(time.time() - r.status_t, 1) if r.status_t else None,
                "run": {k: run[k] for k in ("id", "done", "joint", "target", "q0", "t0", "controller", "gains", "duration")} if run else None,
                "limits": LINUXCNC_SOFT_LIMITS_DEG, "home": HOME_LINUXCNC_DEG, "max_step": MAX_STEP_DEG,
                "now": time.time(),
            })
        elif path == "/api/stream":
            since = float(params.get("since", time.time() - 5))
            self._json({"samples": [(round(ta, 4), q, tq) for ta, q, tq, _tr in r.since(since, 3000)], "now": time.time()})
        elif path == "/api/run":
            run = t.run
            if run is None:
                self._json({"run": None})
            else:
                self._json({"run": run})
        elif path == "/api/runs":
            self._json({"runs": t.list_runs()})
        elif path == "/api/run_file":
            self._json({"run": t.load_run(params.get("id", ""))})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        t = self.tuner
        try:
            if self.path == "/api/step":
                self._json({"ok": True, "id": t.step(body)})
            elif self.path == "/api/stream_sine":
                self._json({"ok": True, "id": t.stream_sine(body)})
            elif self.path == "/api/stream_traj":
                self._json({"ok": True, "id": t.stream_traj(body)})
            elif self.path == "/api/hold":
                self._json({"ok": True, "target": t.hold()})
            elif self.path == "/api/estop":
                self._json({"ok": True, "result": t.estop()})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self._json({"ok": False, "error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-host", default="192.168.50.2", help="Pi over the direct cable (10.0.0.27 = WiFi, 50-125 ms spikes)")
    ap.add_argument("--stream-port", type=int, default=9999)
    ap.add_argument("--cmd-port", type=int, default=9998)
    ap.add_argument("--ssh-host", default="pi@192.168.50.2", help="for the E-STOP (SIGINT robot_hal)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--pi-utc-offset-h", type=float, default=8.0, help="Pi timezone offset (hours) for CSV wall-clock stamps")
    a = ap.parse_args()
    global PI_UTC_OFFSET_S
    PI_UTC_OFFSET_S = a.pi_utc_offset_h * 3600
    robot = Robot(a.robot_host, a.stream_port, a.cmd_port)
    Handler.tuner = Tuner(robot, a.ssh_host)
    srv = ThreadingHTTPServer((a.bind, a.port), Handler)
    print(f"ctrl_tuner: http://{a.bind}:{a.port}  (robot {a.robot_host}:{a.cmd_port}/{a.stream_port})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
