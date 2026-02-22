#!/usr/bin/env python3
"""
PD controller with acceleration integration (same HAL architecture as mpc_hal.py).

This is a PD law, not rigid-body inverse dynamics (no mass/inertia matrix). The law is:
  - Desired acceleration: q̈_d = Kp*e - Kd*q̇ (deg/s²), clipped
  - Integrate: next_pos = q + q̇*dt + 0.5*q̈_d*dt²,  vel_cmd = q̇ + q̈_d*dt

So it is PD (Kp*e - Kd*q̇); the only difference from the other PD path (pd_solve) is
that here we integrate acceleration to get position and velocity setpoints, instead of
applying a direct position step. Both paths output pos_cmd and vel_cmd to HAL; the
downstream PIDs/pro600 turn those into torques.

Writes to HAL pins joint{i}_pos_cmd, joint{i}_vel_cmd. Requires HAL: load invdyn
component and wire invdyn.jointN_pos_cmd to pid.N.command (via mux_generic when
invdyn.enable=1). See elerob_invdyn.hal.
"""

import base64
import queue
import time
import sys
import csv
import json
import os
import socket
import threading
import glob
from datetime import datetime
from functools import wraps
import numpy as np
import hal

try:
    from invdyn_model import (
        load_params as load_invdyn_params,
        linuxcnc_deg_to_rad,
        NUM_JOINTS as MODEL_NUM_JOINTS,
    )
except ImportError:
    load_invdyn_params = None
    linuxcnc_deg_to_rad = None
    MODEL_NUM_JOINTS = 6

try:
    import linuxcnc
    import hal
except ImportError as e:
    print(f"Import error: {e}. Need linuxcnc and hal (run with LinuxCNC).")
    sys.exit(1)

MAX_JOINTS = 6
# InvDyn loop period. With 2 ms the position step 0.5*qdd*dt² is ~0.0003° per step, below
# motor resolution/deadband so the robot doesn't move. Use 20 ms so the step is ~0.03° and
# vel_cmd builds (3 deg/s per step); robot then tracks. PD uses position steps up to 8°.
INVDYN_PERIOD_MS = 20  # 50 Hz
# InvDyn: q̈_d = Kp*e - Kd*q̇ (deg/s²). Clip acceleration for safety.
INVDYN_KP = 144.0      # 1/s² → ωn=12 rad/s in rad; in deg: 144 (deg/s² per deg error)
INVDYN_KD = 24.0      # 1/s
QDD_MAX_DEG = 150.0   # max |q̈_d| deg/s²

# Model-based invdyn: PD gains in Nm/rad, Nm/(rad/s) when params loaded
INVDYN_KP_NM = 5.0
INVDYN_KD_NM = 1.0

# PD fallback (same as mpc_hal)
U_MAX_PER_STEP = 8.0
KP_PD = 0.5
KD_PD = 0.1

SUCTION_PIN = "pro600.digital_out00"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
_timing = {"poll": [], "invdyn_solve": [], "pd_solve": [], "hal_write": [], "sleep": []}

# Loaded by main() when --params points to a valid npz; used by invdyn_control_loop for model-based solve
_invdyn_params = None


def timed(step_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            _timing[step_name].append(elapsed_ms)
            return result
        return wrapper
    return decorator


@timed("invdyn_solve")
def invdyn_solve(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """PD in acceleration: q̈_d = Kp*e - Kd*q̇; integrate to next_pos, vel_cmd."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    # Desired acceleration (deg/s²)
    qdd_d = INVDYN_KP * error - INVDYN_KD * velocity
    qdd_d = np.clip(qdd_d, -QDD_MAX_DEG, QDD_MAX_DEG)
    if dt is None or dt <= 0:
        dt = INVDYN_PERIOD_MS / 1000.0
    next_pos = current + velocity * dt + 0.5 * qdd_d * (dt ** 2)
    vel_cmd = velocity + qdd_d * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


@timed("invdyn_solve")
def invdyn_model_solve(q, target_angles, params, q_vel=None, prev_q=None, dt=None):
    """Drake-aligned: vd_command = Kp*(q_d - q) - Kd*qd (rad/s²); qdd_d = vd_command; integrate to next_pos, vel_cmd. No M/C/G solve."""
    if load_invdyn_params is None or linuxcnc_deg_to_rad is None:
        return invdyn_solve(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    if q_vel is not None:
        velocity_deg = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        velocity_deg = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity_deg = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = INVDYN_PERIOD_MS / 1000.0
    # LinuxCNC deg -> URDF rad
    q_rad = linuxcnc_deg_to_rad(current)
    target_rad = linuxcnc_deg_to_rad(target)
    qd_rad = np.deg2rad(velocity_deg)
    # Drake: PID outputs desired acceleration (rad/s²). Kp=1/s², Kd=1/s.
    vd_command_rad = INVDYN_KP * (target_rad - q_rad) - INVDYN_KD * qd_rad
    qdd_d_rad = vd_command_rad
    # rad/s² -> deg/s², clip, integrate
    deg_per_rad = 180.0 / np.pi
    qdd_d_deg = qdd_d_rad * deg_per_rad
    qdd_d_deg = np.clip(qdd_d_deg, -QDD_MAX_DEG, QDD_MAX_DEG)
    next_pos = current + velocity_deg * dt + 0.5 * qdd_d_deg * (dt ** 2)
    vel_cmd = velocity_deg + qdd_d_deg * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


@timed("pd_solve")
def pd_solve(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """PD controller (fallback): same as mpc_hal."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    u_opt = KP_PD * error - KD_PD * velocity
    u_opt = np.clip(u_opt, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_pos = current + u_opt
    vel_cmd = (u_opt / dt * 0.5) if dt and dt > 0 else np.zeros(MAX_JOINTS)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


# Use PD solver for "invdyn" path so the robot actually moves. invdyn_solve integrates
# acceleration and produces tiny position steps on this hardware; pd_solve uses direct
# position steps (like mpc_hal PD) and works. Set to False to try invdyn_solve again.
INVDYN_USE_PD_SOLVE = True


def invdyn_control_loop(h, s, target_angles, duration_sec=10.0,
                        pos_tol=0.5, vel_tol=1.0, settle_steps=10, controller="invdyn"):
    """Run control loop: poll -> (invdyn_model_solve, pd_solve, or invdyn_solve) -> HAL write."""
    use_model = _invdyn_params is not None
    use_pd = INVDYN_USE_PD_SOLVE and not use_model
    print("InvDyn control loop starting. Target:", target_angles)
    if use_model:
        print("  Using model-based invdyn (M,C,G from npz).")
    elif use_pd:
        print("  Using pd_solve (direct position steps) so robot moves.")
    else:
        print(f"  Kp={INVDYN_KP} Kd={INVDYN_KD} qdd_max={QDD_MAX_DEG} deg/s²")
    print(f"  Early stop: pos_tol={pos_tol}° vel_tol={vel_tol}°/s settle={settle_steps} steps")
    print("Press Ctrl+C to stop.\n")
    h["enable"] = True
    t_start = time.time()
    loop_count = 0
    converged_count = 0
    prev_current = None
    t_prev = None
    for k in _timing:
        _timing[k] = []
    log_rows = []
    period = INVDYN_PERIOD_MS / 1000.0

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else period
        t_prev = t_loop_start

        q, q_vel, hal_vel, hal_torq, t_status = _poll_feedback(s)
        if use_model:
            next_pos, vel_cmd = invdyn_model_solve(q, target_angles, _invdyn_params, q_vel=q_vel, prev_q=prev_current, dt=dt)
        elif use_pd:
            next_pos, vel_cmd = pd_solve(q, target_angles, q_vel=q_vel, prev_q=prev_current, dt=dt)
        else:
            next_pos, vel_cmd = invdyn_solve(q, target_angles, q_vel=q_vel, prev_q=prev_current, dt=dt)
        u = [next_pos[i] - q[i] for i in range(MAX_JOINTS)]
        if prev_current is not None and dt and dt > 0:
            prev = prev_current
            est_vel = [(q[i] - prev[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()

        _write_hal_cmd(h, next_pos, vel_cmd)
        elapsed = time.time() - t_loop_start
        sleep_time = period - elapsed
        if sleep_time > 0:
            t0 = time.perf_counter()
            time.sleep(sleep_time)
            _timing["sleep"].append((time.perf_counter() - t0) * 1000)
        else:
            _timing["sleep"].append(0.0)

        loop_count += 1
        err = sum((t - a) ** 2 for t, a in zip(target_angles, q)) ** 0.5
        vel_norm = sum(v ** 2 for v in vel_cmd) ** 0.5
        if err < pos_tol and vel_norm < vel_tol:
            converged_count += 1
            if converged_count >= settle_steps:
                print(f"  Converged at loop {loop_count}: err={err:.3f}° vel_norm={vel_norm:.3f}°/s")
                break
        else:
            converged_count = 0

        solve_key = "invdyn_solve" if (use_model or not use_pd) else "pd_solve"
        log_rows.append([
            controller, t_status, loop_count,
            *q, *est_vel, *target_angles, *next_pos, *vel_cmd,
            *u, *hal_vel, *hal_torq,
            err,
            _timing["poll"][-1], _timing[solve_key][-1],
            _timing["hal_write"][-1], _timing["sleep"][-1],
        ])
        if loop_count % 10 == 0 or loop_count <= 3:
            print(f"Loop {loop_count}: q={q[:3]}... err={err:.3f} vcmd={[round(v, 1) for v in vel_cmd[:3]]}")

    h["enable"] = False
    print(f"\nDone. Ran {loop_count} InvDyn iterations.")
    _save_log(log_rows, target_angles, controller)


def pd_control_loop(h, s, target_angles, duration_sec=10.0,
                    pos_tol=0.5, vel_tol=1.0, settle_steps=10, controller="pd"):
    """Run PD loop (fallback): poll -> pd_solve -> HAL write."""
    print("PD control loop starting. Target:", target_angles)
    h["enable"] = True
    t_start = time.time()
    loop_count = 0
    converged_count = 0
    prev_current = None
    t_prev = None
    for k in _timing:
        _timing[k] = []
    log_rows = []
    period = INVDYN_PERIOD_MS / 1000.0

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else period
        t_prev = t_loop_start
        q, _q_vel, hal_vel, hal_torq, t_status = _poll_feedback(s)
        next_pos, vel_cmd = pd_solve(q, target_angles, q_vel=None, prev_q=prev_current, dt=dt)
        u = [next_pos[i] - q[i] for i in range(MAX_JOINTS)]
        if prev_current is not None and dt and dt > 0:
            prev = prev_current
            est_vel = [(q[i] - prev[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()
        _write_hal_cmd(h, next_pos, vel_cmd)
        elapsed = time.time() - t_loop_start
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
            _timing["sleep"].append(sleep_time * 1000)
        else:
            _timing["sleep"].append(0.0)
        loop_count += 1
        err = sum((t - a) ** 2 for t, a in zip(target_angles, q)) ** 0.5
        vel_norm = sum(v ** 2 for v in vel_cmd) ** 0.5
        if err < pos_tol and vel_norm < vel_tol:
            converged_count += 1
            if converged_count >= settle_steps:
                break
        else:
            converged_count = 0
        log_rows.append([
            controller, t_status, loop_count, *q, *est_vel, *target_angles, *next_pos, *vel_cmd,
            *u, *hal_vel, *hal_torq,
            err, _timing["poll"][-1], _timing["pd_solve"][-1],
            _timing["hal_write"][-1], _timing["sleep"][-1],
        ])
    h["enable"] = False
    print(f"Done. Ran {loop_count} PD iterations.")
    _save_log(log_rows, target_angles, controller)


def _read_hal_feedback():
    """Read velocity and torque from pro600 HAL pins. Returns (hal_vel, hal_torq) lists."""
    hal_vel = [0.0] * MAX_JOINTS
    hal_torq = [0.0] * MAX_JOINTS
    for i in range(MAX_JOINTS):
        try:
            hal_vel[i] = float(hal.get_value(f"pro600.joint{i}_velfb"))
        except (NameError, TypeError, ValueError, KeyError):
            pass
        try:
            hal_torq[i] = float(hal.get_value(f"pro600.joint{i}_torqfb"))
        except (NameError, TypeError, ValueError, KeyError):
            pass
    return hal_vel, hal_torq


@timed("poll")
def _poll_feedback(s):
    s.poll()
    t_status = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    q = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    q_vel = None
    try:
        q_vel = [s.joint[i]["velocity"] for i in range(MAX_JOINTS)]
    except (KeyError, TypeError, IndexError):
        pass
    hal_vel, hal_torq = _read_hal_feedback()
    return q, q_vel, hal_vel, hal_torq, t_status


@timed("hal_write")
def _write_hal_cmd(h, next_pos, vel_cmd):
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = next_pos[i]
        h[f"joint{i}_vel_cmd"] = vel_cmd[i]
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


_last_log_filename = None  # basename of last saved CSV (for desktop fetch)
_desktop_log_stamp = None  # optional YYYYMMDD_HHMMSS from desktop (Raspi clock may be wrong)


def _save_log(log_rows, target_angles, controller="invdyn"):
    global _last_log_filename, _desktop_log_stamp
    if not log_rows:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = _desktop_log_stamp if _desktop_log_stamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    _desktop_log_stamp = None  # use once per save
    target_str = "_".join(str(int(a)) for a in target_angles)
    filename = os.path.join(LOG_DIR, f"invdyn_{stamp}_t{target_str}.csv")
    header = (
        ["controller", "timestamp", "loop"]
        + [f"q{i}" for i in range(MAX_JOINTS)] + [f"qvel{i}" for i in range(MAX_JOINTS)]
        + [f"target{i}" for i in range(MAX_JOINTS)] + [f"cmd_pos{i}" for i in range(MAX_JOINTS)]
        + [f"cmd_vel{i}" for i in range(MAX_JOINTS)]
        + [f"u{i}" for i in range(MAX_JOINTS)]
        + [f"hal_vel{i}" for i in range(MAX_JOINTS)] + [f"hal_torq{i}" for i in range(MAX_JOINTS)]
        + ["err_norm", "poll_ms", "invdyn_solve_ms", "hal_write_ms", "sleep_ms"]
    )
    with open(filename, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(log_rows)
    _last_log_filename = os.path.basename(filename)
    print(f"  Log saved: {filename}")


import subprocess


def _halcmd_set(pin, value):
    os.system(f"halcmd setp {pin} {value}")


def _halcmd_get(pin):
    result = subprocess.run(["halcmd", "getp", pin], capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _halcmd_get_bool(pin):
    return _halcmd_get(pin) == "TRUE"


def _halcmd_get_float(pin):
    try:
        return float(_halcmd_get(pin))
    except (ValueError, TypeError):
        return 0.0


def power_on_robot():
    print("  Powering on robot via pro600.poweron...")
    _halcmd_set("pro600.poweron", 1)
    time.sleep(1)
    for _ in range(20):
        if _halcmd_get_bool("pro600.svr_poweroned"):
            break
        time.sleep(1)
    for _ in range(20):
        if _halcmd_get_bool("pro600.svr_enabled"):
            break
        time.sleep(1)
    print("  Robot powered on and servos enabled!")
    return True


def suction_pump(on=True):
    _halcmd_set(SUCTION_PIN, 1 if on else 0)


def power_off_robot():
    print("Powering off robot...")
    _halcmd_set("invdyn.enable", 0)
    time.sleep(0.1)
    _halcmd_set("pro600.poweron", 0)
    time.sleep(1)
    try:
        c = linuxcnc.command()
        c.state(linuxcnc.STATE_ESTOP)
    except Exception:
        pass
    print("  Robot powered off.")


def wait_for_stable_feedback(settle_time=3.0, check_interval=0.5):
    time.sleep(settle_time)
    time.sleep(check_interval)
    fb2 = [_halcmd_get_float(f"pro600.joint{i}_posfb") for i in range(MAX_JOINTS)]
    return fb2


def preload_and_enable_invdyn(h, feedback):
    print("  Pre-loading InvDyn commands with stable feedback...")
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = feedback[i]
        h[f"joint{i}_vel_cmd"] = 0.0
    time.sleep(0.02)
    h["enable"] = True
    print("  invdyn.enable = TRUE")


def enable_machine(h, timeout=60.0):
    c = linuxcnc.command()
    s = linuxcnc.stat()
    for _ in range(5):
        s.poll()
        if s.task_state == linuxcnc.STATE_ESTOP:
            c.state(linuxcnc.STATE_ESTOP_RESET)
            time.sleep(0.5)
        else:
            break
    power_on_robot()
    t0 = time.time()
    while (time.time() - t0) < timeout:
        s.poll()
        if s.task_state == linuxcnc.STATE_ESTOP:
            c.state(linuxcnc.STATE_ESTOP_RESET)
            time.sleep(0.5)
        elif s.task_state == linuxcnc.STATE_ESTOP_RESET:
            if _halcmd_get_bool("motion.enable"):
                c.state(linuxcnc.STATE_ON)
            time.sleep(1)
        elif s.task_state in (linuxcnc.STATE_OFF, linuxcnc.STATE_ON):
            if s.task_state == linuxcnc.STATE_OFF:
                c.state(linuxcnc.STATE_ON)
                time.sleep(0.5)
            else:
                _halcmd_set("or2.0.in1", 1)
                c.mode(linuxcnc.MODE_MANUAL)
                c.wait_complete()
                c.teleop_enable(0)
                c.wait_complete()
                for j in range(MAX_JOINTS):
                    if not s.joint[j]["homed"]:
                        c.home(j)
                time.sleep(0.5)
                feedback = wait_for_stable_feedback(settle_time=3.0)
                preload_and_enable_invdyn(h, feedback)
                return True
        time.sleep(0.5)
    return False


# ── Streaming server ─────────────────────────────────────────────────────
STREAM_PORT = 9999
STREAM_RATE_HZ = 50.0


def _stream_server_thread(port: int, rate_hz: float):
    stat = linuxcnc.stat()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(5)
    print(f"[stream] Listening on 0.0.0.0:{port} at {rate_hz} Hz")
    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            try:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with clients_lock:
                    clients.append(conn)
            except OSError:
                break
    threading.Thread(target=accept_loop, daemon=True).start()
    period = 1.0 / rate_hz
    while True:
        t0 = time.time()
        try:
            stat.poll()
            joints_deg = [round(stat.joint_actual_position[i], 4) for i in range(MAX_JOINTS)]
        except (RuntimeError, OSError):
            time.sleep(1.0)
            continue
        msg = json.dumps({"joints_deg": joints_deg, "timestamp": time.time()}) + "\n"
        dead = []
        with clients_lock:
            for conn in clients:
                try:
                    conn.sendall(msg.encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(conn)
            for conn in dead:
                clients.remove(conn)
                conn.close()
        elapsed = time.time() - t0
        if period - elapsed > 0:
            time.sleep(period - elapsed)


def start_stream_server(port=STREAM_PORT, rate_hz=STREAM_RATE_HZ):
    t = threading.Thread(target=_stream_server_thread, args=(port, rate_hz), daemon=True)
    t.start()
    return t


# ── Command server ───────────────────────────────────────────────────────
CMD_PORT = 9998
_cmd_queue = queue.Queue()
_cmd_status = {
    "state": "idle",
    "current_deg": [0.0] * MAX_JOINTS,
    "target_deg": [0.0] * MAX_JOINTS,
    "error_norm": 0.0,
    "last_log_name": None,
}
_cmd_status_lock = threading.Lock()


def _handle_cmd_client(conn, addr):
    _ = addr
    print(f"[cmd] Client connected: {addr}")
    buffer = ""
    conn.settimeout(1.0)
    try:
        while True:
            with _cmd_status_lock:
                status = dict(_cmd_status)
            try:
                conn.sendall((json.dumps(status) + "\n").encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                        if "target_deg" in cmd:
                            _cmd_queue.put(cmd)
                            conn.sendall((json.dumps({"state": "ack", "target_deg": cmd["target_deg"]}) + "\n").encode("utf-8"))
                        elif "get_log" in cmd:
                            name = cmd.get("get_log")
                            if name and isinstance(name, str):
                                name = os.path.basename(name)
                                if name.endswith(".csv"):
                                    path = os.path.join(LOG_DIR, name)
                                    if os.path.isfile(path):
                                        with open(path, "rb") as f:
                                            raw = f.read()
                                        payload = base64.b64encode(raw).decode("ascii")
                                        conn.sendall((json.dumps({
                                            "state": "log", "filename": name,
                                            "log_content_base64": payload,
                                        }) + "\n").encode("utf-8"))
                                    else:
                                        conn.sendall((json.dumps({"state": "log_error", "error": "file_not_found"}) + "\n").encode("utf-8"))
                                else:
                                    conn.sendall((json.dumps({"state": "log_error", "error": "bad_filename"}) + "\n").encode("utf-8"))
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                pass
            time.sleep(0.01)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _command_server_thread(port: int):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(2)
    print(f"[cmd] Command server listening on 0.0.0.0:{port}")
    while True:
        try:
            conn, addr = server_sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=_handle_cmd_client, args=(conn, addr), daemon=True).start()
        except OSError:
            break


def start_command_server(port=CMD_PORT):
    t = threading.Thread(target=_command_server_thread, args=(port,), daemon=True)
    t.start()
    return t


def _update_cmd_status(state, current_deg, target_deg, error_norm, **extra):
    with _cmd_status_lock:
        _cmd_status["state"] = state
        _cmd_status["current_deg"] = [round(v, 3) for v in current_deg]
        _cmd_status["target_deg"] = [round(v, 3) for v in target_deg]
        _cmd_status["error_norm"] = round(error_norm, 4)
        for k, v in extra.items():
            _cmd_status[k] = v


def main():
    global _invdyn_params
    import argparse
    parser = argparse.ArgumentParser(description="InvDyn control for myCobot Pro 630")
    parser.add_argument("--suction", action="store_true", default=False)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT)
    parser.add_argument("--stream-rate", type=float, default=STREAM_RATE_HZ)
    parser.add_argument("--cmd-port", type=int, default=CMD_PORT)
    default_params = os.path.join(LOG_DIR, "invdyn_params.npz")
    parser.add_argument("--params", default=default_params,
                        help="Path to npz from identify_invdyn_from_log.py (default: logs/invdyn_params.npz)")
    parser.add_argument("--urdf", default=None,
                        help="URDF path for Pinocchio model (when npz has pin_theta)")
    args = parser.parse_args()

    # Load inverse-dynamics params (optional)
    params_path = args.params
    if not os.path.isfile(params_path):
        npz_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.npz")))
        if npz_files:
            params_path = npz_files[0]
    if load_invdyn_params is not None and os.path.isfile(params_path):
        _invdyn_params = load_invdyn_params(params_path, urdf_path=args.urdf)
        if _invdyn_params is not None:
            print(f"  Loaded invdyn params from {params_path} (use_pinocchio={_invdyn_params.get('use_pinocchio', False)})")
        else:
            _invdyn_params = None
    else:
        _invdyn_params = None
        if args.params != default_params or (load_invdyn_params is None):
            pass  # user passed --params or no invdyn_model
        elif not os.path.isfile(default_params):
            pass  # no npz in logs, keep PD

    try:
        h = hal.component("invdyn")
        for i in range(MAX_JOINTS):
            h.newpin(f"joint{i}_pos_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
            h.newpin(f"joint{i}_vel_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
        h.newpin("enable", hal.HAL_BIT, hal.HAL_OUT)
        h.ready()
    except Exception as e:
        print(f"HAL component creation failed: {e}")
        sys.exit(1)

    if args.stream_port > 0:
        start_stream_server(port=args.stream_port, rate_hz=args.stream_rate)
    if args.cmd_port > 0:
        start_command_server(port=args.cmd_port)

    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = 0.0
        h[f"joint{i}_vel_cmd"] = 0.0
    h["enable"] = False

    print("Enabling machine...")
    time.sleep(2)
    if not enable_machine(h):
        print("Failed to enable machine. Exiting.")
        sys.exit(1)

    s = linuxcnc.stat()
    s.poll()
    current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    print("Current angles:", current)
    if args.suction:
        suction_pump(on=True)
    _update_cmd_status("idle", current, current, 0.0)

    print("\n" + "=" * 60)
    print("Waiting for commands from desktop (control_robot.py)...")
    print("  Send controller: 'invdyn' or 'pd'")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                cmd = _cmd_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            target = cmd.get("target_deg")
            if target is None or len(target) != MAX_JOINTS:
                continue
            duration = cmd.get("duration", 2.0)
            controller = cmd.get("controller", "invdyn")
            # This HAL only supports pd and invdyn; normalize so CSV matches actual control
            if controller not in ("pd", "invdyn"):
                print(f"[cmd] controller={controller} not supported by invdyn_hal, using pd")
                controller = "pd"
            pos_tol = cmd.get("pos_tol", 0.5)
            settle = cmd.get("settle_steps", 10)
            # Use desktop timestamp for CSV filename (Raspi system date may be wrong)
            global _desktop_log_stamp
            _desktop_log_stamp = cmd.get("log_stamp") if isinstance(cmd.get("log_stamp"), str) else None

            t_cmd_start = time.perf_counter()
            s.poll()
            current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            err = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5
            print(f"\n[cmd] Moving → {[round(v, 1) for v in target]} controller={controller}")
            _update_cmd_status("moving", current, target, err)
            if controller == "invdyn":
                invdyn_control_loop(h, s, target, duration_sec=duration, pos_tol=pos_tol, settle_steps=settle, controller=controller)
            else:
                pd_control_loop(h, s, target, duration_sec=duration, pos_tol=pos_tol, settle_steps=settle, controller=controller)
            robot_exec_ms = (time.perf_counter() - t_cmd_start) * 1000
            s.poll()
            final = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            final_err = sum((t - f) ** 2 for t, f in zip(target, final)) ** 0.5
            n_loops = len(_timing["poll"]) if _timing["poll"] else 1
            solve_key = "invdyn_solve" if controller == "invdyn" else "pd_solve"
            avg_solve = sum(_timing[solve_key][-n_loops:]) / n_loops if _timing[solve_key] else 0
            _update_cmd_status(
                "done", final, target, final_err,
                robot_exec_ms=round(robot_exec_ms, 2), n_loops=n_loops, avg_solve_ms=round(avg_solve, 3),
                last_log_name=_last_log_filename,
            )
            print(f"[cmd] Done. exec={robot_exec_ms:.0f}ms err={final_err:.3f}°")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        h["enable"] = False
        if args.suction:
            suction_pump(on=False)
        power_off_robot()
    sys.exit(0)


if __name__ == "__main__":
    main()
