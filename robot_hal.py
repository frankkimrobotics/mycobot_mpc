#!/usr/bin/env python3
"""
Unified robot HAL: single component "ctrl" with selectable control law (pid, invdyn, pd_velff, mpc).

Run on Raspi with LinuxCNC: linuxcnc elerob.ini (loads elerob.hal which runs this script).
Control law: --controller pid|invdyn|pd_velff|mpc at startup; can be overridden per move via JSON from desktop.
MPC uses TinyMPC (pip install tinympc) when available; else falls back to OSQP-based QP.

HAL pins: ctrl.joint{i}_pos_cmd, ctrl.joint{i}_vel_cmd, ctrl.enable (same as mpc/invdyn).
"""

import base64
import csv
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from functools import wraps

import numpy as np

try:
    import linuxcnc
    import hal
except ImportError as e:
    print(f"Import error: {e}. Need linuxcnc and hal (run with LinuxCNC).")
    sys.exit(1)

try:
    from invdyn_model import (
        load_params as load_invdyn_params,
        linuxcnc_deg_to_rad,
        compute_MCG,
        NUM_JOINTS as MODEL_NUM_JOINTS,
    )
except Exception:   # invdyn_model is optional (gravity comp); also skips its 3.7-incompat syntax / missing joint_conventions on the Pi
    load_invdyn_params = None
    linuxcnc_deg_to_rad = None
    compute_MCG = None
    MODEL_NUM_JOINTS = 6

import controller_solvers as _cs
from controller_solvers import (
    MAX_JOINTS,
    PERIOD_MS,
    PERIOD_SEC,
    pid_solve as _pid_solve_raw,
    invdyn_solve as _invdyn_solve_raw,
    pd_velff_solve as _pd_velff_solve_raw,
    mpc_solve as _mpc_solve_raw,
)

# Command-loop period overridable via --period-ms
SUCTION_PIN = "pro600.digital_out00"
CMD_PORT = 9998
STREAM_PORT = 9999
STREAM_RATE_HZ = 100.0
# SAFETY CLAMPS applied to every HAL write regardless of control law (2026-09-17):
# the drive firmware trips "Position cmd and fb are too far apart" and powers off
# when motor_poscmd runs away from posfb (seen at a 168 deg gap from the yaml PID
# with kp=5000 on a 0.03 deg error). Velocity units of pro600.jointN_poscmd are
# still uncalibrated, so the ceiling starts small; raise per command via
# "vel_cmd_max" once measured.
POS_CMD_GAP_MAX_DEG = 3.0
VEL_CMD_ABS_MAX = 100.0
_vel_cmd_max = VEL_CMD_ABS_MAX
# Hold mode law: bounded LQR-clamp (mpc) at a low velocity clamp instead of the PID.
HOLD_CONTROLLER = "mpc"
HOLD_GAINS = {"vmax": 5.0}

_timing = {"poll": [], "solve": [], "hal_write": [], "sleep": []}
_lat = {}     # per-move latency marks (Pi epoch): t_loop0 (first poll), t_write0 (first HAL write)
_params = None  # invdyn params when --params npz loaded
_last_log_filename = None
_desktop_log_stamp = None


def timed(step_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            _timing[step_name].append((time.perf_counter() - t0) * 1000)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Control law wrappers (timed; solvers live in controller_solvers)
# ---------------------------------------------------------------------------
@timed("solve")
def pid_solve(q, target_angles, integral, q_vel=None, prev_q=None, dt=None):
    return _pid_solve_raw(q, target_angles, integral, q_vel=q_vel, prev_q=prev_q, dt=dt)


@timed("solve")
def invdyn_solve(q, target_angles, params, q_vel=None, prev_q=None, dt=None, prev_target=None):
    return _invdyn_solve_raw(q, target_angles, params, q_vel=q_vel, prev_q=prev_q, dt=dt, prev_target=prev_target)


@timed("solve")
def pd_velff_solve(q, target_angles, prev_target, q_vel=None, prev_q=None, dt=None):
    return _pd_velff_solve_raw(q, target_angles, prev_target, q_vel=q_vel, prev_q=prev_q, dt=dt)


@timed("solve")
def mpc_solve(q, target_angles, dt=None, q_vel=None):
    return _mpc_solve_raw(q, target_angles, dt=dt, q_vel=q_vel)


# ---------------------------------------------------------------------------
# HAL / feedback / logging
# ---------------------------------------------------------------------------
def _read_hal_feedback():
    out_vel = [0.0] * MAX_JOINTS
    out_torq = [0.0] * MAX_JOINTS
    for i in range(MAX_JOINTS):
        try:
            out_vel[i] = float(hal.get_value(f"pro600.joint{i}_velfb"))
        except (NameError, TypeError, ValueError, KeyError):
            pass
        try:
            out_torq[i] = float(hal.get_value(f"pro600.joint{i}_torqfb"))
        except (NameError, TypeError, ValueError, KeyError):
            pass
    return out_vel, out_torq


# Feedback straight from the driver's HAL pins (pro600.jointN_posfb), bypassing the motion
# controller's servo-thread copy + NML status buffer that linuxcnc.stat() reads (~10-20 ms
# older). Velocity = finite difference of posfb over the loop period (2-sample average).
_fb_prev = {"q": None, "t": None, "v": [0.0] * MAX_JOINTS}
USE_STAT_FEEDBACK = False


def _read_posfb():
    return [round(float(hal.get_value(f"pro600.joint{i}_posfb")), 4) for i in range(MAX_JOINTS)]


@timed("poll")
def _poll_feedback(s):
    t_status = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if USE_STAT_FEEDBACK:
        s.poll()
        q = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
        q_vel = None
        try:
            q_vel = [s.joint[i]["velocity"] for i in range(MAX_JOINTS)]
        except (KeyError, TypeError, IndexError):
            pass
    else:
        q = _read_posfb()
        now = time.time()
        if _fb_prev["q"] is not None and now > _fb_prev["t"]:
            dt = now - _fb_prev["t"]
            v_new = [(q[i] - _fb_prev["q"][i]) / dt for i in range(MAX_JOINTS)]
            q_vel = [0.5 * (v_new[i] + _fb_prev["v"][i]) for i in range(MAX_JOINTS)]
            _fb_prev["v"] = v_new
        else:
            q_vel = [0.0] * MAX_JOINTS
        _fb_prev["q"], _fb_prev["t"] = q, now
    hal_vel, hal_torq = _read_hal_feedback()
    return q, q_vel, hal_vel, hal_torq, t_status


@timed("hal_write")
def _write_hal_cmd(h, next_pos, vel_cmd, q=None):
    if q is not None:
        next_pos = [max(q[i] - POS_CMD_GAP_MAX_DEG, min(q[i] + POS_CMD_GAP_MAX_DEG, next_pos[i])) for i in range(MAX_JOINTS)]
    vel_cmd = [max(-_vel_cmd_max, min(_vel_cmd_max, v)) for v in vel_cmd]
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = next_pos[i]
        h[f"joint{i}_vel_cmd"] = vel_cmd[i]
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _save_log(log_rows, target_angles, controller, log_dir, enabled):
    global _last_log_filename, _desktop_log_stamp
    if not enabled or not log_rows:
        return
    os.makedirs(log_dir, exist_ok=True)
    stamp = _desktop_log_stamp if _desktop_log_stamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    _desktop_log_stamp = None
    target_str = "_".join(str(int(a)) for a in target_angles)
    filename = os.path.join(log_dir, f"robot_hal_{stamp}_t{target_str}.csv")
    header = (
        ["controller", "timestamp", "loop"]
        + [f"q{i}" for i in range(MAX_JOINTS)] + [f"qvel{i}" for i in range(MAX_JOINTS)]
        + [f"target{i}" for i in range(MAX_JOINTS)] + [f"cmd_pos{i}" for i in range(MAX_JOINTS)]
        + [f"cmd_vel{i}" for i in range(MAX_JOINTS)] + [f"u{i}" for i in range(MAX_JOINTS)]
        + [f"hal_vel{i}" for i in range(MAX_JOINTS)] + [f"hal_torq{i}" for i in range(MAX_JOINTS)]
        + ["err_norm", "poll_ms", "solve_ms", "hal_write_ms", "sleep_ms", "t_epoch"]
    )
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(log_rows)
    _last_log_filename = os.path.basename(filename)
    print(f"  Log saved: {filename}")


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


def power_off_robot():
    print("Powering off robot...")
    _halcmd_set("ctrl.enable", 0)
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
    return [_halcmd_get_float(f"pro600.joint{i}_posfb") for i in range(MAX_JOINTS)]


def preload_and_enable_ctrl(h, feedback):
    print("  Pre-loading ctrl commands with stable feedback...")
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = feedback[i]
        h[f"joint{i}_vel_cmd"] = 0.0
    time.sleep(0.02)
    h["enable"] = True
    print("  ctrl.enable = TRUE")


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
                preload_and_enable_ctrl(h, feedback)
                return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Single control loop (dispatches to pid / invdyn / pd_velff)
# ---------------------------------------------------------------------------
def run_control_loop(h, s, target_angles, duration_sec, controller, pos_tol=0.5, vel_tol=1.0, settle_steps=10,
                     log_dir=None, log_enabled=True, period_sec=None, abort_on_cmd=False):
    """One loop: poll -> solve (pid|invdyn|pd_velff|mpc) -> HAL write; done_reason = converged | duration."""
    if period_sec is None:
        period_sec = PERIOD_SEC
    h["enable"] = True
    t_start = time.time()
    loop_count = 0
    converged_count = 0
    prev_current = None
    prev_target = None
    t_prev = None
    integral = [0.0] * MAX_JOINTS
    done_reason = "duration"
    for k in _timing:
        _timing[k] = []
    log_rows = []
    period = period_sec

    _lat.clear()
    while (time.time() - t_start) < duration_sec:
        if abort_on_cmd and not _cmd_queue.empty():      # hold mode: yield to a new command at once
            done_reason = "cmd_pending"
            break
        t_loop_start = time.time()
        if loop_count == 0:
            _lat["t_loop0"] = t_loop_start
        dt = (t_loop_start - t_prev) if t_prev is not None else period
        t_prev = t_loop_start

        q, q_vel, hal_vel, hal_torq, t_status = _poll_feedback(s)

        if controller == "pid":
            next_pos, vel_cmd, integral = pid_solve(q, target_angles, integral, q_vel=q_vel, prev_q=prev_current, dt=dt)
        elif controller == "invdyn":
            res = invdyn_solve(q, target_angles, _params, q_vel=q_vel, prev_q=prev_current, dt=dt, prev_target=prev_target)
            next_pos, vel_cmd = res[0], res[1]
        elif controller == "mpc":
            next_pos, vel_cmd = mpc_solve(q, target_angles, dt=dt, q_vel=q_vel)
        else:  # pd_velff
            next_pos, vel_cmd = pd_velff_solve(q, target_angles, prev_target, q_vel=q_vel, prev_q=prev_current, dt=dt)

        u = [next_pos[i] - q[i] for i in range(MAX_JOINTS)]
        if prev_current is not None and dt and dt > 0:
            est_vel = [(q[i] - prev_current[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()
        prev_target = target_angles

        _write_hal_cmd(h, next_pos, vel_cmd, q)
        if loop_count == 0:
            _lat["t_write0"] = time.time()
        elapsed = time.time() - t_loop_start
        sleep_time = max(0, period - elapsed)
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
                done_reason = "converged"
                print(f"  Converged at loop {loop_count}: err={err:.3f}° vel_norm={vel_norm:.3f}°/s")
                break
        else:
            converged_count = 0

        log_rows.append([
            controller, t_status, loop_count,
            *q, *est_vel, *target_angles, *next_pos, *vel_cmd,
            *u, *hal_vel, *hal_torq,
            err,
            _timing["poll"][-1], _timing["solve"][-1],
            _timing["hal_write"][-1], _timing["sleep"][-1],
            t_loop_start,
        ])
        if loop_count % 10 == 0 or loop_count <= 3:
            print(f"Loop {loop_count}: q={q[:3]}... err={err:.3f} controller={controller}")

    print(f"\nDone. Ran {loop_count} iterations. exit_reason={done_reason}")
    _save_log(log_rows, target_angles, controller, log_dir or "logs", log_enabled)
    return done_reason


# ---------------------------------------------------------------------------
# Streamed reference (welded chunks) -- 2026-09-17
#   cmd {"chunk": [[6 deg],...], "traj_dt": s, "t_anchor": epoch, "seq": k, "gains": {...}}
#   chunks are welded into one reference q_ref(t) in the cmd thread (never queued);
#   the first chunk queues a _stream_start so the main loop enters run_stream_loop().
#   law: u = vff*v_ref(t+lead) - K0*(q - q_ref(t+lead)) - K1*(q_vel - v_ref), clamp vmax
# ---------------------------------------------------------------------------
import bisect
_ref_lock = threading.Lock()
_ref_t, _ref_q, _ref_chunks = [], [], []
_stream_requested = False


def _weld_chunk(cmd):
    t_anchor = float(cmd["t_anchor"])
    dt = float(cmd.get("traj_dt", 0.01))
    pts = cmd["chunk"]
    with _ref_lock:
        k = len(_ref_t)
        while k > 0 and _ref_t[k - 1] >= t_anchor - 1e-6:
            k -= 1
        del _ref_t[k:]
        del _ref_q[k:]
        for i, p in enumerate(pts):
            _ref_t.append(t_anchor + i * dt)
            _ref_q.append([float(x) for x in p][:MAX_JOINTS])
        _ref_chunks.append({"seq": cmd.get("seq"), "t_recv": cmd["_t_recv"], "t_anchor": t_anchor, "n": len(pts)})
        return len(_ref_t)


def _sample_ref(t):
    with _ref_lock:
        n = len(_ref_t)
        if n == 0:
            return None, None, "empty"
        if t <= _ref_t[0]:
            return list(_ref_q[0]), [0.0] * MAX_JOINTS, "before"
        if t >= _ref_t[-1]:
            return list(_ref_q[-1]), [0.0] * MAX_JOINTS, "after"
        i = bisect.bisect_right(_ref_t, t) - 1
        t0, t1 = _ref_t[i], _ref_t[i + 1]
        a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        q0, q1 = _ref_q[i], _ref_q[i + 1]
        q = [q0[j] + a * (q1[j] - q0[j]) for j in range(MAX_JOINTS)]
        v = [(q1[j] - q0[j]) / (t1 - t0) if t1 > t0 else 0.0 for j in range(MAX_JOINTS)]
        return q, v, "in"


def _other_cmd_pending():
    with _cmd_queue.mutex:
        return any(not c.get("_stream_start") for c in _cmd_queue.queue)


def run_stream_loop(h, s, gains, period_sec, log_dir, log_enabled, idle_timeout=0.5):
    K0 = float(gains.get("k0", 6.0)); K1 = float(gains.get("k1", 1.2)); VMAX = float(gains.get("vmax", 50.0))
    VS = float(gains.get("vel_scale", 17.0)); VFF = float(gains.get("vff", 1.0)); LEAD = float(gains.get("lead", 0.0))
    h["enable"] = True
    _lat.clear()
    log_rows, loop_count, t_prev, prev_q = [], 0, None, None
    t_last_in = time.time()
    done_reason = "idle_timeout"
    last_ref = None
    for k in _timing:
        _timing[k] = []
    while True:
        t_loop_start = time.time()
        if loop_count == 0:
            _lat["t_loop0"] = t_loop_start
        dt = (t_loop_start - t_prev) if t_prev is not None else period_sec
        t_prev = t_loop_start
        if _other_cmd_pending():
            done_reason = "cmd_pending"
            break
        q, q_vel, hal_vel, hal_torq, t_status = _poll_feedback(s)
        q_ref, v_ref, st = _sample_ref(t_loop_start + LEAD)
        if st == "empty":
            done_reason = "empty"
            break
        if st == "in":
            t_last_in = t_loop_start
        elif st == "after" and (t_loop_start - t_last_in) > idle_timeout:
            done_reason = "end"
            break
        last_ref = q_ref
        e = [q[j] - q_ref[j] for j in range(MAX_JOINTS)]
        vv = q_vel if q_vel is not None else [0.0] * MAX_JOINTS
        u = [max(-VMAX, min(VMAX, VFF * v_ref[j] - K0 * e[j] - K1 * (vv[j] - v_ref[j]))) for j in range(MAX_JOINTS)]
        vel_cmd = [x * VS for x in u]
        next_pos = [q[j] + u[j] * dt for j in range(MAX_JOINTS)]
        _timing["solve"].append(0.0)
        _write_hal_cmd(h, next_pos, vel_cmd, q)
        if loop_count == 0:
            _lat["t_write0"] = time.time()
        est_vel = [(q[j] - prev_q[j]) / dt for j in range(MAX_JOINTS)] if (prev_q is not None and dt > 0) else [0.0] * MAX_JOINTS
        prev_q = q.copy()
        err = sum(x * x for x in e) ** 0.5
        log_rows.append(["stream", t_status, loop_count, *q, *est_vel, *q_ref, *next_pos, *vel_cmd, *u,
                         *hal_vel, *hal_torq, err, _timing["poll"][-1], 0.0, _timing["hal_write"][-1], 0.0, t_loop_start])
        loop_count += 1
        if loop_count % 50 == 0:
            print(f"[stream] loop {loop_count} err={err:.3f} st={st} n_ref={len(_ref_t)}")
        sl = period_sec - (time.time() - t_loop_start)
        if sl > 0:
            time.sleep(sl)
    print(f"[stream] done: {done_reason} after {loop_count} loops")
    _save_log(log_rows, last_ref or [0.0] * MAX_JOINTS, "stream", log_dir or "logs", log_enabled)
    return done_reason, last_ref, loop_count


# ---------------------------------------------------------------------------
# Streaming server
# ---------------------------------------------------------------------------
def _stream_server_thread(port, rate_hz):
    stat = linuxcnc.stat()
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(5)
    print(f"[stream] Listening on 0.0.0.0:{port} at {rate_hz} Hz")
    clients = []
    clients_lock = threading.Lock()
    period = 1.0 / rate_hz

    def accept_loop():
        while True:
            try:
                conn, _ = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(0.5)          # a stalled client must not block the broadcaster
                with clients_lock:
                    clients.append(conn)
            except OSError:
                break
    threading.Thread(target=accept_loop, daemon=True).start()

    while True:
        t0 = time.time()
        try:
            if USE_STAT_FEEDBACK:
                stat.poll()
                joints_deg = [round(stat.joint_actual_position[i], 4) for i in range(MAX_JOINTS)]
            else:
                joints_deg = _read_posfb()          # driver pins: fresher than linuxcnc.stat
        except Exception as e:      # linuxcnc.error (NML timeout under load) is NOT a RuntimeError -- this killed the thread 2026-09-17
            print(f"[stream] poll error: {e!r}")
            time.sleep(0.05)
            continue
        try:
            _, torq = _read_hal_feedback()               # per-joint torque feedback (pro600.joint{i}_torqfb)
        except Exception:
            torq = [0.0] * MAX_JOINTS
        msg = json.dumps({"joints_deg": joints_deg, "torque": [round(t, 4) for t in torq],
                          "timestamp": time.time()}) + "\n"
        dead = []
        with clients_lock:
            for conn in clients:
                try:
                    conn.sendall(msg.encode("utf-8"))
                except Exception:               # incl. socket.timeout: drop the client
                    dead.append(conn)
            for c in dead:
                try:
                    clients.remove(c)
                    c.close()
                except Exception:
                    pass
        if period - (time.time() - t0) > 0:
            time.sleep(period - (time.time() - t0))


# ---------------------------------------------------------------------------
# Command server
# ---------------------------------------------------------------------------
_cmd_queue = queue.Queue()
_cmd_status = {"state": "idle", "current_deg": [0.0] * MAX_JOINTS, "target_deg": [0.0] * MAX_JOINTS, "error_norm": 0.0, "last_log_name": None}
_cmd_status_lock = threading.Lock()


def _update_cmd_status(state, current_deg, target_deg, error_norm, **extra):
    with _cmd_status_lock:
        _cmd_status["state"] = state
        _cmd_status["current_deg"] = [round(v, 3) for v in current_deg]
        _cmd_status["target_deg"] = [round(v, 3) for v in target_deg]
        _cmd_status["error_norm"] = round(error_norm, 4)
        for k, v in extra.items():
            _cmd_status[k] = v


def _handle_cmd_client(conn, addr, log_dir):
    global _stream_requested
    _ = addr
    print(f"[cmd] Client connected: {addr}")
    buffer = ""
    conn.settimeout(0.05)      # was 1.0: status cadence is bounded by this recv timeout
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
                            cmd["_t_recv"] = time.time()          # Pi epoch when the command line was parsed
                            _cmd_queue.put(cmd)
                            conn.sendall((json.dumps({"state": "ack", "target_deg": cmd["target_deg"],
                                                      "t_recv": cmd["_t_recv"], "tag": cmd.get("tag")}) + "\n").encode("utf-8"))
                        elif "chunk" in cmd:
                            cmd["_t_recv"] = time.time()
                            n_ref = _weld_chunk(cmd)
                            if not _stream_requested:
                                _stream_requested = True
                                _cmd_queue.put({"_stream_start": True, "gains": cmd.get("gains") or {},
                                                "period_ms": cmd.get("period_ms"), "vel_cmd_max": cmd.get("vel_cmd_max"),
                                                "log_stamp": cmd.get("log_stamp"), "_t_recv": cmd["_t_recv"]})
                            conn.sendall((json.dumps({"state": "ack_chunk", "seq": cmd.get("seq"), "tag": cmd.get("tag"),
                                                      "t_recv": cmd["_t_recv"], "n_ref": n_ref}) + "\n").encode("utf-8"))
                        elif "get_log" in cmd:
                            name = cmd.get("get_log")
                            if name and isinstance(name, str):
                                name = os.path.basename(name)
                                if name.endswith(".csv"):
                                    path = os.path.join(log_dir, name)
                                    if os.path.isfile(path):
                                        with open(path, "rb") as f:
                                            raw = f.read()
                                        payload = base64.b64encode(raw).decode("ascii")
                                        conn.sendall((json.dumps({"state": "log", "filename": name, "log_content_base64": payload}) + "\n").encode("utf-8"))
                                    else:
                                        conn.sendall((json.dumps({"state": "log_error", "error": "file_not_found"}) + "\n").encode("utf-8"))
                    except json.JSONDecodeError:
                        pass
            except socket.timeout:
                pass
            time.sleep(0.01)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _command_server_thread(port, log_dir):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(2)
    print(f"[cmd] Command server listening on 0.0.0.0:{port}")
    while True:
        try:
            conn, addr = server_sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=_handle_cmd_client, args=(conn, addr, log_dir), daemon=True).start()
        except OSError:
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global _params, _desktop_log_stamp, _vel_cmd_max, _stream_requested
    import argparse
    parser = argparse.ArgumentParser(description="Unified robot HAL: pid, invdyn, pd_velff, mpc")
    parser.add_argument("--controller", choices=["pid", "invdyn", "pd_velff", "mpc"], default="pid",
                        help="Default control law: pid, invdyn, pd_velff, or mpc (TinyMPC, default: pid)")
    parser.add_argument("--params", default=None, help="Path to npz for invdyn (e.g. logs/invdyn_params.npz)")
    parser.add_argument("--urdf", default=None, help="URDF for Pinocchio when npz has pin_theta")
    parser.add_argument("--cmd-port", type=int, default=CMD_PORT)
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT)
    parser.add_argument("--stream-rate", type=float, default=STREAM_RATE_HZ)
    parser.add_argument("--log-dir", default=None, help="Directory for CSV logs (default: script_dir/logs)")
    parser.add_argument("--no-log", action="store_true", help="Do not write CSV logs")
    parser.add_argument("--period-ms", type=float, default=None,
                        help="Command loop period in ms (default: 20). HAL servo can run faster (e.g. 10 ms or sub-3 ms).")
    parser.add_argument("--suction", action="store_true", default=False)
    args = parser.parse_args()

    period_sec = (args.period_ms if args.period_ms is not None else PERIOD_MS) / 1000.0
    if args.period_ms is not None:
        print(f"  Command loop period: {args.period_ms} ms")

    log_dir = args.log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    log_enabled = not args.no_log

    if args.params and os.path.isfile(args.params) and load_invdyn_params is not None:
        _params = load_invdyn_params(args.params, urdf_path=args.urdf)
        if _params is not None:
            print(f"  Loaded invdyn params from {args.params} (used for gravity compensation in invdyn)")
    else:
        _params = None

    try:
        h = hal.component("ctrl")
        for i in range(MAX_JOINTS):
            h.newpin(f"joint{i}_pos_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
            h.newpin(f"joint{i}_vel_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
        h.newpin("enable", hal.HAL_BIT, hal.HAL_OUT)
        h.ready()
    except Exception as e:
        print(f"HAL component creation failed: {e}")
        sys.exit(1)

    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = 0.0
        h[f"joint{i}_vel_cmd"] = 0.0
    h["enable"] = False

    if args.stream_port > 0:
        threading.Thread(target=_stream_server_thread, args=(args.stream_port, args.stream_rate), daemon=True).start()
    if args.cmd_port > 0:
        threading.Thread(target=_command_server_thread, args=(args.cmd_port, log_dir), daemon=True).start()

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
        _halcmd_set(SUCTION_PIN, 1)
    _update_cmd_status("idle", current, current, 0.0)

    if False:  # USE_PD_STEPS_FOR_INVDYN -- undefined global, print-only branch
        print("  invdyn: using PD position steps (robot will move)")
    if args.controller == "mpc":
        try:
            import tinympc as _tm
            _mpc_ok = True
        except ImportError:
            _mpc_ok = False
        print(f"  mpc: TinyMPC={'ok' if _mpc_ok else 'not found, using OSQP fallback'}")
    print("\n" + "=" * 60)
    print("Waiting for commands. Controller: pid | invdyn | pd_velff | mpc")
    print("=" * 60 + "\n")

    default_controller = args.controller
    # POSITION HOLD between commands: without this the 250 Hz loop only runs
    # during a command window; at idle the HAL pins freeze and the arm sags
    # under gravity (~0.4 deg/s) until the STM32 ferror-trips the drives.
    s.poll()
    hold_target = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    try:
        while True:
            try:
                cmd = _cmd_queue.get(timeout=0.05)
            except queue.Empty:
                # duty-cycled hold: continuous bursts starve the stream
                # thread (GIL); 0.6 s hold + 0.4 s breather -> ripple < 0.2 deg
                try:
                    _cs.set_overrides(HOLD_GAINS)
                    run_control_loop(h, s, hold_target, duration_sec=0.2,
                                     controller=HOLD_CONTROLLER, log_dir=log_dir,
                                     log_enabled=False, abort_on_cmd=True)
                except Exception:
                    pass
                finally:
                    _cs.set_overrides({})
                # no breather sleep: the queue.get(timeout=0.05) at the loop top releases the GIL
                # and wakes instantly when a command arrives (a 50 ms sleep here cost 0-50 ms per command)
                continue
            t_dequeue = time.time()
            if cmd.get("_stream_start"):
                _stream_requested = False
                sgains = cmd.get("gains") or {}
                _pm = cmd.get("period_ms")
                sp = (float(_pm) / 1000.0) if _pm else period_sec
                _vcm = cmd.get("vel_cmd_max")
                _vel_cmd_max = float(_vcm) if _vcm else VEL_CMD_ABS_MAX
                _desktop_log_stamp = cmd.get("log_stamp") if isinstance(cmd.get("log_stamp"), str) else None
                s.poll()
                current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
                print(f"\n[stream] start gains={sgains} period={sp*1000:.0f}ms")
                _update_cmd_status("streaming", current, current, 0.0, gains=sgains, t_recv=cmd.get("_t_recv"), t_dequeue=t_dequeue)
                t_cmd_start = time.perf_counter()
                try:
                    done_reason, last_ref, n_loops = run_stream_loop(h, s, sgains, sp, log_dir, log_enabled)
                finally:
                    _vel_cmd_max = VEL_CMD_ABS_MAX
                    with _cmd_queue.mutex:          # drop duplicate starts
                        for c in list(_cmd_queue.queue):
                            if c.get("_stream_start"):
                                _cmd_queue.queue.remove(c)
                    _stream_requested = False
                with _ref_lock:
                    chunks = list(_ref_chunks)
                    _ref_chunks.clear()
                    del _ref_t[:]
                    del _ref_q[:]
                s.poll()
                final = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
                if last_ref is not None:
                    hold_target = list(last_ref)
                _update_cmd_status("done", final, hold_target, 0.0, done_reason=done_reason, n_loops=n_loops,
                                   robot_exec_ms=round((time.perf_counter() - t_cmd_start) * 1000, 1),
                                   last_log_name=_last_log_filename, controller="stream", gains=sgains,
                                   period_ms=round(sp * 1000, 3), chunks=chunks, t_recv=cmd.get("_t_recv"),
                                   t_dequeue=t_dequeue, t_loop0=_lat.get("t_loop0"), t_write0=_lat.get("t_write0"),
                                   t_done=time.time())
                continue
            target = cmd.get("target_deg")
            if target is None or len(target) != MAX_JOINTS:
                continue
            duration = cmd.get("duration", 2.0)
            controller = cmd.get("controller", default_controller)
            if controller not in ("pid", "invdyn", "pd_velff", "mpc"):
                controller = default_controller
            _desktop_log_stamp = cmd.get("log_stamp") if isinstance(cmd.get("log_stamp"), str) else None
            pos_tol = cmd.get("pos_tol", 0.5)
            settle = cmd.get("settle_steps", 10)
            gains = cmd.get("gains")
            if not isinstance(gains, dict):
                gains = cmd.get("params") if isinstance(cmd.get("params"), dict) else {}
            _pm = cmd.get("period_ms")
            move_period = (float(_pm) / 1000.0) if _pm else period_sec
            _cs.set_overrides(gains)          # per-move controller parameter override
            _vcm = cmd.get("vel_cmd_max")
            _vel_cmd_max = float(_vcm) if _vcm else VEL_CMD_ABS_MAX

            t_cmd_start = time.perf_counter()
            s.poll()
            current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            err = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5
            print(f"\n[cmd] Moving -> {[round(v, 1) for v in target]} controller={controller}")
            _update_cmd_status("moving", current, target, err, controller=controller, gains=gains,
                               tag=cmd.get("tag"), t_recv=cmd.get("_t_recv"), t_dequeue=t_dequeue)

            hold_target = list(target)
            try:
                done_reason = run_control_loop(h, s, target, duration_sec=duration, controller=controller,
                                              pos_tol=pos_tol, settle_steps=settle, log_dir=log_dir, log_enabled=log_enabled,
                                              period_sec=move_period)
            finally:
                _cs.set_overrides({})         # hold mode + next move use yaml defaults
                _vel_cmd_max = VEL_CMD_ABS_MAX

            robot_exec_ms = (time.perf_counter() - t_cmd_start) * 1000
            s.poll()
            final = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            final_err = sum((t - f) ** 2 for t, f in zip(target, final)) ** 0.5
            n_loops = len(_timing["poll"]) if _timing["poll"] else 1
            avg_solve = sum(_timing["solve"][-n_loops:]) / n_loops if _timing["solve"] else 0
            _update_cmd_status(
                "done", final, target, final_err,
                robot_exec_ms=round(robot_exec_ms, 2), n_loops=n_loops, done_reason=done_reason,
                avg_solve_ms=round(avg_solve, 3), last_log_name=_last_log_filename,
                controller=controller, gains=gains, period_ms=round(move_period * 1000, 3),
                tag=cmd.get("tag"), t_recv=cmd.get("_t_recv"), t_dequeue=t_dequeue,
                t_loop0=_lat.get("t_loop0"), t_write0=_lat.get("t_write0"), t_done=time.time(),
            )
            print(f"[cmd] Done. exit_reason={done_reason} exec={robot_exec_ms:.0f}ms err={final_err:.3f}°")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        h["enable"] = False
        if args.suction:
            _halcmd_set(SUCTION_PIN, 0)
        power_off_robot()
    sys.exit(0)


if __name__ == "__main__":
    main()
