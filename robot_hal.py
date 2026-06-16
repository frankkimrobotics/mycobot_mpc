#!/usr/bin/env python3
"""
Unified robot HAL: single component "ctrl" with selectable control law (pid, invdyn, pd_velff).

Run on Raspi with LinuxCNC: linuxcnc elerob.ini (loads elerob.hal which runs this script).
Control law: --controller pid|invdyn|pd_velff at startup; can be overridden per move via JSON from desktop.

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
except ImportError:
    load_invdyn_params = None
    linuxcnc_deg_to_rad = None
    compute_MCG = None
    MODEL_NUM_JOINTS = 6

MAX_JOINTS = 6
# Command-loop period (ms). HAL servo can run faster (e.g. SERVO_PERIOD 10 ms or sub-3 ms);
# this is how often we compute and write pos_cmd/vel_cmd. Overridable via --period-ms.
PERIOD_MS = 20
PERIOD_SEC = PERIOD_MS / 1000.0

# PID gains (position space: u = Kp*e - Kd*qd + Ki*integral)
KP_PID = 0.5
KD_PID = 0.1
KI_PID = 0.05
INTEGRAL_CLAMP = 50.0  # anti-windup: clamp |integral| per joint
U_MAX_PER_STEP = 8.0

# InvDyn (acceleration space): qdd_d = Kp*e - Kd*qd, then integrate
# Acceleration integration: step = 0.5*qdd*dt^2; with qdd_max=150 deg/s^2, dt=20ms → ~0.03° per step;
# with dt=3ms → ~0.0007° per step. So robot barely moves at any realistic loop period.
# USE_PD_STEPS_FOR_INVDYN: use direct position-step PD for invdyn so the robot moves.
USE_PD_STEPS_FOR_INVDYN = True
INVDYN_KP = 144.0
INVDYN_KD = 24.0
QDD_MAX_DEG = 150.0
# PD position-step gains when USE_PD_STEPS_FOR_INVDYN (same as mpc_hal / invdyn_hal pd_solve)
KP_PD_INVDYN = 0.5
KD_PD_INVDYN = 0.1
# Gravity compensation from npz: vel_cmd += K_GRAV_COMP * G(q) (G in Nm; scale to deg/s)
K_GRAV_COMP = 0.02

# PD + vel feedforward: same Kp,Kd as invdyn-style for consistency
KP_PD_VELFF = 144.0 / 90.0  # scale to deg/s² per deg error
KD_PD_VELFF = 24.0 / 90.0

SUCTION_PIN = "pro600.digital_out00"
CMD_PORT = 9998
STREAM_PORT = 9999
STREAM_RATE_HZ = 50.0

_timing = {"poll": [], "solve": [], "hal_write": [], "sleep": []}
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
# Control law: PID (with integral, anti-windup)
# ---------------------------------------------------------------------------
@timed("solve")
def pid_solve(q, target_angles, integral, q_vel=None, prev_q=None, dt=None):
    """PID: u = Kp*e - Kd*qd + Ki*integral; next_pos = q + u, vel_cmd = u/dt. Returns (next_pos, vel_cmd, new_integral)."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    integ = np.array(integral, dtype=float)
    integ += error * dt
    integ = np.clip(integ, -INTEGRAL_CLAMP, INTEGRAL_CLAMP)
    u = KP_PID * error - KD_PID * velocity + KI_PID * integ
    u = np.clip(u, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_pos = current + u
    vel_cmd = (u / dt * 0.5) if dt > 0 else np.zeros(MAX_JOINTS)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd], integ.tolist()


# ---------------------------------------------------------------------------
# Control law: InvDyn (model-based when params loaded, else acceleration PD)
# ---------------------------------------------------------------------------
def _invdyn_solve_fallback(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """PD in acceleration: qdd_d = Kp*e - Kd*qd; integrate to next_pos, vel_cmd."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    qdd_d = INVDYN_KP * error - INVDYN_KD * velocity
    qdd_d = np.clip(qdd_d, -QDD_MAX_DEG, QDD_MAX_DEG)
    next_pos = current + velocity * dt + 0.5 * qdd_d * (dt ** 2)
    vel_cmd = velocity + qdd_d * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


def _invdyn_pd_solve(q, target_angles, q_vel=None, prev_q=None, dt=None, prev_target=None, params=None):
    """Direct position-step PD for invdyn; optional vel_ff for trajectory tracking; optional G(q) from npz."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    u = KP_PD_INVDYN * error - KD_PD_INVDYN * velocity
    vel_ff = np.zeros(MAX_JOINTS)
    if prev_target is not None and dt > 0:
        prev_t = np.array(prev_target, dtype=float)
        vel_ff = (target - prev_t) / dt
    u = u + vel_ff * dt  # add velocity feedforward for trajectory tracking
    u = np.clip(u, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_pos = current + u
    vel_cmd = (u / dt * 0.5) if dt > 0 else np.zeros(MAX_JOINTS)
    vel_cmd = np.array(vel_cmd, dtype=float) + vel_ff
    # Use npz model for gravity compensation: vel_cmd += K_GRAV_COMP * G(q) (G in Nm)
    if params is not None and compute_MCG is not None and linuxcnc_deg_to_rad is not None:
        try:
            q_rad = linuxcnc_deg_to_rad(current)
            qd_rad = np.deg2rad(velocity)
            _, _, G = compute_MCG(q_rad, qd_rad, params)
            vel_cmd = vel_cmd + K_GRAV_COMP * np.asarray(G, dtype=float)
        except Exception:
            pass
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


def _invdyn_model_solve_deg(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """Model-aware invdyn in deg: vd_command in rad/s² from Kp*(target_rad - q_rad) - Kd*qd_rad; integrate in deg."""
    if linuxcnc_deg_to_rad is None:
        return _invdyn_solve_fallback(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    if q_vel is not None:
        velocity_deg = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt and dt > 0:
        velocity_deg = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity_deg = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    q_rad = linuxcnc_deg_to_rad(current)
    target_rad = linuxcnc_deg_to_rad(target)
    qd_rad = np.deg2rad(velocity_deg)
    vd_command_rad = INVDYN_KP * (target_rad - q_rad) - INVDYN_KD * qd_rad
    qdd_d_deg = np.rad2deg(vd_command_rad)
    qdd_d_deg = np.clip(qdd_d_deg, -QDD_MAX_DEG, QDD_MAX_DEG)
    next_pos = current + velocity_deg * dt + 0.5 * qdd_d_deg * (dt ** 2)
    vel_cmd = velocity_deg + qdd_d_deg * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


@timed("solve")
def invdyn_solve(q, target_angles, params, q_vel=None, prev_q=None, dt=None, prev_target=None):
    """Invdyn: use PD position steps (so robot moves) when USE_PD_STEPS_FOR_INVDYN; else acceleration integration.
    With prev_target, adds velocity feedforward for trajectory tracking."""
    if USE_PD_STEPS_FOR_INVDYN:
        return _invdyn_pd_solve(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt, prev_target=prev_target, params=params)
    if params is not None and (load_invdyn_params is not None or linuxcnc_deg_to_rad is not None):
        return _invdyn_model_solve_deg(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)
    return _invdyn_solve_fallback(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)


# ---------------------------------------------------------------------------
# Control law: PD + velocity feedforward
# ---------------------------------------------------------------------------
@timed("solve")
def pd_velff_solve(q, target_angles, prev_target, q_vel=None, prev_q=None, dt=None):
    """PD on position + velocity feedforward: vel_ff = (target - prev_target)/dt; next_pos = q + (Kp*e - Kd*qd)*dt + vel_ff*dt."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    vel_ff = np.zeros(MAX_JOINTS)
    if prev_target is not None and dt > 0:
        prev_t = np.array(prev_target, dtype=float)
        vel_ff = (target - prev_t) / dt
    acc_pd = KP_PD_VELFF * error - KD_PD_VELFF * velocity
    next_pos = current + (acc_pd + vel_ff) * dt
    vel_cmd = acc_pd + vel_ff
    next_pos = np.clip(next_pos, current - U_MAX_PER_STEP, current + U_MAX_PER_STEP)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


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
        + ["err_norm", "poll_ms", "solve_ms", "hal_write_ms", "sleep_ms"]
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
                     log_dir=None, log_enabled=True, period_sec=None, trajectory=None, traj_dt=None):
    """One loop: poll -> solve (pid|invdyn|pd_velff) -> HAL write; done_reason = converged | duration.

    If ``trajectory`` (list of N x MAX_JOINTS poses) and ``traj_dt`` (s/sample) are
    given, the setpoint follows trajectory[idx], idx = int(elapsed / traj_dt),
    clamped to the last sample (which equals ``target_angles``). Convergence is
    only tested once the trajectory has been fully played, so the smooth motion
    isn't cut short by an early-stop. Logged target is the live (moving) setpoint.
    """
    if period_sec is None:
        period_sec = PERIOD_SEC
    use_traj = bool(trajectory) and bool(traj_dt)
    final_target = list(target_angles)
    n_traj = len(trajectory) if use_traj else 0
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

    traj_done = not use_traj
    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else period
        t_prev = t_loop_start

        # Advance the setpoint along the trajectory by elapsed time (time-accurate
        # regardless of loop jitter); hold the final sample once exhausted.
        if use_traj:
            idx = int((t_loop_start - t_start) / traj_dt)
            if idx >= n_traj - 1:
                idx = n_traj - 1
                traj_done = True
            target_angles = trajectory[idx]

        q, q_vel, hal_vel, hal_torq, t_status = _poll_feedback(s)

        if controller == "pid":
            next_pos, vel_cmd, integral = pid_solve(q, target_angles, integral, q_vel=q_vel, prev_q=prev_current, dt=dt)
        elif controller == "invdyn":
            res = invdyn_solve(q, target_angles, _params, q_vel=q_vel, prev_q=prev_current, dt=dt, prev_target=prev_target)
            next_pos, vel_cmd = res[0], res[1]
        else:  # pd_velff
            next_pos, vel_cmd = pd_velff_solve(q, target_angles, prev_target, q_vel=q_vel, prev_q=prev_current, dt=dt)

        u = [next_pos[i] - q[i] for i in range(MAX_JOINTS)]
        if prev_current is not None and dt and dt > 0:
            est_vel = [(q[i] - prev_current[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()
        prev_target = target_angles

        _write_hal_cmd(h, next_pos, vel_cmd)
        elapsed = time.time() - t_loop_start
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            t0 = time.perf_counter()
            time.sleep(sleep_time)
            _timing["sleep"].append((time.perf_counter() - t0) * 1000)
        else:
            _timing["sleep"].append(0.0)

        loop_count += 1
        # Convergence is judged against the FINAL target, and only once the
        # trajectory has been fully played (so a smooth move isn't cut short).
        err = sum((t - a) ** 2 for t, a in zip(final_target, q)) ** 0.5
        vel_norm = sum(v ** 2 for v in vel_cmd) ** 0.5
        if traj_done and err < pos_tol and vel_norm < vel_tol:
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
        ])
        if loop_count % 10 == 0 or loop_count <= 3:
            print(f"Loop {loop_count}: q={q[:3]}... err={err:.3f} controller={controller}")

    print(f"\nDone. Ran {loop_count} iterations. exit_reason={done_reason}")
    _save_log(log_rows, target_angles, controller, log_dir or "logs", log_enabled)
    return done_reason


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
                with clients_lock:
                    clients.append(conn)
            except OSError:
                break
    threading.Thread(target=accept_loop, daemon=True).start()

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
            for c in dead:
                clients.remove(c)
                c.close()
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
    global _params, _desktop_log_stamp
    import argparse
    parser = argparse.ArgumentParser(description="Unified robot HAL: pid, invdyn, pd_velff")
    parser.add_argument("--controller", choices=["pid", "invdyn", "pd_velff"], default="pid",
                        help="Default control law (default: pid)")
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

    if USE_PD_STEPS_FOR_INVDYN:
        print("  invdyn: using PD position steps (robot will move)")
    print("\n" + "=" * 60)
    print("Waiting for commands. Controller: pid | invdyn | pd_velff")
    print("=" * 60 + "\n")

    default_controller = args.controller
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
            controller = cmd.get("controller", default_controller)
            if controller not in ("pid", "invdyn", "pd_velff"):
                controller = default_controller
            _desktop_log_stamp = cmd.get("log_stamp") if isinstance(cmd.get("log_stamp"), str) else None
            pos_tol = cmd.get("pos_tol", 0.5)
            settle = cmd.get("settle_steps", 10)
            # Optional B-spline/quintic trajectory: a list of per-sample joint poses
            # (N x 6) plus the seconds between samples. When present, the controller
            # tracks this time-varying setpoint instead of a constant target, in a
            # single control loop with a single log. Final target = last sample.
            trajectory = cmd.get("trajectory")
            traj_dt = cmd.get("traj_dt")
            if trajectory and traj_dt:
                # Run long enough to play the whole trajectory plus a settle margin.
                duration = max(duration, len(trajectory) * traj_dt + 1.0)

            t_cmd_start = time.perf_counter()
            s.poll()
            current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            err = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5
            kind = "trajectory" if (trajectory and traj_dt) else "target"
            print(f"\n[cmd] Moving ({kind}) -> {[round(v, 1) for v in target]} controller={controller}")
            _update_cmd_status("moving", current, target, err)

            done_reason = run_control_loop(h, s, target, duration_sec=duration, controller=controller,
                                          pos_tol=pos_tol, settle_steps=settle, log_dir=log_dir, log_enabled=log_enabled,
                                          period_sec=period_sec, trajectory=trajectory, traj_dt=traj_dt)

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
