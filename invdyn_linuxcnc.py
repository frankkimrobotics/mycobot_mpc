#!/usr/bin/env python3
"""
InvDyn control loop for LinuxCNC (Option 1: InvDyn as reference generator via MDI).
Same architecture as mpc_linuxcnc.py but uses inverse-dynamics law to compute
next waypoint: q̈_d = Kp*e - Kd*q̇, next_pos = q + q̇*dt + 0.5*q̈_d*dt².

Flow: Get state -> InvDyn solve -> Send waypoint via G-code -> Loop
"""

import time
import sys
from datetime import datetime
from functools import wraps
import numpy as np

try:
    import linuxcnc
except ImportError:
    print("linuxcnc module not found. Run this on a system with LinuxCNC installed.")
    sys.exit(1)

MAX_JOINTS = 6
MAX_ANGULAR_SPEED = 6930  # deg/min
INVDYN_PERIOD_MS = 10
INVDYN_KP = 144.0
INVDYN_KD = 24.0
QDD_MAX_DEG = 150.0

_timing = {"poll": [], "invdyn_solve": [], "send_cmd": [], "sleep": []}


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


def angles_to_gcode(angles):
    return (
        "X" + str(round(angles[0], 3)) +
        "Y" + str(round(angles[1], 3)) +
        "Z" + str(round(angles[2], 3)) +
        "A" + str(round(angles[3], 3)) +
        "B" + str(round(angles[4], 3)) +
        "C" + str(round(angles[5], 3))
    )


@timed("invdyn_solve")
def invdyn_solve(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """InvDyn: q̈_d = Kp*e - Kd*q̇, next_pos = q + q̇*dt + 0.5*q̈_d*dt² (deg)."""
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        velocity = (current - np.array(prev_q, dtype=float)) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    qdd_d = INVDYN_KP * error - INVDYN_KD * velocity
    qdd_d = np.clip(qdd_d, -QDD_MAX_DEG, QDD_MAX_DEG)
    if dt is None or dt <= 0:
        dt = INVDYN_PERIOD_MS / 1000.0
    next_pos = current + velocity * dt + 0.5 * qdd_d * (dt ** 2)
    return [float(x) for x in next_pos]


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
    return q, q_vel, t_status


@timed("send_cmd")
def _send_mdi_command(c, next_cmd, speed_pct):
    speed = speed_pct * MAX_ANGULAR_SPEED / 100
    gcode = "G38.3F" + str(speed) + angles_to_gcode(next_cmd)
    c.mdi(gcode)
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _print_timing_summary(loop_count):
    if not _timing["poll"]:
        return
    n = len(_timing["poll"])
    avg = lambda key: sum(_timing[key][-n:]) / n if _timing[key] else 0
    print(f"  [timing] poll={avg('poll'):.2f}ms invdyn_solve={avg('invdyn_solve'):.2f}ms send_cmd={avg('send_cmd'):.2f}ms sleep={avg('sleep'):.2f}ms")


def run_invdyn_loop(target_angles, speed_pct=50.0, duration_sec=10.0):
    """Run InvDyn control loop: poll -> invdyn_solve -> send MDI."""
    c = linuxcnc.command()
    s = linuxcnc.stat()
    print("InvDyn loop starting. Target:", target_angles)
    print("Press Ctrl+C to stop.\n")
    s.poll()
    if s.task_mode != linuxcnc.MODE_MDI:
        c.mode(linuxcnc.MODE_MDI)
        c.wait_complete()
    t_start = time.time()
    loop_count = 0
    prev_current = None
    t_prev = None
    for k in _timing:
        _timing[k] = []

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else None
        t_prev = t_loop_start
        q, q_vel, t_status = _poll_feedback(s)
        next_cmd = invdyn_solve(q, target_angles, q_vel=q_vel, prev_q=prev_current, dt=dt)
        prev_current = q.copy()
        _send_mdi_command(c, next_cmd, speed_pct)
        elapsed = time.time() - t_loop_start
        sleep_time = INVDYN_PERIOD_MS / 1000.0 - elapsed
        if sleep_time > 0:
            t0 = time.perf_counter()
            time.sleep(sleep_time)
            _timing["sleep"].append((time.perf_counter() - t0) * 1000)
        else:
            _timing["sleep"].append(0.0)
        loop_count += 1
        err = sum((t - a) ** 2 for t, a in zip(target_angles, q)) ** 0.5
        if loop_count % 5 == 0 or loop_count <= 3:
            print(f"Loop {loop_count}: q={q[:3]}... err={err:.3f}")
            _print_timing_summary(loop_count)

    print(f"\nDone. Ran {loop_count} InvDyn iterations.")
    _print_timing_summary(loop_count)


def main():
    s = linuxcnc.stat()
    s.poll()
    initial = [-90, -90, 0, -90, 0, 0]
    current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    target = [a + 5.0 for a in current]
    print("Current angles:", current)
    print("Target (current + 5° each):", target)
    run_invdyn_loop(target, speed_pct=15.0, duration_sec=10.0)
    run_invdyn_loop(initial, speed_pct=5.0, duration_sec=10.0)
    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
