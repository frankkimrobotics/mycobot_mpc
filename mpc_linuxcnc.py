#!/usr/bin/env python3
"""
Simple MPC control loop for LinuxCNC (Option 1: MPC as Reference Generator).
Based on the architecture in /home/pi/mpc_control_architecture.md

Flow: Get state -> MPC solve -> Send waypoint via G-code -> Loop
"""

import time
import sys
from datetime import datetime
from functools import wraps
import numpy as np

# LinuxCNC is typically available on Debian/aarch64 (e.g. Raspberry Pi with myCobot Pro 630)
try:
    import linuxcnc
except ImportError:
    print("linuxcnc module not found. Run this on a system with LinuxCNC installed.")
    sys.exit(1)

# Constants from myCobot Pro 630
MAX_JOINTS = 6
MAX_ANGULAR_SPEED = 6930  # deg/min
MPC_PERIOD_MS = 10        # 200 Hz - matches trajectory planner ~10 Hz limitation

# QP: max change per step (degrees) for safety
U_MAX_PER_STEP = 5.0
# PD gains
KP = 0.99  # proportional (Q/(Q+R) with Q=1, R=0.01)
KD = 0.0   # derivative (damping)

# Timing accumulators (ms) for each step
_timing = {"poll": [], "pd_solve": [], "send_cmd": [], "sleep": []}


def timed(step_name):
    """Decorator to measure and record time spent in a step (ms)."""

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


def angles_to_gcode(angles):
    """Convert joint angles to G-code X/Y/Z/A/B/C string."""
    return (
        "X" + str(round(angles[0], 3)) +
        "Y" + str(round(angles[1], 3)) +
        "Z" + str(round(angles[2], 3)) +
        "A" + str(round(angles[3], 3)) +
        "B" + str(round(angles[4], 3)) +
        "C" + str(round(angles[5], 3))
    )


@timed("pd_solve")
def mpc_solve_qp(q, target_angles, speed_pct=50.0, q_vel=None, prev_q=None, dt=None):
    """
    PD controller: u = Kp * e - Kd * q_vel, clipped to bounds.
    q = joint position (deg), q_vel = joint velocity (deg/s) from LinuxCNC or estimated.
    Falls back to estimated velocity from (q - prev_q) / dt if q_vel is None.
    """
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current

    # Derivative term: use q_vel from LinuxCNC s.joint[i]["velocity"], or estimate
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        prev = np.array(prev_q, dtype=float)
        velocity = (current - prev) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    u_opt = KP * error - KD * velocity
    u_opt = np.clip(u_opt, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_cmd = current + u_opt
    return [float(x) for x in next_cmd]


@timed("poll")
def _poll_feedback(s):
    """Poll LinuxCNC and return (q, q_vel, t_status)."""
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
    """Send MDI command to LinuxCNC. Assumes MDI mode already set (done once at loop start).
    Note: c.mdi() uses NML IPC (~20ms). For faster control use Option 2 (HAL direct write)."""
    speed = speed_pct * MAX_ANGULAR_SPEED / 100
    gcode = "G38.3F" + str(speed) + angles_to_gcode(next_cmd)
    c.mdi(gcode)
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _print_timing_summary(loop_count):
    """Print average time (ms) per step for logged iterations."""
    if not _timing["poll"]:
        return
    n = len(_timing["poll"])
    avg = lambda key: sum(_timing[key][-n:]) / n if _timing[key] else 0
    poll_ms = avg("poll")
    solve_ms = avg("pd_solve")
    send_ms = avg("send_cmd")
    sleep_ms = avg("sleep") if _timing["sleep"] else 0
    total_ms = poll_ms + solve_ms + send_ms + sleep_ms
    print(f"  [timing] poll={poll_ms:.2f}ms pd_solve={solve_ms:.2f}ms send_cmd={send_ms:.2f}ms sleep={sleep_ms:.2f}ms total={total_ms:.2f}ms")


def run_mpc_loop(target_angles, speed_pct=50.0, duration_sec=10.0):
    """
    Run the MPC control loop: poll -> solve -> send.
    LinuxCNC must be running (e.g. axis-qt) before starting.
    """
    c = linuxcnc.command()
    s = linuxcnc.stat()

    print("MPC loop starting. Target:", target_angles)
    print("Press Ctrl+C to stop.\n")

    # Ensure MDI mode once at start (avoids mode check + wait_complete inside loop)
    s.poll()
    if s.task_mode != linuxcnc.MODE_MDI:
        c.mode(linuxcnc.MODE_MDI)
        c.wait_complete()

    t_start = time.time()
    loop_count = 0
    prev_current = None
    t_prev = None
    # Reset timing for this run
    for k in _timing:
        _timing[k] = []

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else None
        t_prev = t_loop_start

        # 1. Poll feedback
        q, q_vel, t_status = _poll_feedback(s)

        # 2. PD solve
        next_cmd = mpc_solve_qp(q, target_angles, speed_pct, q_vel=q_vel, prev_q=prev_current, dt=dt)
        prev_current = q.copy()

        # 3. Send command
        t_cmd = _send_mdi_command(c, next_cmd, speed_pct)

        # 4. Sleep to maintain loop period
        elapsed = time.time() - t_loop_start
        sleep_time = MPC_PERIOD_MS / 1000.0 - elapsed
        if sleep_time > 0:
            t0 = time.perf_counter()
            time.sleep(sleep_time)
            _timing["sleep"].append((time.perf_counter() - t0) * 1000)
        else:
            _timing["sleep"].append(0.0)

        loop_count += 1
        err = sum((t - a) ** 2 for t, a in zip(target_angles, q)) ** 0.5
        if loop_count % 5 == 0 or loop_count <= 3:
            vel_str = f" q_vel={[round(v, 3) for v in q_vel[:3]]}" if q_vel else ""
            print(f"Loop {loop_count}: status_recv={t_status} cmd_sent={t_cmd} q={q[:3]}... err={err:.3f}{vel_str}")
            _print_timing_summary(loop_count)

    print(f"\nDone. Ran {loop_count} MPC iterations.")
    _print_timing_summary(loop_count)


def main():
    s = linuxcnc.stat()
    s.poll()
    initial = [-90,-90,0,-90,0,0]
    current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    # Safety: target = current + 5.0 degrees per joint (small move from current pose)
    target = [a + 5.0 for a in current]
    print("Current angles:", current)
    print("Target (current + 5° each):", target)
    # for i in range(10):
        # run_mpc_loop(initial, speed_pct=(i+1)*5.0, duration_sec=10.0)
        # run_mpc_loop(target, speed_pct=(i+1)*5.0, duration_sec=10.0)
    run_mpc_loop(target, speed_pct=15.0, duration_sec=10.0)
    run_mpc_loop(initial, speed_pct=5.0, duration_sec=10.0)
    print("Done.")
    sys.exit(0)



if __name__ == "__main__":
    main()
