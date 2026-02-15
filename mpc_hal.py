#!/usr/bin/env python3
"""
MPC control via HAL direct write (Option 2: MPC Bypasses Trajectory Planner).
Same logic as mpc_linuxcnc.py but writes joint commands to HAL pins instead of MDI.

Requires HAL setup: load mpc component and wire mpc.jointN_pos_cmd to pid.N.command  
(via mux_generic when mpc.enable=1). See mpc_hal_setup.hal and README.
"""

import time
import sys
import csv
import os
from datetime import datetime
from functools import wraps
import numpy as np

# LinuxCNC and HAL
try:
    import linuxcnc
    import hal
except ImportError as e:
    print(f"Import error: {e}. Need linuxcnc and hal (run with LinuxCNC).")
    sys.exit(1)

# Constants from myCobot Pro 630
MAX_JOINTS = 6
MPC_PERIOD_MS = 2   # 100 Hz - HAL write is fast
U_MAX_PER_STEP = 5.0
KP = 0.3
KD = 0.1

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Timing
_timing = {"poll": [], "pd_solve": [], "hal_write": [], "sleep": []}


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


@timed("pd_solve")
def mpc_solve_qp(q, target_angles, q_vel=None, prev_q=None, dt=None):
    """PD controller returning (next_pos, vel_cmd).

    u = Kp * e - Kd * q_vel, clipped to ±U_MAX_PER_STEP.
    next_pos = current + u          (position command)
    vel_cmd  = u / dt               (velocity command in deg/s)
    """
    current = np.array(q, dtype=float)
    target = np.array(target_angles, dtype=float)
    error = target - current
    if q_vel is not None:
        velocity = np.array(q_vel, dtype=float)
    elif prev_q is not None and dt is not None and dt > 0:
        prev = np.array(prev_q, dtype=float)
        velocity = (current - prev) / dt
    else:
        velocity = np.zeros(MAX_JOINTS)
    u_opt = KP * error - KD * velocity
    u_opt = np.clip(u_opt, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_pos = current + u_opt
    # Velocity = position increment / dt (deg/s)
    if dt is not None and dt > 0:
        vel_cmd = u_opt / dt * 0.5
    else:
        vel_cmd = np.zeros(MAX_JOINTS)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


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


@timed("hal_write")
def _write_hal_cmd(h, next_pos, vel_cmd):
    """Write joint position and velocity commands to HAL pins."""
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = next_pos[i]
        h[f"joint{i}_vel_cmd"] = vel_cmd[i]
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _print_timing_summary(loop_count):
    if not _timing["poll"]:
        return
    n = len(_timing["poll"])
    avg = lambda key: sum(_timing[key][-n:]) / n if _timing[key] else 0
    poll_ms = avg("poll")
    solve_ms = avg("pd_solve")
    hal_ms = avg("hal_write")
    sleep_ms = avg("sleep") if _timing["sleep"] else 0
    total_ms = poll_ms + solve_ms + hal_ms + sleep_ms
    print(f"  [timing] poll={poll_ms:.2f}ms pd_solve={solve_ms:.2f}ms hal_write={hal_ms:.2f}ms sleep={sleep_ms:.2f}ms total={total_ms:.2f}ms")


def _save_log(log_rows, target_angles):
    """Write collected log rows to a timestamped CSV file."""
    if not log_rows:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_str = "_".join(str(int(a)) for a in target_angles)
    filename = os.path.join(LOG_DIR, f"mpc_{stamp}_t{target_str}.csv")
    header = (
        ["timestamp", "loop"]
        + [f"q{i}" for i in range(MAX_JOINTS)]
        + [f"qvel{i}" for i in range(MAX_JOINTS)]
        + [f"target{i}" for i in range(MAX_JOINTS)]
        + [f"cmd_pos{i}" for i in range(MAX_JOINTS)]
        + [f"cmd_vel{i}" for i in range(MAX_JOINTS)]
        + ["err_norm", "poll_ms", "pd_solve_ms", "hal_write_ms", "sleep_ms"]
    )
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(log_rows)
    print(f"  Log saved: {filename} ({len(log_rows)} rows)")


def run_mpc_loop(h, s, target_angles, duration_sec=10.0,
                 pos_tol=0.5, vel_tol=1.0, settle_steps=50):
    """Run MPC loop: poll -> PD solve -> HAL write.

    Early-stops when position error norm < pos_tol (deg) AND velocity norm
    < vel_tol (deg/s) for settle_steps consecutive iterations.
    """
    print("MPC HAL loop starting. Target:", target_angles)
    print(f"  Early stop: pos_tol={pos_tol}°, vel_tol={vel_tol}°/s, settle={settle_steps} steps")
    print("Press Ctrl+C to stop.\n")

    # Enable MPC override
    h["enable"] = True

    t_start = time.time()
    loop_count = 0
    converged_count = 0
    prev_current = None
    t_prev = None
    for k in _timing:
        _timing[k] = []

    # Log buffer: collect rows in memory, flush to CSV after loop
    log_rows = []

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else None
        t_prev = t_loop_start

        # 1. Poll feedback
        q, q_vel, t_status = _poll_feedback(s)

        # 2. PD solve → (position, velocity)
        next_pos, vel_cmd = mpc_solve_qp(q, target_angles, q_vel=q_vel, prev_q=prev_current, dt=dt)
        prev_current = q.copy()

        # 3. HAL write — position + velocity
        t_cmd = _write_hal_cmd(h, next_pos, vel_cmd)

        # 4. Sleep
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
        vel_norm = sum(v ** 2 for v in vel_cmd) ** 0.5

        # Early stop: converged if pos error and vel command are both small
        if err < pos_tol and vel_norm < vel_tol:
            converged_count += 1
            if converged_count >= settle_steps:
                print(f"  Converged at loop {loop_count}: err={err:.3f}° vel_norm={vel_norm:.3f}°/s")
                break
        else:
            converged_count = 0

        # Collect log row (no file I/O in the control loop)
        vel_list = q_vel if q_vel else [0.0] * MAX_JOINTS
        log_rows.append([
            t_status, loop_count,
            *q, *vel_list, *target_angles, *next_pos, *vel_cmd,
            err,
            _timing["poll"][-1], _timing["pd_solve"][-1],
            _timing["hal_write"][-1], _timing["sleep"][-1],
        ])

        if loop_count % 10 == 0 or loop_count <= 3:
            vel_str = f" q_vel={[round(v, 3) for v in q_vel[:3]]}" if q_vel else ""
            vcmd_str = f" vcmd={[round(v, 1) for v in vel_cmd[:3]]}"
            print(f"Loop {loop_count}: status_recv={t_status} hal_write={t_cmd} q={q[:3]}... err={err:.3f}{vel_str}{vcmd_str}")
            _print_timing_summary(loop_count)

    h["enable"] = False
    print(f"\nDone. Ran {loop_count} MPC iterations.")
    _print_timing_summary(loop_count)

    # Flush log to CSV
    _save_log(log_rows, target_angles)


import subprocess


def _halcmd_set(pin, value):
    """Set a HAL pin via halcmd."""
    os.system(f"halcmd setp {pin} {value}")


def _halcmd_get(pin):
    """Get a HAL pin value via halcmd. Returns the string value."""
    result = subprocess.run(
        ["halcmd", "getp", pin], capture_output=True, text=True
    )
    return result.stdout.strip()


def _halcmd_get_bool(pin):
    """Get a HAL bit pin value. Returns True/False."""
    return _halcmd_get(pin) == "TRUE"


def _halcmd_get_float(pin):
    """Get a HAL float pin value. Returns float."""
    try:
        return float(_halcmd_get(pin))
    except (ValueError, TypeError):
        return 0.0


def power_on_robot():
    """Power on robot via pro600.poweron pin (direct CAN command to motor controllers)."""
    print("  Powering on robot via pro600.poweron...")

    # Set pro600.poweron = 1 (tells hal_ele_master600 to power on motors via CAN)
    _halcmd_set("pro600.poweron", 1)
    time.sleep(1)

    # Wait for pro600.svr_poweroned
    for i in range(20):
        powered = _halcmd_get_bool("pro600.svr_poweroned")
        print(f"  Power check {i+1}/20: svr_poweroned={powered}")
        if powered:
            break
        time.sleep(1)
    else:
        print("  WARNING: pro600.svr_poweroned not TRUE after 20s")
        # Check individual servo status
        for j in range(MAX_JOINTS):
            enabled = _halcmd_get_bool(f"pro600.svr{j}_enabled")
            print(f"    svr{j}_enabled={enabled}")
        return False

    # Wait for pro600.svr_enabled (all servos)
    for i in range(20):
        enabled = _halcmd_get_bool("pro600.svr_enabled")
        print(f"  Servo check {i+1}/20: svr_enabled={enabled}")
        if enabled:
            break
        time.sleep(1)
    else:
        print("  WARNING: pro600.svr_enabled not TRUE after 20s")
        for j in range(MAX_JOINTS):
            en = _halcmd_get_bool(f"pro600.svr{j}_enabled")
            print(f"    svr{j}_enabled={en}")
        return False

    print("  Robot powered on and servos enabled!")
    return True


def power_off_robot():
    """Power off robot: disable MPC, set ESTOP, power off motors via CAN."""
    print("Powering off robot...")

    # Disable MPC so PIDs stop driving motors
    _halcmd_set("mpc.enable", 0)
    time.sleep(0.1)

    # Power off motors via CAN
    _halcmd_set("pro600.poweron", 0)
    time.sleep(1)

    # Verify power off
    for i in range(10):
        powered = _halcmd_get_bool("pro600.svr_poweroned")
        print(f"  Power-off check {i+1}/10: svr_poweroned={powered}")
        if not powered:
            break
        time.sleep(0.5)

    # Put LinuxCNC into ESTOP
    try:
        c = linuxcnc.command()
        c.state(linuxcnc.STATE_ESTOP)
        time.sleep(0.5)
    except Exception as e:
        print(f"  ESTOP command failed: {e}")

    s = linuxcnc.stat()
    s.poll()
    print(f"  Final state: task_state={s.task_state}, svr_poweroned={_halcmd_get_bool('pro600.svr_poweroned')}")
    print("  Robot powered off.")


def wait_for_stable_feedback(settle_time=3.0, check_interval=0.5):
    """Wait for pro600 encoder feedback to stabilize after motor init.

    Motor init over CAN takes variable time per joint. Joint feedback pins
    start at 0.0 and jump to actual positions once CAN frames arrive.
    We poll twice and confirm values are stable (not changing).
    """
    print(f"  Waiting {settle_time}s for encoder feedback to stabilize...")
    time.sleep(settle_time)

    fb1 = [_halcmd_get_float(f"pro600.joint{i}_posfb") for i in range(MAX_JOINTS)]
    time.sleep(check_interval)
    fb2 = [_halcmd_get_float(f"pro600.joint{i}_posfb") for i in range(MAX_JOINTS)]

    stable = all(abs(a - b) < 0.1 for a, b in zip(fb1, fb2))
    print(f"  Feedback check: {[round(v, 2) for v in fb2]}  stable={stable}")
    if not stable:
        print(f"  Feedback still changing, waiting 2s more...")
        time.sleep(2.0)
        fb2 = [_halcmd_get_float(f"pro600.joint{i}_posfb") for i in range(MAX_JOINTS)]
        print(f"  Feedback final:  {[round(v, 2) for v in fb2]}")
    return fb2


def preload_and_enable_mpc(h, feedback):
    """Pre-load MPC command pins with actual positions, then enable MPC.

    PIDs are gated ONLY by mpc.enable (not motion.motion-enabled).
    So PIDs stay OFF until this function sets mpc.enable = TRUE.
    This prevents the PID spike that was powering off the robot.
    """
    print("  Pre-loading MPC commands with stable feedback...")
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = feedback[i]
        h[f"joint{i}_vel_cmd"] = 0.0  # zero velocity at startup (hold position)
        print(f"    joint{i}: pos={feedback[i]:.3f}  vel=0.0")

    # Small delay to ensure position pins are committed to shared memory
    # before the enable bit activates the PIDs on the next servo cycle
    time.sleep(0.02)

    # NOW enable MPC → PIDs activate with command ≈ actual → zero error
    h["enable"] = True
    print("  mpc.enable = TRUE → PIDs now active (command matches feedback)")


def enable_machine(h, timeout=60.0):
    """Power on robot, then bring machine from ESTOP → ON → stable PIDs.

    Sequence (PIDs are gated by mpc.enable, NOT motion.motion-enabled):
      1. Clear ESTOP
      2. Power on robot hardware (pro600.poweron)
      3. STATE_ON + home joints  (PIDs still OFF — mpc.enable is FALSE)
      4. Wait for encoder feedback to stabilize (CAN init completes)
      5. Pre-load MPC commands with valid feedback
      6. Set mpc.enable = TRUE → PIDs enable with zero error → safe
    """
    c = linuxcnc.command()
    s = linuxcnc.stat()

    # Step 1: Clear ESTOP first (needed before pro600 can enable)
    print("  Clearing ESTOP...")
    for _ in range(5):
        s.poll()
        if s.task_state == linuxcnc.STATE_ESTOP:
            c.state(linuxcnc.STATE_ESTOP_RESET)
            time.sleep(0.5)
        else:
            break
    s.poll()
    print(f"  After ESTOP clear: task_state={s.task_state}")

    # Step 2: Power on the robot hardware
    if not power_on_robot():
        print("  Robot power-on failed, trying to continue anyway...")

    # Step 3: State machine: ESTOP_RESET → ON (PIDs stay OFF, mpc.enable=FALSE)
    t0 = time.time()
    machine_on = False
    while (time.time() - t0) < timeout:
        s.poll()
        state = s.task_state

        if state == linuxcnc.STATE_ESTOP:
            print(f"  task_state={state} (ESTOP) → sending ESTOP_RESET...")
            c.state(linuxcnc.STATE_ESTOP_RESET)
            time.sleep(0.5)

        elif state == linuxcnc.STATE_ESTOP_RESET:
            motion_en = _halcmd_get_bool("motion.enable")
            svr_en = _halcmd_get_bool("pro600.svr_enabled")
            print(f"  task_state={state} (ESTOP_RESET), motion.enable={motion_en}, svr_enabled={svr_en}")
            if motion_en:
                print(f"  → sending STATE_ON...")
                c.state(linuxcnc.STATE_ON)
            time.sleep(1)

        elif state == linuxcnc.STATE_OFF:
            print(f"  task_state={state} (OFF) → sending STATE_ON...")
            c.state(linuxcnc.STATE_ON)
            time.sleep(0.5)

        elif state == linuxcnc.STATE_ON:
            print(f"  task_state={state} (ON) — machine enabled!")
            _halcmd_set("or2.0.in1", 1)
            # Home all joints (absolute encoders, instant)
            s.poll()
            c.mode(linuxcnc.MODE_MANUAL)
            c.wait_complete()
            c.teleop_enable(0)
            c.wait_complete()
            for j in range(MAX_JOINTS):
                if not s.joint[j]["homed"]:
                    c.home(j)
            time.sleep(0.5)
            s.poll()
            homed = all(s.joint[j]["homed"] for j in range(MAX_JOINTS))
            print(f"  All joints homed: {homed}")
            machine_on = True
            break

        else:
            print(f"  task_state={state} (unknown), waiting...")
            time.sleep(0.5)

    if not machine_on:
        print(f"ERROR: Could not enable machine within {timeout}s")
        return False

    # Step 4: Wait for encoder feedback to stabilize (CAN init must complete)
    # PIDs are still OFF (mpc.enable=FALSE), so no spike even if feedback jumps
    feedback = wait_for_stable_feedback(settle_time=3.0)

    # Step 5: Pre-load MPC commands and enable PIDs
    preload_and_enable_mpc(h, feedback)

    return True


def main():
    # Create HAL component
    try:
        h = hal.component("mpc")
        for i in range(MAX_JOINTS):
            h.newpin(f"joint{i}_pos_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
            h.newpin(f"joint{i}_vel_cmd", hal.HAL_FLOAT, hal.HAL_OUT)
        h.newpin("enable", hal.HAL_BIT, hal.HAL_OUT)
        h.ready()
    except Exception as e:
        print(f"HAL component creation failed: {e}")
        print("Ensure LinuxCNC is running and mpc component not already loaded.")
        sys.exit(1)

    # Initialize pins
    for i in range(MAX_JOINTS):
        h[f"joint{i}_pos_cmd"] = 0.0
        h[f"joint{i}_vel_cmd"] = 0.0
    h["enable"] = False

    # Enable machine (ESTOP → ON) — no GUI needed
    print("Enabling machine...")
    time.sleep(2)  # wait for LinuxCNC startup to settle
    if not enable_machine(h):
        print("Failed to enable machine. Exiting.")
        sys.exit(1)

    # Get current position from LinuxCNC
    s = linuxcnc.stat()
    s.poll()
    current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
    init = [-90,-90,0,-90,0,0]
    target = [a + 15.0 for a in current]
    print("Current angles:", current)
    print("Target (current + 5° each):", target)

    # Run (Ctrl+C to stop)
    try:
        init = [-90, -90, 0, -90, 0, 0]
        for i in range(5):
            target = [a + np.random.uniform(0, 5) for a in init]
            print(f"\n=== Run {i+1}/5: target={[round(t,1) for t in target]} ===")
            run_mpc_loop(h, s, init, duration_sec=5.0)
            run_mpc_loop(h, s, target, duration_sec=5.0)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        h["enable"] = False
        power_off_robot()

    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
