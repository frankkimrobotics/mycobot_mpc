#!/usr/bin/env python3
"""
MPC control via HAL direct write (Option 2: MPC Bypasses Trajectory Planner).
Same logic as mpc_linuxcnc.py but writes joint commands to HAL pins instead of MDI.

Requires HAL setup: load mpc component and wire mpc.jointN_pos_cmd to pid.N.command  
(via mux_generic when mpc.enable=1). See mpc_hal_setup.hal and README.
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
U_MAX_PER_STEP = 8.0
KP = 0.5
KD = 0.1

# MPC parameters
MPC_HORIZON = 20    # prediction horizon (N steps)
MPC_Q = 10.0        # state cost weight (tracking error)
MPC_R = 0.1         # control effort weight
MPC_Q_TERMINAL = 50.0  # terminal state cost weight

# Suction pump
SUCTION_PIN = "pro600.digital_out00"  # HAL pin for suction pump

# Logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Timing
_timing = {"poll": [], "pd_solve": [], "mpc_solve": [], "hal_write": [], "sleep": []}


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
def pd_solve(q, target_angles, q_vel=None, prev_q=None, dt=None):
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


# ── MPC (QP-based) solver ────────────────────────────────────────────────────
# Uses OSQP for real-time QP solving (~0.1ms per solve on Pi).
# Formulation per joint (independent, solved in batch):
#   Plant:  x(k+1) = x(k) + u(k)       (single integrator, position += step)
#   Cost:   sum_{k=0}^{N-1} [ Q * (x(k) - x_ref)^2 + R * u(k)^2 ]
#           + Q_terminal * (x(N) - x_ref)^2
#   Constraints: -U_MAX <= u(k) <= U_MAX  for all k
#
# Decision variable: u = [u(0), u(1), ..., u(N-1)]  per joint
# State is eliminated: x(k) = x(0) + sum_{i=0}^{k-1} u(i)

try:
    import osqp
    from scipy import sparse
    _HAS_OSQP = True
except ImportError:
    _HAS_OSQP = False


def _build_mpc_qp(N, Q, R, Q_term, u_max):
    """Pre-build QP matrices for a single-joint MPC (reused across solves).

    Returns (P, A, l_template, u_template, solver_settings) that only need
    q-vector update per solve (depends on current state and target).
    """
    # State propagation: x(k) = x0 + sum_{i<k} u(i)
    # So x(k) - x_ref = (x0 - x_ref) + sum_{i<k} u(i)
    # Let e0 = x0 - x_ref (scalar, updated each solve)
    #
    # Cost = sum_{k=1}^{N} Q_k * (e0 + sum_{i<k} u(i))^2 + sum_{k=0}^{N-1} R * u(k)^2
    # where Q_k = Q for k<N, Q_term for k=N
    #
    # Expanding: quadratic in u → P matrix, linear in u → q vector (depends on e0)

    # Build cumulative sum matrix S: S[k,i] = 1 if i < k+1 (for k=0..N-1 representing x(1)..x(N))
    S = np.tril(np.ones((N, N)))  # S[k,i] = 1 for i <= k

    # Weight vector for states x(1)..x(N)
    w = np.full(N, Q)
    w[-1] = Q_term

    W = np.diag(w)

    # P = S^T W S + R * I  (Hessian, N x N)
    P = S.T @ W @ S + R * np.eye(N)
    P = sparse.csc_matrix(P)

    # q = S^T W @ ones * e0  → q_vec = (S^T @ w) * e0 (computed per solve)
    q_coeffs = S.T @ w  # N-vector, multiply by e0 each solve

    # Constraints: -u_max <= u(k) <= u_max
    A = sparse.eye(N, format="csc")
    l_bound = np.full(N, -u_max)
    u_bound = np.full(N, u_max)

    return P, A, l_bound, u_bound, q_coeffs


class MPCSolver:
    """Pre-compiled OSQP solver for single-integrator MPC, one per joint.

    Warm-starts between solves for speed.
    """

    def __init__(self, N=MPC_HORIZON, Q=MPC_Q, R=MPC_R, Q_term=MPC_Q_TERMINAL,
                 u_max=U_MAX_PER_STEP):
        self.N = N
        self.q_coeffs = None
        self.solvers = []  # one OSQP instance per joint

        if not _HAS_OSQP:
            raise ImportError("osqp not installed. Run: pip install osqp")

        P, A, l_bound, u_bound, self.q_coeffs = _build_mpc_qp(N, Q, R, Q_term, u_max)

        # Create one solver per joint (same structure, different q vector each solve)
        for _ in range(MAX_JOINTS):
            solver = osqp.OSQP()
            solver.setup(P, np.zeros(N), A, l_bound, u_bound,
                         warm_start=True, verbose=False,
                         eps_abs=1e-4, eps_rel=1e-4,
                         max_iter=200, polish=False)
            self.solvers.append(solver)

    def solve(self, q_current, target, dt):
        """Solve MPC for all joints. Returns (next_pos, vel_cmd) lists."""
        next_pos = []
        vel_cmd = []
        for j in range(MAX_JOINTS):
            e0 = q_current[j] - target[j]
            q_vec = self.q_coeffs * e0  # linear cost term

            self.solvers[j].update(q=q_vec)
            result = self.solvers[j].solve()

            if result.info.status == "solved" or result.info.status == "solved_inaccurate":
                u0 = result.x[0]
            else:
                # Fallback: simple proportional step
                u0 = np.clip(-KP * e0, -U_MAX_PER_STEP, U_MAX_PER_STEP)

            u0 = float(u0)
            next_pos.append(q_current[j] + u0)
            vel_cmd.append(u0 / dt if dt > 0 else 0.0)

        return next_pos, vel_cmd


# Global MPC solver instance (lazy init)
_mpc_solver = None


@timed("mpc_solve")
def mpc_solve(q, target_angles, dt):
    """Solve MPC QP for all joints. Returns (next_pos, vel_cmd).

    Uses OSQP with warm-starting. Falls back to PD if OSQP unavailable.
    """
    global _mpc_solver
    if _mpc_solver is None:
        _mpc_solver = MPCSolver()
    if dt is None or dt <= 0:
        dt = MPC_PERIOD_MS / 1000.0
    return _mpc_solver.solve(q, target_angles, dt)


def mpc_control_loop(h, s, target_angles, duration_sec=10.0,
                     pos_tol=0.5, vel_tol=1.0, settle_steps=10):
    """Run MPC control loop: poll -> QP solve (N-step horizon) -> HAL write.

    Uses OSQP to solve a QP over MPC_HORIZON steps per iteration,
    applies only the first control action (receding horizon).

    Early-stops when position error norm < pos_tol (deg) AND velocity norm
    < vel_tol (deg/s) for settle_steps consecutive iterations.
    """
    print("MPC control loop starting. Target:", target_angles)
    print(f"  Horizon={MPC_HORIZON}, Q={MPC_Q}, R={MPC_R}, Q_term={MPC_Q_TERMINAL}")
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

    # Log buffer
    log_rows = []

    while (time.time() - t_start) < duration_sec:
        t_loop_start = time.time()
        dt = (t_loop_start - t_prev) if t_prev is not None else None
        t_prev = t_loop_start

        # 1. Poll feedback
        q, q_vel, t_status = _poll_feedback(s)

        # 2. MPC solve (N-step QP, apply first action)
        next_pos, vel_cmd = mpc_solve(q, target_angles,
                                      dt if dt else MPC_PERIOD_MS / 1000.0)
        # Estimate velocity for logging (before overwriting prev_current)
        if prev_current is not None and dt is not None and dt > 0:
            est_vel = [(q[i] - prev_current[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()

        # 3. HAL write
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

        # Early stop
        if err < pos_tol and vel_norm < vel_tol:
            converged_count += 1
            if converged_count >= settle_steps:
                print(f"  Converged at loop {loop_count}: err={err:.3f}° vel_norm={vel_norm:.3f}°/s")
                break
        else:
            converged_count = 0

        # Collect log row
        log_rows.append([
            t_status, loop_count,
            *q, *est_vel, *target_angles, *next_pos, *vel_cmd,
            err,
            _timing["poll"][-1], _timing["mpc_solve"][-1],
            _timing["hal_write"][-1], _timing["sleep"][-1],
        ])

        if loop_count % 10 == 0 or loop_count <= 3:
            vel_str = f" q_vel={[round(v, 3) for v in q_vel[:3]]}" if q_vel else ""
            vcmd_str = f" vcmd={[round(v, 1) for v in vel_cmd[:3]]}"
            print(f"Loop {loop_count}: status_recv={t_status} hal_write={t_cmd} q={q[:3]}... err={err:.3f}{vel_str}{vcmd_str}")
            if _timing["mpc_solve"]:
                n = len(_timing["mpc_solve"])
                avg_ms = sum(_timing["mpc_solve"][-n:]) / n
                print(f"  [timing] mpc_solve={avg_ms:.2f}ms")

    h["enable"] = False
    print(f"\nDone. Ran {loop_count} MPC iterations.")
    _save_log(log_rows, target_angles)


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


_last_log_filename = None  # basename of last saved CSV (for desktop fetch)


def _save_log(log_rows, target_angles):
    """Write collected log rows to a timestamped CSV file. Sets _last_log_filename for fetch."""
    global _last_log_filename
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
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(log_rows)
    _last_log_filename = os.path.basename(filename)
    print(f"  Log saved: {filename} ({len(log_rows)} rows)")


def pd_control_loop(h, s, target_angles, duration_sec=10.0,
                 pos_tol=0.5, vel_tol=1.0, settle_steps=10):
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
        # q_vel from LinuxCNC is ~0 (motion planner bypassed), use prev_q estimation instead
        next_pos, vel_cmd = pd_solve(q, target_angles, q_vel=None, prev_q=prev_current, dt=dt)
        # Estimate velocity for logging (before overwriting prev_current)
        if prev_current is not None and dt is not None and dt > 0:
            est_vel = [(q[i] - prev_current[i]) / dt for i in range(MAX_JOINTS)]
        else:
            est_vel = [0.0] * MAX_JOINTS
        prev_current = q.copy()

        # 3. HAL write — position + velocity
        t_cmd = _write_hal_cmd(h, next_pos, vel_cmd)

        # 4. Sleep
        elapsed = time.time() - t_loop_start
        # sleep_time = MPC_PERIOD_MS / 1000.0 - elapsed
        sleep_time = 0.0
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
        log_rows.append([
            t_status, loop_count,
            *q, *est_vel, *target_angles, *next_pos, *vel_cmd,
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


def suction_pump(on=True):
    """Turn suction pump on or off via HAL GPIO pin."""
    val = 1 if on else 0
    print(f"  Suction pump: {'ON' if on else 'OFF'} ({SUCTION_PIN}={val})")
    _halcmd_set(SUCTION_PIN, val)


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Integrated TCP streaming server (runs as background daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════

STREAM_PORT = 9999
STREAM_RATE_HZ = 50.0


def _stream_server_thread(port: int, rate_hz: float):
    """Background thread: poll LinuxCNC joint positions and broadcast over TCP.

    Desktop client (robot_pose_stream_ros2.py client) connects here to feed
    rviz2 with live joint angles. Multiple clients supported.
    """
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
                print(f"[stream] Client connected: {addr} (total: {len(clients)})")
            except OSError:
                break

    threading.Thread(target=accept_loop, daemon=True).start()

    period = 1.0 / rate_hz
    loop_count = 0

    while True:
        t0 = time.time()
        try:
            stat.poll()
            joints_deg = [round(stat.joint_actual_position[i], 4)
                          for i in range(MAX_JOINTS)]
        except (RuntimeError, OSError):
            time.sleep(1.0)
            continue

        msg = json.dumps({
            "joints_deg": joints_deg,
            "timestamp": time.time(),
        }) + "\n"
        msg_bytes = msg.encode("utf-8")

        dead = []
        with clients_lock:
            for conn in clients:
                try:
                    conn.sendall(msg_bytes)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(conn)
            for conn in dead:
                clients.remove(conn)
                conn.close()

        if dead:
            print(f"[stream] {len(dead)} client(s) disconnected (remaining: {len(clients)})")

        loop_count += 1
        if loop_count % (int(rate_hz) * 10) == 0:
            print(f"[stream] loop={loop_count} q={[round(j, 1) for j in joints_deg]} clients={len(clients)}")

        elapsed = time.time() - t0
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def start_stream_server(port: int = STREAM_PORT, rate_hz: float = STREAM_RATE_HZ):
    """Launch the TCP streaming server as a daemon thread (non-blocking)."""
    t = threading.Thread(target=_stream_server_thread, args=(port, rate_hz), daemon=True)
    t.start()
    print(f"[stream] Background streaming server started on port {port}")
    return t


# ═══════════════════════════════════════════════════════════════════════════════
#  Command server: receives target joint angles from desktop (control_robot.py)
# ═══════════════════════════════════════════════════════════════════════════════

CMD_PORT = 9998

# Shared command queue: desktop sends commands, main loop consumes them
_cmd_queue = queue.Queue()

# Shared status dict: main loop writes, command server reads & sends to client
_cmd_status = {
    "state": "idle",
    "current_deg": [0.0] * MAX_JOINTS,
    "target_deg": [0.0] * MAX_JOINTS,
    "error_norm": 0.0,
    "last_log_name": None,  # basename of last CSV on robot (for desktop fetch)
}
_cmd_status_lock = threading.Lock()


def _handle_cmd_client(conn, addr):
    """Handle a single command client connection.

    Protocol (JSON lines over TCP):
      Desktop → Robot:  {"target_deg": [j1..j6], "duration": 5.0, "controller": "pd"}\n
      Robot → Desktop:  {"state": "ack|moving|done|error", ...}\n  (periodic updates)
    """
    print(f"[cmd] Client connected: {addr}")
    buffer = ""
    conn.settimeout(1.0)

    try:
        while True:
            # Send periodic status updates
            with _cmd_status_lock:
                status = dict(_cmd_status)
            try:
                conn.sendall((json.dumps(status) + "\n").encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                break

            # Check for incoming commands
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
                    except json.JSONDecodeError:
                        continue

                    if "target_deg" in cmd:
                        print(f"[cmd] Received target: {[round(v, 1) for v in cmd['target_deg']]}")
                        _cmd_queue.put(cmd)
                        ack = {"state": "ack", "target_deg": cmd["target_deg"]}
                        conn.sendall((json.dumps(ack) + "\n").encode("utf-8"))
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
            except socket.timeout:
                pass

            time.sleep(0.01)  # status update rate ~100 Hz
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()
        print(f"[cmd] Client disconnected: {addr}")


def _command_server_thread(port: int):
    """Background thread: accept command connections from desktop."""
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


def start_command_server(port: int = CMD_PORT):
    """Launch the command server as a daemon thread (non-blocking)."""
    t = threading.Thread(target=_command_server_thread, args=(port,), daemon=True)
    t.start()
    print(f"[cmd] Background command server started on port {port}")
    return t


def _update_cmd_status(state, current_deg, target_deg, error_norm, **extra):
    """Update shared status dict (thread-safe)."""
    with _cmd_status_lock:
        _cmd_status["state"] = state
        _cmd_status["current_deg"] = [round(v, 3) for v in current_deg]
        _cmd_status["target_deg"] = [round(v, 3) for v in target_deg]
        _cmd_status["error_norm"] = round(error_norm, 4)
        for k, v in extra.items():
            _cmd_status[k] = v


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MPC control for myCobot Pro 630")
    parser.add_argument("--suction", action="store_true", default=False,
                        help="Turn on suction pump during operation (default: off)")
    parser.add_argument("--stream-port", type=int, default=STREAM_PORT,
                        help=f"TCP port for rviz2 streaming server (default: {STREAM_PORT}, 0=disable)")
    parser.add_argument("--stream-rate", type=float, default=STREAM_RATE_HZ,
                        help=f"Streaming rate in Hz (default: {STREAM_RATE_HZ})")
    parser.add_argument("--cmd-port", type=int, default=CMD_PORT,
                        help=f"TCP port for command server (default: {CMD_PORT}, 0=disable)")
    args = parser.parse_args()

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

    # Start integrated streaming server for rviz2 visualization
    if args.stream_port > 0:
        start_stream_server(port=args.stream_port, rate_hz=args.stream_rate)

    # Start command server for desktop control (control_robot.py)
    if args.cmd_port > 0:
        start_command_server(port=args.cmd_port)

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
    print("Current angles:", current)

    # Suction pump
    if args.suction:
        suction_pump(on=True)

    _update_cmd_status("idle", current, current, 0.0)

    # Main loop: wait for commands from the desktop (control_robot.py)
    print("\n" + "=" * 60)
    print("Waiting for commands from desktop (control_robot.py)...")
    print(f"  Command port: {args.cmd_port}")
    print(f"  Stream port:  {args.stream_port}")
    print("  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    try:
        while True:
            try:
                cmd = _cmd_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            target = cmd.get("target_deg")
            if target is None or len(target) != MAX_JOINTS:
                print(f"[cmd] Invalid target: {target}")
                continue

            duration = cmd.get("duration", 2.0)
            controller = cmd.get("controller", "pd")
            pos_tol = cmd.get("pos_tol", 0.5)
            settle = cmd.get("settle_steps", 10)

            t_cmd_start = time.perf_counter()

            s.poll()
            current = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            err = sum((t - c) ** 2 for t, c in zip(target, current)) ** 0.5
            print(f"\n[cmd] Moving: {[round(v, 1) for v in current]} → {[round(v, 1) for v in target]}")
            print(f"       distance={err:.1f}° duration={duration}s controller={controller}"
                  f" pos_tol={pos_tol}° settle={settle}")

            _update_cmd_status("moving", current, target, err)

            if controller == "mpc":
                mpc_control_loop(h, s, target, duration_sec=duration,
                                 pos_tol=pos_tol, settle_steps=settle)
            else:
                pd_control_loop(h, s, target, duration_sec=duration,
                                pos_tol=pos_tol, settle_steps=settle)

            robot_exec_ms = (time.perf_counter() - t_cmd_start) * 1000

            s.poll()
            final = [round(s.joint_actual_position[i], 3) for i in range(MAX_JOINTS)]
            final_err = sum((t - f) ** 2 for t, f in zip(target, final)) ** 0.5

            # Compute per-loop timing averages for this command
            n_loops = len(_timing["poll"]) if _timing["poll"] else 1
            avg_poll = sum(_timing["poll"][-n_loops:]) / n_loops if _timing["poll"] else 0
            solve_key = "mpc_solve" if controller == "mpc" else "pd_solve"
            avg_solve = sum(_timing[solve_key][-n_loops:]) / n_loops if _timing[solve_key] else 0
            avg_hal = sum(_timing["hal_write"][-n_loops:]) / n_loops if _timing["hal_write"] else 0
            avg_sleep = sum(_timing["sleep"][-n_loops:]) / n_loops if _timing["sleep"] else 0

            _update_cmd_status(
                "done", final, target, final_err,
                robot_exec_ms=round(robot_exec_ms, 2),
                n_loops=n_loops,
                avg_poll_ms=round(avg_poll, 3),
                avg_solve_ms=round(avg_solve, 3),
                avg_hal_write_ms=round(avg_hal, 3),
                avg_sleep_ms=round(avg_sleep, 3),
                last_log_name=_last_log_filename,
            )

            print(f"[cmd] Done. exec={robot_exec_ms:.0f}ms loops={n_loops} err={final_err:.3f}°"
                  f" [avg poll={avg_poll:.2f} solve={avg_solve:.2f} hal={avg_hal:.2f} sleep={avg_sleep:.2f} ms]")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        h["enable"] = False
        if args.suction:
            suction_pump(on=False)
        power_off_robot()

    print("Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
