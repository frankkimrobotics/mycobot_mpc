"""
Shared control solvers (no HAL/LinuxCNC). Used by robot_hal.py and mujoco_viewer.py.

All angles and angular velocities are in degrees (LinuxCNC convention).
Optional: invdyn_model for gravity comp; tinympc/osqp for MPC.
Gains are loaded from controller_params.yaml via controller_params.get_controller_params().
"""
from __future__ import annotations  # keep PEP 585/604 hints lazy for Pi/LinuxCNC Python 3.7.3

import numpy as np

from controller_params import get_controller_params

try:
    from invdyn_model import (
        load_params as load_invdyn_params,
        linuxcnc_deg_to_rad,
        compute_MCG,
        NUM_JOINTS as MODEL_NUM_JOINTS,
    )
except Exception:   # invdyn_model is optional (gravity comp); also skips its 3.7-incompat syntax on the Pi
    load_invdyn_params = None
    linuxcnc_deg_to_rad = None
    compute_MCG = None
    MODEL_NUM_JOINTS = 6

# Load once at import from controller_params.yaml
_P = get_controller_params()
MAX_JOINTS = int(_P.get("max_joints", 6))
PERIOD_MS = float(_P.get("period_ms", 20))
PERIOD_SEC = PERIOD_MS / 1000.0


def _gain_array(scalar: float, per_joint: list | None, nj: int = MAX_JOINTS) -> np.ndarray:
    """Return (nj,) gain array from scalar or per-joint list (length >= nj)."""
    if per_joint is not None and len(per_joint) >= nj:
        return np.asarray(per_joint[:nj], dtype=float)
    return np.full(nj, scalar, dtype=float)


# ---------------------------------------------------------------------------
# Per-command runtime overrides (set by robot_hal from the "gains" field of a
# command; cleared after the move). Keys are controller-specific:
#   pid:      kp, kd (scalar or 6-list), ki, u_max, integral_clamp
#   pd_velff: kp, kd
#   mpc:      k0, k1, vmax (deg/s), vel_scale (drive units per deg/s)
# ---------------------------------------------------------------------------
_OVR = {}


def set_overrides(d):
    global _OVR
    _OVR = dict(d) if isinstance(d, dict) else {}


def get_overrides():
    return dict(_OVR)


def _ov(name, default):
    v = _OVR.get(name)
    if v is None:
        return default
    if isinstance(default, np.ndarray):
        if isinstance(v, (list, tuple)):
            return np.asarray(v, dtype=float)[:MAX_JOINTS]
        return np.full(MAX_JOINTS, float(v))
    return float(v)


# PID
_pid = _P.get("pid", {})
KP_PID = float(_pid.get("kp", 0.5))
KD_PID = float(_pid.get("kd", 0.1))
KP_PID_ARR = _gain_array(KP_PID, _pid.get("kp_per_joint"))
KD_PID_ARR = _gain_array(KD_PID, _pid.get("kd_per_joint"))
KI_PID = float(_pid.get("ki", 0.05))
INTEGRAL_CLAMP = float(_pid.get("integral_clamp", 50.0))
U_MAX_PER_STEP = float(_pid.get("u_max_per_step", 8.0))

# InvDyn
_invdyn = _P.get("invdyn", {})
USE_PD_STEPS_FOR_INVDYN = bool(_invdyn.get("use_pd_steps", True))
INVDYN_KP = float(_invdyn.get("kp", 144.0))
INVDYN_KD = float(_invdyn.get("kd", 24.0))
INVDYN_KP_ARR = _gain_array(INVDYN_KP, _invdyn.get("kp_per_joint"))
INVDYN_KD_ARR = _gain_array(INVDYN_KD, _invdyn.get("kd_per_joint"))
QDD_MAX_DEG = float(_invdyn.get("qdd_max_deg", 150.0))
KP_PD_INVDYN = float(_invdyn.get("kp_pd", 0.5))
KD_PD_INVDYN = float(_invdyn.get("kd_pd", 0.1))
KP_PD_INVDYN_ARR = _gain_array(KP_PD_INVDYN, _invdyn.get("kp_pd_per_joint"))
KD_PD_INVDYN_ARR = _gain_array(KD_PD_INVDYN, _invdyn.get("kd_pd_per_joint"))
K_GRAV_COMP = float(_invdyn.get("k_grav_comp", 0.02))

# PD + vel feedforward
_pd_velff = _P.get("pd_velff", {})
KP_PD_VELFF = float(_pd_velff.get("kp", 144.0 / 90.0))
KD_PD_VELFF = float(_pd_velff.get("kd", 24.0 / 90.0))
KP_PD_VELFF_ARR = _gain_array(KP_PD_VELFF, _pd_velff.get("kp_per_joint"))
KD_PD_VELFF_ARR = _gain_array(KD_PD_VELFF, _pd_velff.get("kd_per_joint"))
if "u_max_per_step" in _pd_velff:
    U_MAX_PER_STEP_PD_VELFF = float(_pd_velff["u_max_per_step"])
else:
    U_MAX_PER_STEP_PD_VELFF = U_MAX_PER_STEP

# MPC
_mpc = _P.get("mpc", {})
MPC_HORIZON = int(_mpc.get("horizon", 20))
MPC_Q = float(_mpc.get("q", 10.0))
MPC_R = float(_mpc.get("r", 0.1))
MPC_Q_TERMINAL = float(_mpc.get("q_terminal", 50.0))
MPC_RHO = float(_mpc.get("rho", 1.0))
if "u_max_per_step" in _mpc:
    U_MAX_PER_STEP_MPC = float(_mpc["u_max_per_step"])
else:
    U_MAX_PER_STEP_MPC = U_MAX_PER_STEP

_mpc_solver = None
_mpc_use_tinympc = None


# ---------------------------------------------------------------------------
# PID
# ---------------------------------------------------------------------------
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
    _ic = _ov("integral_clamp", INTEGRAL_CLAMP)
    integ = np.clip(integ, -_ic, _ic)
    u = _ov("kp", KP_PID_ARR) * error - _ov("kd", KD_PID_ARR) * velocity + _ov("ki", KI_PID) * integ
    _um = _ov("u_max", U_MAX_PER_STEP)
    u = np.clip(u, -_um, _um)
    next_pos = current + u
    vel_cmd = (u / dt * 0.5) if dt > 0 else np.zeros(MAX_JOINTS)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd], integ.tolist()


# ---------------------------------------------------------------------------
# InvDyn helpers
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
    qdd_d = INVDYN_KP_ARR * error - INVDYN_KD_ARR * velocity
    qdd_d = np.clip(qdd_d, -QDD_MAX_DEG, QDD_MAX_DEG)
    next_pos = current + velocity * dt + 0.5 * qdd_d * (dt ** 2)
    vel_cmd = velocity + qdd_d * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


def _invdyn_pd_solve(q, target_angles, q_vel=None, prev_q=None, dt=None, prev_target=None, params=None):
    """Direct position-step PD for invdyn; optional vel_ff; optional G(q) from npz."""
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
    u = KP_PD_INVDYN_ARR * error - KD_PD_INVDYN_ARR * velocity
    vel_ff = np.zeros(MAX_JOINTS)
    if prev_target is not None and dt > 0:
        prev_t = np.array(prev_target, dtype=float)
        vel_ff = (target - prev_t) / dt
    u = u + vel_ff * dt
    u = np.clip(u, -U_MAX_PER_STEP, U_MAX_PER_STEP)
    next_pos = current + u
    vel_cmd = (u / dt * 0.5) if dt > 0 else np.zeros(MAX_JOINTS)
    vel_cmd = np.array(vel_cmd, dtype=float) + vel_ff
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
    """Model-aware invdyn in deg; uses linuxcnc_deg_to_rad when available."""
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
    vd_command_rad = INVDYN_KP_ARR * (target_rad - q_rad) - INVDYN_KD_ARR * qd_rad
    qdd_d_deg = np.rad2deg(vd_command_rad)
    qdd_d_deg = np.clip(qdd_d_deg, -QDD_MAX_DEG, QDD_MAX_DEG)
    next_pos = current + velocity_deg * dt + 0.5 * qdd_d_deg * (dt ** 2)
    vel_cmd = velocity_deg + qdd_d_deg * dt
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


def invdyn_solve(q, target_angles, params, q_vel=None, prev_q=None, dt=None, prev_target=None):
    """Invdyn: PD position steps when USE_PD_STEPS_FOR_INVDYN; else acceleration integration."""
    if USE_PD_STEPS_FOR_INVDYN:
        return _invdyn_pd_solve(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt, prev_target=prev_target, params=params)
    if params is not None and (load_invdyn_params is not None or linuxcnc_deg_to_rad is not None):
        return _invdyn_model_solve_deg(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)
    return _invdyn_solve_fallback(q, target_angles, q_vel=q_vel, prev_q=prev_q, dt=dt)


# ---------------------------------------------------------------------------
# PD + velocity feedforward
# ---------------------------------------------------------------------------
def pd_velff_solve(q, target_angles, prev_target, q_vel=None, prev_q=None, dt=None):
    """PD on position + velocity feedforward. Returns (next_pos, vel_cmd)."""
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
    acc_pd = _ov("kp", KP_PD_VELFF_ARR) * error - _ov("kd", KD_PD_VELFF_ARR) * velocity
    next_pos = current + (acc_pd + vel_ff) * dt
    vel_cmd = acc_pd + vel_ff
    next_pos = np.clip(next_pos, current - U_MAX_PER_STEP_PD_VELFF, current + U_MAX_PER_STEP_PD_VELFF)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


# ---------------------------------------------------------------------------
# MPC (TinyMPC or OSQP fallback)
# ---------------------------------------------------------------------------
def _mpc_solve_tinympc(q, target_angles, dt):
    """Solve MPC using TinyMPC. Returns (next_pos, vel_cmd)."""
    import tinympc
    global _mpc_solver, _mpc_use_tinympc
    if _mpc_solver is None:
        nx = nu = MAX_JOINTS
        A = np.eye(nx, dtype=np.float64)
        B = np.eye(nu, dtype=np.float64)
        Q = np.diag(np.full(nx, MPC_Q))
        Q[-1] = MPC_Q_TERMINAL
        R = np.diag(np.full(nu, MPC_R))
        _mpc_solver = tinympc.TinyMPC()
        _mpc_solver.setup(A, B, Q, R, MPC_HORIZON, rho=MPC_RHO, verbose=False)
        u_min = np.full((nu, MPC_HORIZON - 1), -U_MAX_PER_STEP_MPC)
        u_max = np.full((nu, MPC_HORIZON - 1), U_MAX_PER_STEP_MPC)
        _mpc_solver.set_bound_constraints([], [], u_min, u_max)
    nx, nu = MAX_JOINTS, MAX_JOINTS
    x0 = np.asarray(q, dtype=np.float64)
    x_ref = np.tile(np.asarray(target_angles, dtype=np.float64).reshape(-1, 1), (1, MPC_HORIZON))
    u_ref = np.zeros((nu, MPC_HORIZON - 1))
    _mpc_solver.set_x0(x0)
    _mpc_solver.set_x_ref(x_ref)
    _mpc_solver.set_u_ref(u_ref)
    solution = _mpc_solver.solve()
    u0 = np.asarray(solution["controls"]).ravel()
    u0 = np.clip(u0, -U_MAX_PER_STEP_MPC, U_MAX_PER_STEP_MPC)
    next_pos = x0 + u0
    vel_cmd = (u0 / dt) if dt and dt > 0 else np.zeros(nu)
    return [float(x) for x in next_pos], [float(x) for x in vel_cmd]


def _mpc_solve_osqp(q, target_angles, dt):
    """Fallback: OSQP-based QP. Returns (next_pos, vel_cmd)."""
    try:
        import osqp
        from scipy import sparse
    except ImportError:
        e = np.array(target_angles, dtype=float) - np.array(q, dtype=float)
        u0 = np.clip(KP_PD_INVDYN * e, -U_MAX_PER_STEP_MPC, U_MAX_PER_STEP_MPC)
        next_pos = np.array(q, dtype=float) + u0
        vel_cmd = (u0 / dt) if dt and dt > 0 else np.zeros(MAX_JOINTS)
        return [float(x) for x in next_pos], [float(x) for x in vel_cmd]
    N = MPC_HORIZON
    Q, R, Q_term = MPC_Q, MPC_R, MPC_Q_TERMINAL
    S = np.tril(np.ones((N, N)))
    w = np.full(N, Q)
    w[-1] = Q_term
    P = S.T @ np.diag(w) @ S + R * np.eye(N)
    P = sparse.csc_matrix(P)
    q_coeffs = S.T @ w
    A = sparse.eye(N, format="csc")
    l_bound = np.full(N, -U_MAX_PER_STEP_MPC)
    u_bound = np.full(N, U_MAX_PER_STEP_MPC)
    next_pos_list = []
    vel_cmd_list = []
    for j in range(MAX_JOINTS):
        e0 = q[j] - target_angles[j]
        q_vec = q_coeffs * e0
        solver = osqp.OSQP()
        solver.setup(P, q_vec, A, l_bound, u_bound, warm_start=False, verbose=False, max_iter=200)
        result = solver.solve()
        if result.info.status in ("solved", "solved_inaccurate"):
            u0_j = float(np.clip(result.x[0], -U_MAX_PER_STEP_MPC, U_MAX_PER_STEP_MPC))
        else:
            u0_j = float(np.clip(-KP_PD_INVDYN * e0, -U_MAX_PER_STEP_MPC, U_MAX_PER_STEP_MPC))
        next_pos_list.append(q[j] + u0_j)
        vel_cmd_list.append((u0_j / dt) if dt and dt > 0 else 0.0)
    return next_pos_list, vel_cmd_list


def mpc_solve(q, target_angles, dt=None, q_vel=None):
    # LAG-AWARE LQR-CLAMP (sim-tuned 2026-08-24 on the MuJoCo twin):
    # u = -K @ [err, vel] clamped to +-VMAX deg/s. K from the DARE of the
    # 2-state drive-lag model (kv=40/s, dt=4ms, qe=100 qv=1 r=0.005).
    # Replaces the integrator MPC whose model ignored drive lag ->
    # ~5% overshoot (= drive braking distance). Solver-free: Pi-safe.
    _K0 = _ov("k0", 6.0)        # hardware-tuned 2026-09-17 (53.48 limit-cycled on the real drive)
    _K1 = _ov("k1", 0.0)        # true velocity feedback now; K1 only for streaming (0.2-0.3)
    _VMAX = _ov("vmax", 50.0)
    _VS = _ov("vel_scale", 17.0)         # drive velocity units per deg/s (measured 2026-09-17)
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    next_pos, vel_cmd = [], []
    for _j in range(MAX_JOINTS):
        _e = float(q[_j]) - float(target_angles[_j])
        _v = float(q_vel[_j]) if q_vel is not None else 0.0
        _u = -(_K0 * _e + _K1 * _v)
        _u = max(-_VMAX, min(_VMAX, _u))
        vel_cmd.append(_u * _VS)
        next_pos.append(float(q[_j]) + _u * dt)
    return next_pos, vel_cmd


def _mpc_solve_legacy(q, target_angles, dt=None):
    """MPC: TinyMPC if available, else OSQP fallback. Returns (next_pos, vel_cmd)."""
    global _mpc_use_tinympc
    if dt is None or dt <= 0:
        dt = PERIOD_SEC
    if _mpc_use_tinympc is None:
        try:
            import tinympc as _tm
            _mpc_use_tinympc = True
        except ImportError:
            _mpc_use_tinympc = False
    if _mpc_use_tinympc:
        return _mpc_solve_tinympc(q, target_angles, dt)
    return _mpc_solve_osqp(q, target_angles, dt)
