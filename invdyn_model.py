#!/usr/bin/env python3
"""
Shared inverse-dynamics model: load npz (from identify_invdyn_from_log.py),
evaluate M(q), C(q,qd), G(q), and compute desired acceleration for control.

Uses the same LinuxCNC ↔ URDF convention as identify_invdyn_from_log.py
(JOINT_SIGNS, JOINT_OFFSETS_DEG). All dynamics internally in Nm, rad, rad/s, rad/s².

Reference (Drake InverseDynamicsController and InverseDynamics):
  https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1controllers_1_1_inverse_dynamics_controller.html
  Drake: PID outputs desired acceleration vd_command = kp*(q_d - q) + kd*(v_d - v) + ki*∫(q_d - q) + vd_d;
  then generalized_force = inverse_dynamics(q, v, vd_command) = M*vd_command + C*v + G
  (see inverse_dynamics.cc CalcOutputForce). We follow the same pattern: PID output is
  vd_command (rad/s²); we use it as qdd_d and integrate to pos/vel (HAL has no torque port).
  Optional: compute_tau_from_desired_acceleration() gives Drake-style generalized force.
"""

import math
import os
from typing import Any

import numpy as np

NUM_JOINTS = 6

# Same as identify_invdyn_from_log.py (LinuxCNC ↔ URDF)
JOINT_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_OFFSETS_DEG = [0.0, 90.0, 0.0, 90.0, 0.0, 0.0]

DEFAULT_URDF_PATH = os.path.join(
    os.path.expanduser("~"),
    "ros2_ws/src/mycobot_description/urdf/mycobot_pro_630.urdf",
)


def linuxcnc_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    """LinuxCNC joint angles (deg) → URDF/convention (rad)."""
    deg = np.asarray(deg, dtype=float)
    return np.array([
        JOINT_SIGNS[i] * np.deg2rad(deg[i] + JOINT_OFFSETS_DEG[i])
        for i in range(NUM_JOINTS)
    ])


def rad_to_linuxcnc_deg(rad: np.ndarray) -> np.ndarray:
    """URDF/convention (rad) → LinuxCNC joint angles (deg)."""
    rad = np.asarray(rad, dtype=float)
    return np.array([
        np.rad2deg(rad[i]) / JOINT_SIGNS[i] - JOINT_OFFSETS_DEG[i]
        for i in range(NUM_JOINTS)
    ])


def load_params(
    npz_path: str,
    urdf_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Load inverse-dynamics parameters from npz (from identify_invdyn_from_log.py).

    If npz contains pin_theta and Pinocchio + URDF are available, uses Pinocchio
    model. Otherwise uses simple model (M_diag, G_coeff). Returns None if file
    missing or load fails.
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception:
        return None

    params = {"torque_scale": float(data.get("torque_scale", 1.0))}

    use_pinocchio = False
    if "pin_theta" in data:
        theta = np.asarray(data["pin_theta"])
        urdf = urdf_path or DEFAULT_URDF_PATH
        try:
            import pinocchio as pin
            if os.path.isfile(urdf):
                model = pin.buildModelFromUrdf(urdf)
                if model.nq == NUM_JOINTS and model.nv == NUM_JOINTS:
                    data_obj = model.createData()
                    params["model"] = model
                    params["data"] = data_obj
                    params["theta"] = theta
                    use_pinocchio = True
        except ImportError:
            pass
        except Exception:
            pass

    params["use_pinocchio"] = use_pinocchio
    if not use_pinocchio:
        if "M_diag" not in data or "G_coeff" not in data:
            return None
        params["M_diag"] = np.asarray(data["M_diag"])
        params["G_coeff"] = np.asarray(data["G_coeff"])
    # Friction (optional; from identify_invdyn_from_log with Fv/Fc)
    params["Fv"] = np.asarray(data["Fv"]) if "Fv" in data else np.zeros(NUM_JOINTS)
    params["Fc"] = np.asarray(data["Fc"]) if "Fc" in data else np.zeros(NUM_JOINTS)

    return params


def compute_MCG(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate M(q), C(q,qd), G(q). q and qd in URDF rad convention.

    Returns M (6x6), C (6x6), G (6,) in Nm, kg·m², etc.
    """
    q_rad = np.asarray(q_rad, dtype=float)
    qd_rad = np.asarray(qd_rad, dtype=float)

    if params.get("use_pinocchio"):
        return _compute_MCG_pinocchio(q_rad, qd_rad, params)
    return _compute_MCG_simple(q_rad, qd_rad, params)


def _compute_MCG_simple(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,  # unused in simple model (C=0)
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple model: M = diag(M_diag), C = 0, G from G_coeff."""
    _ = qd_rad
    M = np.diag(np.maximum(params["M_diag"], 1e-8))
    C = np.zeros((NUM_JOINTS, NUM_JOINTS))
    coeff = params["G_coeff"]
    G = np.array([
        coeff[j, 0] * math.cos(q_rad[j]) + coeff[j, 1] * math.sin(q_rad[j]) + coeff[j, 2]
        for j in range(NUM_JOINTS)
    ])
    return M, C, G


def _compute_MCG_pinocchio(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinocchio: M and G from regressor @ theta, C from nominal Coriolis."""
    import pinocchio as pin
    model = params["model"]
    data = params["data"]
    theta = params["theta"]
    nv = model.nv

    def get_reg():
        r = getattr(data, "joint_torque_regressor", getattr(data, "jointTorqueRegressor", None))
        return np.asarray(r)

    pin.computeJointTorqueRegressor(model, data, q_rad, np.zeros(nv), np.zeros(nv))
    G = get_reg() @ theta

    M = np.zeros((nv, nv))
    for j in range(nv):
        e_j = np.zeros(nv)
        e_j[j] = 1.0
        pin.computeJointTorqueRegressor(model, data, q_rad, np.zeros(nv), e_j)
        M[:, j] = get_reg() @ theta

    pin.computeCoriolisMatrix(model, data, q_rad, qd_rad)
    C = np.asarray(data.C)
    return M, C, G


def _friction_torque(qd_rad: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Fv*qd + Fc*sign(qd) from identified params."""
    Fv = params.get("Fv", np.zeros(NUM_JOINTS))
    Fc = params.get("Fc", np.zeros(NUM_JOINTS))
    qd = np.asarray(qd_rad, dtype=float)
    return Fv * qd + Fc * np.tanh(qd / 1e-6)


def compute_tau_from_desired_acceleration(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,
    vd_command_rad: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """
    Inverse dynamics: tau = M(q)*vd + C(q,qd)*qd + G(q) + Fv*qd + Fc*sign(qd).

    Same idea as Drake's InverseDynamics; includes identified friction. Returns tau in model units (use params['torque_scale'] for hal_torq conversion).
    """
    M, C, G = compute_MCG(q_rad, qd_rad, params)
    tau = M @ vd_command_rad + C @ qd_rad + G
    tau = tau + _friction_torque(qd_rad, params)
    return tau.astype(float)


def compute_qdd_d(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,
    tau_pd_Nm: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """
    Solve for qdd_d such that M*qdd_d + C*qd + G + friction = tau_pd.
    Drake-aligned controller uses vd_command (PID as acceleration) as qdd_d instead.
    """
    M, C, G = compute_MCG(q_rad, qd_rad, params)
    friction = _friction_torque(qd_rad, params)
    rhs = tau_pd_Nm - C @ qd_rad - G - friction
    try:
        qdd_d = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        M_diag = np.diag(M)
        M_diag = np.maximum(np.abs(M_diag), 1e-8)
        qdd_d = rhs / M_diag
    return np.asarray(qdd_d, dtype=float)
