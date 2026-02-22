#!/usr/bin/env python3
"""
Estimate inverse-dynamics parameters (M, C, G) from robot CSV log data.

Reads mpc_*.csv or invdyn_*.csv with columns: q0..q5, qvel/hal_vel, hal_torq,
and optionally timestamp/loop. Computes q̈ from velocity finite differences,
then either:

  (1) With Pinocchio: builds dynamics regressor Φ(q,q̇,q̈), solves τ = Φ θ
      via least squares, and reports identified inertial parameters. You can
      then compute M(q), C(q,q̇), G(q) from the model and θ.

  (2) Without Pinocchio: fits a simple gravity + diagonal-inertia model
      (G from cos/sin basis per joint, M diagonal constant) from the same data.

Units: CSV q in degrees, qvel in deg/s; internally we convert to rad, rad/s, rad/s².
Torque (hal_torq) is in driver units. We estimate M, G, viscous friction (Fv), Coulomb
friction (Fc), and a force/torque scaling factor (torque_scale) so that
  torque_scale * hal_torq ≈ M q̈ + G(q) + Fv*q̇ + Fc*sign(q̇).
All terms needed for the inverse-dynamics controller are identified. Use --torque-scale
as an initial guess when loading data; the script refines or estimates the scale.

Usage:
  python identify_invdyn_from_log.py logs/mpc_20260211_*.csv
  python identify_invdyn_from_log.py logs/invdyn_*.csv --period-ms 20
  python identify_invdyn_from_log.py --csv logs/mpc_latest.csv --out params.npz
  python identify_invdyn_from_log.py logs/mpc_*.csv -o logs/invdyn_params.npz   # invdyn_hal picks this up by default

Note: Use robot logs (mpc_*.csv, invdyn_*.csv, or robot_hal_*.csv), not desktop control_calibrate_*.csv.
      The latter has one row per move and no hal_torq; fetch robot CSVs from the Raspi after calibration.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

NUM_JOINTS = 6
DEG2RAD = math.pi / 180.0

# Default control period (ms) for building time from loop index if no elapsed_s
DEFAULT_PERIOD_MS = 20.0

# URDF path (same default as ik_pyroki)
DEFAULT_URDF_PATH = os.path.join(
    os.path.expanduser("~"),
    "ros2_ws/src/mycobot_description/urdf/mycobot_pro_630.urdf",
)

# LinuxCNC ↔ URDF calibration (same as ik_pyroki)
JOINT_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_OFFSETS_DEG = [0.0, 90.0, 0.0, 90.0, 0.0, 0.0]


def linuxcnc_deg_to_rad(deg: np.ndarray) -> np.ndarray:
    """LinuxCNC joint angles (deg) → URDF/convention (rad)."""
    deg = np.asarray(deg, dtype=float)
    return np.array([
        JOINT_SIGNS[i] * np.deg2rad(deg[i] + JOINT_OFFSETS_DEG[i])
        for i in range(NUM_JOINTS)
    ])


def load_robot_log(
    csv_path: str,
    period_ms: float = DEFAULT_PERIOD_MS,
    torque_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Load robot CSV and add columns: elapsed_s, q_rad, qd_rad, qdd_rad, tau.
    Prefers hal_vel over qvel; requires hal_torq for tau.
    torque_scale: multiply hal_torq by this to get Nm (e.g. from pro600 datasheet).
    """
    df = pd.read_csv(csv_path)

    # Time: use elapsed_s if present (from plot_log-style), else build from loop and period
    if "elapsed_s" in df.columns:
        t = df["elapsed_s"].values
    elif "loop" in df.columns:
        t = (df["loop"].values - df["loop"].iloc[0]) * (period_ms / 1000.0)
        df["elapsed_s"] = t
    else:
        t = np.arange(len(df)) * (period_ms / 1000.0)
        df["elapsed_s"] = t

    # Position (degrees) → rad
    q_deg = np.column_stack([df[f"q{i}"].values for i in range(NUM_JOINTS)])
    q_rad = np.deg2rad(q_deg)
    # Calibrate to same convention as URDF if we use Pinocchio
    q_rad_urdf = np.array([linuxcnc_deg_to_rad(q_deg[k]) for k in range(len(q_deg))])
    df["q_rad"] = list(q_rad)
    df["q_rad_urdf"] = list(q_rad_urdf)

    # Velocity: prefer hal_vel, else qvel (deg/s) → rad/s
    if all(f"hal_vel{i}" in df.columns for i in range(NUM_JOINTS)):
        v_deg = np.column_stack([df[f"hal_vel{i}"].values for i in range(NUM_JOINTS)])
    else:
        v_deg = np.column_stack([df[f"qvel{i}"].values for i in range(NUM_JOINTS)])
    v_rad = np.deg2rad(v_deg)
    df["qd_rad"] = list(v_rad)

    # Acceleration: finite difference of velocity
    dt = np.diff(t)
    dt = np.concatenate([[dt[0] if len(dt) else period_ms / 1000.0], dt])
    qdd_rad = np.zeros_like(v_rad)
    qdd_rad[0] = 0.0
    for i in range(1, len(t)):
        if dt[i] > 0:
            qdd_rad[i] = (v_rad[i] - v_rad[i - 1]) / dt[i]
        else:
            qdd_rad[i] = qdd_rad[i - 1]
    df["qdd_rad"] = list(qdd_rad)

    # Torque (required); optional scale to convert to Nm
    if not all(f"hal_torq{i}" in df.columns for i in range(NUM_JOINTS)):
        raise ValueError(
            f"CSV must contain hal_torq0..hal_torq5 for dynamics ID. Columns: {list(df.columns)}"
        )
    tau = np.column_stack([df[f"hal_torq{i}"].values for i in range(NUM_JOINTS)]) * torque_scale
    df["tau"] = list(tau)

    return df


def _sign_smooth(qd: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Smooth sign for Coulomb friction; sign(0)=0."""
    return np.tanh(qd / eps)


def fit_simple_model(df: pd.DataFrame, torque_scale_init: float = 1.0) -> dict:
    """
    Fit inverse-dynamics model: τ = M_diag @ q̈ + G(q) + Fv*q̇ + Fc*sign(q̇).

    G_i(q) = a_i cos(q_i) + b_i sin(q_i) + c_i. Friction: viscous Fv_j * qd_j,
    Coulomb Fc_j * sign(qd_j) per joint. We fit tau (in loaded units) = W @ theta,
    then estimate torque_scale so that torque_scale * hal_torq ≈ W @ theta (best
    scalar fit). All terms needed for the inverse-dynamics controller are identified.
    """
    q = np.array(df["q_rad_urdf"].tolist())
    qd = np.array(df["qd_rad"].tolist())
    qdd = np.array(df["qdd_rad"].tolist())
    tau = np.array(df["tau"].tolist())
    n = len(q)

    # theta = [M_00..M_55, a_0,b_0,c_0, ..., a_5,b_5,c_5, Fv_0..Fv_5, Fc_0..Fc_5]
    nparams = 6 + 6 * 3 + 6 + 6  # M_diag + G_coeff(3 per joint) + Fv + Fc
    W = np.zeros((n * NUM_JOINTS, nparams))
    tau_vec = tau.ravel()
    for k in range(n):
        for j in range(NUM_JOINTS):
            row = k * NUM_JOINTS + j
            # M_ii
            W[row, j] = qdd[k, j]
            # G: a_j cos(q_j), b_j sin(q_j), c_j
            base_g = 6 + j * 3
            W[row, base_g] = math.cos(q[k, j])
            W[row, base_g + 1] = math.sin(q[k, j])
            W[row, base_g + 2] = 1.0
            # Friction: Fv_j * qd_j, Fc_j * sign(qd_j)
            W[row, 6 + 18 + j] = qd[k, j]
            W[row, 6 + 18 + 6 + j] = _sign_smooth(qd[k, j])

    # Regularization: small for M/G, slightly larger for Fv/Fc to avoid runaway
    reg = 1e-6 * np.eye(nparams)
    for j in range(6):
        reg[6 + 18 + j, 6 + 18 + j] = 1e-4
        reg[6 + 18 + 6 + j, 6 + 18 + 6 + j] = 1e-4
    theta = np.linalg.lstsq(W.T @ W + reg, W.T @ tau_vec, rcond=None)[0]
    M_diag = np.maximum(theta[:6], 1e-8)
    g_params = theta[6 : 6 + 18].reshape(NUM_JOINTS, 3)
    Fv = theta[6 + 18 : 6 + 24]
    Fc = theta[6 + 24 : 6 + 30]

    tau_pred = (W @ theta).reshape(n, NUM_JOINTS)
    tau_vec_sq = np.dot(tau_vec, tau_vec)
    if tau_vec_sq > 1e-20:
        torque_scale = np.dot(tau_pred.ravel(), tau_vec) / tau_vec_sq
    else:
        torque_scale = float(torque_scale_init)
    # Alternating LS: refine torque_scale and theta so that torque_scale * hal_torq ≈ W @ theta
    for _ in range(3):
        # Fix theta, fit torque_scale: torque_scale * tau_vec = tau_pred
        tau_pred_flat = (W @ theta).ravel()
        if tau_vec_sq > 1e-20:
            torque_scale = np.dot(tau_pred_flat, tau_vec) / tau_vec_sq
        # Fix torque_scale, fit theta: W @ theta = torque_scale * tau_vec
        target = torque_scale * tau_vec
        theta = np.linalg.lstsq(W.T @ W + reg, W.T @ target, rcond=None)[0]
        M_diag = np.maximum(theta[:6], 1e-8)
        g_params = theta[6 : 6 + 18].reshape(NUM_JOINTS, 3)
        Fv = theta[6 + 18 : 6 + 24]
        Fc = theta[6 + 24 : 6 + 30]
    tau_pred = (W @ theta).reshape(n, NUM_JOINTS)
    err = tau - (tau_pred / torque_scale) if torque_scale != 0 else tau - tau_pred
    rmse = np.sqrt(np.mean(err ** 2))
    return {
        "M_diag": M_diag,
        "G_coeff": g_params,
        "Fv": Fv,
        "Fc": Fc,
        "torque_scale": torque_scale,
        "tau_pred": tau_pred,
        "rmse": rmse,
        "theta": theta,
    }


def eval_simple_MCG(q_rad: np.ndarray, qd_rad: np.ndarray, res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate M (diagonal), C (zero in this simple model), G from fitted simple model. Friction is applied separately at tau level."""
    M = np.diag(res["M_diag"])
    C = np.zeros((NUM_JOINTS, NUM_JOINTS))
    G = np.zeros(NUM_JOINTS)
    coeff = res["G_coeff"]
    for j in range(NUM_JOINTS):
        G[j] = coeff[j, 0] * math.cos(q_rad[j]) + coeff[j, 1] * math.sin(q_rad[j]) + coeff[j, 2]
    return M, C, G


def eval_simple_friction(qd_rad: np.ndarray, res: dict) -> np.ndarray:
    """Evaluate friction torque: Fv*qd + Fc*sign(qd)."""
    Fv = res.get("Fv", np.zeros(NUM_JOINTS))
    Fc = res.get("Fc", np.zeros(NUM_JOINTS))
    return Fv * qd_rad + Fc * _sign_smooth(qd_rad)


def eval_pinocchio_MCG(
    q_rad: np.ndarray,
    qd_rad: np.ndarray,
    pin_result: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate M(q), C(q,qd), G(q) from Pinocchio identified result.
    M and G come from the regressor and identified theta; C is from the nominal
    model (pin.computeCoriolisMatrix) since the regressor gives only C*v, not C.
    """
    try:
        import pinocchio as pin
    except ImportError:
        raise RuntimeError("Pinocchio required for eval_pinocchio_MCG")
    model = pin_result["model"]
    data = pin_result["data"]
    theta = pin_result["theta"]
    nv = model.nv

    def get_reg():
        r = getattr(data, "joint_torque_regressor", getattr(data, "jointTorqueRegressor", None))
        return np.asarray(r)

    # G(q) = Phi(q, 0, 0) * theta
    pin.computeJointTorqueRegressor(model, data, q_rad, np.zeros(nv), np.zeros(nv))
    G = get_reg() @ theta

    # M(q): column j = Phi(q, 0, e_j) * theta
    M = np.zeros((nv, nv))
    for j in range(nv):
        e_j = np.zeros(nv)
        e_j[j] = 1.0
        pin.computeJointTorqueRegressor(model, data, q_rad, np.zeros(nv), e_j)
        M[:, j] = get_reg() @ theta

    # C from nominal model (Coriolis matrix at (q, qd))
    pin.computeCoriolisMatrix(model, data, q_rad, qd_rad)
    C = np.asarray(data.C)
    return M, C, G


def try_pinocchio_identification(df: pd.DataFrame, urdf_path: str) -> dict | None:
    """
    If Pinocchio is available, build model from URDF and run regressor-based
    least-squares identification. Returns dict with model, data, theta, or None.
    """
    try:
        import pinocchio as pin
    except ImportError:
        return None

    if not os.path.isfile(urdf_path):
        print(f"  URDF not found: {urdf_path}")
        return None

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    nq = model.nq
    nv = model.nv
    # Handle quaternion joint (e.g. free flyer) vs all revolute
    if nq != NUM_JOINTS or nv != NUM_JOINTS:
        print(f"  Pinocchio model nq={nq} nv={nv}; expected 6. Skipping Pinocchio ID.")
        return None

    q = np.array(df["q_rad_urdf"].tolist())
    qd = np.array(df["qd_rad"].tolist())
    qdd = np.array(df["qdd_rad"].tolist())
    tau = np.array(df["tau"].tolist())
    n = len(q)

    # Joint torque regressor: tau = Phi(q, v, a) * pi (pi = inertial params)
    # Phi is nv x nparams
    pin.computeJointTorqueRegressor(model, data, q[0], qd[0], qdd[0])
    reg = getattr(data, "joint_torque_regressor", getattr(data, "jointTorqueRegressor", None))
    if reg is None:
        return None
    regressor = np.asarray(reg)
    nparams = regressor.shape[1]
    W = np.zeros((n * nv, nparams))
    tau_vec = tau.ravel()
    for k in range(n):
        pin.computeJointTorqueRegressor(model, data, q[k], qd[k], qdd[k])
        reg = getattr(data, "joint_torque_regressor", getattr(data, "jointTorqueRegressor", None))
        W[k * nv : (k + 1) * nv] = np.asarray(reg)
    reg = 1e-8 * np.eye(nparams)
    theta = np.linalg.lstsq(W.T @ W + reg, W.T @ tau_vec, rcond=None)[0]
    tau_pred = (W @ theta).reshape(n, NUM_JOINTS)
    err = tau - tau_pred
    rmse = np.sqrt(np.mean(err ** 2))

    return {
        "model": model,
        "data": data,
        "theta": theta,
        "tau_pred": tau_pred,
        "rmse": rmse,
        "nparams": nparams,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Estimate M, C, G from robot CSV log (invdyn/mpc_*.csv)"
    )
    ap.add_argument("csv", nargs="*", help="Path(s) to robot log CSV (mpc_*.csv or invdyn_*.csv)")
    ap.add_argument("--csv", dest="csv_single", help="Single CSV path (alternative to positional)")
    ap.add_argument("--period-ms", type=float, default=DEFAULT_PERIOD_MS,
                    help="Control loop period in ms (for time from loop index)")
    ap.add_argument("--urdf", default=DEFAULT_URDF_PATH, help="URDF for Pinocchio (if available)")
    ap.add_argument("--no-pinocchio", action="store_true", help="Skip Pinocchio even if installed")
    ap.add_argument("--out", "-o", help="Save identified params to this .npz file (e.g. logs/invdyn_params.npz for invdyn_hal default)")
    ap.add_argument("--torque-scale", type=float, default=1.0,
                    help="Initial scale when loading: tau = hal_torq * scale. Script also estimates torque_scale jointly with M, G, Fv, Fc.")
    args = ap.parse_args()

    files = args.csv or args.csv_single
    if not files:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        import glob
        mpc_files = sorted(glob.glob(os.path.join(log_dir, "mpc_*.csv")))
        inv_files = sorted(glob.glob(os.path.join(log_dir, "invdyn_*.csv")))
        robot_hal_files = sorted(glob.glob(os.path.join(log_dir, "robot_hal_*.csv")))
        files = mpc_files + inv_files + robot_hal_files
        if not files:
            print("No CSV given and no mpc_*.csv / invdyn_*.csv / robot_hal_*.csv in logs/. Use: identify_invdyn_from_log.py <file.csv>")
            sys.exit(1)
        files = files[-3:]  # last 3

    # Load first file (or merge multiple)
    if len(files) == 1:
        csv_path = files[0]
    else:
        csv_path = files[0]
    print(f"Loading: {csv_path}")
    df = load_robot_log(csv_path, period_ms=args.period_ms, torque_scale=args.torque_scale)
    if len(files) > 1:
        for f in files[1:]:
            df2 = load_robot_log(f, period_ms=args.period_ms, torque_scale=args.torque_scale)
            df = pd.concat([df, df2], ignore_index=True)
        print(f"Merged {len(files)} files → {len(df)} rows")
    if args.torque_scale != 1.0:
        print(f"  Torque scale: {args.torque_scale} (tau in Nm → M in kg·m²)")

    print(f"  Rows: {len(df)}, time span: {df['elapsed_s'].min():.3f} – {df['elapsed_s'].max():.3f} s")
    q_all = np.array(df["q_rad"].tolist())
    print(f"  q range (rad): {q_all.min():.3f} – {q_all.max():.3f}")

    # Simple model (M, G, Fv, Fc) and torque_scale
    print("\n--- Simple model (M, G, viscous Fv, Coulomb Fc, torque_scale) ---")
    simple = fit_simple_model(df, torque_scale_init=args.torque_scale)
    print(f"  M_diag (diagonal inertia, ≥ 0): {simple['M_diag']}")
    print(f"  G coeffs (a*cos+b*sin+c per joint):\n{simple['G_coeff']}")
    print(f"  Fv (viscous, per joint): {simple['Fv']}")
    print(f"  Fc (Coulomb, per joint): {simple['Fc']}")
    print(f"  torque_scale (hal_torq * torque_scale ≈ model torque): {simple['torque_scale']:.6f}")
    print(f"  RMSE torque: {simple['rmse']:.6f}")

    # Pinocchio if available
    pin_result = None
    if not args.no_pinocchio:
        print("\n--- Pinocchio regressor-based ID ---")
        pin_result = try_pinocchio_identification(df, args.urdf)
        if pin_result:
            print(f"  Identified {pin_result['nparams']} inertial parameters")
            print(f"  RMSE torque: {pin_result['rmse']:.6f}")
        else:
            print("  Pinocchio not used (not installed or URDF not found)")

    # Summary: how to get M, C, G, friction
    print("\n--- Using the identified model ---")
    print("  tau = M @ qdd + G(q) + Fv*qd + Fc*sign(qd); torque_scale converts hal_torq to model units.")
    if pin_result:
        print("  Pinocchio: tau = Phi(q,qd,qdd)*theta. To get M,C,G: use pin.computeMassMatrix(model,data,q),")
        print("            pin.computeCoriolisMatrix(model,data,q,v), pin.computeGeneralizedGravity(model,data,q)")
        print("            after updating model inertias from theta (or use regressor at (q,0,e_i) for M columns).")

    if args.out:
        out = {
            "M_diag": simple["M_diag"],
            "G_coeff": simple["G_coeff"],
            "Fv": simple["Fv"],
            "Fc": simple["Fc"],
            "torque_scale": np.float64(simple["torque_scale"]),
            "simple_rmse": simple["rmse"],
        }
        if pin_result:
            out["pin_theta"] = pin_result["theta"]
            out["pin_rmse"] = pin_result["rmse"]
            out["pin_nparams"] = pin_result["nparams"]
        np.savez(args.out, **out)
        print(f"\nSaved to {args.out}")
        if args.out.endswith("invdyn_params.npz") or "invdyn_params" in args.out:
            print("  robot_hal.py: run with --params logs/invdyn_params.npz for model-based invdyn.")


if __name__ == "__main__":
    main()
