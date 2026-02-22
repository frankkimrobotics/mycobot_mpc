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
Torque (hal_torq) is in pro600 driver units by default. Use --torque-scale to convert
to Nm so M comes out in kg·m² and G in Nm. The robot total mass (e.g. 8.8 kg) is not
the diagonal of M(q): M(q) is joint-space inertia matrix; total mass is from the full
inertial parameter vector (e.g. Pinocchio theta), not from M_diag.

Usage:
  python identify_invdyn_from_log.py logs/mpc_20260211_*.csv
  python identify_invdyn_from_log.py logs/invdyn_*.csv --period-ms 20
  python identify_invdyn_from_log.py --csv logs/mpc_latest.csv --out params.npz
  python identify_invdyn_from_log.py logs/mpc_*.csv -o logs/invdyn_params.npz   # invdyn_hal picks this up by default

Note: Use robot logs (mpc_*.csv or invdyn_*.csv), not desktop control_calibrate_*.csv.
      The latter has one row per move and no hal_torq; fetch robot CSVs with fetch_robot_logs.py.
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


def fit_simple_model(df: pd.DataFrame) -> dict:
    """
    Fit a simple inverse-dynamics model without Pinocchio:
      τ ≈ M_diag @ q̈ + G(q)
    with G(q) = g_coeff @ [cos(q); sin(q)] per-joint basis (10 params per joint),
    and M_diag a constant diagonal (6 params). Total 6*10 + 6 = 66 params for G + M.
    Simplified: G_i(q) ≈ a_i cos(q_i) + b_i sin(q_i) + c_i (3 per joint), M diagonal (6).
    So τ_i ≈ M_ii q̈_i + a_i cos(q_i) + b_i sin(q_i) + c_i.
    """
    q = np.array(df["q_rad_urdf"].tolist())   # use URDF convention for consistency
    qd = np.array(df["qd_rad"].tolist())
    qdd = np.array(df["qdd_rad"].tolist())
    tau = np.array(df["tau"].tolist())
    n = len(q)

    # Regressor: for each sample, row i is [q̈_i, cos(q_i), sin(q_i), 1] for joint i (block diagonal)
    # τ_i = M_ii q̈_i + a_i cos(q_i) + b_i sin(q_i) + c_i
    # Stack: tau_vec = W @ theta, theta = [M_00, M_11, ..., M_55, a_0, b_0, c_0, ..., a_5, b_5, c_5]
    nparams = 6 + 6 * 3  # 6 M_ii + 6*(a,b,c)
    W = np.zeros((n * NUM_JOINTS, nparams))
    tau_vec = tau.ravel()
    for k in range(n):
        for j in range(NUM_JOINTS):
            row = k * NUM_JOINTS + j
            # M_ii column
            W[row, j] = qdd[k, j]
            # a_j cos(q_j), b_j sin(q_j), c_j for joint j
            base = 6 + j * 3
            W[row, base] = math.cos(q[k, j])
            W[row, base + 1] = math.sin(q[k, j])
            W[row, base + 2] = 1.0

    # Least squares with small regularization
    reg = 1e-6 * np.eye(nparams)
    theta = np.linalg.lstsq(W.T @ W + reg, W.T @ tau_vec, rcond=None)[0]
    M_diag = np.maximum(theta[:6], 1e-8)  # clamp to positive (inertia must be > 0)
    g_params = theta[6:].reshape(NUM_JOINTS, 3)  # (a, b, c) per joint

    # Residual
    tau_pred = (W @ theta).reshape(n, NUM_JOINTS)
    err = tau - tau_pred
    rmse = np.sqrt(np.mean(err ** 2))
    return {
        "M_diag": M_diag,
        "G_coeff": g_params,
        "tau_pred": tau_pred,
        "rmse": rmse,
        "theta": theta,
    }


def eval_simple_MCG(q_rad: np.ndarray, qd_rad: np.ndarray, res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate M (diagonal), C (zero in this simple model), G from fitted simple model."""
    M = np.diag(res["M_diag"])
    C = np.zeros((NUM_JOINTS, NUM_JOINTS))
    G = np.zeros(NUM_JOINTS)
    coeff = res["G_coeff"]
    for j in range(NUM_JOINTS):
        G[j] = coeff[j, 0] * math.cos(q_rad[j]) + coeff[j, 1] * math.sin(q_rad[j]) + coeff[j, 2]
    return M, C, G


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
                    help="Scale hal_torq to Nm: tau_Nm = hal_torq * scale (then M is in kg·m²)")
    args = ap.parse_args()

    files = args.csv or args.csv_single
    if not files:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        import glob
        mpc_files = sorted(glob.glob(os.path.join(log_dir, "mpc_*.csv")))
        inv_files = sorted(glob.glob(os.path.join(log_dir, "invdyn_*.csv")))
        files = mpc_files + inv_files
        if not files:
            print("No CSV given and no mpc_*.csv / invdyn_*.csv in logs/. Use: identify_invdyn_from_log.py <file.csv>")
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

    # Simple model (always)
    print("\n--- Simple model (M diagonal + G cos/sin per joint) ---")
    simple = fit_simple_model(df)
    print(f"  M_diag (diagonal inertia, clamped ≥ 0): {simple['M_diag']}")
    print(f"  G coeffs (a*cos+b*sin+c per joint):\n{simple['G_coeff']}")
    print(f"  RMSE torque: {simple['rmse']:.6f}")
    print("  Note: M(q) is joint-space inertia (kg·m² if --torque-scale gives Nm), not total mass.")
    print("  Robot total mass (e.g. 8.8 kg) is in the full inertial params (Pinocchio theta), not M_diag.")

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

    # Summary: how to get M, C, G
    print("\n--- Using the identified model ---")
    print("  Simple model: M = np.diag(M_diag), C = 0, G = G_coeff[:,0]*cos(q) + G_coeff[:,1]*sin(q) + G_coeff[:,2]")
    print("  Inverse dynamics (simple): tau = M @ qdd + G(q)")
    if pin_result:
        print("  Pinocchio: tau = Phi(q,qd,qdd)*theta. To get M,C,G: use pin.computeMassMatrix(model,data,q),")
        print("            pin.computeCoriolisMatrix(model,data,q,v), pin.computeGeneralizedGravity(model,data,q)")
        print("            after updating model inertias from theta (or use regressor at (q,0,e_i) for M columns).")

    if args.out:
        out = {
            "M_diag": simple["M_diag"],
            "G_coeff": simple["G_coeff"],
            "simple_rmse": simple["rmse"],
            "torque_scale": np.float64(args.torque_scale),
        }
        if pin_result:
            out["pin_theta"] = pin_result["theta"]
            out["pin_rmse"] = pin_result["rmse"]
            out["pin_nparams"] = pin_result["nparams"]
        np.savez(args.out, **out)
        print(f"\nSaved to {args.out}")
        if args.out.endswith("invdyn_params.npz") or "invdyn_params" in args.out:
            print("  invdyn_hal.py and invdyn_linuxcnc.py use logs/invdyn_params.npz by default.")


if __name__ == "__main__":
    main()
