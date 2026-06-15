#!/usr/bin/env python3
"""
Smooth, time-scaled joint-space trajectories for the myCobot Pro 630.

Given a current pose and a target pose (optionally with via-points), build a
trajectory q(t), q̇(t), q̈(t) that is:
  * smooth   — continuous velocity and acceleration (no instantaneous jumps), and
  * fast     — time-scaled so the binding joint rides the velocity/accel limits.

Two path types:
  * "quintic"  — minimum-jerk point-to-point (current → target). Simplest; zero
                 velocity and acceleration at both ends. Best for 2 endpoints.
  * "bspline"  — quintic B-spline through current + via-points + target, with
                 zero end velocity/accel. Use when there are intermediate
                 waypoints (Cartesian paths, calibration tours, blended moves).

"linspace" (constant-velocity, the old planner behaviour) is included only for
comparison — it has velocity steps at the ends (unbounded acceleration).

All angles in LinuxCNC degrees; limits default to the INI joint limits
(MAX_VEL=200 deg/s, MAX_ACCEL=300 deg/s²) scaled by vel_frac/acc_frac for safety.

Stream the result into the inner loop as (pos_cmd, vel_cmd) at the control rate;
the velocity column is the feedforward term for the pd_velff / invdyn controller.
"""

import numpy as np

# INI joint limits (mycobot_params.md §8.3)
MAX_VEL_DEG_S = 200.0
MAX_ACC_DEG_S2 = 300.0

# Min-jerk quintic s(τ)=10τ³−15τ⁴+6τ⁵ peak derivatives (τ∈[0,1]):
_QUINTIC_VEL_PEAK = 1.875          # max ds/dτ  (at τ=0.5)
_QUINTIC_ACC_PEAK = 5.7735026919   # max |d²s/dτ²|


def _quintic_s(tau):
    """Min-jerk scaling s and its time-normalized derivatives, τ∈[0,1]."""
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4      # ds/dτ
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3      # d²s/dτ²
    return s, ds, dds


def _min_time(dq, vmax, amax):
    """Shortest quintic duration T (s) so no joint exceeds vmax / amax."""
    dq_max = float(np.max(np.abs(dq)))
    if dq_max < 1e-9:
        return 0.0
    t_v = _QUINTIC_VEL_PEAK * dq_max / vmax
    t_a = np.sqrt(_QUINTIC_ACC_PEAK * dq_max / amax)
    return float(max(t_v, t_a))


def quintic_trajectory(q0, qf, vmax, amax, rate_hz):
    """Minimum-jerk point-to-point trajectory. Returns (t, q, qd, qdd, T)."""
    q0 = np.asarray(q0, float)
    qf = np.asarray(qf, float)
    dq = qf - q0
    T = _min_time(dq, vmax, amax)
    n = max(2, int(round(T * rate_hz)) + 1)
    t = np.linspace(0.0, T, n)
    tau = t / T if T > 0 else np.zeros(n)
    s, ds, dds = _quintic_s(tau)
    q = q0[None, :] + np.outer(s, dq)
    qd = np.outer(ds / T, dq) if T > 0 else np.zeros((n, len(dq)))
    qdd = np.outer(dds / T**2, dq) if T > 0 else np.zeros((n, len(dq)))
    return t, q, qd, qdd, T


def bspline_trajectory(waypoints, vmax, amax, rate_hz, degree=5):
    """B-spline path through waypoints (>=2 rows) traversed with a min-jerk time
    profile, so velocity and acceleration are zero at both ends regardless of
    waypoint count, and the binding joint rides the limits. Returns (t,q,qd,qdd,T).

    Path q(u), u in [0,1], from the spline; timing u = s(τ), τ = t/T, with the
    min-jerk s (s'(0)=s'(1)=0, s''(0)=s''(1)=0). Then
        q̇ = q'(u)·s'(τ)/T,
        q̈ = [q''(u)·s'(τ)² + q'(u)·s''(τ)] / T²,
    both zero at the ends because s'(τ) and s''(τ) vanish there.
    """
    from scipy.interpolate import make_interp_spline

    W = np.asarray(waypoints, float)
    if W.ndim != 2 or len(W) < 2:
        raise ValueError("waypoints must be (N>=2, n_joints)")
    if len(W) == 2:
        return quintic_trajectory(W[0], W[1], vmax, amax, rate_hz)

    k = min(degree, len(W) - 1)
    # Chord-length parameterization in [0,1] (better spacing than uniform).
    seg = np.linalg.norm(np.diff(W, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)])
    u = u / u[-1] if u[-1] > 0 else np.linspace(0, 1, len(W))
    spl = make_interp_spline(u, W, k=k)
    d1, d2 = spl.derivative(1), spl.derivative(2)

    # Find T so peak vel/accel hit the limits, using the min-jerk timing shape.
    tau = np.linspace(0.0, 1.0, 600)
    s, ds, dds = _quintic_s(tau)
    qp = d1(s); qpp = d2(s)                                   # (M, n_joints)
    vel_shape = qp * ds[:, None]                             # per-unit-T velocity
    acc_shape = qpp * ds[:, None] ** 2 + qp * dds[:, None]   # per-unit-T² accel
    Pv = float(np.max(np.abs(vel_shape)))
    Pa = float(np.max(np.abs(acc_shape)))
    T = max(Pv / vmax, np.sqrt(Pa / amax)) if (Pv > 0 or Pa > 0) else 0.0

    n = max(2, int(round(T * rate_hz)) + 1)
    t = np.linspace(0.0, T, n)
    tau_t = t / T if T > 0 else np.zeros(n)
    s_t, ds_t, dds_t = _quintic_s(tau_t)
    q = spl(s_t)
    if T > 0:
        qd = d1(s_t) * ds_t[:, None] / T
        qdd = (d2(s_t) * ds_t[:, None] ** 2 + d1(s_t) * dds_t[:, None]) / T**2
    else:
        qd = np.zeros_like(q); qdd = np.zeros_like(q)
    return t, q, qd, qdd, T


def linspace_trajectory(q0, qf, T, rate_hz):
    """Constant-velocity interpolation (old planner). For comparison only —
    velocity steps from 0 to const at the ends (unbounded acceleration)."""
    q0 = np.asarray(q0, float); qf = np.asarray(qf, float)
    n = max(2, int(round(T * rate_hz)) + 1)
    t = np.linspace(0.0, T, n)
    q = np.linspace(q0, qf, n)
    qd = np.gradient(q, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    return t, q, qd, qdd, T


def plan(q0, qf, via=None, kind="quintic", rate_hz=50.0,
         vel_frac=0.6, acc_frac=0.6):
    """Convenience entry point. Returns dict(t, q, qd, qdd, T, kind).

    vel_frac/acc_frac scale the INI limits (default 60% for safety headroom).
    kind: "quintic" (q0→qf), "bspline" (q0→via...→qf), or "linspace".
    """
    vmax = MAX_VEL_DEG_S * vel_frac
    amax = MAX_ACC_DEG_S2 * acc_frac
    if kind == "bspline":
        wps = [q0] + (list(via) if via is not None else []) + [qf]
        t, q, qd, qdd, T = bspline_trajectory(np.array(wps, float), vmax, amax, rate_hz)
    elif kind == "linspace":
        T = _min_time(np.asarray(qf, float) - np.asarray(q0, float), vmax, amax)
        t, q, qd, qdd, T = linspace_trajectory(q0, qf, T, rate_hz)
    else:
        t, q, qd, qdd, T = quintic_trajectory(q0, qf, vmax, amax, rate_hz)
    return {"t": t, "q": q, "qd": qd, "qdd": qdd, "T": T, "kind": kind}


def _report(name, tr, vmax, amax):
    qd = tr["qd"]
    pk_v = np.max(np.abs(qd)); pk_a = np.max(np.abs(tr["qdd"]))
    # Endpoint speed: ~0 means a smooth start/stop; nonzero means a velocity STEP
    # (instantaneous, unbounded acceleration) — the linspace flaw gradient can't show.
    end_v = max(float(np.max(np.abs(qd[0]))), float(np.max(np.abs(qd[-1]))))
    smooth = "smooth start/stop" if end_v < 1.0 else f"VEL STEP at ends ({end_v:.0f}°/s → inf accel)"
    print(f"  {name:<9} T={tr['T']:5.2f}s  peak|q̇|={pk_v:6.1f}°/s ({100*pk_v/vmax:4.0f}% lim)  "
          f"peak|q̈|={pk_a:7.1f}°/s² ({100*pk_a/amax:4.0f}% lim)  {smooth}")


def main():
    """Offline demo: compare linspace vs quintic vs B-spline for a sample move."""
    rate = 50.0
    vmax = MAX_VEL_DEG_S * 0.6
    amax = MAX_ACC_DEG_S2 * 0.6
    q0 = np.array([0.0, -110.3, 111.4, -90.1, -90.3, 0.0])   # measured base pose
    qf = np.array([-90.0, -90.0, 0.0, -90.0, 0.0, 0.0])      # upright/home

    print("Trajectory comparison: base → home  (limits 60% of INI: "
          f"{vmax:.0f}°/s, {amax:.0f}°/s²)\n")
    lin = plan(q0, qf, kind="linspace", rate_hz=rate)
    qui = plan(q0, qf, kind="quintic", rate_hz=rate)
    via = [(q0 + qf) / 2 + np.array([0, 5, -5, 0, 0, 0])]    # a nudge via-point
    bsp = plan(q0, qf, via=via, kind="bspline", rate_hz=rate)
    for name, tr in (("linspace", lin), ("quintic", qui), ("bspline", bsp)):
        _report(name, tr, vmax, amax)
    print("\n  linspace has a velocity step at each end (huge peak accel/jerk) — not smooth.")
    print("  quintic/bspline ramp smoothly and finish within the duration, riding the limits.")

    # Optional plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for name, tr, ls in (("linspace", lin, ":"), ("quintic", qui, "-"), ("bspline", bsp, "--")):
            ax[0].plot(tr["t"], tr["q"][:, 1], ls, label=name)
            ax[1].plot(tr["t"], tr["qd"][:, 1], ls, label=name)
            ax[2].plot(tr["t"], tr["qdd"][:, 1], ls, label=name)
        ax[0].set_ylabel("J1 angle (°)"); ax[1].set_ylabel("J1 vel (°/s)")
        ax[2].set_ylabel("J1 acc (°/s²)"); ax[2].set_xlabel("time (s)")
        for a in ax: a.grid(True, alpha=0.3); a.legend(fontsize=8)
        ax[1].axhline(vmax, color="r", ls=":", lw=0.8); ax[1].axhline(-vmax, color="r", ls=":", lw=0.8)
        fig.suptitle("Joint 1: linspace vs quintic vs B-spline")
        fig.tight_layout(); plt.show()
    except ImportError:
        print("\n  (matplotlib not installed — numeric report only)")


if __name__ == "__main__":
    main()
