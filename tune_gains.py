#!/usr/bin/env python3
"""
Per-controller gain tuning from robot logs: analyze the motion response to the
control command and suggest gain changes.

For each controller present in the logs (pid | invdyn | pd_velff | mpc) and each
joint, it measures the step/move response:
    rise time, settle time, steady-state error, overshoot, oscillation,
    and peak velocity as a fraction of the joint limit,
then maps those symptoms to concrete gain suggestions for that controller's
entries in controller_params.yaml.

Tuning logic (classic, response-based):
    sluggish (low vel%, long rise, doesn't settle)  -> raise proportional gain
    overshoot / oscillation                          -> raise derivative gain (damping)
    persistent steady-state error                    -> add / raise integral gain

Usage:
    python tune_gains.py                       # all robot logs in logs/
    python tune_gains.py --controller invdyn   # only invdyn runs
    python tune_gains.py logs/robot_hal_*.csv  # specific files
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np

NJ = 6
VMAX_DEG_S = 200.0          # INI MAX_VEL
MOVE_THRESH_DEG = 2.0       # a joint "moved" if |target-q| at start exceeds this
SETTLE_TOL_DEG = 1.0

# Which yaml gains each controller exposes (for the suggestions).
GAIN_NAMES = {
    "pid":      {"P": "pid.kp / kp_per_joint", "D": "pid.kd", "I": "pid.ki"},
    "invdyn":   {"P": "invdyn.kp_pd",          "D": "invdyn.kd_pd", "I": "(none; rely on k_grav_comp)"},
    "pd_velff": {"P": "pd_velff.kp",           "D": "pd_velff.kd", "I": "(none)"},
    "mpc":      {"P": "mpc.q (state weight)",  "D": "mpc.r (input weight)", "I": "(none)"},
}


def _tsec(ts):
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


def analyze_run(rows):
    """Return per-joint metrics for one run, only for joints that actually moved.

    Yields (joint, dict_of_metrics).
    """
    if len(rows) < 20:
        return
    try:
        t = np.array([_tsec(r["timestamp"]) for r in rows]); t -= t[0]
    except (KeyError, ValueError):
        t = np.arange(len(rows)) * 0.02
    if t[-1] < 1.0:
        return
    n = len(rows)
    tail = slice(int(n * 0.85), n)
    half = slice(int(n * 0.5), n)
    for j in range(NJ):
        try:
            q = np.array([float(r[f"q{j}"]) for r in rows])
            tgt = np.array([float(r[f"target{j}"]) for r in rows])
        except (KeyError, ValueError):
            continue
        amp = abs(tgt[0] - q[0])
        if amp < MOVE_THRESH_DEG:
            continue
        err = tgt - q
        d = np.sign(tgt[0] - q[0])
        # rise to 90% of the move
        prog = (q - q[0]) * d / amp
        idx = np.where(prog >= 0.9)[0]
        rise = t[idx[0]] if len(idx) else np.nan
        # settle: first time within tol and stays
        within = np.abs(err) < SETTLE_TOL_DEG
        settle = np.nan
        for i in range(n):
            if within[i:].all():
                settle = t[i]; break
        # steady-state error (last 15%)
        ss = float(np.mean(np.abs(err[tail])))
        # overshoot past target (in approach direction)
        overshoot = max(0.0, float(np.max((q - tgt) * d)))
        # oscillation: error sign changes in second half
        sgn = np.sign(err[half]); sgn = sgn[sgn != 0]
        osc = int(np.sum(np.abs(np.diff(sgn)) > 0)) if len(sgn) > 1 else 0
        # peak velocity utilization
        if all(f"hal_vel{j}" in rows[0] for _ in [0]):
            v = np.array([float(r[f"hal_vel{j}"]) for r in rows])
        else:
            v = np.gradient(q, t)
        vel_util = float(np.max(np.abs(v))) / VMAX_DEG_S
        yield j, {"amp": amp, "rise": rise, "settle": settle, "ss": ss,
                  "overshoot": overshoot, "osc": osc, "vel_util": vel_util}


def _med(x):
    x = [v for v in x if v == v]
    return float(np.median(x)) if x else float("nan")


def diagnose(controller, per_joint):
    """Per-joint symptom → gain suggestion for this controller."""
    names = GAIN_NAMES.get(controller, GAIN_NAMES["pid"])
    print(f"\n=== Controller: {controller}  (gains: P={names['P']}, D={names['D']}, I={names['I']}) ===")
    hdr = f"{'J':<3}{'#':>4}{'rise_s':>8}{'settle_s':>9}{'ss°':>7}{'overshoot°':>11}{'osc':>5}{'vel%':>6}  suggestion"
    print(hdr); print("-" * len(hdr))
    for j in range(NJ):
        runs = per_joint[j]
        if not runs:
            continue
        rise = _med([r["rise"] for r in runs]); settle = _med([r["settle"] for r in runs])
        ss = _med([r["ss"] for r in runs]); ov = _med([r["overshoot"] for r in runs])
        osc = _med([r["osc"] for r in runs]); vu = _med([r["vel_util"] for r in runs])
        # response-based rule
        sug = []
        if ov > 2.0 or osc >= 3:
            sug.append(f"overshoot/oscillation → raise {names['D']} (more damping) or lower {names['P']}")
        elif (settle != settle or settle > 3.0 or vu < 0.15) and ov <= 2.0 and osc < 3:
            sug.append(f"sluggish/overdamped (vel {100*vu:.0f}% of limit) → raise {names['P']}")
        if ss > SETTLE_TOL_DEG:
            sug.append(f"steady-state err {ss:.1f}° → raise {names['I']}")
        if not sug:
            sug.append("well-tuned (fast, no overshoot, low error) — leave as is")
        print(f"{j:<3}{len(runs):>4}{rise:>8.2f}{settle:>9.2f}{ss:>7.2f}{ov:>11.2f}{osc:>5.0f}{100*vu:>5.0f}%  {'; '.join(sug)}")


def main():
    ap = argparse.ArgumentParser(description="Per-controller gain tuning from robot logs")
    ap.add_argument("files", nargs="*", help="CSV files (default: logs/{robot_hal,mpc,invdyn}_*.csv)")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--controller", default=None, help="Only analyze this controller")
    args = ap.parse_args()

    files = args.files
    if not files:
        for pat in ("robot_hal_*.csv", "mpc_*.csv", "invdyn_*.csv"):
            files += glob.glob(os.path.join(args.logdir, pat))
    files = sorted(set(files))
    if not files:
        print(f"No log files found in {args.logdir}/", file=sys.stderr)
        sys.exit(1)

    # group metrics by controller -> joint -> list of run-metrics
    by_ctrl = {}
    used = 0
    for f in files:
        rows = load(f)
        if not rows:
            continue
        ctrl = rows[0].get("controller", "").strip() or os.path.basename(f).split("_")[0]
        if args.controller and ctrl != args.controller:
            continue
        got = False
        for j, m in analyze_run(rows):
            by_ctrl.setdefault(ctrl, {jj: [] for jj in range(NJ)})[j].append(m)
            got = True
        used += got
    print(f"Analyzed {used} runs across controllers: {sorted(by_ctrl)}")
    if not by_ctrl:
        print("No moving-joint runs found.", file=sys.stderr); sys.exit(1)

    for ctrl in sorted(by_ctrl):
        diagnose(ctrl, by_ctrl[ctrl])
    print("\nNote: suggestions are response-based heuristics. Apply one change at a time,")
    print("re-run a calibration/perturbation move, and re-analyze the fresh logs.")


if __name__ == "__main__":
    main()
