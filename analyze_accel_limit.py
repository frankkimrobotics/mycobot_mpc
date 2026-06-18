#!/usr/bin/env python3
"""
analyze_accel_limit :: estimate the per-joint ACCELERATION limit (and the
saturated torque level) from a sinusoid velocity-sweep raw log.

In the acceleration-saturated runs the velocity feedback becomes a TRIANGLE
wave: the joint ramps at a constant slope = a_max, the physical ceiling. We
read that ceiling two independent ways and cross-check:

  (1) slope method:   a = 95th-pct of |d(vel_fb)/dt|   (the ramp slope itself)
  (2) triangle method: a = 4 * Vpk_achieved * f         (geometry of a triangle
                          velocity wave: -Vpk -> +Vpk in a half period 1/(2f))

a_max is the PLATEAU these reach once the command saturates (high % runs). The
railed torque |torqfb| at that point is tau_sat (drive units); their ratio is
the effective inertia J_eff = tau_sat / a_max (drive-torque * s^2 / deg).

Usage:  python3 analyze_accel_limit.py                 # newest raw log per joint
        python3 analyze_accel_limit.py --files raw_j0.jsonl ...
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joint_conventions import MAX_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")


def newest_raw_per_joint():
    out = {}
    for p in glob.glob(os.path.join(LOGDIR, "sin_velsweep_raw_j*_*.jsonl")):
        m = re.search(r"sin_velsweep_raw_j(\d)_", os.path.basename(p))
        if m:
            j = int(m.group(1))
            if j not in out or os.path.getmtime(p) > os.path.getmtime(out[j]):
                out[j] = p
    return out


def analyze(path):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    runs = [r for r in runs if r.get("samples")]
    runs.sort(key=lambda r: r["pct"])
    j = runs[0]["joint"]
    res = []
    for r in runs:
        s = r["samples"]
        t = np.array([x[0] for x in s])
        vel = np.array([x[2][j] for x in s])    # active-joint velocity fb (deg/s)
        tor = np.array([x[3][j] for x in s])    # active-joint torque fb
        # keep the steady window (skip 0.7 s ramp-in)
        mask = t >= 0.7
        if mask.sum() < 10:
            mask = np.ones_like(t, bool)
        t, vel, tor = t[mask], vel[mask], tor[mask]
        f = r["freq"]
        vpk_ach = float(np.percentile(np.abs(vel), 98))         # achieved peak speed
        # (1) ramp-slope acceleration: low-pass the noisy 100 Hz vel fb first,
        #     then take the ramp slope as a high percentile of |d(vel)/dt|
        k = 7
        vel_s = np.convolve(vel, np.ones(k) / k, mode="same")
        dvdt = np.gradient(vel_s, t)
        a_slope = float(np.percentile(np.abs(dvdt), 90))
        # (2) triangle-geometry acceleration
        a_tri = 4.0 * vpk_ach * f
        tau_sat = float(np.percentile(np.abs(tor), 95))
        res.append(dict(pct=r["pct"], vpk_cmd=r["vpk"], f=f,
                        vpk_ach=vpk_ach, a_slope=a_slope, a_tri=a_tri, tau=tau_sat))
    return j, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+")
    ap.add_argument("--sat-from", type=float, default=60.0,
                    help="treat runs at >= this %% as saturated (plateau estimate)")
    args = ap.parse_args()
    paths = args.files or [newest_raw_per_joint()[j] for j in sorted(newest_raw_per_joint())]
    if not paths:
        print("no raw logs found"); return

    summary = []
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    cmap = plt.get_cmap("viridis")
    print(f"\n{'J':>2} {'a_max[deg/s^2]':>15} {'tau_sat':>9} {'J_eff[tau.s2/deg]':>18}")
    for p in paths:
        j, res = analyze(p)
        pct = [x["pct"] for x in res]
        a_slope = [x["a_slope"] for x in res]
        a_tri = [x["a_tri"] for x in res]
        tau = [x["tau"] for x in res]
        # plateau estimate over the saturated runs
        sat = [x for x in res if x["pct"] >= args.sat_from]
        a_max = float(np.median([x["a_slope"] for x in sat])) if sat else float("nan")
        a_max_tri = float(np.median([x["a_tri"] for x in sat])) if sat else float("nan")
        tau_sat = float(np.median([x["tau"] for x in sat])) if sat else float("nan")
        j_eff = tau_sat / a_max if a_max else float("nan")
        summary.append(dict(joint=j, a_max=a_max, a_max_tri=a_max_tri,
                            tau_sat=tau_sat, j_eff=j_eff))
        print(f"{j:>2} {a_max:>8.0f}(slope) {tau_sat:>9.4f} {j_eff:>18.2e}   "
              f"[triangle {a_max_tri:.0f}]")
        c = cmap(j / 5.0)
        ax[0].plot(pct, a_slope, "o-", color=c, label=f"J{j} slope")
        ax[0].plot(pct, a_tri, "x--", color=c, alpha=0.5)
        ax[1].plot(pct, tau, "o-", color=c, label=f"J{j}")
    ax[0].axvline(args.sat_from, ls=":", color="0.6")
    ax[0].set_ylabel("acceleration [deg/s^2]")
    ax[0].set_title("Estimated acceleration vs commanded velocity  "
                    "(o solid = ramp-slope, x dashed = triangle-geometry; plateau = a_max)")
    ax[0].grid(alpha=.3); ax[0].legend(ncol=3, fontsize=8)
    ax[1].axvline(args.sat_from, ls=":", color="0.6")
    ax[1].set_xlabel("commanded peak velocity [% of limit]")
    ax[1].set_ylabel("|torque fb| (drive units)")
    ax[1].set_title("Saturated torque level vs commanded velocity")
    ax[1].grid(alpha=.3); ax[1].legend(ncol=3, fontsize=8)
    fig.suptitle("Acceleration / torque limit extracted from velocity-sweep saturation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(LOGDIR, "sin_velsweep_accel_limit.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")
    json.dump(summary, open(os.path.join(LOGDIR, "accel_limit_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
