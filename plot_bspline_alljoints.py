#!/usr/bin/env python3
"""
plot_bspline_alljoints :: grid comparison of the s-curve vs sinusoid move for
every joint (from bspline_vs_sine_j<N>_*.json), run at the 10 ms loop period.

Rows = joints J0..J5, columns = [position cmd vs actual, velocity, tracking gap].
Sine = blue, s-curve = orange; command = dashed. Also prints a metrics table
(completion time, RMS error, peak cmd-fb gap) per joint per profile.

Usage: python3 plot_bspline_alljoints.py            # newest json per joint
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")
COL = {"sine": "C0", "scurve": "C1"}


def newest_per_joint():
    out = {}
    for p in glob.glob(os.path.join(LOGDIR, "bspline_vs_sine_j*_*.json")):
        m = re.search(r"bspline_vs_sine_j(\d)_", os.path.basename(p))
        if m:
            j = int(m.group(1))
            if j not in out or os.path.getmtime(p) > os.path.getmtime(out[j]):
                out[j] = p
    return out


def main():
    paths = newest_per_joint()
    if not paths:
        print("no bspline_vs_sine logs found")
        return
    joints = sorted(paths)
    n = len(joints)
    fig, ax = plt.subplots(n, 3, figsize=(15, 2.6 * n), squeeze=False)

    print(f"\n{'J':>2} {'profile':>7} {'t_complete[s]':>13} {'RMSerr[deg]':>11} {'peak_gap[deg]':>13} {'Vpk':>6}")
    for ri, j in enumerate(joints):
        rows = json.load(open(paths[j]))
        for r in rows:
            c = COL.get(r["label"], "C2")
            t = np.array(r["t"])
            ax[ri][0].plot(t, r["cmd"], "--", color=c, lw=1, alpha=.6)
            ax[ri][0].plot(t, r["pos"], "-", color=c, lw=1.6, label=r["label"])
            ax[ri][1].plot(t, r["vel"], "-", color=c, lw=1.3, label=r["label"])
            ax[ri][2].plot(t, np.abs(np.array(r["cmd"]) - np.array(r["pos"])), "-",
                           color=c, lw=1.3, label=r["label"])
            tc = r.get("t_complete")
            tc_s = f"{tc:.2f}" if isinstance(tc, (int, float)) and tc == tc else "nan"
            print(f"{j:>2} {r['label']:>7} {tc_s:>13} {r['rms']:>11.2f} {r['peak_gap']:>13.2f} {r['vpk']:>6.1f}")
        ax[ri][0].set_ylabel(f"J{j}\ndeg from base", fontsize=9)
        ax[ri][0].grid(alpha=.3); ax[ri][1].grid(alpha=.3); ax[ri][2].grid(alpha=.3)
        if ri == 0:
            ax[ri][0].legend(fontsize=8); ax[ri][0].set_title("position: cmd (dashed) vs actual")
            ax[ri][1].set_title("velocity [deg/s]"); ax[ri][2].set_title("tracking gap |cmd-actual| [deg]")
    for c in range(3):
        ax[-1][c].set_xlabel("time [s]")
    fig.suptitle("S-curve (orange) vs Sinusoid (blue) - all joints, 30 deg move @25 deg/s, 10 ms loop",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(LOGDIR, "bspline_vs_sine_alljoints.png")
    fig.savefig(out, dpi=100)
    print(f"\nused logs: " + ", ".join(os.path.basename(paths[j]) for j in joints))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
