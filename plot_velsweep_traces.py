#!/usr/bin/env python3
"""
plot_velsweep_traces :: per-joint, per-run time-series of angle / velocity /
torque from a sinusoid velocity-sweep raw log (sin_velsweep_raw_j<N>_*.jsonl).

For each swept joint it builds ONE figure: rows = runs (10 .. 100 % velocity),
cols = [angle, velocity, torque]. Every panel overlays all 6 joints (the swept
joint bold, the others thin) so coupling shows up. The angle panel also draws
the commanded sinusoid (dashed) for the swept joint.

Usage:
    python3 plot_velsweep_traces.py                       # newest raw log per joint
    python3 plot_velsweep_traces.py --files raw_j0.jsonl  # specific log(s)
"""
import argparse
import glob
import json
import math
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
        if not m:
            continue
        j = int(m.group(1))
        if j not in out or os.path.getmtime(p) > os.path.getmtime(out[j]):
            out[j] = p
    return out


def plot_one(path):
    runs = [json.loads(l) for l in open(path) if l.strip()]
    runs = [r for r in runs if r.get("samples")]
    if not runs:
        print(f"  {os.path.basename(path)}: no samples"); return None
    runs.sort(key=lambda r: r["pct"])
    jact = runs[0]["joint"]
    base = runs[0]["base"]
    n = len(runs)
    fig, ax = plt.subplots(n, 3, figsize=(15, 2.4 * n), sharex=False, squeeze=False)
    cmap = plt.get_cmap("tab10")
    for ri, r in enumerate(runs):
        s = r["samples"]
        t = np.array([x[0] for x in s])
        pos = np.array([x[1] for x in s])   # (N,6) deg
        vel = np.array([x[2] for x in s])   # (N,6) deg/s
        tor = np.array([x[3] for x in s])   # (N,6) torque
        for j in range(MAX_JOINTS):
            lw = 2.0 if j == jact else 0.8
            a = 1.0 if j == jact else 0.45
            c = cmap(j)
            ax[ri][0].plot(t, pos[:, j] - base[j], color=c, lw=lw, alpha=a,
                           label=f"J{j}" if ri == 0 else None)
            ax[ri][1].plot(t, vel[:, j], color=c, lw=lw, alpha=a)
            ax[ri][2].plot(t, tor[:, j], color=c, lw=lw, alpha=a)
        # commanded sinusoid (angle) + its derivative (velocity) for the swept joint
        w = 2 * math.pi * r["freq"]; amp = r["amp_cmd"]
        ax[ri][0].plot(t, amp * np.sin(w * t), "k--", lw=1.0, alpha=0.7,
                       label="cmd" if ri == 0 else None)
        ax[ri][1].plot(t, amp * w * np.cos(w * t), "k--", lw=1.0, alpha=0.7)
        ax[ri][0].set_ylabel(f"{r['pct']:.0f}%  ({r['vpk']:.0f}°/s)\nangle dev [deg]", fontsize=8)
        ax[ri][1].set_ylabel("vel [deg/s]", fontsize=8)
        ax[ri][2].set_ylabel("torque", fontsize=8)
        for c in range(3):
            ax[ri][c].grid(alpha=.3); ax[ri][c].tick_params(labelsize=7)
    ax[0][0].set_title("angle deviation from base")
    ax[0][1].set_title("velocity")
    ax[0][2].set_title("torque (effort)")
    for c in range(3):
        ax[-1][c].set_xlabel("time [s]")
    ax[0][0].legend(ncol=4, fontsize=7, loc="upper right")
    fig.suptitle(f"Velocity sweep raw traces - swept joint J{jact} "
                 f"(±{runs[0]['amp_cmd']:.0f}°, vff_scale=1); bold=J{jact}, dashed=command",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = os.path.join(LOGDIR, f"sin_velsweep_traces_j{jact}.png")
    fig.savefig(out, dpi=100); plt.close(fig)
    print(f"  J{jact}: {os.path.basename(path)} -> {os.path.basename(out)} ({n} runs)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="raw jsonl logs (else newest per joint)")
    args = ap.parse_args()
    paths = args.files if args.files else [newest_raw_per_joint()[j]
                                           for j in sorted(newest_raw_per_joint())]
    if not paths:
        print("no sin_velsweep_raw logs found"); return
    print("plotting raw traces:")
    for p in paths:
        plot_one(p)


if __name__ == "__main__":
    main()
