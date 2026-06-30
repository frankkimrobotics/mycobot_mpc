#!/usr/bin/env python3
"""
plot_velsweep_alljoints :: overlay the per-joint sinusoid velocity-sweep logs
(logs/sin_velsweep_j<N>_*.jsonl) on one figure so the closed-loop tracking
bandwidth of every joint can be compared at a glance.

Picks the NEWEST log per joint by default. Plots tracked amplitude (% of
commanded) and phase lag vs commanded peak velocity (% of limit), one curve
per joint.

Usage:
    python3 plot_velsweep_alljoints.py                # newest per joint
    python3 plot_velsweep_alljoints.py --files a.jsonl b.jsonl ...
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOGDIR = os.path.join(HERE, "logs")


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["pct"])
    return rows


def newest_per_joint():
    out = {}
    for p in glob.glob(os.path.join(LOGDIR, "sin_velsweep_j*_*.jsonl")):
        m = re.search(r"sin_velsweep_j(\d)_", os.path.basename(p))
        if not m:
            continue
        j = int(m.group(1))
        if j not in out or os.path.getmtime(p) > os.path.getmtime(out[j]):
            out[j] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="explicit jsonl logs (else newest per joint)")
    ap.add_argument("--out", default=os.path.join(LOGDIR, "sin_velsweep_alljoints.png"))
    args = ap.parse_args()

    if args.files:
        paths = {}
        for p in args.files:
            m = re.search(r"_j(\d)_", os.path.basename(p))
            paths[int(m.group(1)) if m else len(paths)] = p
    else:
        paths = newest_per_joint()
    if not paths:
        print("no sin_velsweep logs found")
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.get_cmap("viridis")
    for j in sorted(paths):
        rows = load(paths[j])
        pct = [r["pct"] for r in rows]
        c = cmap(j / 5.0)
        ax[0].plot(pct, [r["amp_pct"] for r in rows], "o-", color=c, label=f"J{j}")
        ax[1].plot(pct, [abs(r["phase_lag"]) for r in rows], "o-", color=c, label=f"J{j}")

    ax[0].axhline(100, ls=":", color="0.6")
    ax[0].axhline(70.7, ls="--", color="0.7")
    ax[0].text(11, 72.5, "-3 dB (70.7%)", fontsize=8, color="0.4")
    ax[0].set_ylim(0, 105)
    ax[0].set_xlabel("commanded peak velocity [% of limit]")
    ax[0].set_ylabel("tracked amplitude [% of commanded]")
    ax[0].set_title("amplitude vs velocity")
    ax[0].grid(alpha=.3); ax[0].legend(ncol=2, fontsize=9)

    ax[1].set_xlabel("commanded peak velocity [% of limit]")
    ax[1].set_ylabel("phase lag [deg]")
    ax[1].set_title("phase lag vs velocity")
    ax[1].grid(alpha=.3); ax[1].legend(ncol=2, fontsize=9)

    fig.suptitle("Sinusoid tracking vs commanded velocity - all joints (velocity-FF, vff_scale=1)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=110)
    print(f"used logs:")
    for j in sorted(paths):
        print(f"  J{j}: {os.path.basename(paths[j])}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
