#!/usr/bin/env python3
"""
plot_servo_char :: plot the low-level drive feedback (posfb/velfb/torqfb) from a
characterize_servo.py log against the synchronized clock.

Usage: python3 plot_servo_char.py [servo_char_*.jsonl]   (default: newest)
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def newest():
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "servo_char_*.jsonl")))
    if not logs:
        raise SystemExit("no servo_char logs")
    return logs[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows = [json.loads(l) for l in open(path)]
    fb = [r for r in rows if r.get("type") == "drive_feedback"]
    met = next((r for r in rows if r.get("type") == "metrics"), {})
    j = fb[0]["joint"] if fb else 0
    t = np.array([r["t_sync"] for r in fb]); t = t - t[0]
    pos = np.array([r["posfb"] for r in fb])
    vel = np.array([r["velfb"] for r in fb])
    torq = np.array([r["torqfb"] for r in fb])

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, pos, color="C0"); ax[0].set_ylabel("posfb [deg]")
    ax[1].plot(t, vel, color="C1"); ax[1].set_ylabel("velfb [deg/s]")
    ax[2].plot(t, torq, color="C3"); ax[2].set_ylabel("torqfb")
    ax[2].set_xlabel("time [s] (synchronized clock)")
    for a in ax:
        a.grid(True, alpha=0.3)
    sub = (f"vel BW~{met.get('vel_bw_hz', float('nan')):.0f} Hz (rise "
           f"{met.get('vel_rise_s', float('nan'))*1e3:.0f} ms, sampling-limited), "
           f"plateau {met.get('plateau_vel_deg_s', float('nan')):.1f} deg/s, "
           f"peak torq {met.get('peak_torque', float('nan')):.3f}") if met else ""
    fig.suptitle(f"myCobot Pro 630 - low-level drive feedback, joint {j}\n{os.path.basename(path)}\n{sub}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = path.replace(".jsonl", ".png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
