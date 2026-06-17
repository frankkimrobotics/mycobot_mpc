#!/usr/bin/env python3
"""
plot_calibrate_ros2 :: plot joint command vs robot response from a
calibrate_ros2.py log (synchronized timestamps).

Reads logs/calibrate_ros2_<stamp>.jsonl and draws, per joint, the commanded
setpoint (step) over the measured response, on the shared `t_sync` clock.

  * command  : target_deg from /mycobot/cmd/move records (LinuxCNC degrees)
  * response : /joint_states (URDF radians) converted back to LinuxCNC degrees
               so it overlays the command in the same frame

Usage:
    python3 plot_calibrate_ros2.py [log.jsonl]   # default: newest log
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from joint_conventions import MAX_JOINTS, rad_to_linuxcnc_deg

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_log():
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "calibrate_ros2_*.jsonl")))
    if not logs:
        raise SystemExit("no calibrate_ros2_*.jsonl logs found")
    return logs[-1]


def main():
    ap = argparse.ArgumentParser(description="Plot command vs response from a calibrate_ros2 log.")
    ap.add_argument("log", nargs="?", default=None, help="JSONL log (default: newest)")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the log)")
    args = ap.parse_args()

    path = args.log or newest_log()
    rows = [json.loads(l) for l in open(path)]
    t0 = min(r["t_sync"] for r in rows if "t_sync" in r)

    cmds = [r for r in rows if r.get("type") == "command"]
    resp = [r for r in rows if r.get("source") == "joint_states" and r.get("joints_rad")]
    if not resp:
        raise SystemExit("no /joint_states responses in log")

    rt = np.array([r["t_sync"] - t0 for r in resp])
    rdeg = np.array([rad_to_linuxcnc_deg(r["joints_rad"]) for r in resp])  # N x 6, LinuxCNC deg
    ct = np.array([r["t_sync"] - t0 for r in cmds])
    ctgt = np.array([r["target_deg"] for r in cmds])                        # M x 6

    fig, axes = plt.subplots(MAX_JOINTS, 1, figsize=(12, 15), sharex=True)
    for j in range(MAX_JOINTS):
        ax = axes[j]
        ax.plot(rt, rdeg[:, j], "-", lw=1.0, color="C0", label="response (actual)")
        ax.step(ct, ctgt[:, j], where="post", lw=1.6, color="C3", label="command (target)")
        # mark command instants
        for tc in ct:
            ax.axvline(tc, color="0.85", lw=0.6, zorder=0)
        ax.set_ylabel(f"J{j}  [deg]")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time since start  [s]  (synchronized t_sync clock)")
    fig.suptitle(
        "myCobot Pro 630 — calibration command vs response\n"
        f"{os.path.basename(path)}   ({len(cmds)} commands, {len(resp)} responses)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out = args.out or path.replace(".jsonl", ".png")
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    # quick tracking summary
    print("per-joint command range (deg):")
    for j in range(MAX_JOINTS):
        print(f"  J{j}: {ctgt[:, j].min():+.1f} .. {ctgt[:, j].max():+.1f}")


if __name__ == "__main__":
    main()
