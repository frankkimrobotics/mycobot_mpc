#!/usr/bin/env python3
"""
Plot HAL and joint information per joint from robot log CSVs.

One figure per joint (6 figures). Each figure has 4 panels:
  - Joint angle (line) + cmd_pos (scatter at timestamp)
  - Joint velocity (line) + cmd_vel (scatter at timestamp)
  - HAL velocity (line)
  - HAL torque (line)

All CSV files in logs/ (mpc_*.csv, invdyn_*.csv) are overlaid on each plot.

Usage:
    python plot_hal.py                    # all robot CSVs in logs/
    python plot_hal.py logs/mpc_*.csv      # specific files
    python plot_hal.py --no-show           # save only, no interactive window
"""

import os
import sys
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Shared dirs, palette, and CSV loader.
from plot_common import LOG_DIR, FIG_DIR, NUM_JOINTS, JOINT_COLORS, load_robot_csv


def plot_hal_all(csv_files, no_show=False):
    """For each joint, one figure with 4 panels: angle, velocity, hal_vel, hal_torq. All CSVs overlaid."""
    if not csv_files:
        print("No robot CSV files found.")
        return

    dfs = [load_robot_csv(fp) for fp in csv_files]
    n_files = len(dfs)
    alpha_line = max(0.15, 1.0 - (n_files - 1) * 0.008)
    alpha_scatter = max(0.2, 0.8 - (n_files - 1) * 0.01)
    scatter_size = max(2, 8 - n_files // 10)

    os.makedirs(FIG_DIR, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for j in range(NUM_JOINTS):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
        fig.suptitle(f"Joint {j} — angle, velocity, HAL vel, HAL torq ({n_files} runs)", fontsize=12, fontweight="bold")

        # 1) Joint angle (line) + cmd_pos (scatter)
        ax_angle = axes[0, 0]
        for i, df in enumerate(dfs):
            t = df["elapsed_s"].values
            ax_angle.plot(t, df[f"q{j}"], color="C0", alpha=alpha_line, linewidth=0.8)
            ax_angle.scatter(t, df[f"cmd_pos{j}"], s=scatter_size, color="C1", alpha=alpha_scatter, label="cmd_pos" if i == 0 else None)
        ax_angle.set_ylabel("Angle (deg)")
        ax_angle.set_title("Joint angle (line) & cmd_pos (scatter)")
        ax_angle.grid(True, alpha=0.3)
        ax_angle.legend(loc="best", fontsize=8)

        # 2) Joint velocity (line) + cmd_vel (scatter)
        ax_vel = axes[0, 1]
        for i, df in enumerate(dfs):
            t = df["elapsed_s"].values
            ax_vel.plot(t, df[f"qvel{j}"], color="C0", alpha=alpha_line, linewidth=0.8)
            ax_vel.scatter(t, df[f"cmd_vel{j}"], s=scatter_size, color="C1", alpha=alpha_scatter, label="cmd_vel" if i == 0 else None)
        ax_vel.set_ylabel("Velocity (deg/s)")
        ax_vel.set_title("Joint velocity (line) & cmd_vel (scatter)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(loc="best", fontsize=8)

        # 3) HAL velocity
        ax_halv = axes[1, 0]
        for df in dfs:
            t = df["elapsed_s"].values
            ax_halv.plot(t, df[f"hal_vel{j}"], color="C2", alpha=alpha_line, linewidth=0.8)
        ax_halv.set_ylabel("HAL velocity")
        ax_halv.set_xlabel("Time (s)")
        ax_halv.set_title("HAL velocity")
        ax_halv.grid(True, alpha=0.3)

        # 4) HAL torque
        ax_halt = axes[1, 1]
        for df in dfs:
            t = df["elapsed_s"].values
            ax_halt.plot(t, df[f"hal_torq{j}"], color="C3", alpha=alpha_line, linewidth=0.8)
        ax_halt.set_ylabel("HAL torque")
        ax_halt.set_xlabel("Time (s)")
        ax_halt.set_title("HAL torque")
        ax_halt.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = os.path.join(FIG_DIR, f"hal_joint{j}_{stamp}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
        if no_show:
            plt.close(fig)

    if not no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot HAL & joint info per joint from robot CSVs")
    parser.add_argument("files", nargs="*", help="CSV paths (default: all mpc_*.csv, invdyn_*.csv in logs/)")
    parser.add_argument("--no-show", action="store_true", help="Save figures only, no interactive window")
    args = parser.parse_args()

    if args.files:
        csv_files = [f for f in args.files if os.path.isfile(f)]
    else:
        csv_files = sorted(
            glob.glob(os.path.join(LOG_DIR, "mpc_*.csv")) +
            glob.glob(os.path.join(LOG_DIR, "invdyn_*.csv"))
        )

    if not csv_files:
        print(f"No robot CSV files in {LOG_DIR} or given as arguments.")
        sys.exit(1)

    print(f"Plotting {len(csv_files)} CSV file(s), 6 figures (one per joint)...")
    plot_hal_all(csv_files, no_show=args.no_show)


if __name__ == "__main__":
    main()
