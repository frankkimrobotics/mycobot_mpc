#!/usr/bin/env python3
"""
Visualize MPC log CSV files.

Usage:
    python plot_log.py                          # plot all CSVs in logs/
    python plot_log.py logs/mpc_20260215_*.csv  # plot specific files
    python plot_log.py --latest                 # plot only the most recent file
    python plot_log.py --latest 3               # plot the 3 most recent files
"""

import sys
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
NUM_JOINTS = 6


def load_csv(filepath):
    """Load a log CSV and add an elapsed-time column (seconds from start)."""
    df = pd.read_csv(filepath)
    # Convert timestamp string (HH:MM:SS.mmm) to elapsed seconds
    t0_parts = df["timestamp"].iloc[0].split(":")
    t0_sec = int(t0_parts[0]) * 3600 + int(t0_parts[1]) * 60 + float(t0_parts[2])

    def to_elapsed(ts):
        p = ts.split(":")
        sec = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
        return sec - t0_sec

    df["elapsed_s"] = df["timestamp"].apply(to_elapsed)
    return df


def plot_trajectory(df, title, axes_pos, axes_vel, ax_err):
    """Plot joint position and velocity trajectories for one run."""
    t = df["elapsed_s"]

    # Joint positions + targets
    for j in range(NUM_JOINTS):
        axes_pos.plot(t, df[f"q{j}"], label=f"q{j}", linewidth=1)
        axes_pos.axhline(
            y=df[f"target{j}"].iloc[0], color=f"C{j}",
            linestyle="--", alpha=0.4, linewidth=0.8,
        )
    axes_pos.set_ylabel("Joint angle (deg)")
    axes_pos.set_title(title, fontsize=10)
    axes_pos.legend(fontsize=7, ncol=3, loc="upper right")
    axes_pos.grid(True, alpha=0.3)

    # Joint velocities
    for j in range(NUM_JOINTS):
        axes_vel.plot(t, df[f"qvel{j}"], label=f"qvel{j}", linewidth=1)
    axes_vel.set_ylabel("Joint velocity (deg/s)")
    axes_vel.set_xlabel("Time (s)")
    axes_vel.legend(fontsize=7, ncol=3, loc="upper right")
    axes_vel.grid(True, alpha=0.3)

    # Error norm
    ax_err.plot(t, df["err_norm"], color="red", linewidth=1)
    ax_err.set_ylabel("Error norm (deg)")
    ax_err.set_xlabel("Time (s)")
    ax_err.grid(True, alpha=0.3)


def plot_timing_bar(df, title, ax):
    """Bar chart of average processing time per step for one run."""
    steps = ["poll_ms", "pd_solve_ms", "hal_write_ms", "sleep_ms"]
    labels = ["Poll", "PD Solve", "HAL Write", "Sleep"]
    means = [df[s].mean() for s in steps]
    stds = [df[s].std() for s in steps]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bar values
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=8,
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize MPC log CSVs")
    parser.add_argument("files", nargs="*", help="CSV file paths (default: all in logs/)")
    parser.add_argument("--latest", nargs="?", const=1, type=int, metavar="N",
                        help="Plot only the N most recent log files (default: 1)")
    args = parser.parse_args()

    # Resolve file list
    if args.files:
        csv_files = args.files
    else:
        csv_files = sorted(glob.glob(os.path.join(LOG_DIR, "mpc_*.csv")))

    if not csv_files:
        print(f"No CSV files found in {LOG_DIR}")
        sys.exit(1)

    if args.latest is not None:
        csv_files = csv_files[-args.latest:]

    print(f"Plotting {len(csv_files)} log file(s)...")

    # --- Figure 1: Trajectories ---
    n = len(csv_files)
    fig_traj, axes = plt.subplots(n * 3, 1, figsize=(12, 4 * n), squeeze=False)
    fig_traj.suptitle("MPC Joint Trajectories", fontsize=13, fontweight="bold")

    for i, fp in enumerate(csv_files):
        df = load_csv(fp)
        label = os.path.basename(fp).replace(".csv", "")
        ax_pos = axes[i * 3, 0]
        ax_vel = axes[i * 3 + 1, 0]
        ax_err = axes[i * 3 + 2, 0]
        plot_trajectory(df, label, ax_pos, ax_vel, ax_err)

    fig_traj.tight_layout(rect=[0, 0, 1, 0.97])

    # --- Figure 2: Timing bar charts ---
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig_bar, axes_bar = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    fig_bar.suptitle("Average Processing Time per Step", fontsize=13, fontweight="bold")

    for i, fp in enumerate(csv_files):
        df = load_csv(fp)
        label = os.path.basename(fp).replace(".csv", "")
        r, c = divmod(i, cols)
        plot_timing_bar(df, label, axes_bar[r, c])

    # Hide unused subplots
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes_bar[r, c].set_visible(False)

    fig_bar.tight_layout(rect=[0, 0, 1, 0.95])

    # --- Figure 3: Timing comparison across runs (if multiple) ---
    if n > 1:
        fig_cmp, ax_cmp = plt.subplots(figsize=(max(8, n * 1.5), 5))
        fig_cmp.suptitle("Timing Comparison Across Runs", fontsize=13, fontweight="bold")

        steps = ["poll_ms", "pd_solve_ms", "hal_write_ms", "sleep_ms"]
        labels = ["Poll", "PD Solve", "HAL Write", "Sleep"]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
        x = np.arange(n)
        width = 0.18

        for j, (step, lbl, clr) in enumerate(zip(steps, labels, colors)):
            means = []
            for fp in csv_files:
                df = load_csv(fp)
                means.append(df[step].mean())
            ax_cmp.bar(x + j * width, means, width, label=lbl, color=clr, edgecolor="black", linewidth=0.5)

        ax_cmp.set_xticks(x + width * 1.5)
        run_labels = [os.path.basename(fp).replace("mpc_", "").replace(".csv", "") for fp in csv_files]
        ax_cmp.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
        ax_cmp.set_ylabel("Time (ms)")
        ax_cmp.legend()
        ax_cmp.grid(True, axis="y", alpha=0.3)
        fig_cmp.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    main()
