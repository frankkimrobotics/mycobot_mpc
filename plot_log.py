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
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NUM_JOINTS = 6


def _save_fig(fig, name):
    """Save figure to figures/ directory as PNG and PDF."""
    os.makedirs(FIG_DIR, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(FIG_DIR, f"{name}_{stamp}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")


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


def plot_merged_trajectories(dfs, labels, fig, axes):
    """Plot all runs merged: position error (target - actual) and velocity per joint."""
    ax_pos, ax_vel = axes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(dfs), 10)))

    for i, (df, label) in enumerate(zip(dfs, labels)):
        t = df["elapsed_s"]
        clr = colors[i % len(colors)]

        # Position error per joint → compute norm of (target - q) per timestep
        pos_err = np.zeros(len(df))
        for j in range(NUM_JOINTS):
            pos_err += (df[f"target{j}"] - df[f"q{j}"]) ** 2
        pos_err = np.sqrt(pos_err)
        ax_pos.plot(t, pos_err, color=clr, linewidth=1, alpha=0.8, label=label)

        # Velocity norm per timestep
        vel_norm = np.zeros(len(df))
        for j in range(NUM_JOINTS):
            vel_norm += df[f"qvel{j}"] ** 2
        vel_norm = np.sqrt(vel_norm)
        ax_vel.plot(t, vel_norm, color=clr, linewidth=1, alpha=0.8, label=label)

    ax_pos.set_ylabel("Position error norm (deg)")
    ax_pos.set_title("Position Error (‖target − actual‖) — all runs")
    ax_pos.legend(fontsize=6, ncol=2, loc="upper right")
    ax_pos.grid(True, alpha=0.3)

    ax_vel.set_ylabel("Velocity norm (deg/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_title("Velocity Norm (‖q_vel‖) — all runs")
    ax_vel.legend(fontsize=6, ncol=2, loc="upper right")
    ax_vel.grid(True, alpha=0.3)


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

    # Load all CSVs
    n = len(csv_files)
    dfs = [load_csv(fp) for fp in csv_files]
    run_labels = [os.path.basename(fp).replace("mpc_", "").replace(".csv", "") for fp in csv_files]

    # --- Figure 1: Merged trajectories (pos error + vel norm) ---
    fig_traj, axes_traj = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig_traj.suptitle("MPC Trajectories — All Runs", fontsize=13, fontweight="bold")
    plot_merged_trajectories(dfs, run_labels, fig_traj, axes_traj)
    fig_traj.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig_traj, "trajectories")

    # --- Figure 2: Timing bar chart (averaged across all runs) ---
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    fig_bar.suptitle("Average Processing Time per Step (all runs)", fontsize=13, fontweight="bold")
    all_df = pd.concat(dfs, ignore_index=True)
    plot_timing_bar(all_df, f"{n} runs combined", ax_bar)
    fig_bar.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig_bar, "timing_bars")

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
            means = [df[step].mean() for df in dfs]
            ax_cmp.bar(x + j * width, means, width, label=lbl, color=clr, edgecolor="black", linewidth=0.5)

        ax_cmp.set_xticks(x + width * 1.5)
        ax_cmp.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=8)
        ax_cmp.set_ylabel("Time (ms)")
        ax_cmp.legend()
        ax_cmp.grid(True, axis="y", alpha=0.3)
        fig_cmp.tight_layout(rect=[0, 0, 1, 0.95])
        _save_fig(fig_cmp, "timing_comparison")

    plt.show()


if __name__ == "__main__":
    main()
