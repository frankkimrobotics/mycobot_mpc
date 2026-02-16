#!/usr/bin/env python3
"""
Visualize MPC log CSV files (robot-side and desktop-side).

Robot logs  (mpc_*.csv)     — per-loop timing: poll, solve, hal_write, sleep
Desktop logs (control_*.csv) — per-move pipeline: IK, FK, network, robot exec

Usage:
    python plot_log.py                          # plot all CSVs in logs/
    python plot_log.py logs/mpc_20260215_*.csv  # plot specific files
    python plot_log.py --latest                 # plot only the most recent file
    python plot_log.py --latest 3               # plot the 3 most recent files
    python plot_log.py --pipeline               # plot only desktop pipeline logs
    python plot_log.py --robot                  # plot only robot-side logs
"""

import sys
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NUM_JOINTS = 6


def _save_fig(fig, name):
    """Save figure to figures/ directory as PNG."""
    os.makedirs(FIG_DIR, exist_ok=True)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(FIG_DIR, f"{name}_{stamp}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Robot-side logs (mpc_*.csv)
# ═══════════════════════════════════════════════════════════════════════════════

def load_robot_csv(filepath):
    """Load a robot log CSV and add an elapsed-time column (seconds from start)."""
    df = pd.read_csv(filepath)
    t0_parts = df["timestamp"].iloc[0].split(":")
    t0_sec = int(t0_parts[0]) * 3600 + int(t0_parts[1]) * 60 + float(t0_parts[2])

    def to_elapsed(ts):
        p = ts.split(":")
        sec = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
        return sec - t0_sec

    df["elapsed_s"] = df["timestamp"].apply(to_elapsed)
    return df


def plot_merged_trajectories(dfs, axes):
    """Plot all runs merged: signed per-joint position error and cmd velocity."""
    joint_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for j in range(NUM_JOINTS):
        ax_pos = axes[0, j]
        ax_vel = axes[1, j]

        for i, df in enumerate(dfs):
            t = df["elapsed_s"]
            alpha = max(0.3, 1.0 - i * 0.05)

            pos_err = df[f"target{j}"] - df[f"q{j}"]
            ax_pos.plot(t, pos_err, color=joint_colors[j], linewidth=0.8, alpha=alpha)

            vel_err = df[f"cmd_vel{j}"] - df[f"qvel{j}"]
            ax_vel.plot(t, vel_err, color=joint_colors[j], linewidth=0.8, alpha=alpha)

        ax_pos.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
        ax_pos.set_title(f"J{j}", fontsize=10)
        ax_pos.grid(True, alpha=0.3)
        if j == 0:
            ax_pos.set_ylabel("Pos error (deg)")

        ax_vel.axhline(y=0, color="k", linewidth=0.5, alpha=0.5)
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True, alpha=0.3)
        if j == 0:
            ax_vel.set_ylabel("Vel error (deg/s)")


def plot_robot_timing_bar(df, title, ax):
    """Bar chart of average per-loop processing time for robot-side logs."""
    steps = ["poll_ms", "pd_solve_ms", "hal_write_ms", "sleep_ms"]
    labels = ["Poll", "PD Solve", "HAL Write", "Sleep"]
    means = [df[s].mean() for s in steps]
    stds = [df[s].std() for s in steps]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Time (ms)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=8,
        )


def plot_robot_logs(csv_files):
    """Generate all plots for robot-side (mpc_*.csv) logs."""
    n = len(csv_files)
    dfs = [load_robot_csv(fp) for fp in csv_files]
    print(f"Plotting {n} robot log file(s)...")

    # Figure 1: Per-joint trajectories
    fig_traj, axes_traj = plt.subplots(2, NUM_JOINTS, figsize=(18, 6), sharex=True)
    fig_traj.suptitle("Robot Trajectories — All Runs (top: pos error, bottom: vel error)",
                      fontsize=13, fontweight="bold")
    plot_merged_trajectories(dfs, axes_traj)
    fig_traj.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig_traj, "trajectories")

    # Figure 2: Per-loop timing bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    fig_bar.suptitle("Robot: Average Per-Loop Timing (all runs)", fontsize=13, fontweight="bold")
    all_df = pd.concat(dfs, ignore_index=True)
    plot_robot_timing_bar(all_df, f"{n} runs combined", ax_bar)
    fig_bar.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig_bar, "robot_loop_timing")


# ═══════════════════════════════════════════════════════════════════════════════
#  Desktop-side logs (control_*.csv) — full pipeline timing
# ═══════════════════════════════════════════════════════════════════════════════

def load_control_csv(filepath):
    """Load a desktop control log CSV."""
    df = pd.read_csv(filepath)
    return df


def plot_pipeline_waterfall(df, ax):
    """Stacked horizontal bar (waterfall) showing per-move pipeline stages.

    Each bar is one move, stacked segments = IK, FK, network, robot exec.
    """
    stages = [
        ("ik_solve_ms",            "IK Solve",      "#FF6B6B"),
        ("fk_verify_ms",           "FK Verify",     "#FFA07A"),
        ("cmd_send_ms",            "Cmd Send",      "#4ECDC4"),
        ("robot_exec_ms",          "Robot Exec",    "#45B7D1"),
    ]

    n = len(df)
    y_pos = np.arange(n)
    left = np.zeros(n)

    for col, label, color in stages:
        if col in df.columns:
            vals = df[col].fillna(0).values
            ax.barh(y_pos, vals, left=left, label=label, color=color,
                    edgecolor="white", linewidth=0.5, height=0.7)
            left += vals

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Move {i+1}" for i in range(n)], fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Pipeline Waterfall (per move)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()


def plot_pipeline_bar(df, ax):
    """Grouped bar chart: average time per pipeline stage across all moves."""
    desktop_stages = [
        ("ik_solve_ms",   "IK Solve"),
        ("fk_verify_ms",  "FK Verify"),
        ("cmd_send_ms",   "Cmd Send\n(network)"),
        ("ack_rtt_ms",    "Ack RTT"),
        ("wait_done_ms",  "Wait Done\n(exec)"),
    ]
    robot_stages = [
        ("robot_exec_ms",         "Robot\nExec"),
        ("robot_avg_poll_ms",     "Robot\nAvg Poll"),
        ("robot_avg_solve_ms",    "Robot\nAvg Solve"),
        ("robot_avg_hal_write_ms", "Robot\nAvg HAL"),
        ("robot_avg_sleep_ms",    "Robot\nAvg Sleep"),
    ]

    all_stages = desktop_stages + robot_stages
    cols = [s[0] for s in all_stages]
    labels = [s[1] for s in all_stages]

    # Use blue palette for desktop, green palette for robot
    n_desktop = len(desktop_stages)
    n_robot = len(robot_stages)
    desktop_colors = plt.colormaps["Blues"](np.linspace(0.4, 0.8, n_desktop))
    robot_colors = plt.colormaps["Greens"](np.linspace(0.4, 0.8, n_robot))
    colors = list(desktop_colors) + list(robot_colors)

    means = []
    stds = []
    for col in cols:
        if col in df.columns:
            means.append(df[col].mean())
            stds.append(df[col].std())
        else:
            means.append(0)
            stds.append(0)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=3,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Average Pipeline Timing Breakdown", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate
    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=7)

    # Legend for desktop vs robot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=desktop_colors[2], label="Desktop"),
        Patch(facecolor=robot_colors[2], label="Robot"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


def plot_pipeline_total_vs_move(df, ax):
    """Line plot showing total time and components for each move."""
    move_ids = df["move_id"].values if "move_id" in df.columns else np.arange(1, len(df) + 1)

    lines = [
        ("total_ms",      "Total",       "#2C3E50", 2.0, "-"),
        ("ik_solve_ms",   "IK Solve",    "#FF6B6B", 1.2, "--"),
        ("cmd_send_ms",   "Network",     "#4ECDC4", 1.2, "--"),
        ("robot_exec_ms", "Robot Exec",  "#45B7D1", 1.5, "-"),
        ("wait_done_ms",  "Wait Done",   "#9B59B6", 1.0, ":"),
    ]

    for col, label, color, lw, ls in lines:
        if col in df.columns:
            ax.plot(move_ids, df[col].values, label=label, color=color,
                    linewidth=lw, linestyle=ls, marker="o", markersize=3)

    ax.set_xlabel("Move #")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Pipeline Timing per Move", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def plot_pipeline_accuracy(df, ax):
    """Plot position error (mm) and joint error (deg) per move."""
    move_ids = df["move_id"].values if "move_id" in df.columns else np.arange(1, len(df) + 1)

    ax2 = ax.twinx()

    if "pos_error_mm" in df.columns:
        ax.bar(move_ids - 0.15, df["pos_error_mm"].values, width=0.3,
               color="#E74C3C", alpha=0.7, label="Pos error (mm)")
    if "joint_error_deg" in df.columns:
        ax2.bar(move_ids + 0.15, df["joint_error_deg"].values, width=0.3,
                color="#3498DB", alpha=0.7, label="Joint error (deg)")

    ax.set_xlabel("Move #")
    ax.set_ylabel("Position Error (mm)", color="#E74C3C")
    ax2.set_ylabel("Joint Error (deg)", color="#3498DB")
    ax.set_title("Accuracy per Move", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")


def plot_control_logs(csv_files):
    """Generate all plots for desktop-side (control_*.csv) logs."""
    n = len(csv_files)
    dfs = [load_control_csv(fp) for fp in csv_files]
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Plotting {n} desktop control log file(s) ({len(all_df)} moves total)...")

    # Figure: 2x2 layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Desktop Pipeline Timing — {n} session(s), {len(all_df)} moves",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_pipeline_bar(all_df, ax1)
    plot_pipeline_waterfall(all_df, ax2)
    plot_pipeline_total_vs_move(all_df, ax3)
    plot_pipeline_accuracy(all_df, ax4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, "pipeline_timing")

    # Print summary table
    print("\n  Pipeline Timing Summary:")
    print(f"  {'Stage':<25s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'Min (ms)':>10s} {'Max (ms)':>10s}")
    print("  " + "-" * 65)
    for col in ["ik_solve_ms", "fk_verify_ms", "cmd_send_ms", "ack_rtt_ms",
                "wait_done_ms", "total_ms", "robot_exec_ms",
                "robot_avg_poll_ms", "robot_avg_solve_ms",
                "robot_avg_hal_write_ms", "robot_avg_sleep_ms"]:
        if col in all_df.columns:
            vals = all_df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:<25s} {vals.mean():>10.2f} {vals.std():>10.2f} "
                      f"{vals.min():>10.2f} {vals.max():>10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MPC & pipeline log CSVs")
    parser.add_argument("files", nargs="*", help="CSV file paths (default: all in logs/)")
    parser.add_argument("--latest", nargs="?", const=1, type=int, metavar="N",
                        help="Plot only the N most recent log files (default: 1)")
    parser.add_argument("--pipeline", action="store_true",
                        help="Plot only desktop pipeline logs (control_*.csv)")
    parser.add_argument("--robot", action="store_true",
                        help="Plot only robot-side logs (mpc_*.csv)")
    args = parser.parse_args()

    # Resolve file lists
    if args.files:
        all_files = args.files
    else:
        all_files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))

    if not all_files:
        print(f"No CSV files found in {LOG_DIR}")
        sys.exit(1)

    # Split into robot and control logs
    robot_files = [f for f in all_files if os.path.basename(f).startswith("mpc_")]
    control_files = [f for f in all_files if os.path.basename(f).startswith("control_")]

    if args.latest is not None:
        robot_files = robot_files[-args.latest:]
        control_files = control_files[-args.latest:]

    # Plot based on flags
    plotted = False

    if not args.pipeline and robot_files:
        plot_robot_logs(robot_files)
        plotted = True

    if not args.robot and control_files:
        plot_control_logs(control_files)
        plotted = True

    if not plotted:
        kind = "pipeline" if args.pipeline else ("robot" if args.robot else "any")
        print(f"No {kind} log files found in {LOG_DIR}")
        sys.exit(1)

    plt.show()


if __name__ == "__main__":
    main()
