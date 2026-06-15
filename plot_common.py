"""
Shared helpers for the log-plotting scripts (plot_log.py, plot_hal.py, sim_mpc.py).

Holds the figure/log directories, the per-joint color palette, the robot-CSV
loader, and the figure-saving helper that were previously duplicated.
"""

import os
from datetime import datetime

import pandas as pd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
NUM_JOINTS = 6

# Per-joint color palette (matplotlib tab10 order), shared across all plots.
JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def load_robot_csv(filepath):
    """Load a robot log CSV and add an elapsed-time column (seconds from start)."""
    df = pd.read_csv(filepath)
    t0_parts = df["timestamp"].iloc[0].split(":")
    t0_sec = int(t0_parts[0]) * 3600 + int(t0_parts[1]) * 60 + float(t0_parts[2])

    def to_elapsed(ts):
        p = ts.split(":")
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2]) - t0_sec

    df["elapsed_s"] = df["timestamp"].apply(to_elapsed)
    return df


def save_fig(fig, name, dpi=200):
    """Save a figure to figures/ as <name>_<timestamp>.png and return the path."""
    os.makedirs(FIG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(FIG_DIR, f"{name}_{stamp}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    return path
