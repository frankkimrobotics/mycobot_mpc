#!/usr/bin/env python3
"""
Copy CSV log files from the Raspi's mycobot_mpc logs folder to the laptop.

Pulls from the robot's logs/ (and log/ if present) into the local mycobot_mpc/logs/
using rsync over SSH, or scp if rsync is not available.

Usage:
  # Use ROBOT_IP from environment (e.g. set by setup_robot_ip.sh)
  python fetch_robot_logs.py

  # Explicit host
  python fetch_robot_logs.py --host 10.0.0.27

  # Custom Raspi repo path and SSH user
  python fetch_robot_logs.py --host 10.0.0.27 --remote-dir /home/pi/Desktop/mpc --user pi
"""

import argparse
import os
import subprocess
import sys

# Local logs directory (same as control_robot.py, plot_log.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")

# Default Raspi path (from elerob.hal)
DEFAULT_REMOTE_DIR = "/home/pi/Desktop/mpc"
DEFAULT_SSH_USER = "pi"


def main():
    ap = argparse.ArgumentParser(
        description="Copy robot CSV logs from Raspi to this laptop's logs/ folder."
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="Raspi IP or hostname (default: ROBOT_IP env)",
    )
    ap.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help=f"Path to mycobot_mpc repo on Raspi (default: {DEFAULT_REMOTE_DIR})",
    )
    ap.add_argument(
        "--user",
        default=DEFAULT_SSH_USER,
        help=f"SSH user on Raspi (default: {DEFAULT_SSH_USER})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the rsync/scp command, do not run it",
    )
    args = ap.parse_args()

    if not args.host:
        print("Error: --host or ROBOT_IP must be set.", file=sys.stderr)
        print("  Example: export ROBOT_IP=10.0.0.27  or  python fetch_robot_logs.py --host 10.0.0.27", file=sys.stderr)
        sys.exit(1)

    os.makedirs(LOG_DIR, exist_ok=True)
    remote_logs = f"{args.user}@{args.host}:{args.remote_dir.rstrip('/')}/logs"
    remote_log_singular = f"{args.user}@{args.host}:{args.remote_dir.rstrip('/')}/log"

    # Prefer rsync (only copies new/changed files)
    rsync_cmd = [
        "rsync",
        "-avz",
        "--include=*.csv",
        "--exclude=*",
        f"{remote_logs}/",
        f"{LOG_DIR}/",
    ]
    rsync_cmd_log = [
        "rsync",
        "-avz",
        "--include=*.csv",
        "--exclude=*",
        f"{remote_log_singular}/",
        f"{LOG_DIR}/",
    ]

    if args.dry_run:
        print("Would run:")
        print("  ", " ".join(rsync_cmd))
        print("  ", " ".join(rsync_cmd_log))
        return

    use_rsync = True
    try:
        r = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"Synced {remote_logs}/ -> {LOG_DIR}/")
        elif r.returncode == 23:
            print(f"Synced {remote_logs}/ -> {LOG_DIR}/ (some paths may be missing on robot)")
        else:
            print(f"rsync failed (exit {r.returncode}): {r.stderr or r.stdout}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        use_rsync = False

    if use_rsync:
        # Try singular 'log' folder on robot if it exists
        try:
            subprocess.run(rsync_cmd_log, check=False, capture_output=True)
        except FileNotFoundError:
            pass
    else:
        # Fallback: scp (requires listing remote files first)
        print("rsync not found, using scp...", file=sys.stderr)
        list_cmd = ["ssh", f"{args.user}@{args.host}", f"ls {args.remote_dir}/logs/*.csv 2>/dev/null || true"]
        result = subprocess.run(list_cmd, capture_output=True, text=True)
        files = [f.strip() for f in result.stdout.strip().split() if f.strip()]
        if not files:
            print("No CSV files found on robot.", file=sys.stderr)
            return
        for f in files:
            subprocess.run(["scp", f"{args.user}@{args.host}:{f}", f"{LOG_DIR}/"], check=True)
        print(f"Copied {len(files)} file(s) to {LOG_DIR}/")


if __name__ == "__main__":
    main()
