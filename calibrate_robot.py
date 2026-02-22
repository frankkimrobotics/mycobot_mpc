#!/usr/bin/env python3
"""
Calibration motions for parameter estimation: move each joint one by one
within ±max_deg of the current pose. Other joints stay fixed.

Uses the same PID controller and connection as control_robot.py. The robot
CSV logs from these moves can be used with identify_invdyn_from_log.py to
estimate M, C, G.

Usage:
  python calibrate_robot.py --host $ROBOT_IP --controller pid
  python calibrate_robot.py --host 10.0.0.27 --duration 3 --step-deg 10 --max-deg 30
  python calibrate_robot.py --host $ROBOT_IP --start -90 -90 0 -90 0 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Reuse connection and move logic from control_robot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import control_robot as cr

MAX_JOINTS = cr.MAX_JOINTS
HOME_LINUXCNC_DEG = np.array(cr.HOME_LINUXCNC_DEG)


def build_sweep_offsets(step_deg: float, max_deg: float) -> list[float]:
    """Offsets in degrees: 0 -> +step -> ... -> +max -> ... -> 0 -> -step -> ... -> -max -> ... -> 0."""
    if step_deg <= 0 or max_deg <= 0:
        return [0.0]
    offsets = [0.0]
    # Positive sweep
    x = step_deg
    while x <= max_deg:
        offsets.append(x)
        x += step_deg
    x = max_deg - step_deg
    while x >= 0:
        offsets.append(x)
        x -= step_deg
    # Negative sweep
    x = -step_deg
    while x >= -max_deg:
        offsets.append(x)
        x -= step_deg
    x = -max_deg + step_deg
    while x <= 0:
        offsets.append(x)
        x += step_deg
    return offsets


def main():
    ap = argparse.ArgumentParser(
        description="Calibration: move each joint one-by-one within ±max_deg for parameter ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--host", default=os.environ.get("ROBOT_IP", "").strip(),
                    help="Robot IP (default: ROBOT_IP)")
    ap.add_argument("--cmd-port", type=int, default=9998, help="Command port (default: 9998)")
    ap.add_argument("--controller", choices=["pid", "mpc", "invdyn"], default="pid",
                    help="Controller (default: pid)")
    ap.add_argument("--duration", type=float, default=3.0,
                    help="Move duration per waypoint in seconds (default: 3)")
    ap.add_argument("--step-deg", type=float, default=10.0,
                    help="Step size in degrees per joint (default: 10)")
    ap.add_argument("--max-deg", type=float, default=30.0,
                    help="Max joint offset in degrees, symmetric ± (default: 30)")
    ap.add_argument("--pos-tol", type=float, default=0.5, help="Position tolerance for early stop (deg)")
    ap.add_argument("--settle-steps", type=int, default=10, help="Settle steps for early stop")
    ap.add_argument("--no-fetch-logs", action="store_true",
                    help="Do not fetch robot CSV logs after each move")
    start_group = ap.add_mutually_exclusive_group()
    start_group.add_argument("--start", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                             help="Start pose (6 joint angles in deg). First move goes here.")
    start_group.add_argument("--home", action="store_true",
                             help="Start from home pose (-90 -90 0 -90 0 0) (default)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print planned moves only, do not connect or move")
    args = ap.parse_args()

    if not args.host:
        print("Error: --host or ROBOT_IP required.", file=sys.stderr)
        sys.exit(1)

    # Start pose
    if args.start is not None:
        start_deg = np.array(args.start, dtype=float)
    else:
        start_deg = HOME_LINUXCNC_DEG.copy()

    # Sweep offsets (same for every joint)
    offsets = build_sweep_offsets(args.step_deg, args.max_deg)
    n_waypoints = len(offsets)
    n_moves_total = MAX_JOINTS * n_waypoints
    print(f"Calibration plan: {MAX_JOINTS} joints × {n_waypoints} waypoints = {n_moves_total} moves")
    print(f"  Offsets (deg): {offsets}")
    print(f"  Start pose: {start_deg.tolist()} deg")
    print(f"  Duration per move: {args.duration}s, controller: {args.controller}")
    if args.dry_run:
        for j in range(MAX_JOINTS):
            print(f"  Joint {j}: ", end="")
            for i, off in enumerate(offsets):
                t = start_deg.copy()
                t[j] = start_deg[j] + off
                print(f"{t[j]:.0f}°", end=("\n    " if (i + 1) % 6 == 0 and i else " "))
            print()
        return

    conn = cr.RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (OSError, ConnectionRefusedError) as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    logger = cr.MoveLogger()
    current_deg = start_deg.copy()

    # First move: go to start pose
    print("\n--- Moving to start pose ---")
    status = cr.move_to_joints(
        conn, current_deg,
        duration=args.duration, controller=args.controller,
        pos_tol=args.pos_tol, settle_steps=args.settle_steps,
        logger=logger,
    )
    cr._maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)
    if status.get("current_deg"):
        current_deg = np.array(status["current_deg"])

    # Per-joint sweeps: each joint sweeps, then move back to start pose before next joint
    for j in range(MAX_JOINTS):
        print(f"\n--- Joint {j} sweep ({n_waypoints} waypoints) ---")
        for i, offset_deg in enumerate(offsets):
            target = current_deg.copy()
            target[j] = start_deg[j] + offset_deg
            print(f"  [{i+1}/{n_waypoints}] J{j} = {target[j]:.1f}° (offset {offset_deg:+.1f}°)")
            status = cr.move_to_joints(
                conn, target,
                duration=args.duration, controller=args.controller,
                pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                logger=logger,
            )
            cr._maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)
            if status.get("current_deg"):
                current_deg = np.array(status["current_deg"])

        # Move back to start pose so the next joint starts from the same initial pose
        print(f"  Return to start pose before joint {j+1}...")
        status = cr.move_to_joints(
            conn, start_deg,
            duration=args.duration, controller=args.controller,
            pos_tol=args.pos_tol, settle_steps=args.settle_steps,
            logger=logger,
        )
        cr._maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)
        current_deg = start_deg.copy()
        if status.get("current_deg"):
            current_deg = np.array(status["current_deg"])

    conn.close()

    # Save desktop-side calibration log (control_calibrate_<stamp>.csv in logs/)
    logger.save(tag="calibrate")
    print(f"\nCalibration done. Desktop log in {cr.LOG_DIR}/ (control_calibrate_*.csv)")
    print("Fetch robot CSVs (if not auto-fetched) with: python fetch_robot_logs.py")
    print("Then run: python identify_invdyn_from_log.py logs/mpc_*.csv logs/invdyn_*.csv")


if __name__ == "__main__":
    main()
