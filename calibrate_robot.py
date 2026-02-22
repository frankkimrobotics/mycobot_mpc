#!/usr/bin/env python3
"""
Calibration motions for parameter estimation: move each joint one by one
within ±max_deg of the current pose. Other joints stay fixed.

Uses the same PID controller and connection as control_robot.py. The robot
CSV logs from these moves can be used with identify_invdyn_from_log.py to
estimate M, C, G.

Usage:
  python calibrate_robot.py --host $ROBOT_IP --controller pid
  python calibrate_robot.py --host 10.0.0.27 --duration 3 --step-deg 10 --max-deg 80
  python calibrate_robot.py --host $ROBOT_IP --start -90 -90 0 -90 0 0

  Phase 1: one move to rest pose (duration 3 s, so robot holds there for 3 s); then per-joint sweep ±max_deg (default ±80°); each move 3 s. No log fetch after the initial rest move so the robot does not sit with PIDs off.
  Phase 2: 10 random poses — joint0 ±10°, joints 1–5 ±30° from rest pose.
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

# Phase 2: random poses (without joint0) — joint0 ±10°, joints 1–5 ±30° from rest
RANDOM_POSES = 10
JOINT0_RANGE_DEG = 10.0
OTHER_JOINTS_RANGE_DEG = 30.0


def sample_random_poses(rest_deg: np.ndarray, n_poses: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Sample n_poses random joint angles: joint0 within ±JOINT0_RANGE_DEG, joints 1–5 within ±OTHER_JOINTS_RANGE_DEG of rest_deg."""
    poses = []
    for _ in range(n_poses):
        target = rest_deg.copy().astype(float)
        target[0] = rest_deg[0] + rng.uniform(-JOINT0_RANGE_DEG, JOINT0_RANGE_DEG)
        for j in range(1, MAX_JOINTS):
            target[j] = rest_deg[j] + rng.uniform(-OTHER_JOINTS_RANGE_DEG, OTHER_JOINTS_RANGE_DEG)
        poses.append(target)
    return poses


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
                    help="Move duration per waypoint in seconds (default: 3); used for every calibration move")
    ap.add_argument("--step-deg", type=float, default=10.0,
                    help="Step size in degrees per joint (default: 10)")
    ap.add_argument("--max-deg", type=float, default=80.0,
                    help="Max joint offset in degrees per joint (symmetric ±) for sweep phase (default: 80, stay within limits)")
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
    ap.add_argument("--random-poses", type=int, default=RANDOM_POSES,
                    help=f"Number of random poses after sweep (joint0 ±{JOINT0_RANGE_DEG}°, others ±{OTHER_JOINTS_RANGE_DEG}°) (default: {RANDOM_POSES})")
    ap.add_argument("--no-random-poses", action="store_true",
                    help="Skip the random-pose phase after the per-joint sweep")
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
    n_random = 0 if args.no_random_poses else args.random_poses
    print(f"Calibration plan: {MAX_JOINTS} joints × {n_waypoints} waypoints = {n_moves_total} moves (sweep ±{args.max_deg}°)")
    if n_random:
        print(f"  Then {n_random} random poses: J0 ±{JOINT0_RANGE_DEG}°, J1–J5 ±{OTHER_JOINTS_RANGE_DEG}°")
    print(f"  Offsets (deg): {offsets}")
    print(f"  Start/rest pose: {start_deg.tolist()} deg")
    print(f"  Duration per move: {args.duration}s, controller: {args.controller}")
    if args.dry_run:
        for j in range(MAX_JOINTS):
            print(f"  Joint {j}: ", end="")
            for i, off in enumerate(offsets):
                t = start_deg.copy()
                t[j] = start_deg[j] + off
                print(f"{t[j]:.0f}°", end=("\n    " if (i + 1) % 6 == 0 and i else " "))
            print()
        if n_random:
            rng = np.random.default_rng(42)
            for i, p in enumerate(sample_random_poses(start_deg, n_random, rng)):
                print(f"  Random pose {i+1}: {p.tolist()}")
        return

    conn = cr.RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (OSError, ConnectionRefusedError) as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    logger = cr.MoveLogger()
    current_deg = start_deg.copy()

    # Single move to rest pose (duration 3 s): robot goes there and holds for the move duration.
    # Do not fetch log after this move so the next command is sent immediately; otherwise the
    # robot sits with PIDs off (mpc.enable=False after each move) and can appear powered off.
    print("\n--- Moving to rest pose (hold 3 s) ---")
    status = cr.move_to_joints(
        conn, start_deg,
        duration=args.duration, controller=args.controller,
        pos_tol=args.pos_tol, settle_steps=args.settle_steps,
        logger=logger,
    )
    # Skip fetch after rest move to avoid long gap with robot PIDs off
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

    # Phase 2: random poses — joint0 ±10°, joints 1–5 ±30° from rest (start_deg)
    if not args.no_random_poses and args.random_poses > 0:
        rng = np.random.default_rng()
        random_poses = sample_random_poses(start_deg, args.random_poses, rng)
        print(f"\n--- Random poses ({len(random_poses)} poses, J0 ±{JOINT0_RANGE_DEG}°, J1–J5 ±{OTHER_JOINTS_RANGE_DEG}°) ---")
        for i, target in enumerate(random_poses):
            print(f"  [{i+1}/{len(random_poses)}] {target.tolist()}")
            status = cr.move_to_joints(
                conn, target,
                duration=args.duration, controller=args.controller,
                pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                logger=logger,
            )
            cr._maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)
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
