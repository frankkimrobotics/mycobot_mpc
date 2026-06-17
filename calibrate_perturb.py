#!/usr/bin/env python3
"""
Local perturbation calibration: ±step° per joint about a fixed base pose.

Unlike calibrate_robot.py (which sweeps ±max_deg about the upright rest pose),
this moves to a given BASE pose first, then perturbs each joint by +step and
-step relative to that base, returning to base between perturbations. Other
joints are held at the base angle throughout.

The default base is the pose the robot was measured at on power-up
(LinuxCNC degrees):  [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]
so the initial "move to base" is small (no large home swing).

Run ON the robot (Raspberry Pi) against the local robot_hal command server:

    python calibrate_perturb.py --host 127.0.0.1
    python calibrate_perturb.py --host 127.0.0.1 --step-deg 10 --duration 6
    python calibrate_perturb.py --host 127.0.0.1 --base 0 -110.3 111.4 -90.1 -90.3 0
    python calibrate_perturb.py --host 127.0.0.1 --dry-run        # print the plan only

Sequence (per joint j, others held at base):
    base -> base[j]+step -> base -> base[j]-step -> base

Logs (robot CSVs) feed identify_invdyn_from_log.py just like calibrate_robot.py.
"""

import argparse
import os
import sys

import numpy as np

# Reuse connection and move logic from control_robot (same as calibrate_robot.py)
import control_robot as cr

MAX_JOINTS = cr.MAX_JOINTS

# Pose the robot was measured at on power-up (LinuxCNC degrees).
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]


def build_plan(base_deg, step_deg):
    """Return a list of (label, target_deg) for the full perturbation sequence.

    For each joint: +step about base, back to base, -step about base, back to base.
    """
    plan = [("base", base_deg.copy())]
    for j in range(MAX_JOINTS):
        for sign in (+1.0, -1.0):
            t = base_deg.copy()
            t[j] = base_deg[j] + sign * step_deg
            plan.append((f"J{j} {sign*step_deg:+.0f}", t))
            plan.append((f"J{j} return", base_deg.copy()))
    return plan


def main():
    ap = argparse.ArgumentParser(
        description="±step° per-joint perturbation about a base pose (move to base first).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host", default=os.environ.get("ROBOT_IP", "").strip(),
                    help="Robot command-server IP (use 127.0.0.1 when running on the Pi)")
    ap.add_argument("--cmd-port", type=int, default=9998, help="Command port (default: 9998)")
    ap.add_argument("--base", nargs=6, type=float, default=None,
                    help=f"Base pose in LinuxCNC deg (default: {DEFAULT_BASE_DEG})")
    ap.add_argument("--step-deg", type=float, default=10.0,
                    help="Perturbation magnitude per joint in degrees (default: 10)")
    ap.add_argument("--duration", type=float, default=6.0,
                    help="Max move duration per waypoint (s); long enough to reach the pose (default: 6)")
    ap.add_argument("--base-duration", type=float, default=None,
                    help="Duration for the initial move to base (default: same as --duration; raise if the "
                         "robot starts far from base)")
    ap.add_argument("--controller", choices=["pid", "invdyn", "pd_velff"], default="pid",
                    help="Controller for the moves (default: pid)")
    ap.add_argument("--pos-tol", type=float, default=0.5, help="Early-stop position tolerance (deg)")
    ap.add_argument("--settle-steps", type=int, default=10, help="Settle steps for early stop")
    ap.add_argument("--smooth", action="store_true",
                    help="Move along a smooth, time-scaled B-spline/quintic trajectory between "
                         "poses (uses the robot trajectory mode). Recommended.")
    ap.add_argument("--traj-kind", choices=["quintic", "bspline"], default="quintic",
                    help="Trajectory type when --smooth (default: quintic, ideal for 2 endpoints)")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="Trajectory sample rate when --smooth (default: 50)")
    ap.add_argument("--vel-frac", type=float, default=0.6, help="Fraction of joint vel limit when --smooth (default: 0.6)")
    ap.add_argument("--acc-frac", type=float, default=0.6, help="Fraction of joint accel limit when --smooth (default: 0.6)")
    ap.add_argument("--no-fetch-logs", action="store_true", help="Do not auto-fetch robot logs after each move")
    ap.add_argument("--dry-run", action="store_true", help="Print the planned moves only; do not connect or move")
    args = ap.parse_args()

    if not args.host:
        print("Error: --host or ROBOT_IP required (use --host 127.0.0.1 on the Pi).", file=sys.stderr)
        sys.exit(1)

    base_deg = np.array(args.base if args.base is not None else DEFAULT_BASE_DEG, dtype=float)
    plan = build_plan(base_deg, args.step_deg)
    n_perturb = MAX_JOINTS * 2

    print(f"Perturbation calibration: ±{args.step_deg:.0f}° on each of {MAX_JOINTS} joints "
          f"= {n_perturb} perturbations ({len(plan)} moves incl. returns)")
    print(f"  Base pose (LinuxCNC deg): {base_deg.tolist()}")
    print(f"  Duration per move: {args.duration}s, controller: {args.controller}")

    if args.dry_run:
        for label, t in plan:
            print(f"  {label:<12} -> {np.round(t, 1).tolist()}")
        return

    conn = cr.RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (OSError, ConnectionRefusedError) as e:
        print(f"Failed to connect: {e}", file=sys.stderr)
        sys.exit(1)

    logger = cr.MoveLogger()
    current_deg = base_deg.copy()

    def do_move(target, duration, smooth):
        """Execute one move (smooth trajectory or point-to-point) and track current pose."""
        nonlocal current_deg
        if smooth:
            status = cr.move_smooth(
                conn, current_deg, np.asarray(target, float), kind=args.traj_kind,
                controller=args.controller, rate_hz=args.rate_hz,
                vel_frac=args.vel_frac, acc_frac=args.acc_frac,
                pos_tol=args.pos_tol, settle_steps=args.settle_steps, logger=logger,
            )
        else:
            status = cr.move_to_joints(
                conn, np.asarray(target, float), duration=duration, controller=args.controller,
                pos_tol=args.pos_tol, settle_steps=args.settle_steps, logger=logger,
            )
        cr._maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)
        cur = status.get("current_deg")
        current_deg = np.array(cur, float) if cur else np.asarray(target, float)
        return status

    # 1) Move to base pose first. Point-to-point: the start pose is unknown to the
    #    desktop, and base ≈ current so this opening move is small either way.
    base_dur = args.base_duration if args.base_duration is not None else args.duration
    print("\n--- Moving to base pose ---")
    do_move(base_deg, base_dur, smooth=False)

    # 2) Perturbation sequence (skip the leading "base" entry already done above).
    mode = f"smooth {args.traj_kind}" if args.smooth else "point-to-point"
    print(f"\n--- Perturbation sequence ({mode}) ---")
    for i, (label, target) in enumerate(plan[1:], start=1):
        print(f"  [{i}/{len(plan)-1}] {label:<12} J-targets = {np.round(target, 1).tolist()}")
        do_move(target, args.duration, smooth=args.smooth)

    conn.close()
    logger.save(tag="perturb")
    print(f"\nPerturbation calibration done. Desktop log in {cr.LOG_DIR}/ (control_perturb_*.csv)")
    print("Fetch robot CSVs (if not auto-fetched): python fetch_robot_logs.py")
    print("Then identify: python identify_invdyn_from_log.py logs/*.csv -o logs/invdyn_params.npz")


if __name__ == "__main__":
    main()
