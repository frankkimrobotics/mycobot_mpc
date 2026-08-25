#!/usr/bin/env python3
"""perturb_plan :: FK current pose -> sample 5cm-perturbed EE pose -> cuRobo IK + B-spline.

Runs in the `curobo2` conda env (cuRobo v0.8.0). Pure planning, NO robot motion.

  1. FK the given start joint config -> current EE (tcp) SE3 pose.
  2. Sample a goal pose perturbed by `--perturb` m in a random direction
     (position-only by default; orientation held fixed).
  3. cuRobo MotionPlanner.plan_pose -> collision-free IK + dynamics-aware
     B-spline trajectory. Retries new random directions until a plan succeeds.
  4. Write the plan (trajectory rad, dt, B-spline control points, poses) to JSON.

Usage (curobo2 env):
  python perturb_plan.py --start-q "[0,-0.354,1.945,-0.001,-1.576,-0.031]" \
      --perturb 0.05 --out /tmp/perturb_plan.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CUROBO_DIR = "/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
sys.path.insert(0, CUROBO_DIR)

from curobo_planner_server_v2 import Planner           # noqa: E402
from curobo._src.state.state_joint import JointState   # noqa: E402


def fk(planner, q):
    qt = torch.tensor([q], dtype=torch.float32, device=planner.device)
    st = planner.mp.compute_kinematics(
        JointState.from_position(qt, joint_names=planner.joint_names))
    pose = st.tool_poses.get_link_pose(planner.tool_frame)
    pos = pose.position.detach().cpu().numpy()[0]        # xyz (m)
    quat = pose.quaternion.detach().cpu().numpy()[0]     # wxyz
    return pos, quat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-q", required=True, help="JSON list of 6 joint angles (URDF rad)")
    ap.add_argument("--perturb", type=float, default=0.05, help="EE position perturbation (m)")
    ap.add_argument("--full-se3", action="store_true",
                    help="also perturb orientation (default: position only)")
    ap.add_argument("--rot-deg", type=float, default=15.0, help="orientation perturb if --full-se3")
    ap.add_argument("--max-tries", type=int, default=12)
    ap.add_argument("--max-attempts", type=int, default=8, help="cuRobo plan attempts per try")
    ap.add_argument("--ground-z", type=float, default=-0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/tmp/perturb_plan.json")
    args = ap.parse_args()

    q = list(map(float, json.loads(args.start_q)))
    rng = np.random.default_rng(args.seed)

    print("[plan] building cuRobo MotionPlanner (curobo2, warming up) ...", flush=True)
    planner = Planner(ground_z=args.ground_z)

    pos0, quat0 = fk(planner, q)
    print(f"[plan] current EE (tcp) pos(m)=[{pos0[0]:+.4f},{pos0[1]:+.4f},{pos0[2]:+.4f}] "
          f"quat(wxyz)=[{quat0[0]:+.4f},{quat0[1]:+.4f},{quat0[2]:+.4f},{quat0[3]:+.4f}]")

    res = None
    goal_pos = goal_quat = perturb = None
    for t in range(args.max_tries):
        d = rng.normal(size=3)
        d = d / (np.linalg.norm(d) + 1e-9) * args.perturb
        gpos = pos0 + d
        gquat = quat0.copy()
        if args.full_se3:  # small random rotation about a random axis
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9
            ang = np.deg2rad(args.rot_deg)
            dq = np.array([np.cos(ang/2), *(np.sin(ang/2)*ax)])  # wxyz
            w0, x0, y0, z0 = quat0; w1, x1, y1, z1 = dq
            gquat = np.array([
                w0*w1 - x0*x1 - y0*y1 - z0*z1,
                w0*x1 + x0*w1 + y0*z1 - z0*y1,
                w0*y1 - x0*z1 + y0*w1 + z0*x1,
                w0*z1 + x0*y1 - y0*x1 + z0*w1])
        goal_pose = [float(v) for v in (*gpos, *gquat)]
        r = planner.plan_pose(q, goal_pose, max_attempts=args.max_attempts)
        if r.get("success"):
            res, goal_pos, goal_quat, perturb = r, gpos, gquat, d
            print(f"[plan] try {t+1}: SUCCESS  |perturb|={np.linalg.norm(d)*100:.1f} cm")
            break
        print(f"[plan] try {t+1}: plan failed ({r.get('status')}), resampling direction")

    if res is None:
        print("[plan] FAILED: no collision-free plan found"); sys.exit(2)

    traj = np.array(res["trajectory"])           # [N, 6] rad
    cps = np.array(res["control_points"])         # [n_ctrl, 6] rad (B-spline)
    deltas_deg = np.rad2deg(traj[-1] - traj[0])

    out = {
        "joint_names": planner.joint_names,
        "start_q": q,
        "ee_current": {"pos": pos0.tolist(), "quat_wxyz": quat0.tolist()},
        "ee_goal": {"pos": goal_pos.tolist(), "quat_wxyz": goal_quat.tolist()},
        "perturb_vec_m": perturb.tolist(),
        "perturb_norm_m": float(np.linalg.norm(perturb)),
        "success": True,
        "dt": res["dt"],
        "motion_time": res["motion_time"],
        "n_waypoints": int(traj.shape[0]),
        "n_control_points": int(cps.shape[0]),
        "trajectory": res["trajectory"],
        "control_points": res["control_points"],
    }
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"[plan] goal EE pos(m)=[{goal_pos[0]:+.4f},{goal_pos[1]:+.4f},{goal_pos[2]:+.4f}]")
    print(f"[plan] trajectory: {out['n_waypoints']} waypts @ dt={out['dt']:.3f}s "
          f"| motion_time={out['motion_time']:.2f}s | B-spline ctrl pts={out['n_control_points']}")
    print("[plan] per-joint start->end (deg): "
          + ", ".join(f"j{i+1}:{deltas_deg[i]:+.1f}" for i in range(6)))
    print(f"[plan] saved -> {args.out}")


if __name__ == "__main__":
    main()
