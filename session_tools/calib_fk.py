#!/usr/bin/env python3
"""Compute the cuRobo tool-frame (TCP) pose in base_link for every captured
record -> tcp_poses.json (the gripper2base input for hand-eye).

Run in the curobo2 env (no cv2 needed here):
  ~/miniconda3/envs/curobo2/bin/python session_tools/calib_fk.py \
        --session captures/calib_session_XXXX/session.json
"""
import sys, os, json, argparse
import numpy as np, torch

CUROBO_DIR = "/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
sys.path.insert(0, CUROBO_DIR)
from curobo_planner_server_v2 import Planner
from curobo._src.state.state_joint import JointState

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--ground-z", type=float, default=-0.1)
a = ap.parse_args()

planner = Planner(ground_z=a.ground_z)
sess = json.load(open(a.session))
out = {}
for rec in sess["records"]:
    q = rec.get("q_rad")
    if q is None:
        continue
    qt = torch.tensor([q], dtype=torch.float32, device=planner.device)
    st = planner.mp.compute_kinematics(JointState.from_position(qt, joint_names=planner.joint_names))
    p = st.tool_poses.get_link_pose(planner.tool_frame)
    pos = p.position.detach().cpu().numpy()[0]
    quat = p.quaternion.detach().cpu().numpy()[0]          # wxyz
    out[rec["name"]] = {"pos": pos.tolist(), "quat_wxyz": quat.tolist()}

outpath = os.path.join(os.path.dirname(a.session), "tcp_poses.json")
json.dump(out, open(outpath, "w"), indent=2)
print(f"wrote {len(out)} TCP poses -> {outpath}")
