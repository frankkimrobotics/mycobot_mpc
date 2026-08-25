#!/usr/bin/env python3
"""FK the FLANGE link (link6) pose in base_link for every captured record ->
flange_poses.json. Lets the hand-eye solver return T_flange_cam405 directly,
without trusting the repo's two conflicting link6->tcp definitions.

Prints the link names cuRobo exposes; pass --link to pick the flange link if the
default guess is wrong.

  ~/miniconda3/envs/curobo2/bin/python session_tools/calib_fk_flange.py \
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
ap.add_argument("--link", default=None, help="flange link name (default: auto-pick link6 / last non-tcp link)")
ap.add_argument("--ground-z", type=float, default=-0.1)
a = ap.parse_args()

planner = Planner(ground_z=a.ground_z)
sess = json.load(open(a.session))

# discover available link names from one FK
q0 = next(r["q_rad"] for r in sess["records"] if r.get("q_rad"))
st0 = planner.mp.compute_kinematics(
    JointState.from_position(torch.tensor([q0], dtype=torch.float32, device=planner.device),
                             joint_names=planner.joint_names))
links = list(getattr(st0, "link_poses", {}) or {})
print("available link_poses:", links, " tool_frame:", planner.tool_frame)

link = a.link
if link is None:
    for cand in ["link6", "joint6_output", "flange", "link_6"]:
        if cand in links:
            link = cand; break
    if link is None:  # fall back: last link that isn't the tool frame
        non_tool = [l for l in links if l != planner.tool_frame]
        link = non_tool[-1] if non_tool else planner.tool_frame
print(f"using flange link: {link}")

out = {}
for rec in sess["records"]:
    q = rec.get("q_rad")
    if q is None:
        continue
    st = planner.mp.compute_kinematics(
        JointState.from_position(torch.tensor([q], dtype=torch.float32, device=planner.device),
                                 joint_names=planner.joint_names))
    p = st.link_poses[link]
    pos = p.position.detach().cpu().numpy()[0]
    quat = p.quaternion.detach().cpu().numpy()[0]
    out[rec["name"]] = {"pos": pos.tolist(), "quat_wxyz": quat.tolist()}

outpath = os.path.join(os.path.dirname(a.session), "flange_poses.json")
json.dump(out, open(outpath, "w"), indent=2)
print(f"wrote {len(out)} flange ({link}) poses -> {outpath}")
