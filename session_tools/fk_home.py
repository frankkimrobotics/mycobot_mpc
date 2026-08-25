import sys, os, math
import numpy as np
import torch
CUROBO_DIR = "/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
sys.path.insert(0, CUROBO_DIR)
from curobo_planner_server_v2 import Planner            # noqa
from curobo._src.state.state_joint import JointState    # noqa

planner = Planner(ground_z=-0.1)

def fk(q):
    qt = torch.tensor([q], dtype=torch.float32, device=planner.device)
    st = planner.mp.compute_kinematics(
        JointState.from_position(qt, joint_names=planner.joint_names))
    pose = st.tool_poses.get_link_pose(planner.tool_frame)
    pos = pose.position.detach().cpu().numpy()[0]
    quat = pose.quaternion.detach().cpu().numpy()[0]     # wxyz
    return pos, quat

def quat_to_rpy(w, x, y, z):
    # ZYX intrinsic (roll x, pitch y, yaw z), degrees
    r = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    p = math.asin(max(-1, min(1, 2*(w*y - z*x))))
    yw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return [math.degrees(a) for a in (r, p, yw)]

def lcnc_to_rad(deg):
    SIGN = [1, 1, 1, 1, 1, 1]; OFF = [0, 90, 0, 90, 0, 0]
    return [SIGN[i]*math.radians(deg[i] + OFF[i]) for i in range(6)]

configs = {
    "cuRobo default / retract_config  (q = 0,0,0,0,0,0)": [0.0]*6,
    "Robot HOME  (HOME_LINUXCNC_DEG = -90,-90,0,-90,0,0)": lcnc_to_rad([-90, -90, 0, -90, 0, 0]),
}
print(f"tool_frame = {planner.tool_frame} ; base = base_link\n")
for name, q in configs.items():
    pos, quat = fk(q)
    rpy = quat_to_rpy(*quat)
    print(name)
    print("  q_urdf_rad   : [" + ", ".join(f"{v:+.5f}" for v in q) + "]")
    print(f"  EE position  : x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}  (m)")
    print(f"  EE quat wxyz : [{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]")
    print(f"  EE rpy (deg) : roll={rpy[0]:+.2f}  pitch={rpy[1]:+.2f}  yaw={rpy[2]:+.2f}\n")
