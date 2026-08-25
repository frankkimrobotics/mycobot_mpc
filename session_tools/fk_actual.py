import sys, math, numpy as np, torch
CUROBO_DIR="/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
sys.path.insert(0, CUROBO_DIR)
from curobo_planner_server_v2 import Planner
from curobo._src.state.state_joint import JointState
planner = Planner(ground_z=-0.1)
def fk(q):
    qt=torch.tensor([q],dtype=torch.float32,device=planner.device)
    st=planner.mp.compute_kinematics(JointState.from_position(qt,joint_names=planner.joint_names))
    p=st.tool_poses.get_link_pose(planner.tool_frame)
    return p.position.detach().cpu().numpy()[0], p.quaternion.detach().cpu().numpy()[0]
def rpy(w,x,y,z):
    r=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)); p=math.asin(max(-1,min(1,2*(w*y-z*x)))); yw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    return [math.degrees(a) for a in (r,p,yw)]
actual=[-1.54546,-0.00774,0.02935,0.0028,-0.0219,-0.0006]   # real robot reached home (URDF rad)
ideal=[-1.5708,0,0,0,0,0]                                    # exact HOME target
for name,q in [("REAL robot home (measured)",actual),("Exact HOME target",ideal)]:
    pos,quat=fk(q); e=rpy(*quat)
    print(f"\n{name}")
    print(f"  EE pos (m)  : x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")
    print(f"  quat wxyz   : [{quat[0]:+.4f}, {quat[1]:+.4f}, {quat[2]:+.4f}, {quat[3]:+.4f}]")
    print(f"  rpy (deg)   : roll={e[0]:+.2f} pitch={e[1]:+.2f} yaw={e[2]:+.2f}")
