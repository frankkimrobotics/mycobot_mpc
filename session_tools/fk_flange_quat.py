import sys, math, json, numpy as np, torch
CUROBO_DIR="/home/lisc-frank/Desktop/2026/frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo"
sys.path.insert(0,CUROBO_DIR)
from curobo_planner_server_v2 import Planner
from curobo._src.state.state_joint import JointState
planner=Planner(ground_z=-0.1)
q=json.load(open("/tmp/cur_q.json"))
qt=torch.tensor([q],dtype=torch.float32,device=planner.device)
st=planner.mp.compute_kinematics(JointState.from_position(qt,joint_names=planner.joint_names))
p=st.tool_poses.get_link_pose(planner.tool_frame)
pt=p.position.detach().cpu().numpy()[0]; qtcp=p.quaternion.detach().cpu().numpy()[0]
def q2R(w,x,y,z):return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2;w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
def RotY(t):c,s=math.cos(t),math.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
T_bt=np.eye(4);T_bt[:3,:3]=q2R(*qtcp);T_bt[:3,3]=pt
T_lt=np.eye(4);T_lt[:3,:3]=RotY(math.pi/2)@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]]);T_lt[:3,3]=[-0.145,0,0]
T_bl=T_bt@np.linalg.inv(T_lt)
pf=T_bl[:3,3]; qf=R2q(T_bl[:3,:3])
json.dump(qf,open("/tmp/flange_quat.json","w"))
print(f"current FLANGE pos: [{pf[0]:.4f},{pf[1]:.4f},{pf[2]:.4f}]  quat wxyz: [{qf[0]:.4f},{qf[1]:.4f},{qf[2]:.4f},{qf[3]:.4f}]")
