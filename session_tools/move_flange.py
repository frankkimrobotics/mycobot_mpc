#!/usr/bin/env python3
"""Move the FLANGE center (link6) to a target. cuRobo plans the tcp, so we map
flange goal -> tcp goal via the fixed link6->tcp transform (RotY(90deg), +0.06 m)."""
import argparse, numpy as np, math, json, rclpy
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
ap=argparse.ArgumentParser()
ap.add_argument("--pos", required=True, help="flange center x,y,z (m)")
ap.add_argument("--quat", default=None, help="flange orientation wxyz; default tries candidates")
ap.add_argument("--max-vel-deg", type=float, default=30.0)
ap.add_argument("--duration", type=float, default=5.0)
a=ap.parse_args()
pos_f=np.array([float(v) for v in a.pos.split(",")])
def Rz(t):c,s=math.cos(t),math.sin(t);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Rx(t):c,s=math.cos(t),math.sin(t);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def RotY(t):c,s=math.cos(t),math.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def q2R(w,x,y,z):return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2; w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
T_l6_tcp_R=RotY(math.pi/2)@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]]); T_l6_tcp_t=np.array([-0.145,0,0])      # link6 -> tcp (fixed)
def flange_to_tcp(pf, Rf):                                       # compose
    return pf + Rf@T_l6_tcp_t, Rf@T_l6_tcp_R
# flange orientation candidates (link6 frame in base)
if a.quat:
    qf=[float(v) for v in a.quat.split(",")]; cands=[("user-quat",q2R(*qf))]
else:
    cands=[(f"flangeZ-down-yaw{y}", Rz(math.radians(y))@Rx(math.pi)) for y in [0,90,180,270]]
    cands+= [(f"flangeZ-up-yaw{y}",   Rz(math.radians(y)))            for y in [0,90,180,270]]
pc=PlannerClient(); print("[flange] planner:",pc.rpc({'type':'ping'}).get('backend'))
rclpy.init(); node=rclpy.create_node("move_flange")
pub=node.create_publisher(String,"/mycobot/cmd/move",10); state=RobotState(node); q=state.get_q()
print("[flange] current q deg:",np.round(np.rad2deg(q),1)); print(f"[flange] FLANGE target pos: {pos_f.tolist()}")
chosen=None
for name,Rf in cands:
    p_tcp,R_tcp=flange_to_tcp(pos_f,Rf); goal=[*p_tcp,*R2q(R_tcp)]
    r=pc.plan_pose(q,goal)
    print(f"   try {name:18s}: success={r.get('success')}")
    if r.get("success"): chosen=(name,r); break
if not chosen: print("ABORT: no feasible flange orientation for that position"); rclpy.shutdown(); exit(1)
name,r=chosen; print(f"[flange] -> orientation '{name}' ({len(r['trajectory'])} wpts)")
ex=execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",a.max_vel_deg,a.duration,"flange",
           track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0})
qf=state.get_q(); json.dump([float(x) for x in qf],open("/tmp/cur_q.json","w"))
print(f"[flange] settled={ex['ok']} reach_err={ex['reach_err']:.2f} deg")
print("[flange] reached URDF deg:",np.round(np.rad2deg(qf),2))
