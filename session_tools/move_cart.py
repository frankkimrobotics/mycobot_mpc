import numpy as np, math, json, rclpy
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from joint_conventions import rad_to_linuxcnc_deg
TARGET=[0.45,0.08,0.18]
def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Rx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2; w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
# tool-DOWN first (tcp +Z -> world -Z), several yaws; keep-current as last resort
cands=[("keep-current(model-up=real-DOWN)",[0.6957,0.0054,0.0081,0.7182])]
pc=PlannerClient()
print("[cart] planner:",pc.rpc({'type':'ping'}).get('backend'))
rclpy.init(); node=rclpy.create_node("move_cart")
pub=node.create_publisher(String,"/mycobot/cmd/move",10)
state=RobotState(node); q=state.get_q()
if q is None: print("ABORT: no /joint_states"); rclpy.shutdown(); exit(1)
print("[cart] current q deg:",np.round(np.rad2deg(q),1))
print(f"[cart] target tcp position: {TARGET}")
chosen=None
for name,quat in cands:
    r=pc.plan_pose(q,[*TARGET,*quat])
    print(f"   try {name:18s}: success={r.get('success')}  status={r.get('status')}")
    if r.get("success"): chosen=(name,quat,r); break
if not chosen: print("ABORT: no feasible orientation"); rclpy.shutdown(); exit(1)
name,quat,r=chosen
print(f"[cart] -> using orientation '{name}' ({len(r['trajectory'])} wpts)")
track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0}
ex=execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",30,5,"cart",track=track)
qf=state.get_q()
json.dump([float(x) for x in qf], open("/tmp/cur_q.json","w"))
print(f"[cart] settled={ex['ok']} reach_err={ex['reach_err']:.2f} deg")
print("[cart] reached URDF deg:",np.round(np.rad2deg(qf),2))
