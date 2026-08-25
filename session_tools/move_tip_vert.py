import sys, math, numpy as np, rclpy, json
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
def Rz(a):c,s=math.cos(a),math.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def Rx(a):c,s=math.cos(a),math.sin(a);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2;w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
TARGET=[float(x) for x in (sys.argv[1].split(",") if len(sys.argv)>1 else ["0.35","0.08","0.01"])]
VMAX=float(sys.argv[2]) if len(sys.argv)>2 else 15.0
pc=PlannerClient(); print("[tip] planner:",pc.rpc({'type':'ping'}).get('backend'))
rclpy.init(); node=rclpy.create_node("tipvert"); pub=node.create_publisher(String,"/mycobot/cmd/move",10)
state=RobotState(node); q=state.get_q(); qd=np.rad2deg(q)
# VERTICAL (cup-down) orientation for every yaw; pick the NATURAL (least joint-travel) config.
res=[]
for yaw in range(-180,180,15):
    quat=R2q(Rz(math.radians(yaw))@Rx(math.pi))
    r=pc.plan_pose(q,[*TARGET,*quat])
    if r.get("success"):
        goal=np.rad2deg(np.array(r["trajectory"][-1])); mv=np.abs(goal-qd).max()
        res.append((yaw,mv,r))
if not res: print("ABORT: target not reachable vertical"); rclpy.shutdown(); exit(1)
yaw,mv,r = min(res, key=lambda x: x[1])
print(f"[tip] TARGET={TARGET} vertical | yaw={yaw} max-joint-move={mv:.0f} deg (natural branch)")
ex=execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",VMAX,3,"tipvert",track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0})
json.dump([float(x) for x in state.get_q()],open("/tmp/cur_q.json","w"))
print(f"[tip] settled={ex['ok']} reach_err={ex['reach_err']:.2f} | reached URDF deg:",np.round(np.rad2deg(state.get_q()),2))
rclpy.shutdown()
