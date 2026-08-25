import argparse, json, numpy as np, rclpy
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from joint_conventions import rad_to_linuxcnc_deg
ap=argparse.ArgumentParser()
ap.add_argument("--target", required=True)
ap.add_argument("--max-vel-deg", type=float, default=30.0)
ap.add_argument("--duration", type=float, default=6.0)
a=ap.parse_args()
goal=np.array(json.loads(a.target))
pc=PlannerClient()
print("[move] planner:", pc.rpc({'type':'ping'}).get('backend'))
rclpy.init(); node=rclpy.create_node("move_to_q")
pub=node.create_publisher(String,"/mycobot/cmd/move",10)
state=RobotState(node); q=state.get_q()
if q is None: print("ABORT: no /joint_states"); rclpy.shutdown(); exit(1)
print("[move] current URDF deg:", np.round(np.rad2deg(q),1))
print("[move] goal    URDF deg:", np.round(np.rad2deg(goal),1))
print("[move] max |delta|:", round(float(np.abs(np.rad2deg(q-goal)).max()),1), "deg")
r=pc.plan_joint(q,goal)
if not r.get("success"): print("ABORT: plan failed", r.get("status")); rclpy.shutdown(); exit(1)
track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0}
ex=execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",a.max_vel_deg,a.duration,"move",track=track)
qf=state.get_q()
print(f"[move] settled={ex['ok']} reach_err={ex['reach_err']:.2f} deg")
print("[move] reached URDF deg:    ", np.round(np.rad2deg(qf),2))
print("[move] reached LinuxCNC deg:", np.round(rad_to_linuxcnc_deg(qf),2))
rclpy.shutdown()
