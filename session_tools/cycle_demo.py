import argparse, time, math, json, os, numpy as np, rclpy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from joint_conventions import JOINT_NAMES
ap=argparse.ArgumentParser()
ap.add_argument("--cycles",type=int,default=3)
ap.add_argument("--max-vel-deg",type=float,default=30.0)
ap.add_argument("--duration",type=float,default=1.0)
ap.add_argument("--vff-scale",type=float,default=1.0)
a=ap.parse_args()
def RotY(t):c,s=math.cos(t),math.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def q2R(w,x,y,z):return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2;w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
fq=json.load(open("/tmp/flange_quat.json")); Rf=q2R(*fq)
p_tcp=np.array([0.45,0.08,0.15])+Rf@np.array([-0.145,0,0]); tcp_goal=[*p_tcp,*R2q(Rf@RotY(math.pi/2)@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]]))]
base_q=[0.0,-0.349066,1.919862,0.0,-1.570796,0.0]
track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":a.vff_scale}
pc=PlannerClient(); print("[cycle] planner:",pc.rpc({'type':'ping'}).get('backend'))
rclpy.init(); node=rclpy.create_node("cycle"); pub=node.create_publisher(String,"/mycobot/cmd/move",10)
state=RobotState(node); state.get_q()
legs=[]
def leg(plan,label):
    ex=execute(state,pub,np.array(plan["trajectory"]),plan["dt"],"pid",a.max_vel_deg,a.duration,label,track=track)
    legs.append({"label":label,"cmd":np.array(ex["cmd_traj"]),"sdt":ex["sdt"],
                 "at":np.array(ex["trace_t"]),"aq":np.array(ex["trace_q"]),"err":ex["reach_err"]})
    return ex
print(f"[cycle] max_vel={a.max_vel_deg} deg/s duration={a.duration}s vff={a.vff_scale}")
for c in range(a.cycles):
    q=state.get_q(); r1=pc.plan_pose(q,tcp_goal)
    if not r1.get("success"): print(f"  cycle{c}: plan->target FAILED"); break
    e1=leg(r1,f"c{c}-fwd"); q=state.get_q(); r2=pc.plan_joint(q,base_q); e2=leg(r2,f"c{c}-back")
    print(f"  CYCLE {c}: fwd err {e1['reach_err']:.1f}deg | back err {e2['reach_err']:.1f}deg")
rclpy.shutdown()
# ---- concatenated cmd-vs-actual plot ----
stamp=time.strftime("%Y%m%d_%H%M%S")
outdir=os.path.expanduser("~/Desktop/2026/mycobot_mpc/captures"); os.makedirs(outdir,exist_ok=True)
fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
off=0.0; bounds=[]
for L in legs:
    ct=np.arange(L["cmd"].shape[0])*L["sdt"]+off
    at=L["at"]+off
    for j in range(6):
        ax=axes[j//3][j%3]
        ax.plot(ct,np.rad2deg(L["cmd"][:,j]),"--",color="C1",lw=1.6)
        if L["aq"].size: ax.plot(at,np.rad2deg(L["aq"][:,j]),"-",color="C0",lw=1.3)
    end=max(ct[-1], at[-1] if at.size else ct[-1]); bounds.append((off,end,L["label"])); off=end+0.25
for j in range(6):
    ax=axes[j//3][j%3]
    for (s,e,lab) in bounds: ax.axvline(e,color="0.85",lw=0.8)
    ax.set_title(JOINT_NAMES[j]); ax.set_ylabel("angle (deg)"); ax.grid(alpha=0.3)
    if j>=3: ax.set_xlabel("time (s)")
axes[0][0].plot([],[],"--",color="C1",label="command"); axes[0][0].plot([],[],"-",color="C0",label="actual"); axes[0][0].legend(loc="best")
for (s,e,lab) in bounds: axes[0][1].text((s+e)/2,axes[0][1].get_ylim()[1],lab,ha="center",va="bottom",fontsize=7,rotation=0)
fig.suptitle(f"Cycle base<->[0.45,0.08,0.15] x{len(legs)//2}  |  max_vel={a.max_vel_deg} deg/s, dur={a.duration}s  (cmd vs actual)")
fig.tight_layout()
path=os.path.join(outdir,f"cycle_cmd_vs_actual_{stamp}.png"); fig.savefig(path,dpi=120); plt.close(fig)
print("saved",path)
