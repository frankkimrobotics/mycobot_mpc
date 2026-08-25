"""3x5 flange grid x 4 tilted orientations (30deg from vertical along +-x,+-y).
   vel 60, NO lead (tau=0, to stay under saturation), abort-on-fault.
   Records reach_err (error at last moment) + time-to-goal per (cell,orientation)."""
import math, json, time, os, numpy as np, rclpy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
def RotX(t):c,s=math.cos(t),math.sin(t);return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def RotY(t):c,s=math.cos(t),math.sin(t);return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def q2R(w,x,y,z):return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2;w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
fq=np.array(json.load(open("/tmp/flange_quat.json"))); Rf=q2R(*fq)
TILT=math.radians(30.0)
ORIENTS=[("tilt+x",RotY(+TILT)@Rf),("tilt-x",RotY(-TILT)@Rf),("tilt+y",RotX(-TILT)@Rf),("tilt-y",RotX(+TILT)@Rf)]
def tcpgoal(x,y,z,R): return [*(np.array([x,y,z])+R@np.array([-0.145,0,0])),*R2q(R@RotY(math.pi/2)@np.array([[1.,0,0],[0,-1.,0],[0,0,-1.]]))]
base_q=[0.0,-0.349066,1.919862,0.0,-1.570796,0.0]
Z=0.20; VMAX=60.0
GX=[0.25,0.35,0.45]; GY=[-0.32,-0.12,0.08,0.28,0.48]; NX=len(GX); NY=len(GY)
trk={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0}   # tau=0 (no lead) -- not passing lead_time
stamp=time.strftime("%Y%m%d_%H%M%S"); outdir=os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/gridorient_{stamp}"); os.makedirs(outdir,exist_ok=True)
pc=PlannerClient(); print("planner:",pc.rpc({'type':'ping'}).get('backend'),flush=True)
rclpy.init(); node=rclpy.create_node("gridorient"); pub=node.create_publisher(String,"/mycobot/cmd/move",10); state=RobotState(node)
FAULT=False
def chk(ex,lbl):
    global FAULT
    if (not ex.get("ok")) or ex.get("reach_err",999)>12:
        print(f"  !! FAULT/STALL {lbl}: settled={ex.get('ok')} reach_err={ex.get('reach_err'):.1f} -> ABORT",flush=True); FAULT=True
    return FAULT
def reset_base():
    q=state.get_q(); r=pc.plan_joint(q,base_q); return execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",VMAX,0.0,"to-base",track=trk)
res={}   # (ix,iy,oi) -> dict
for iy,y in enumerate(GY):
  for ix,x in enumerate(GX):
    if FAULT: break
    for oi,(oname,R) in enumerate(ORIENTS):
        if FAULT: break
        if chk(reset_base(),"reset"): break
        q=state.get_q(); r=pc.plan_pose(q,tcpgoal(x,y,Z,R))
        if not r.get("success"):
            res[(ix,iy,oi)]={"reach":False}; print(f"  x={x} y={y:+.2f} {oname}: UNREACHABLE",flush=True); continue
        traj=np.array(r["trajectory"]); N=traj.shape[0]
        ex=execute(state,pub,traj,r["dt"],"pid",VMAX,0.0,f"x{x}_y{y}_{oname}",track=trk)
        res[(ix,iy,oi)]={"reach":True,"reach_err":ex["reach_err"],"t":ex["sdt"]*(N-1)}
        print(f"  x={x} y={y:+.2f} {oname}: reach_err {ex['reach_err']:.1f} | t {ex['sdt']*(N-1):.2f}s",flush=True)
        if chk(ex,f"{oname}@[{x},{y}]"): break
if not FAULT: reset_base()
json.dump({f"{k[0]}_{k[1]}_{k[2]}":v for k,v in res.items()},open(os.path.join(outdir,"grid_log.json"),"w"),indent=2)
# plot: 2 rows (reach_err, time) x 4 cols (orientations), each 3x5 heatmap
fig,axs=plt.subplots(2,4,figsize=(20,9))
for oi,(oname,_) in enumerate(ORIENTS):
    RE=np.full((NY,NX),np.nan); TT=np.full((NY,NX),np.nan)
    for (ix,iy,o),v in res.items():
        if o==oi and v.get("reach"): RE[NY-1-iy,ix]=v["reach_err"]; TT[NY-1-iy,ix]=v["t"]
    for row,(M,ttl,cm) in enumerate([(RE,"reach_err (deg)","viridis"),(TT,"time (s)","plasma")]):
        ax=axs[row][oi]; im=ax.imshow(M,cmap=cm,aspect="auto")
        ax.set_xticks(range(NX)); ax.set_xticklabels([str(v) for v in GX]); ax.set_yticks(range(NY)); ax.set_yticklabels([f"{v:+.2f}" for v in GY[::-1]])
        for rr in range(NY):
            for cc in range(NX): ax.text(cc,rr,("n/a" if np.isnan(M[rr,cc]) else (f"{M[rr,cc]:.1f}" if row==0 else f"{M[rr,cc]:.2f}")),ha="center",va="center",color="w",fontsize=8)
        ax.set_title(f"{oname}\n{ttl}",fontsize=10); fig.colorbar(im,ax=ax,shrink=.8)
fig.suptitle(f"3x5 grid x 4 tilted orientations (30deg) | vmax={VMAX} no-lead | reach_err (top) & time (bottom)",fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(outdir,"grid_orient_summary.png"),dpi=110); plt.close(fig)
print(("FAULT" if FAULT else "OK"),"OUTDIR",outdir,flush=True)
rclpy.shutdown()
