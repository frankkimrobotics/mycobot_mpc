#!/usr/bin/env python3
"""multiview_fuse :: 8-view segmented point-cloud fusion in the robot base frame.

Moves the arm so the D405 (eye-in-hand) views an object at OBJ from 8 azimuths on a
hemisphere (slant range R_SLANT, optical axis aimed at OBJ). At each pose it:
  1. settles, reads the ACTUAL joint config, FK -> T_base_tcp (planner 'fk' RPC),
  2. extrinsic  T_base_cam = T_base_tcp @ T_tcp_cam   (hand-eye result),
  3. captures aligned color+depth, segments the object (SAM 3),
  4. deprojects the masked depth -> camera-frame points, transforms to base,
  5. crops to a box around OBJ, accumulates, and writes the CUMULATIVE cloud
     after each view (cloud_cum_1.ply .. cloud_cum_8.ply) + per-view extrinsics.

Run (infra up: ROS bridge, planner :9997 with fk, SAM3 :5599):
  source ~/Desktop/2026/ros2node/config/ros2node.env
  PYTHONPATH=~/librealsense/build/release python3 mycobot_mpc/multiview_fuse.py \
      --obj 0.35,0.08,0.0 --prompt object
"""
import sys, os, math, json, time, argparse
sys.path.insert(0, os.path.expanduser("~/Desktop/2026/mycobot_mpc"))
sys.path.insert(0, os.path.expanduser("~/Desktop/2026/ros2node/perception"))
import numpy as np, rclpy
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from object_pointclouds import capture_aligned, deproject_mask, write_ply
from capture_and_plot import segment

HANDEYE = os.path.expanduser(
    "~/Desktop/2026/mycobot_mpc/captures/calib_20260622_215855/handeye_result.json")

def Rz(a): c,s=math.cos(a),math.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1.]])
def R2q(R):
    w=math.sqrt(max(0,1+R[0,0]+R[1,1]+R[2,2]))/2; w=max(w,1e-9)
    return [w,(R[2,1]-R[1,2])/(4*w),(R[0,2]-R[2,0])/(4*w),(R[1,0]-R[0,1])/(4*w)]
def q2R(w,x,y,z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def pick_res(serial):
    """Largest color+depth@30 res the current USB link supports (848 needs USB3)."""
    import pyrealsense2 as rs
    d=[x for x in rs.context().query_devices() if x.get_info(rs.camera_info.serial_number)==serial][0]
    avail=set()
    for s in d.query_sensors():
        for p in s.get_stream_profiles():
            vp=p.as_video_stream_profile()
            if vp and vp.fps()==30: avail.add((p.stream_type(),vp.width(),vp.height()))
    for w,h in [(848,480),(640,480),(424,240)]:
        if (rs.stream.color,w,h) in avail and (rs.stream.depth,w,h) in avail:
            return w,h
    return 640,480

def lookat_R(p_cam, target, up_ref=(0,0,1.)):
    """Camera (optical) frame looking from p_cam at target. +Z fwd, +Y down, +X right."""
    z=np.array(target)-np.array(p_cam); z/=np.linalg.norm(z)
    up=np.array(up_ref,float)
    if abs(np.dot(up,z))>0.95: up=np.array([0,1.,0])
    x=np.cross(up,z); x/=np.linalg.norm(x)
    y=np.cross(z,x)
    return np.column_stack([x,y,z])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--obj",default="0.35,0.08,0.0")
    ap.add_argument("--prompt",default="object")
    ap.add_argument("--n",type=int,default=8)
    ap.add_argument("--slant",type=float,default=0.25)
    ap.add_argument("--elev-deg",type=float,default=50.0)
    ap.add_argument("--serial",default="218622271300")
    ap.add_argument("--vmax",type=float,default=25.0)
    ap.add_argument("--crop",type=float,default=0.15)   # half-box around OBJ (x,y)
    ap.add_argument("--zmax",type=float,default=0.35)
    args=ap.parse_args()
    OBJ=np.array([float(v) for v in args.obj.split(",")])

    he=json.load(open(HANDEYE)); T_tcp_cam=np.array(he["T_tcp_cam"])
    T_cam_tcp=np.linalg.inv(T_tcp_cam)
    stamp=time.strftime("%Y%m%d_%H%M%S")
    out=os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/multiview_{stamp}")
    os.makedirs(out+"/img",exist_ok=True)

    CW,CH=pick_res(args.serial); print(f"camera res: {CW}x{CH}",flush=True)
    pc=PlannerClient(); print("planner:",pc.rpc({'type':'ping'}).get('backend'),flush=True)
    def fk(q):
        r=pc.rpc({"type":"fk","q":list(map(float,q))})
        T=np.eye(4); T[:3,:3]=q2R(*r["quat"][0]); T[:3,3]=r["pos"][0]; return T
    rclpy.init(); node=rclpy.create_node("mvfuse")
    pub=node.create_publisher(String,"/mycobot/cmd/move",10); state=RobotState(node)

    el=math.radians(args.elev_deg)
    accP=np.empty((0,3),np.float32); accC=np.empty((0,3),np.uint8)
    extr=[]; nok=0
    for k in range(args.n):
        az=2*math.pi*k/args.n
        d=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        p_cam=OBJ+args.slant*d
        Rbc=lookat_R(p_cam,OBJ)
        q_cur=state.get_q()
        # roll sweep about optical axis -> pick reachable plan with least joint travel
        best=None
        for th in range(0,360,30):
            Rc=Rbc@Rz(math.radians(th))
            Tbc=np.eye(4); Tbc[:3,:3]=Rc; Tbc[:3,3]=p_cam
            Tbt=Tbc@T_cam_tcp
            goal=list(Tbt[:3,3])+R2q(Tbt[:3,:3])
            r=pc.plan_pose(q_cur,goal)
            if not r.get("success"): continue
            travel=float(np.rad2deg(np.abs(np.array(r["trajectory"][-1])-q_cur)).max())
            if best is None or travel<best[0]: best=(travel,r,th)
        if best is None:
            print(f"view {k} az{math.degrees(az):.0f}: UNREACHABLE (all rolls)",flush=True); continue
        travel,r,th=best
        ex=execute(state,pub,np.array(r["trajectory"]),r["dt"],"pid",args.vmax,2.0,
                   f"v{k}",track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0})
        if not ex.get("ok") or ex.get("reach_err",999)>12:
            print(f"view {k}: move failed reach_err={ex.get('reach_err'):.1f}",flush=True); continue
        time.sleep(0.4)
        q_act=state.get_q()
        T_base_cam=fk(q_act)@T_tcp_cam        # ACTUAL extrinsic
        rgb,depth,K,_=capture_aligned(args.serial,CW,CH,30,30)
        bgr=rgb[:,:,::-1].copy()
        try:
            label,inst=segment("tcp://127.0.0.1:5599",bgr,args.prompt,20,20000)
        except Exception as e:
            print(f"view {k}: SAM3 fail {e}",flush=True); inst=[]
        # choose instance whose base-frame centroid is nearest OBJ; fall back to all
        bestpts=None; bestd=1e9
        for ins in inst:
            m=label==ins["id"]
            pts,cols=deproject_mask(m,depth,rgb,K,0.05,2.0)
            if len(pts)==0: continue
            Pb=(T_base_cam[:3,:3]@pts.T).T+T_base_cam[:3,3]
            # crop to workspace box around OBJ
            sel=(np.abs(Pb[:,0]-OBJ[0])<args.crop)&(np.abs(Pb[:,1]-OBJ[1])<args.crop)&\
                (Pb[:,2]>-0.03)&(Pb[:,2]<args.zmax)
            if sel.sum()<30: continue
            cen=Pb[sel].mean(0); dd=np.linalg.norm(cen[:2]-OBJ[:2])
            if dd<bestd: bestd=dd; bestpts=(Pb[sel].astype(np.float32),cols[sel])
        cv_fn=f"img/view{k}_az{int(math.degrees(az))}.png"
        import cv2; cv2.imwrite(os.path.join(out,cv_fn),bgr)
        if bestpts is None:
            print(f"view {k} az{math.degrees(az):.0f}: no object points in crop",flush=True)
            extr.append({"view":k,"az_deg":math.degrees(az),"roll_deg":th,"q":list(map(float,q_act)),
                         "T_base_cam":T_base_cam.tolist(),"image":cv_fn,"obj_pts":0}); continue
        Pb,Cb=bestpts; accP=np.concatenate([accP,Pb]); accC=np.concatenate([accC,Cb])
        nok+=1
        write_ply(os.path.join(out,f"cloud_cum_{nok}.ply"),accP,accC)
        write_ply(os.path.join(out,f"view{k}_obj.ply"),Pb,Cb)
        cen=Pb.mean(0)
        print(f"view {k} az{math.degrees(az):3.0f} roll{th:3d}: +{len(Pb):5d} pts "
              f"(cum {len(accP)}) cen=[{cen[0]:+.3f},{cen[1]:+.3f},{cen[2]:+.3f}] travel{travel:.0f}",flush=True)
        extr.append({"view":k,"az_deg":math.degrees(az),"roll_deg":th,"q":list(map(float,q_act)),
                     "T_base_cam":T_base_cam.tolist(),"image":cv_fn,"obj_pts":int(len(Pb)),
                     "cum_pts":int(len(accP)),"cum_ply":f"cloud_cum_{nok}.ply"})

    # return to base
    q=state.get_q(); rb=pc.plan_joint(q,[0,-0.349066,1.919862,0,-1.570796,0])
    if rb.get("success"):
        execute(state,pub,np.array(rb["trajectory"]),rb["dt"],"pid",25,3,"to-base",
                track={"ramp_time":0.15,"pos_gain":1.0,"vff_scale":1.0})
    json.dump({"obj":OBJ.tolist(),"slant":args.slant,"elev_deg":args.elev_deg,"prompt":args.prompt,
               "handeye":HANDEYE,"frame":"base_link","views":extr,
               "total_pts":int(len(accP))},open(out+"/extrinsics.json","w"),indent=2)
    print(f"\nDONE views_ok={nok}/{args.n} total_pts={len(accP)} OUT={out}",flush=True)
    rclpy.shutdown()

if __name__=="__main__":
    main()
