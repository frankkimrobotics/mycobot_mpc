#!/usr/bin/env python3
"""refine_icp :: tighten the 8-view fused cloud with CONSTRAINED colored ICP.

The per-view clouds are already in the base frame via calibration
(T_base_cam = FK(q)@T_tcp_cam) -- a good init (~5mm) but with residual per-view
offset. Naive multiway ICP OVER-corrects here: the bottle is locally slab/cylinder
-like, so views slide along the degenerate axis (100mm 'corrections', no RMSE gain).

Fix: register each view to the UNION OF ALL OTHER views (leave-one-out anchor, so
the object's ends pin the slide), colored ICP (color fights the geometric
degeneracy), and CAP each correction at the calibration uncertainty (~15mm / 10deg)
so ICP cannot run away. Metric = mean inter-view nearest-neighbour distance
(= double-wall thickness), the honest measure of fusion tightness.

Run:  python3 mycobot_mpc/refine_icp.py captures/multiview_20260622_234245
"""
import sys, os, glob, numpy as np, open3d as o3d
reg = o3d.pipelines.registration

D = sys.argv[1] if len(sys.argv)>1 else "captures/multiview_20260622_234245"
VOXEL=0.003; SCALES=[0.010,0.006,0.003]; ITERS=[50,40,30]
LAMBDA=0.5            # colored-ICP geometric/color balance (0.5 = equal-ish)
CAP_T=0.015          # max translation correction per view (m)
CAP_R=np.deg2rad(10) # max rotation correction per view
PASSES=3

def load(p):
    c=o3d.io.read_point_cloud(p)
    c,_=c.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
    return c
def prep(c,v):
    d=c.voxel_down_sample(v)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2.5,max_nn=30))
    d.colors=c.voxel_down_sample(v).colors
    return d
def cap(T):
    t=T[:3,3]; nt=np.linalg.norm(t)
    if nt>CAP_T: T[:3,3]=t*(CAP_T/nt)
    ang=np.arccos(np.clip((np.trace(T[:3,:3])-1)/2,-1,1))
    if ang>CAP_R:           # scale rotation back toward identity
        s=CAP_R/ang
        rv,_=__import__("cv2").Rodrigues(T[:3,:3]);
        T[:3,:3],_=__import__("cv2").Rodrigues(rv*s)
    return T

files=sorted(glob.glob(os.path.join(D,"view*_obj.ply")))
clouds=[load(f) for f in files]; n=len(clouds)
poses=[np.eye(4) for _ in range(n)]
print(f"{n} views ({[len(c.points) for c in clouds]} pts)")

def placed(i):
    c=o3d.geometry.PointCloud(clouds[i]); c.transform(poses[i]); return c

def interview_nn():
    """mean nearest-neighbour distance from each view to the union of the others."""
    ds=[]
    pls=[prep(placed(i),VOXEL) for i in range(n)]
    for i in range(n):
        others=o3d.geometry.PointCloud()
        for j in range(n):
            if j!=i: others+=pls[j]
        d=np.asarray(pls[i].compute_point_cloud_distance(others))
        d=d[d<0.02]                      # ignore non-overlap (>2cm) regions
        if len(d): ds.append(d.mean())
    return np.mean(ds)

pre=interview_nn(); print(f"inter-view NN distance (init/calibration): {pre*1000:.2f} mm")

for p in range(PASSES):
    for i in range(n):
        tgt=o3d.geometry.PointCloud()
        for j in range(n):
            if j!=i: tgt+=placed(j)
        T=np.eye(4)
        for v,it in zip(SCALES,ITERS):
            s=prep(placed(i),v); t=prep(tgt,v)
            try:
                res=reg.registration_colored_icp(s,t,v*1.5,T,
                    reg.TransformationEstimationForColoredICP(lambda_geometric=LAMBDA),
                    reg.ICPConvergenceCriteria(max_iteration=it))
            except Exception:
                res=reg.registration_icp(s,t,v*1.5,T,
                    reg.TransformationEstimationPointToPlane(),
                    reg.ICPConvergenceCriteria(max_iteration=it))
            T=res.transformation
        poses[i]=cap(T.copy())@poses[i]
    print(f"  pass {p+1}: inter-view NN = {interview_nn()*1000:.2f} mm")

post=interview_nn()
corr=[np.linalg.norm(poses[i][:3,3]) for i in range(n)]
print(f"\ninter-view NN: {pre*1000:.2f} -> {post*1000:.2f} mm "
      f"({100*(1-post/pre):.0f}% tighter)")
print("per-view applied correction (mm):",[round(c*1000,1) for c in corr])

# merge + save (refined and raw, same voxel, for fair comparison)
def merge(transform):
    m=o3d.geometry.PointCloud()
    for i,f in enumerate(files):
        c=load(f)
        if transform: c.transform(poses[i])
        m+=c
    return m.voxel_down_sample(VOXEL)
ref=merge(True); raw=merge(False)
o3d.io.write_point_cloud(os.path.join(D,"fused_icp.ply"),ref)
o3d.io.write_point_cloud(os.path.join(D,"fused_raw.ply"),raw)
print(f"saved fused_icp.ply ({len(ref.points)} pts), fused_raw.ply ({len(raw.points)} pts)")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig=plt.figure(figsize=(16,8))
for col,(name,c) in enumerate([("RAW (calibration only)",raw),("ICP-refined (anchored+capped)",ref)]):
    P=np.asarray(c.points); C=np.asarray(c.colors)
    if len(P)>60000: idx=np.linspace(0,len(P)-1,60000).astype(int); P,C=P[idx],C[idx]
    ax=fig.add_subplot(2,2,col+1); ax.scatter(P[:,0],P[:,1],c=C,s=2); ax.set_aspect("equal")
    ax.set_title(f"{name} - top XY"); ax.set_xlabel("X");ax.set_ylabel("Y")
    ax2=fig.add_subplot(2,2,col+3); ax2.scatter(P[:,1],P[:,2],c=C,s=2); ax2.set_aspect("equal")
    ax2.set_title(f"{name} - side YZ"); ax2.set_xlabel("Y");ax2.set_ylabel("Z")
fig.tight_layout(); fig.savefig(os.path.join(D,"icp_before_after.png"),dpi=110)
print("saved icp_before_after.png")
