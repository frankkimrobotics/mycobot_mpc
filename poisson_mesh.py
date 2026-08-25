#!/usr/bin/env python3
"""Poisson-mesh the DUSt3R segmented object cloud into a watertight surface."""
import sys, os, numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D=sys.argv[1]; SRC=os.path.join(D,"dust3r_object_segmented.ply")
pcd=o3d.io.read_point_cloud(SRC)
pcd,_=pcd.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.0)
pcd=pcd.voxel_down_sample(0.0015)
print(f"cloud: {len(pcd.points)} pts after clean/downsample")

# normals oriented toward the cameras (object seen from above) so Poisson faces outward
pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.006,max_nn=40))
pcd.orient_normals_towards_camera_location(np.array([0.35,0.08,0.5]))  # above the object

mesh,dens=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd,depth=9,scale=1.1,linear_fit=True)
dens=np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens,0.06))        # trim balloon/low-support
# crop to the cloud's bbox (Poisson over-extends past the data)
bb=pcd.get_axis_aligned_bounding_box(); bb=bb.scale(1.05,bb.get_center())
mesh=mesh.crop(bb)
mesh.compute_vertex_normals()
mesh=mesh.filter_smooth_taubin(number_of_iterations=8)
mesh.compute_vertex_normals()
out=os.path.join(D,"dust3r_object_mesh.ply"); o3d.io.write_triangle_mesh(out,mesh)
V=np.asarray(mesh.vertices); T=np.asarray(mesh.triangles)
print(f"mesh: {len(V)} verts, {len(T)} faces  extent(mm) {(np.ptp(V,0)*1000).round(0)} -> {os.path.basename(out)}")

# render: shaded mesh from 3 angles + the cloud
P=np.asarray(pcd.points); Cc=np.asarray(pcd.colors)
fig=plt.figure(figsize=(18,5))
ax=fig.add_subplot(141,projection="3d"); ax.scatter(P[:,0],P[:,1],P[:,2],c=np.clip(Cc,0,1),s=2)
ax.set_title("segmented cloud"); ax.view_init(30,-60)
for k,(el,az) in enumerate([(35,-60),(15,30),(75,-90)]):
    a=fig.add_subplot(1,4,k+2,projection="3d")
    a.plot_trisurf(V[:,0],V[:,1],T,V[:,2],cmap="copper",linewidth=0,antialiased=True,shade=True)
    a.set_title(f"Poisson mesh (view {k+1})"); a.view_init(el,az)
    for axis in (a.set_xlabel,a.set_ylabel): axis("")
fig.tight_layout(); fig.savefig(os.path.join(D,"dust3r_mesh.png"),dpi=120)
print("saved dust3r_object_mesh.ply + dust3r_mesh.png")
