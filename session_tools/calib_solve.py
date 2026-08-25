#!/usr/bin/env python3
"""Solve hand-eye extrinsics from a capture session.

  D405 (arm)   : eye-in-hand  -> T_tcp_cam405  (camera pose w.r.t. cuRobo tool frame)
  D435 (fixed) : if it co-saw the SAME board -> T_base_cam435 via the board chain

Quality metric = spread of the recovered board pose in base_link across all
views (a perfect calibration puts the static board at exactly one place).

  python3 session_tools/calib_solve.py --session captures/calib_session_XXXX/session.json
"""
import argparse, json, os
import numpy as np, cv2
from calib_utils import (rt_to_T, posquat_to_T, T_inv, quat_to_R,
                         R_to_quat, R_to_rpy_deg, avg_pose)

ap = argparse.ArgumentParser()
ap.add_argument("--session", required=True)
ap.add_argument("--min-reproj", type=float, default=1.5, help="drop dets worse than this (px)")
ap.add_argument("--gripper-poses", default="tcp_poses.json",
                help="gripper<-base poses file in the session dir (tcp_poses.json or flange_poses.json)")
ap.add_argument("--gripper-name", default="tcp", help="label for the gripper frame in output keys")
a = ap.parse_args()

sess = json.load(open(a.session))
d = os.path.dirname(a.session)
tcp = json.load(open(os.path.join(d, a.gripper_poses)))

# ---- collect eye-in-hand samples for D405 ----
samples, dropped = [], 0
for rec in sess["records"]:
    if not rec.get("settled"):
        continue
    det = rec.get("d405")
    if det is None:
        continue
    if det["reproj_px"] > a.min_reproj:
        dropped += 1; continue
    if rec["name"] not in tcp:
        continue
    g = posquat_to_T(tcp[rec["name"]]["pos"], tcp[rec["name"]]["quat_wxyz"])  # base<-tcp
    c = rt_to_T(det["rvec"], det["tvec"])                                     # cam<-target
    samples.append((g, c))

print(f"D405 eye-in-hand samples: {len(samples)} used, {dropped} dropped (>{a.min_reproj}px)")
if len(samples) < 3:
    raise SystemExit("ABORT: need >=3 (ideally >=8) good D405 detections")

Rg = [s[0][:3, :3] for s in samples]
tg = [s[0][:3, 3].reshape(3, 1) for s in samples]
Rt = [s[1][:3, :3] for s in samples]
tt = [s[1][:3, 3].reshape(3, 1) for s in samples]

methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK,
           "HORAUD": cv2.CALIB_HAND_EYE_HORAUD, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
best = None
for mname, m in methods.items():
    try:
        Rcg, tcg = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
    except cv2.error:
        continue
    X = np.eye(4); X[:3, :3] = Rcg; X[:3, 3] = tcg.flatten()        # tcp<-cam
    boards = [g @ X @ c for (g, c) in samples]                      # base<-target per view
    _, spread = avg_pose(boards)
    print(f"  {mname:11s} board-in-base spread = {spread:5.1f} mm")
    if best is None or spread < best[1]:
        best = (mname, spread, X, boards)

mname, spread, X405, boards = best
T_base_target, _ = avg_pose(boards)
print(f"\n== D405 (eye-in-hand) ==  method={mname}  board-in-base spread={spread:.1f} mm "
      f"({'great' if spread < 3 else 'ok' if spread < 8 else 'POOR - add rotation diversity'})")
print(f"  T_tcp_cam405  pos(m)={np.round(X405[:3,3],4).tolist()}  "
      f"rpy(deg)={[round(v,2) for v in R_to_rpy_deg(X405[:3,:3])]}")

out = {"frame_convention": "gripper = cuRobo tool frame (TCP) reported by FK; "
                           "camera pose at runtime = T_base_tcp(q) @ T_tcp_cam405",
       "board": sess["board"],
       "T_tcp_cam405": {"pos": X405[:3, 3].tolist(), "quat_wxyz": R_to_quat(X405[:3, :3]),
                        "method": mname, "board_in_base_spread_mm": spread, "n": len(samples)},
       "T_base_target": {"pos": T_base_target[:3, 3].tolist(),
                         "quat_wxyz": R_to_quat(T_base_target[:3, :3])}}

# ---- D435 (fixed) via the shared static board ----
cams435 = []
for rec in sess["records"]:
    det = rec.get("d435")
    if det is None or det["reproj_px"] > 3.0:
        continue
    c435 = rt_to_T(det["rvec"], det["tvec"])            # cam435<-target
    cams435.append(T_base_target @ T_inv(c435))         # base<-cam435
if cams435:
    T435, spread435 = avg_pose(cams435)
    print(f"\n== D435 (fixed, co-vis) ==  {len(cams435)} views  spread={spread435:.1f} mm")
    print(f"  T_base_cam435  pos(m)={np.round(T435[:3,3],4).tolist()}  "
          f"rpy(deg)={[round(v,2) for v in R_to_rpy_deg(T435[:3,:3])]}")
    out["T_base_cam435"] = {"pos": T435[:3, 3].tolist(), "quat_wxyz": R_to_quat(T435[:3, :3]),
                            "spread_mm": spread435, "n": len(cams435), "mode": "covis"}
else:
    print("\n== D435 ==  no usable detections of the shared board.")
    print("  -> not co-visible; run a separate eye-to-hand session (board on flange).")
    out["T_base_cam435"] = None

json.dump(out, open(os.path.join(d, "extrinsics.json"), "w"), indent=2)
print(f"\nsaved -> {os.path.join(d, 'extrinsics.json')}")
