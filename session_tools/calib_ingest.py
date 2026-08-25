#!/usr/bin/env python3
"""Ingest an externally-captured calibration set (images + q_urdf_rad) into the
session.json schema that calib_fk.py / calib_solve.py consume.

Expects <capture>/poses.json with records: {image, q_urdf_rad, reach, reach_err}.
Detects the ChArUco board in each image, attaches D405 factory intrinsics (at the
image resolution), and writes <capture>/session.json.

  source ~/Desktop/2026/ros2node/config/ros2node.env
  export PYTHONPATH=~/Desktop/2026/mycobot_mpc:~/librealsense/build/release:$PYTHONPATH
  python3 session_tools/calib_ingest.py --capture captures/calib_20260622_215855
"""
import argparse, json, os, glob
import numpy as np, cv2
from calib_utils import make_board, detect_charuco

ap = argparse.ArgumentParser()
ap.add_argument("--capture", required=True, help="capture dir holding poses.json + img/")
ap.add_argument("--square", type=float, default=0.035)
ap.add_argument("--marker", type=float, default=0.026)
ap.add_argument("--d405", default="218622271300")
ap.add_argument("--intrinsics", default=None, help="optional intrinsics.json {K,dist}; else D405 factory")
a = ap.parse_args()

doc = json.load(open(os.path.join(a.capture, "poses.json")))
recs_in = doc["records"]
# image size (intrinsics are resolution-specific)
sample = next(r for r in recs_in if r.get("image"))
H, W = cv2.imread(os.path.join(a.capture, sample["image"])).shape[:2]

# ---- intrinsics ----
if a.intrinsics:
    j = json.load(open(a.intrinsics)); K, dist = j["K"], j["dist"]
    print(f"intrinsics from {a.intrinsics}")
else:
    import pyrealsense2 as rs
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(a.d405)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipe.stop()
    K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
    dist = list(intr.coeffs)
    print(f"D405 factory intrinsics @ {W}x{H}: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

board = make_board(a.square, a.marker)
records, n_det = [], 0
for r in recs_in:
    if not r.get("image") or r.get("q_urdf_rad") is None:
        continue
    name = os.path.splitext(os.path.basename(r["image"]))[0]
    bgr = cv2.imread(os.path.join(a.capture, r["image"]))
    det = detect_charuco(board, bgr, K, dist) if bgr is not None else None
    n_det += det is not None
    records.append({"name": name, "i": r.get("idx"),
                    "settled": bool(r.get("reach", True)),
                    "reach_err": float(r.get("reach_err", 0.0)),
                    "q_rad": [float(v) for v in r["q_urdf_rad"]],
                    "d405": det})

sess = {"board": {"square": a.square, "marker": a.marker, "dict": "DICT_4X4_50", "squares": [5, 7]},
        "target": doc.get("target"),
        "intrinsics": {"d405": {"K": K, "dist": dist}},
        "records": records}
out = os.path.join(a.capture, "session.json")
json.dump(sess, open(out, "w"), indent=2)
print(f"ingested {len(records)} shots, {n_det} board detections -> {out}")
print("next:")
print(f"  ~/miniconda3/envs/curobo2/bin/python session_tools/calib_fk.py --session {out}")
print(f"  python3 session_tools/calib_solve.py --session {out}")
