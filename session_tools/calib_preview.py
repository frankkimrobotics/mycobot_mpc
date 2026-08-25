#!/usr/bin/env python3
"""No-motion preflight: grab ONE frame from each camera, detect the ChArUco
board, draw the overlay + board-frame axes, save PNGs, and report. Use this to
confirm the fixed D435i (and, if the arm is parked over the board, the D405)
actually see the board before launching the full capture run.

  source ~/Desktop/2026/ros2node/config/ros2node.env
  export PYTHONPATH=~/Desktop/2026/mycobot_mpc:~/librealsense/build/release:$PYTHONPATH
  python3 session_tools/calib_preview.py --square 0.035 --marker 0.026
"""
import argparse, os, time
import numpy as np, cv2
import pyrealsense2 as rs
from calib_utils import make_board, detect_charuco

ap = argparse.ArgumentParser()
ap.add_argument("--square", type=float, default=0.035)
ap.add_argument("--marker", type=float, default=0.026)
ap.add_argument("--d405", default="218622271300")
ap.add_argument("--d435", default="043422070101")
ap.add_argument("--out", default=None)
a = ap.parse_args()

board = make_board(a.square, a.marker)
stamp = time.strftime("%Y%m%d_%H%M%S")
outdir = a.out or os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/preview_{stamp}")
os.makedirs(outdir, exist_ok=True)


def grab(serial, w=1280, h=720, fps=15):
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    prof = pipe.start(cfg)
    intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
    dist = list(intr.coeffs)
    bgr = None
    try:
        for _ in range(15):
            try: fr = pipe.wait_for_frames(2000)
            except RuntimeError: continue
            c = fr.get_color_frame()
            if c: bgr = np.asanyarray(c.get_data()).copy()
    finally:
        pipe.stop()
    return bgr, K, dist


for name, serial in [("d405", a.d405), ("d435", a.d435)]:
    try:
        bgr, K, dist = grab(serial)
    except Exception as e:
        print(f"{name}: camera open FAILED ({e})"); continue
    if bgr is None:
        print(f"{name}: no frame"); continue
    det = detect_charuco(board, bgr, K, dist)
    vis = bgr.copy()
    if det:
        cv2.drawFrameAxes(vis, np.asarray(K, float), np.asarray(dist, float),
                          np.asarray(det["rvec"]), np.asarray(det["tvec"]), a.square * 3)
        dist_m = float(np.linalg.norm(det["tvec"]))
        print(f"{name}: DETECTED  corners={det['n_corners']}  reproj={det['reproj_px']:.2f}px  "
              f"board~{dist_m:.2f} m away  {'(GOOD)' if det['reproj_px'] < 1.5 else '(high reproj - check focus/glare)'}")
    else:
        print(f"{name}: board NOT detected - reposition camera/board or check lighting")
    p = os.path.join(outdir, f"{name}.png")
    cv2.imwrite(p, vis)
    print(f"      saved {p}")

print(f"\noverlays in {outdir}")
