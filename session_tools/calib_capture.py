#!/usr/bin/env python3
"""Step the arm through the saved calibration poses; at each SETTLED pose grab
D405 (arm) + D435 (fixed) color frames, detect the ChArUco board, solvePnP ->
T_cam_target, and record the actual joint config. Writes a session dir consumed
by calib_fk.py (curobo2 env) then calib_solve.py.

Env (python3 with realsense + ROS):
  source ~/Desktop/2026/ros2node/config/ros2node.env
  export PYTHONPATH=~/Desktop/2026/mycobot_mpc:~/librealsense/build/release:$PYTHONPATH
  python3 session_tools/calib_capture.py --poses captures/calib_poses_XXXX.json \
        --square 0.035 --marker 0.026
"""
import argparse, json, os, time
import numpy as np, cv2, rclpy
import pyrealsense2 as rs
from std_msgs.msg import String
from perturb_loop import PlannerClient, RobotState, execute
from calib_utils import make_board, detect_charuco

ap = argparse.ArgumentParser()
ap.add_argument("--poses", required=True)
ap.add_argument("--square", type=float, default=0.035, help="square side (m)")
ap.add_argument("--marker", type=float, default=0.026, help="marker side (m)")
ap.add_argument("--d405", default="218622271300", help="arm camera serial")
ap.add_argument("--d435", default="043422070101", help="fixed camera serial")
ap.add_argument("--max-vel-deg", type=float, default=18.0)
ap.add_argument("--settle", type=float, default=0.6, help="dwell after settle before grab (s)")
ap.add_argument("--out", default=None)
a = ap.parse_args()


class Cam:
    def __init__(self, name, serial, w=1280, h=720, fps=15):
        self.name = name; self.serial = serial
        self.pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        self.prof = self.pipe.start(cfg)
        intr = self.prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
        self.dist = list(intr.coeffs)
        for _ in range(10):                       # warm up auto-exposure
            try: self.pipe.wait_for_frames(2000)
            except RuntimeError: pass

    def grab(self):
        for _ in range(3):
            try: fr = self.pipe.wait_for_frames(2000)
            except RuntimeError: continue
            c = fr.get_color_frame()
            if c: return np.asanyarray(c.get_data()).copy()
        return None

    def stop(self):
        try: self.pipe.stop()
        except Exception: pass


poses_doc = json.load(open(a.poses))
poses = poses_doc["poses"]
board = make_board(a.square, a.marker)
stamp = time.strftime("%Y%m%d_%H%M%S")
outdir = a.out or os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/calib_session_{stamp}")
imgdir = os.path.join(outdir, "img"); os.makedirs(imgdir, exist_ok=True)

print("opening cameras ...")
cams = {"d405": Cam("d405", a.d405)}              # arm cam is required
try:
    cams["d435"] = Cam("d435", a.d435)
except Exception as e:
    print(f"[warn] D435 not opened ({e}) -- continuing D405-only")

pc = PlannerClient(); print("planner:", pc.rpc({'type': 'ping'}).get('backend'))
rclpy.init(); node = rclpy.create_node("calib_capture")
pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
state = RobotState(node)
track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}

records = []
for i, P in enumerate(poses):
    name = P["name"]; goal = np.array(P["q_rad"])
    q = state.get_q()
    if q is None:
        print("  no /joint_states -- abort"); break
    r = pc.plan_joint(q, goal)
    if not r.get("success"):
        print(f"[{i:02d}] {name}: plan FAILED ({r.get('status')}), skip"); continue
    ex = execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                 a.max_vel_deg, 3.0, name, track=track)
    time.sleep(a.settle)                          # kill residual vibration
    qf = state.get_q()
    rec = {"name": name, "i": i, "settled": bool(ex.get("ok")),
           "reach_err": float(ex.get("reach_err", 999)),
           "q_rad": [float(x) for x in qf]}
    for cn, cam in cams.items():
        bgr = cam.grab(); det = None
        if bgr is not None:
            cv2.imwrite(os.path.join(imgdir, f"{name}_{cn}.png"), bgr)
            det = detect_charuco(board, bgr, cam.K, cam.dist)
        rec[cn] = det
    d405 = rec.get("d405"); d435 = rec.get("d435")
    print(f"[{i:02d}] {name}: settled={rec['settled']} reach_err={rec['reach_err']:.1f} "
          f"d405={'Y %dc %.2fpx' % (d405['n_corners'], d405['reproj_px']) if d405 else 'NO'} "
          f"d435={'Y %dc' % d435['n_corners'] if d435 else 'no'}")
    records.append(rec)

for cam in cams.values():
    cam.stop()

sess = {"board": {"square": a.square, "marker": a.marker, "dict": "DICT_4X4_50", "squares": [5, 7]},
        "target": poses_doc.get("target"),
        "intrinsics": {cn: {"K": cam.K, "dist": cam.dist} for cn, cam in cams.items()},
        "records": records}
json.dump(sess, open(os.path.join(outdir, "session.json"), "w"), indent=2)

n405 = sum(1 for r in records if r.get("d405") and r["settled"])
n435 = sum(1 for r in records if r.get("d435") and r["settled"])
print(f"\nsaved -> {outdir}/session.json")
print(f"good detections (settled):  D405={n405}  D435={n435}")
print("next:")
print(f"  ~/miniconda3/envs/curobo2/bin/python session_tools/calib_fk.py --session {outdir}/session.json")
print(f"  python3 session_tools/calib_solve.py --session {outdir}/session.json")
rclpy.shutdown()
