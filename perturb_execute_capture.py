#!/usr/bin/env python3
"""perturb_execute_capture :: execute a cuRobo B-spline plan + capture camera frames.

Runs in Python 3.8 (ROS 2 Humble + libusb pyrealsense2). Loads the plan written
by perturb_plan.py, time-scales it to respect a --max-vel-deg cap (40+ deg/s verified fault-free)
following-error ceiling, commands the robot over the ROS bridge, and captures
color frames from the D405/D435 throughout the motion.

Safety:
  * verifies the live joint state matches the plan's start_q before moving
  * time-scales so peak joint speed <= --max-vel-deg (default 28 deg/s)
  * the bridge clamps all targets to LinuxCNC soft limits

Usage (after sourcing ros2node.env, PYTHONPATH=~/librealsense/build/release):
  python3 perturb_execute_capture.py --plan /tmp/perturb_plan.json --duration 2.0
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import cv2
import pyrealsense2 as rs

from joint_conventions import rad_to_linuxcnc_deg, JOINT_NAMES

CAMERAS = {"d405": "218622271300", "d435": "043422070101"}


def read_current_q(node, timeout=8.0):
    got = {}
    for rel in (ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT):
        node.create_subscription(
            JointState, "/joint_states",
            lambda m: got.__setitem__("q", list(m.position)),
            QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=10))
    t = time.time()
    while "q" not in got and time.time() - t < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    return got.get("q")


class CameraCapture(threading.Thread):
    def __init__(self, name, serial, outdir, w=848, h=480, fps=30):
        super().__init__(daemon=True)
        self.name, self.serial, self.outdir = name, serial, outdir
        self.w, self.h, self.fps = w, h, fps
        self.stop_evt = threading.Event()
        self.frames = 0
        self.ok = False

    def run(self):
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"[cap:{self.name}] start failed: {e}"); return
        self.ok = True
        try:
            while not self.stop_evt.is_set():
                try:
                    fr = pipe.wait_for_frames(1000)
                except RuntimeError:
                    continue
                c = fr.get_color_frame()
                if not c:
                    continue
                img = np.asanyarray(c.get_data())
                ts = time.time()
                cv2.imwrite(os.path.join(self.outdir, f"{self.name}_{self.frames:04d}_{ts:.3f}.png"), img)
                self.frames += 1
        finally:
            pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/tmp/perturb_plan.json")
    ap.add_argument("--duration", type=float, default=2.0, help="total move time (s) after scaling")
    ap.add_argument("--max-vel-deg", type=float, default=28.0, help="peak joint speed cap (deg/s)")
    ap.add_argument("--controller", default="pid")
    ap.add_argument("--q-tol-deg", type=float, default=3.0, help="abort if live q far from plan start")
    ap.add_argument("--cameras", nargs="+", default=["d405", "d435"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="capture only, do NOT command the robot")
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    traj = np.array(plan["trajectory"])          # [N,6] URDF rad
    N = traj.shape[0]
    start_q = np.array(plan["start_q"])

    # time-scale: stretch so peak joint speed <= max_vel
    base_dt = float(plan["dt"])
    step_max = np.abs(np.diff(traj, axis=0)).max() if N > 1 else 0.0  # rad per step
    peak_vel_base = np.rad2deg(step_max) / base_dt if base_dt > 0 else 0.0
    scale_vel = peak_vel_base / args.max_vel_deg if peak_vel_base > 0 else 1.0
    scale_dur = args.duration / (base_dt * (N - 1)) if N > 1 else 1.0
    scale = max(1.0, scale_vel, scale_dur)       # only ever slow down
    dt = base_dt * scale
    peak_vel = peak_vel_base / scale
    print(f"[exec] {N} waypoints | base peak {peak_vel_base:.1f} deg/s -> scaled "
          f"{peak_vel:.1f} deg/s | dt {base_dt:.3f}->{dt:.3f}s | total {dt*(N-1):.2f}s")

    rclpy.init()
    node = rclpy.create_node("perturb_exec")

    # safety: live q must match the plan's start
    q = read_current_q(node)
    if q is None:
        print("[exec] ABORT: no /joint_states"); rclpy.shutdown(); return
    dq = np.rad2deg(np.abs(np.array(q) - start_q))
    print(f"[exec] live-vs-plan start max joint diff {dq.max():.2f} deg")
    if dq.max() > args.q_tol_deg:
        print(f"[exec] ABORT: robot moved since planning (> {args.q_tol_deg} deg). Re-plan."); rclpy.shutdown(); return

    # output dir
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/move_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    # start cameras
    caps = [CameraCapture(n, CAMERAS[n], outdir) for n in args.cameras if n in CAMERAS]
    for c in caps:
        c.start()
    time.sleep(1.5)  # let pipelines + auto-exposure settle, capture a few pre-move frames

    # build + send the trajectory command (LinuxCNC deg)
    traj_deg = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    cmd = {"trajectory": traj_deg, "traj_dt": dt,
           "target_deg": traj_deg[-1], "controller": args.controller}
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    status = {"done": False}
    def on_status(m):
        try:
            if json.loads(m.data).get("state") == "done":
                status["done"] = True
        except Exception:
            pass
    node.create_subscription(String, "/mycobot/status", on_status,
                             QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                        history=HistoryPolicy.KEEP_LAST, depth=10))

    if args.dry_run:
        print("[exec] DRY RUN: not sending move command")
    else:
        time.sleep(0.3)
        pub.publish(String(data=json.dumps(cmd)))
        print(f"[exec] sent trajectory to /mycobot/cmd/move (controller={args.controller})")

    # wait for motion to finish (+ tail frames)
    move_t = dt * (N - 1)
    deadline = time.time() + move_t + 4.0
    while time.time() < deadline and not (status["done"] and not args.dry_run):
        rclpy.spin_once(node, timeout_sec=0.05)
    time.sleep(0.8)  # capture a few post-move frames

    for c in caps:
        c.stop_evt.set()
    for c in caps:
        c.join(timeout=3.0)

    print(f"[exec] done. status_done={status['done']}")
    for c in caps:
        print(f"[exec] {c.name}: {c.frames} frames {'OK' if c.ok else 'FAILED'} -> {outdir}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
