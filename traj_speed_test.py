#!/usr/bin/env python3
"""
traj_speed_test :: does trajectory-style commanding let robot_hal exceed the
~14 deg/s ceiling while keeping the realtime (2.1 ms) loop?

Commands robot_hal's traj_move mode (smoothly advancing trapezoidal pos_cmd +
velocity feedforward) on one joint at increasing commanded velocities, and
measures the achieved peak velfb from /mycobot/drive_feedback. Returns to base
between, aborts on under-track or fault.

Usage:
    python3 traj_speed_test.py --joint 0 --distance 20 --vels 30 50 70
"""
import argparse
import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]


class TrajTest(Node):
    def __init__(self, joint, log_path):
        super().__init__("mycobot_traj_test")
        self._j = joint
        self._log = open(log_path, "w", buffering=1)
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 200)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._status = None
        self._collect = False
        self._buf = []
        self._tag = None

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, msg):
        try:
            self._status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_fb(self, msg: JointState):
        if not msg.position or len(msg.position) <= self._j:
            return
        v = msg.velocity[self._j] if len(msg.velocity) > self._j else 0.0
        p = msg.position[self._j]
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "drive_feedback",
                                    "tag": self._tag, "joint": self._j,
                                    "posfb": round(p, 5), "velfb": round(v, 5)}) + "\n")
        if self._collect:
            self._buf.append((self._now(), p, v))

    def _spin(self, secs):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _send(self, obj, wait_done=True, timeout=8.0):
        self._status = None
        m = String(); m.data = json.dumps(obj)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(m)
        if not wait_done:
            return
        t0 = time.time()
        while rclpy.ok() and time.time() < t0 + timeout:
            rclpy.spin_once(self, timeout_sec=0.03)
            if self._status and self._status.get("state") == "done":
                return

    def probe(self, base, V, dist, tag):
        self._tag = tag
        self._send({"target_deg": base, "duration": 6, "controller": "pid", "gains": {"u_max": 8}})
        self._spin(0.4)
        self._buf = []; self._collect = True
        self._send({"traj_move": {"joint": self._j, "distance_deg": dist,
                                  "velocity_deg_s": V, "accel_deg_s2": 300}}, timeout=8)
        self._spin(0.3)
        self._collect = False
        arr = np.array(self._buf) if len(self._buf) > 10 else None
        self._send({"target_deg": base, "duration": 6, "controller": "pid", "gains": {"u_max": 8}})
        if arr is None:
            return None
        v = np.abs(arr[:, 2])
        return dict(cmd_V=V, peak_velfb=float(np.max(v)), moved=float(arr[-1, 1] - arr[0, 1]))

    def close(self):
        self._log.close()


def main():
    ap = argparse.ArgumentParser(description="Trajectory-style realtime speed test.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--distance", type=float, default=20.0)
    ap.add_argument("--vels", type=float, nargs="+", default=[30, 50, 70])
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"traj_speed_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = TrajTest(args.joint, out)
    results = []
    try:
        for V in args.vels:
            node.get_logger().info(f"traj_move V_cmd={V} deg/s, {args.distance} deg")
            r = node.probe(base, V, args.distance, tag=f"v{int(V)}")
            if r is None:
                node.get_logger().warn(f"V={V}: no data (fault?) - ABORTING")
                break
            results.append(r)
            node.get_logger().info(f"  commanded {V:.0f} -> achieved peak {r['peak_velfb']:.1f} deg/s "
                                   f"(moved {r['moved']:.1f} deg)")
            if abs(r["moved"]) < 0.5 * args.distance:
                node.get_logger().warn("  under-tracked - ABORTING"); break
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n=== trajectory-style realtime speed test: joint {args.joint}, {args.distance:.0f} deg ===")
    print(f"{'cmd V':>8}{'peak velfb':>12}{'moved':>8}   (robot_hal ceiling was ~14 deg/s)")
    for r in results:
        print(f"{r['cmd_V']:>8.0f}{r['peak_velfb']:>12.1f}{r['moved']:>8.1f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
