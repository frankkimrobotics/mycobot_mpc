#!/usr/bin/env python3
"""
traj_speed_test :: per-joint realtime velocity sweep via robot_hal's
trajectory-style command (advancing pos_cmd + tracking-error vel_cmd).

For each joint, commands traj_move at 20/40/60/80/100 % of the joint's
configured velocity limit (180 deg/s for J0-J2, 200 for J3-J5) and measures the
achieved peak velfb. The level where achieved stops tracking the command = that
joint's realtime ceiling. Returns to base between moves; aborts a joint on
under-track / suspected fault.

Usage:
    python3 traj_speed_test.py --joints 0 1 2 3 4 5 --distance 25 --accel 1000
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
# configured joint velocity limits (deg/s) from elerob.ini
JOINT_LIMIT = [180.0, 180.0, 180.0, 200.0, 200.0, 200.0]


class TrajTest(Node):
    def __init__(self, log_path):
        super().__init__("mycobot_traj_test")
        self._j = 0
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

    def probe(self, base, joint, V, dist, accel, tag):
        self._j = joint
        self._tag = tag
        self._send({"target_deg": base, "duration": 6, "controller": "pid", "gains": {"u_max": 8}})
        self._spin(0.4)
        self._buf = []; self._collect = True
        self._send({"traj_move": {"joint": joint, "distance_deg": dist,
                                  "velocity_deg_s": V, "accel_deg_s2": accel}}, timeout=8)
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
    ap = argparse.ArgumentParser(description="Per-joint realtime trajectory velocity sweep.")
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(MAX_JOINTS)))
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--distance", type=float, default=25.0)
    ap.add_argument("--accel", type=float, default=1000.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"traj_speed_alljoints_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = TrajTest(out)
    results = {j: [] for j in args.joints}
    try:
        for j in args.joints:
            lim = JOINT_LIMIT[j]
            node.get_logger().info(f"=== joint {j}  (limit {lim:.0f} deg/s) ===")
            for f in args.fractions:
                V = round(f * lim, 1)
                r = node.probe(base, j, V, args.distance, args.accel, tag=f"j{j}_{int(f*100)}pct")
                if r is None:
                    node.get_logger().warn(f"J{j} {int(f*100)}%: no data (fault?) - skipping joint")
                    break
                r["frac"] = f; r["limit"] = lim
                results[j].append(r)
                node.get_logger().info(f"  J{j} {int(f*100)}%: cmd {V:.0f} -> achieved {r['peak_velfb']:.1f} "
                                       f"deg/s (moved {r['moved']:.1f})")
                if abs(r["moved"]) < 0.4 * args.distance:
                    node.get_logger().warn(f"  J{j}: under-tracked - stopping this joint"); break
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n=== per-joint realtime trajectory velocity (25 deg move, accel {args.accel:.0f}) ===")
    print(f"{'joint':>6}{'limit':>7}{'20%':>8}{'40%':>8}{'60%':>8}{'80%':>8}{'100%':>8}   peak")
    for j in sorted(results):
        rs = results[j]
        if not rs:
            continue
        ach = {round(r["frac"], 2): r["peak_velfb"] for r in rs}
        row = "".join(f"{ach.get(f, float('nan')):>8.0f}" for f in [0.2, 0.4, 0.6, 0.8, 1.0])
        peak = max(r["peak_velfb"] for r in rs)
        print(f"{('J'+str(j)):>6}{JOINT_LIMIT[j]:>7.0f}{row}   {peak:.0f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
