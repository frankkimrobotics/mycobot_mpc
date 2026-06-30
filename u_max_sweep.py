#!/usr/bin/env python3
"""
u_max_sweep :: find the achievable joint velocity by raising the control law's
per-step throttle (U_MAX_PER_STEP) and reading the plateau velfb from NORMAL
trapezoidal PID moves (no 500 Hz raw commanding -> gentle on the drive bus).

For each U_MAX value: move to base, command a single-joint move of `distance`
deg WITH gains={"u_max": X}, capture /mycobot/drive_feedback, measure the
plateau (cruise) |velfb|, then return to base at the safe baseline u_max.

Careful by design: ramps U_MAX gradually, ABORTS if a move fails to track
(suspected fault) or if plateau velocity exceeds --vmax-stop.

Usage:
    python3 u_max_sweep.py --joint 0 --distance 20 \
        --u-max 8 12 16 24 32 48 --vmax-stop 90
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
BASE_U_MAX = 8.0  # safe default throttle for return moves


class UMaxSweep(Node):
    def __init__(self, joint, log_path):
        super().__init__("mycobot_umax_sweep")
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

    def move(self, target, duration, u_max, collect=False):
        cmd = {"target_deg": [float(v) for v in target], "duration": float(duration),
               "controller": "pid", "gains": {"u_max": u_max}}
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        if collect:
            self._buf = []; self._collect = True
        self._status = None
        self._pub.publish(m)
        t0 = time.time()
        while rclpy.ok() and time.time() < t0 + duration + 3:
            rclpy.spin_once(self, timeout_sec=0.03)
            if self._status and self._status.get("state") == "done":
                break
        self._spin(0.2)
        self._collect = False

    def probe(self, base, u_max, dist, duration, tag):
        self._tag = tag
        self.move(base, 6, BASE_U_MAX)               # to base (gentle)
        self._spin(0.4)
        target = list(base); target[self._j] = base[self._j] + dist
        self.move(target, duration, u_max, collect=True)   # the test move
        arr = np.array(self._buf) if len(self._buf) > 10 else None
        self.move(base, 6, BASE_U_MAX)               # return to base (gentle)
        if arr is None:
            return None
        v = np.abs(arr[:, 2]); moved = float(arr[-1, 1] - arr[0, 1])
        peak = float(np.max(v))
        cruise = v[v > 0.8 * peak]
        plateau = float(np.median(cruise)) if len(cruise) else peak
        return dict(u_max=u_max, plateau=plateau, peak=peak, moved=moved)

    def close(self):
        self._log.close()


def main():
    ap = argparse.ArgumentParser(description="Raise U_MAX_PER_STEP and measure achievable velocity.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--distance", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--u-max", type=float, nargs="+", default=[8, 12, 16, 24, 32, 48])
    ap.add_argument("--vmax-stop", type=float, default=90.0, help="abort if plateau exceeds this (deg/s)")
    ap.add_argument("--limit", type=float, default=180.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"umax_sweep_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = UMaxSweep(args.joint, out)
    results = []
    try:
        for u in args.u_max:
            node.get_logger().info(f"U_MAX={u} -> {args.distance} deg move on joint {args.joint}")
            r = node.probe(base, u, args.distance, args.duration, tag=f"umax{int(u)}")
            if r is None:
                node.get_logger().warn(f"U_MAX={u}: no data (fault?) - ABORTING")
                break
            results.append(r)
            node.get_logger().info(
                f"  plateau {r['plateau']:.1f} deg/s  peak {r['peak']:.1f}  moved {r['moved']:.1f} deg")
            if abs(r["moved"]) < 0.5 * args.distance:
                node.get_logger().warn(f"  move under-tracked ({r['moved']:.1f}/{args.distance}) - ABORTING")
                break
            if r["plateau"] > args.vmax_stop:
                node.get_logger().warn(f"  plateau {r['plateau']:.0f} > vmax-stop {args.vmax_stop} - stopping (safety)")
                break
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n=== U_MAX sweep: joint {args.joint}, {args.distance:.0f} deg moves, limit {args.limit:.0f} deg/s ===")
    print(f"{'U_MAX':>7}{'plateau v':>11}{'peak v':>9}{'% limit':>9}{'moved':>8}")
    for r in results:
        print(f"{r['u_max']:>7.0f}{r['plateau']:>11.1f}{r['peak']:>9.1f}"
              f"{100*r['plateau']/args.limit:>8.0f}%{r['moved']:>8.1f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
