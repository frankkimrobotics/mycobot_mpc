#!/usr/bin/env python3
"""
probe_max_velocity :: how fast can a joint actually go?

For each requested fraction of the configured joint velocity limit, ramps
pos_cmd at that COMMANDED velocity over a fixed perturbation (default 20 deg)
via robot_hal's vel_probe mode, and measures the achieved peak |velfb| from
/mycobot/drive_feedback. Returns to base between probes; aborts if a fault is
suspected (joint stops tracking).

Default joint = 0 (base rotation, no gravity load). Limit defaults to the J0/J1/J2
value (180 deg/s); J3-J5 are 200.

Usage:
    python3 probe_max_velocity.py --joint 0 --limit 180 \
        --fractions 0.30 0.50 0.70 0.80 0.90 --distance 20
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


class VelProbe(Node):
    def __init__(self, joint, log_path):
        super().__init__("mycobot_vel_probe")
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
        e = msg.effort[self._j] if len(msg.effort) > self._j else 0.0
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "drive_feedback",
                                    "tag": self._tag, "joint": self._j, "posfb": round(p, 5),
                                    "velfb": round(v, 5), "torqfb": round(e, 7)}) + "\n")
        if self._collect:
            self._buf.append((self._now(), p, v, e))

    def _spin(self, secs):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.02)

    def move(self, target, duration):
        self._status = None
        m = String(); m.data = json.dumps({"target_deg": [float(v) for v in target],
                                           "duration": float(duration), "controller": "pid"})
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(m)
        t0 = time.time()
        while rclpy.ok() and time.time() < t0 + duration + 3:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._status and self._status.get("state") == "done":
                return

    def probe(self, base, V, dist, tag):
        self._tag = tag
        self.move(base, 6)              # to base
        self._spin(0.6)
        self._buf = []; self._collect = True
        m = String(); m.data = json.dumps({"vel_probe": {"joint": self._j,
                                           "velocity_deg_s": V, "distance_deg": dist}})
        self._status = None
        self._pub.publish(m)
        t0 = time.time()
        while rclpy.ok() and time.time() < t0 + 7:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self._status and self._status.get("state") == "done":
                break
        self._spin(0.3)
        self._collect = False
        arr = np.array(self._buf) if len(self._buf) > 10 else None
        self.move(base, 6)             # return to base
        if arr is None:
            return None
        v = np.abs(arr[:, 2])
        return dict(cmd_V=V, peak_velfb=float(np.max(v)),
                    moved_deg=float(arr[-1, 1] - arr[0, 1]))

    def close(self):
        self._log.close()


def main():
    ap = argparse.ArgumentParser(description="Probe achievable joint velocity vs commanded fraction of limit.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--limit", type=float, default=180.0, help="configured joint velocity limit (deg/s)")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.30, 0.50, 0.70, 0.80, 0.90])
    ap.add_argument("--distance", type=float, default=20.0, help="perturbation distance (deg)")
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"vel_probe_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = VelProbe(args.joint, out)
    results = []
    try:
        for f in args.fractions:
            V = round(f * args.limit, 1)
            node.get_logger().info(f"probe {int(f*100)}% -> command {V} deg/s, {args.distance} deg")
            r = node.probe(base, V, args.distance, tag=f"{int(f*100)}pct")
            if r is None:
                node.get_logger().warn(f"{int(f*100)}%: no data (fault?) - aborting")
                break
            r["frac"] = f
            results.append(r)
            node.get_logger().info(f"  commanded {V:.0f} -> achieved peak {r['peak_velfb']:.1f} deg/s "
                                   f"(moved {r['moved_deg']:.1f} deg)")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n=== max-velocity probe: joint {args.joint}, limit {args.limit:.0f} deg/s, {args.distance:.0f} deg step ===")
    print(f"{'%limit':>7}{'cmd V':>9}{'peak velfb':>12}{'achieved %':>12}{'moved deg':>11}")
    for r in results:
        print(f"{int(r['frac']*100):>6}%{r['cmd_V']:>9.0f}{r['peak_velfb']:>12.1f}"
              f"{100*r['peak_velfb']/args.limit:>11.0f}%{r['moved_deg']:>11.1f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
