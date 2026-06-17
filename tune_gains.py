#!/usr/bin/env python3
"""
tune_gains :: sweep PID gains over ROS 2 and pick the set that minimizes
settling time, overshoot, and steady-state error on a step.

Needs the gain-override build of robot_hal.py (accepts a "gains" field) and the
bridge running. For each candidate (kp, kd, ki) it: moves to base, commands a
step (default joint 0, +step deg) WITH those gains, measures the response, then
returns to base. Ranks candidates by a weighted cost and reports the best.

    cost = overshoot_pct + 3*settling_s + 30*|sse_deg|

Usage:
    python3 tune_gains.py                 # default candidate set, joint 0, 20 deg
    python3 tune_gains.py --joint 0 --step-deg 20 --duration 6
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS, rad_to_linuxcnc_deg

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]
SETTLE_BAND_DEG = 0.5  # practical settling band for this robot

# Curated candidate gains (scalar Kp/Kd/Ki). Baseline first.
DEFAULT_CANDIDATES = [
    {"kp": 0.5, "kd": 0.1, "ki": 0.05},   # baseline
    {"kp": 0.5, "kd": 0.3, "ki": 0.05},   # more damping
    {"kp": 0.5, "kd": 0.6, "ki": 0.05},
    {"kp": 0.5, "kd": 1.0, "ki": 0.05},
    {"kp": 0.5, "kd": 0.6, "ki": 0.15},   # damping + more integral (kill SSE)
    {"kp": 0.5, "kd": 0.6, "ki": 0.30},
    {"kp": 0.5, "kd": 1.0, "ki": 0.30},
    {"kp": 0.8, "kd": 0.6, "ki": 0.20},   # a bit more proportional
]


def step_metrics(t, y, y0, target):
    step = target - y0
    if abs(step) < 1e-6:
        return None
    band = max(0.02 * abs(step), SETTLE_BAND_DEG)
    osh = (np.max(y) - target) if step > 0 else (target - np.min(y))
    osh_pct = 100.0 * max(0.0, float(osh)) / abs(step)
    outside = np.abs(y - target) > band
    settle = float(t[np.where(outside)[0][-1]]) if outside.any() else 0.0
    tail = y[t >= (t[-1] - 0.5)]
    sse = target - float(np.mean(tail)) if len(tail) else float("nan")
    return dict(step=float(step), overshoot_pct=osh_pct, settling_s=settle, sse_deg=float(sse))


def cost_of(m):
    return m["overshoot_pct"] + 3.0 * m["settling_s"] + 30.0 * abs(m["sse_deg"])


class Tuner(Node):
    def __init__(self, joint, log_path):
        super().__init__("mycobot_tuner")
        self._j = joint
        self._log = open(log_path, "w", buffering=1)
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._status = None
        self._collect = False
        self._buf = []  # (t_sync, deg_j)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_js(self, msg: JointState):
        if msg.position and self._collect:
            self._buf.append((self._now(), float(rad_to_linuxcnc_deg(msg.position)[self._j])))

    def _on_status(self, msg: String):
        try:
            self._status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _send(self, target_deg, duration, gains):
        cmd = {"target_deg": [float(v) for v in target_deg], "duration": float(duration),
               "controller": "pid", "gains": gains}
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(m)
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "command",
                                    "target_deg": cmd["target_deg"], "gains": gains}) + "\n")

    def _wait_settled(self, target, duration, pos_tol=1.0):
        deadline = time.time() + duration + 3.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            st = self._status
            if st and st.get("state") == "done" and st.get("target_deg"):
                if all(abs(a - b) < pos_tol for a, b in zip(st["target_deg"], target)):
                    return
        return

    def run_candidate(self, base, step_deg, duration, gains):
        # to base with these gains, then measure the step
        self._send(base, duration, gains)
        self._wait_settled(base, duration)
        target = list(base); target[self._j] = base[self._j] + step_deg
        self._buf = []
        self._collect = True
        self._send(target, duration, gains)
        self._wait_settled(target, duration)
        # collect a touch longer to capture steady state
        t_extra = time.time() + 1.0
        while rclpy.ok() and time.time() < t_extra:
            rclpy.spin_once(self, timeout_sec=0.05)
        self._collect = False
        # return to base
        self._send(base, duration, gains)
        self._wait_settled(base, duration)
        if len(self._buf) < 10:
            return None
        arr = np.array(self._buf)
        t = arr[:, 0] - arr[0, 0]
        y = arr[:, 1]
        m = step_metrics(t, y, y[0], target[self._j])
        if m:
            self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "result",
                                        "gains": gains, "metrics": m, "cost": cost_of(m)}) + "\n")
        return m

    def close(self):
        self._log.close()


def main():
    ap = argparse.ArgumentParser(description="Sweep PID gains to minimize overshoot/settling/SSE.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--step-deg", type=float, default=20.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    ap.add_argument("--candidates", default=None,
                    help='JSON list of gain dicts, e.g. \'[{"kp":0.5,"kd":1.0,"ki":0.1}]\'')
    args = ap.parse_args()

    candidates = json.loads(args.candidates) if args.candidates else DEFAULT_CANDIDATES

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"tune_gains_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = Tuner(args.joint, out)
    results = []
    try:
        for i, g in enumerate(candidates):
            node.get_logger().info(f"[{i+1}/{len(candidates)}] testing gains {g}")
            m = node.run_candidate(base, args.step_deg, args.duration, g)
            if m:
                results.append((g, m, cost_of(m)))
                node.get_logger().info(
                    f"   overshoot={m['overshoot_pct']:.1f}%  settle={m['settling_s']:.2f}s  "
                    f"sse={m['sse_deg']:+.2f}deg  cost={cost_of(m):.1f}")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not results:
        print("no results")
        return
    results.sort(key=lambda r: r[2])
    print(f"\n=== gain sweep on joint {args.joint}, {args.step_deg:.0f} deg step ===")
    print(f"{'Kp':>5}{'Kd':>6}{'Ki':>7}{'over%':>8}{'settle':>8}{'sse':>8}{'cost':>8}")
    for g, m, c in results:
        print(f"{g['kp']:>5}{g['kd']:>6}{g['ki']:>7}{m['overshoot_pct']:>8.1f}"
              f"{m['settling_s']:>8.2f}{m['sse_deg']:>+8.2f}{c:>8.1f}")
    best = results[0]
    print(f"\nBEST: Kp={best[0]['kp']} Kd={best[0]['kd']} Ki={best[0]['ki']}  -> "
          f"overshoot={best[1]['overshoot_pct']:.1f}%  settle={best[1]['settling_s']:.2f}s  "
          f"sse={best[1]['sse_deg']:+.2f}deg  (cost {best[2]:.1f})")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
