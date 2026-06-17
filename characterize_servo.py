#!/usr/bin/env python3
"""
characterize_servo :: low-level (drive) step-response characterization.

Commands a direct position-command STEP on one joint with the Python PID
*bypassed* (robot_hal.py raw_step mode), so what's measured is the servo
drive's own closed-loop response. Records all HAL feedback pins from
/mycobot/drive_feedback (position=posfb deg, velocity=velfb deg/s,
effort=torqfb) under the synchronized clock, then fits step metrics:

  * rise time (10->90%)   * overshoot %   * settling time (2%)
  * estimated bandwidth   omega_n ~= 1.8 / t_rise  (rad/s),  f ~= omega_n/2pi

Log -> logs/servo_char_<stamp>.jsonl. Use small amplitudes (1-3 deg).

Usage:
    python3 characterize_servo.py --joint 0 --amplitude 2 --hold 2.5
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


class ServoChar(Node):
    def __init__(self, joint, log_path):
        super().__init__("mycobot_servo_char")
        self._j = joint
        self._log = open(log_path, "w", buffering=1)
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 200)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._status = None
        self._collect = False
        self._buf = []  # (t, posfb, velfb, torqfb) for the joint

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, msg: String):
        try:
            self._status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_fb(self, msg: JointState):
        if not msg.position or len(msg.position) <= self._j:
            return
        p = msg.position[self._j]
        v = msg.velocity[self._j] if len(msg.velocity) > self._j else 0.0
        e = msg.effort[self._j] if len(msg.effort) > self._j else 0.0
        rec = {"t_sync": round(self._now(), 6), "type": "drive_feedback", "joint": self._j,
               "posfb": round(p, 6), "velfb": round(v, 6), "torqfb": round(e, 8)}
        self._log.write(json.dumps(rec) + "\n")
        if self._collect:
            self._buf.append((rec["t_sync"], p, v, e))

    def run_step(self, amp, hold, settle):
        spec = {"joint": self._j, "amplitude_deg": amp, "hold_sec": hold,
                "settle_sec": settle, "pre_sec": 0.5}
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "command",
                                    "raw_step": spec}) + "\n")
        m = String(); m.data = json.dumps({"raw_step": spec})
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._buf = []
        self._collect = True
        self._pub.publish(m)
        # collect through the whole raw_step (pre + hold + settle + margin)
        deadline = time.time() + 0.5 + hold + settle + 2.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
        self._collect = False
        return np.array(self._buf) if len(self._buf) > 10 else None

    def close(self):
        self._log.close()


def step_metrics(t, y):
    """t (s), y posfb; auto-detects the step from the baseline pre-hold."""
    t = t - t[0]
    y0 = float(np.median(y[t < 0.4]))            # baseline before step
    yf = float(np.median(y[(t > t[-1] - 0.6) & (t < t[-1] - 0.1)]))  # before return... use plateau
    # plateau = the held-step region: take median of the middle third
    mid = y[(t > 0.6) & (t < 0.6 + (t[-1] - 0.6) * 0.5)]
    plateau = float(np.median(mid)) if len(mid) else yf
    step = plateau - y0
    if abs(step) < 0.05:
        return None
    # restrict to the rising window: from step command (~0.5s) to plateau
    win = (t >= 0.5)
    tw, yw = t[win] - 0.5, y[win]
    # normalize
    yn = (yw - y0) / step
    # rise 10->90%
    def crossing(frac):
        idx = np.where(yn >= frac)[0]
        return float(tw[idx[0]]) if len(idx) else float("nan")
    t10, t90 = crossing(0.1), crossing(0.9)
    rise = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    peak = float(np.max(yn[tw < (tw[-1] * 0.6)])) if len(yn) else 1.0
    overshoot = max(0.0, (peak - 1.0)) * 100.0
    band = 0.02
    outside = np.abs(yn - 1.0) > band
    settle = float(tw[np.where(outside & (tw < tw[-1] * 0.7))[0][-1]]) if outside.any() else 0.0
    wn = (1.8 / rise) if rise and np.isfinite(rise) and rise > 0 else float("nan")
    return dict(step_deg=step, rise_s=rise, overshoot_pct=overshoot, settling_s=settle,
                omega_n_rad_s=wn, bw_hz=(wn / (2 * np.pi) if np.isfinite(wn) else float("nan")))


def main():
    ap = argparse.ArgumentParser(description="Low-level servo step/bandwidth characterization.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=2.0, help="step size (deg), keep small")
    ap.add_argument("--hold", type=float, default=2.5)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(HERE, "logs", f"servo_char_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = ServoChar(args.joint, out)
    try:
        node.get_logger().info(f"raw step: joint {args.joint}, {args.amplitude} deg")
        arr = node.run_step(args.amplitude, args.hold, args.settle)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if arr is None:
        print("no/insufficient feedback captured")
        return
    m = step_metrics(arr[:, 0], arr[:, 1])
    print(f"\n=== servo step response: joint {args.joint}, {args.amplitude} deg ===")
    if m:
        print(f"  rise (10-90%) : {m['rise_s']*1e3:7.1f} ms")
        print(f"  overshoot     : {m['overshoot_pct']:7.1f} %")
        print(f"  settling (2%) : {m['settling_s']*1e3:7.1f} ms")
        print(f"  bandwidth est : {m['bw_hz']:7.2f} Hz  (omega_n ~= {m['omega_n_rad_s']:.1f} rad/s)")
        with open(out, "a") as f:
            f.write(json.dumps({"type": "metrics", "joint": args.joint, **m}) + "\n")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
