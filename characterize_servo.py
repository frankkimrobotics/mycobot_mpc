#!/usr/bin/env python3
"""
characterize_servo :: low-level (drive) velocity-loop step characterization.

The myCobot drives are velocity-commanded (a direct pos_cmd with vel_cmd=0 does
not move them). A normal PID move saturates the velocity command, so the rising
edge of velfb is the drive's velocity-loop step response.

For each requested joint this script: moves the FULL arm to the default base
pose (calibrate_perturb pose), perturbs ONE joint by +amp (clamped to +/-25 deg
and the soft limits, all other joints held at base), captures all HAL feedback
pins from /mycobot/drive_feedback (posfb/velfb/torqfb) at ~100 Hz on the synced
clock, then RETURNS the joint to base. Fits per joint:

  * velocity-loop rise (10->90%) -> bandwidth  BW ~= 0.35 / t_rise
  * peak/plateau velocity, peak torque, position move

Log -> logs/servo_char_<stamp>.jsonl  (one run, all joints).

Usage:
    python3 characterize_servo.py --joints 1 2 3 4 5 --amplitude 15
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

from joint_conventions import MAX_JOINTS, LINUXCNC_SOFT_LIMITS_DEG

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]
PERTURB_LIMIT_DEG = 25.0


class ServoChar(Node):
    def __init__(self, log_path):
        super().__init__("mycobot_servo_char")
        self._j = 0
        self._log = open(log_path, "w", buffering=1)
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 200)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._status = None
        self._collect = False
        self._buf = []

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

    def _spin(self, secs):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _move(self, target, duration):
        """Publish a move and wait for status 'done' (or timeout)."""
        cmd = {"target_deg": [float(v) for v in target], "duration": float(duration),
               "controller": "pid"}
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "command", **cmd}) + "\n")
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._status = None
        self._pub.publish(m)
        deadline = time.time() + duration + 3.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._status and self._status.get("state") == "done":
                return

    def characterize_joint(self, base, j, amp, step_dur, base_dur=6.0):
        self._j = j
        self._move(base, base_dur)      # ensure at base
        self._spin(0.8)                 # settle
        self._buf = []                  # clean capture window
        self._collect = True
        target = list(base); target[j] = base[j] + amp
        self._move(target, step_dur)    # capture the step
        self._spin(0.3)
        self._collect = False
        arr = np.array(self._buf) if len(self._buf) > 20 else None
        self._move(base, base_dur)      # ALWAYS return to base
        return arr

    def close(self):
        self._log.close()


def velocity_loop_metrics(t, vel, pos, torq):
    t = t - t[0]
    av = np.abs(vel)
    peak_v = float(np.max(av))
    if peak_v < 1.0:
        return None
    sign = 1.0 if np.max(vel) >= abs(np.min(vel)) else -1.0
    sv = vel * sign
    start_idx = int(np.argmax(av > 0.1 * peak_v))
    t0 = t[start_idx]
    cruise = sv[sv > 0.8 * peak_v]
    plateau = float(np.median(cruise)) if len(cruise) else peak_v

    def cross(frac):
        thr = frac * plateau
        idx = np.where((t >= t0) & (sv >= thr))[0]
        return float(t[idx[0]]) if len(idx) else float("nan")
    t10, t90 = cross(0.1), cross(0.9)
    rise = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    bw = (0.35 / rise) if rise and np.isfinite(rise) and rise > 0 else float("nan")
    return dict(peak_vel_deg_s=peak_v, plateau_vel_deg_s=plateau, vel_rise_s=rise,
                vel_bw_hz=bw, peak_torque=float(np.max(np.abs(torq))),
                pos_move_deg=float(pos[-1] - pos[0]))


def main():
    ap = argparse.ArgumentParser(description="Low-level velocity-loop step characterization (per joint).")
    ap.add_argument("--joints", type=int, nargs="+", default=[0])
    ap.add_argument("--amplitude", type=float, default=15.0, help="step (deg), clamped to +/-25 + soft limits")
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(HERE, "logs", f"servo_char_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = ServoChar(out)
    results = {}
    try:
        for j in args.joints:
            amp = max(-PERTURB_LIMIT_DEG, min(PERTURB_LIMIT_DEG, args.amplitude))
            lo, hi = LINUXCNC_SOFT_LIMITS_DEG[j]
            amp = max(lo, min(hi, base[j] + amp)) - base[j]
            node.get_logger().info(f"joint {j}: step {amp:+.1f} deg from base")
            arr = node.characterize_joint(base, j, amp, args.duration)
            if arr is None:
                node.get_logger().warn(f"joint {j}: insufficient feedback")
                continue
            m = velocity_loop_metrics(arr[:, 0], arr[:, 2], arr[:, 1], arr[:, 3])
            if m:
                m["amp_deg"] = amp
                results[j] = m
                node.get_logger().info(
                    f"  J{j}: plateau {m['plateau_vel_deg_s']:.1f} deg/s, rise "
                    f"{m['vel_rise_s']*1e3:.0f} ms, BW~{m['vel_bw_hz']:.1f} Hz, "
                    f"peak torq {m['peak_torque']:.3f}")
                node._log.write(json.dumps({"type": "metrics", "joint": j, **m}) + "\n")
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n=== velocity-loop step response per joint ({args.amplitude:.0f} deg) ===")
    print(f"{'joint':>6}{'plateau v':>11}{'rise ms':>9}{'BW Hz':>8}{'pk torq':>9}{'move deg':>9}")
    for j in sorted(results):
        m = results[j]
        print(f"{('J'+str(j)):>6}{m['plateau_vel_deg_s']:>11.1f}{m['vel_rise_s']*1e3:>9.0f}"
              f"{m['vel_bw_hz']:>8.1f}{m['peak_torque']:>9.4f}{m['pos_move_deg']:>9.2f}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
