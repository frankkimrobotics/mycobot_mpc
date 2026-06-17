#!/usr/bin/env python3
"""
characterize_servo :: low-level (drive) characterization from the velocity-loop
step response.

The myCobot drives are velocity-commanded (a direct pos_cmd with vel_cmd=0 does
not move them). A normal PID move saturates the velocity command to max almost
instantly, so the rising edge of velfb IS the drive's velocity-loop step
response. This script commands a step on one joint, captures all HAL feedback
pins from /mycobot/drive_feedback (posfb/velfb/torqfb) at ~100 Hz under the
synchronized clock, and fits:

  * velocity-loop rise time (10->90%)  ->  bandwidth  BW ~= 0.35 / t_rise
  * peak velocity, peak torque
  * position move + settling

Log -> logs/servo_char_j<joint>_<stamp>.jsonl (+ plot via the analysis).

Usage: python3 characterize_servo.py --joint 0 --amplitude 15 --duration 3
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
# Default base pose (LinuxCNC deg) - same as calibrate_perturb.py.
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]
PERTURB_LIMIT_DEG = 25.0  # never perturb a joint more than this from base


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

    def go_to_base(self, base, duration=8.0):
        """Move the FULL arm to the base pose and wait for it to settle."""
        cmd = {"target_deg": [float(v) for v in base], "duration": float(duration),
               "controller": "pid"}
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(m)
        deadline = time.time() + duration + 3.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._status and self._status.get("state") == "done":
                return

    def run_step(self, base, amp, duration):
        target = list(base); target[self._j] = base[self._j] + amp
        cmd = {"target_deg": [float(v) for v in target], "duration": float(duration),
               "controller": "pid"}
        self._log.write(json.dumps({"t_sync": round(self._now(), 6), "type": "command",
                                    **cmd}) + "\n")
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._buf = []
        self._collect = True
        self._pub.publish(m)
        deadline = time.time() + duration + 3.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self._status and self._status.get("state") == "done":
                # grab a little tail then stop
                t_end = time.time() + 0.5
                while rclpy.ok() and time.time() < t_end:
                    rclpy.spin_once(self, timeout_sec=0.02)
                break
        self._collect = False
        return np.array(self._buf) if len(self._buf) > 20 else None

    def close(self):
        self._log.close()


def velocity_loop_metrics(t, vel, pos, torq):
    """Fit the velocity-loop step response from the leading edge of |velfb|."""
    t = t - t[0]
    av = np.abs(vel)
    peak_v = float(np.max(av))
    if peak_v < 1.0:
        return None
    sign = 1.0 if np.max(vel) >= abs(np.min(vel)) else -1.0
    sv = vel * sign  # make the motion positive
    # move start = first time |vel| exceeds 10% of peak
    start_idx = int(np.argmax(av > 0.1 * peak_v))
    t0 = t[start_idx]
    # plateau velocity = median of samples within 80-100% of peak (the cruise)
    cruise = sv[sv > 0.8 * peak_v]
    plateau = float(np.median(cruise)) if len(cruise) else peak_v

    def cross(frac):
        thr = frac * plateau
        idx = np.where((t >= t0) & (sv >= thr))[0]
        return float(t[idx[0]]) if len(idx) else float("nan")
    t10, t90 = cross(0.1), cross(0.9)
    rise = t90 - t10 if np.isfinite(t10) and np.isfinite(t90) else float("nan")
    bw = (0.35 / rise) if rise and np.isfinite(rise) and rise > 0 else float("nan")
    return dict(
        peak_vel_deg_s=peak_v,
        plateau_vel_deg_s=plateau,
        vel_rise_s=rise,
        vel_bw_hz=bw,
        peak_torque=float(np.max(np.abs(torq))),
        pos_move_deg=float(pos[-1] - pos[0]),
    )


def main():
    ap = argparse.ArgumentParser(description="Low-level velocity-loop step characterization.")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=15.0, help="step size (deg); large enough to saturate velocity")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG,
                    help="default pose to start from (LinuxCNC deg)")
    ap.add_argument("--base-duration", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(HERE, "logs", f"servo_char_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    base = [float(v) for v in args.base]
    # Safety: clamp perturbation to +/-PERTURB_LIMIT_DEG of base AND the joint
    # soft limits. Other joints are always held at base (run_step copies base).
    amp = max(-PERTURB_LIMIT_DEG, min(PERTURB_LIMIT_DEG, args.amplitude))
    lo, hi = LINUXCNC_SOFT_LIMITS_DEG[args.joint]
    tgt = max(lo, min(hi, base[args.joint] + amp))
    amp = tgt - base[args.joint]
    if abs(amp - args.amplitude) > 1e-6:
        print(f"amplitude clamped to {amp:+.1f} deg (limit +/-{PERTURB_LIMIT_DEG}, soft [{lo},{hi}])")

    rclpy.init()
    node = ServoChar(args.joint, out)
    try:
        node.get_logger().info(f"moving to default base pose {base}")
        node.go_to_base(base, duration=args.base_duration)
        node.get_logger().info(f"step: joint {args.joint}, {amp:+.1f} deg from base")
        arr = node.run_step(base, amp, args.duration)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if arr is None:
        print("no/insufficient feedback captured")
        return
    m = velocity_loop_metrics(arr[:, 0], arr[:, 2], arr[:, 1], arr[:, 3])
    print(f"\n=== velocity-loop step response: joint {args.joint}, {args.amplitude} deg ===")
    if m:
        print(f"  peak velocity    : {m['peak_vel_deg_s']:7.1f} deg/s")
        print(f"  plateau velocity : {m['plateau_vel_deg_s']:7.1f} deg/s")
        print(f"  vel rise (10-90%): {m['vel_rise_s']*1e3:7.1f} ms")
        print(f"  vel-loop BW est  : {m['vel_bw_hz']:7.2f} Hz")
        print(f"  peak torque fb   : {m['peak_torque']:7.4f}")
        print(f"  position move    : {m['pos_move_deg']:7.2f} deg")
        with open(out, "a") as f:
            f.write(json.dumps({"type": "metrics", "joint": args.joint, **m}) + "\n")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
