#!/usr/bin/env python3
"""
calibrate_ros2 :: run the calibrate_perturb.py calibration THROUGH the ROS 2
bridge and log command input + robot response under a synchronized timestamp.

This is the integration test for mycobot_ros2_bridge.py. Instead of talking raw
TCP (like calibrate_perturb.py), it commands the arm over ROS 2 topics and
records both sides of every move on one synchronized clock:

  * command  -> published to /mycobot/cmd/move, logged with the desktop clock
                (NTP/chrony-synced to the Pi)
  * response -> /joint_states (URDF rad) + /mycobot/status (deg, state), logged
                with the ROBOT's source timestamp carried in the message header

Sequence mirrors calibrate_perturb.py: move to a measured BASE pose, then for
each joint perturb +step and -step about base, returning to base between each:

    base -> base[j]+step -> base -> base[j]-step -> base

The log is logs/calibrate_ros2_<stamp>.jsonl, one JSON object per line, each
carrying `t_sync` (epoch seconds on the shared clock) so command and response
rows interleave on the same timeline.

SAFETY: this moves a real robot arm. Clear the workspace, keep an e-stop in
reach. Default perturbation is a gentle +/-10 deg about the power-up pose.
"""

import argparse
import json
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS

# Pose the robot was measured at on power-up (LinuxCNC degrees) - same as
# calibrate_perturb.py, so the initial "move to base" is small.
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]


def build_plan(base_deg, step_deg):
    """(label, target_deg) sequence: per joint +step, return, -step, return."""
    plan = [("base", list(base_deg))]
    for j in range(MAX_JOINTS):
        for sign in (+1.0, -1.0):
            t = list(base_deg)
            t[j] = base_deg[j] + sign * step_deg
            plan.append((f"J{j}{sign*step_deg:+.0f}", t))
            plan.append((f"J{j}_return", list(base_deg)))
    return plan


class CalibLogger(Node):
    def __init__(self, log_path, controller):
        super().__init__("mycobot_calibrate")
        self._controller = controller
        self._log = open(log_path, "w", buffering=1)  # line-buffered
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._last_status = None
        self._wp = None
        self._label = None
        self.get_logger().info(f"logging to {log_path}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _stamp_to_epoch(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _write(self, record):
        self._log.write(json.dumps(record) + "\n")

    # --- responses (robot -> ROS) ---
    # t_sync is ALWAYS the desktop clock (the bridge runs here, so every row -
    # command and response - shares this single clock and is synchronized by
    # construction). t_robot preserves the Pi's source sample time; once the
    # Pi/desktop clocks are chrony-synced, t_robot matches t_sync within ~ms.
    def _on_js(self, msg: JointState):
        if not msg.position:
            return
        self._write({
            "t_sync": round(self._now(), 6),
            "t_robot": round(self._stamp_to_epoch(msg.header.stamp), 6),
            "type": "response", "source": "joint_states",
            "waypoint": self._wp, "label": self._label,
            "joints_rad": [round(v, 5) for v in msg.position],
        })

    def _on_status(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self._last_status = obj
        self._write({
            "t_sync": round(self._now(), 6),
            "type": "response", "source": "status",
            "waypoint": self._wp, "label": self._label,
            "state": obj.get("state"), "current_deg": obj.get("current_deg"),
            "target_deg": obj.get("target_deg"), "error_norm": obj.get("error_norm"),
        })

    # --- commands (ROS -> robot), stamped with the shared clock ---
    def send_waypoint(self, wp_id, label, target_deg, duration):
        self._wp, self._label = wp_id, label
        cmd = {"target_deg": [round(float(v), 3) for v in target_deg],
               "duration": float(duration), "controller": self._controller}
        self._write({
            "t_sync": round(self._now(), 6),
            "type": "command", "waypoint": wp_id, "label": label,
            "target_deg": cmd["target_deg"], "controller": self._controller,
            "duration": duration,
        })
        m = String(); m.data = json.dumps(cmd)
        for _ in range(30):  # wait for the bridge to subscribe
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(m)
        self.get_logger().info(f"wp{wp_id} {label} -> {cmd['target_deg']}")

    def wait_settled(self, target_deg, duration, pos_tol=1.0):
        deadline = time.time() + duration + 3.0
        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            st = self._last_status
            if st and st.get("state") == "done" and st.get("target_deg"):
                if all(abs(a - b) < pos_tol for a, b in zip(st["target_deg"], target_deg)):
                    return True
        return False

    def close(self):
        self._log.close()


def main():
    ap = argparse.ArgumentParser(description="ROS 2 perturbation calibration with synced-timestamp logging.")
    ap.add_argument("--controller", default="pid", choices=["pid", "invdyn", "pd_velff", "mpc"])
    ap.add_argument("--step-deg", type=float, default=10.0, help="perturbation magnitude per joint (deg)")
    ap.add_argument("--duration", type=float, default=6.0, help="max move time per waypoint (s)")
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG,
                    help=f"base pose in LinuxCNC deg (default: {DEFAULT_BASE_DEG})")
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(MAX_JOINTS)),
                    help="restrict to these joints (default all)")
    ap.add_argument("--quick", action="store_true", help="smoke test: joint 0 only, +/-5 deg")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only; do not move/log")
    ap.add_argument("--out", default=None, help="log path (default logs/calibrate_ros2_<stamp>.jsonl)")
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    step = 5.0 if args.quick else args.step_deg
    plan = build_plan(base, step)
    if args.quick:
        plan = [("base", base)] + [p for p in plan if p[0].startswith("J0")]
    elif args.joints != list(range(MAX_JOINTS)):
        plan = [("base", base)] + [p for p in plan
                                   if any(p[0].startswith(f"J{j}") for j in args.joints)]

    print(f"Perturbation calibration: +/-{step:.0f} deg about base {base}")
    print(f"  {len(plan)} moves, controller={args.controller}, duration={args.duration}s")
    if args.dry_run:
        for label, t in plan:
            print(f"  {label:<12} -> {[round(v,1) for v in t]}")
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "logs", f"calibrate_ros2_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = CalibLogger(out, args.controller)
    node._write({"t_sync": round(node._now(), 6), "type": "session_start",
                 "base_deg": base, "step_deg": step, "controller": args.controller,
                 "duration": args.duration, "n_moves": len(plan)})
    wp = 0
    try:
        for label, target in plan:
            node.send_waypoint(wp, label, target, args.duration)
            node.wait_settled(target, args.duration)
            wp += 1
        node._write({"t_sync": round(node._now(), 6), "type": "session_end", "n_waypoints": wp})
        node.get_logger().info(f"calibration complete: {wp} moves -> {out}")
    except KeyboardInterrupt:
        node._write({"t_sync": round(node._now(), 6), "type": "aborted", "n_waypoints": wp})
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
