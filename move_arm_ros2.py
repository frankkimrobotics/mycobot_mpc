#!/usr/bin/env python3
"""
move_arm_ros2 :: command the myCobot Pro 630 arm over ROS 2.

Publishes a single move command to the bridge node (mycobot_ros2_bridge.py) and
optionally waits until the controller reports the move is done.

Examples
--------
    # move to an explicit LinuxCNC joint pose (degrees)
    python3 move_arm_ros2.py --deg -90 -90 0 -90 0 0 --duration 3

    # go to the home/upright pose
    python3 move_arm_ros2.py --home

    # pick a controller and wait for completion
    python3 move_arm_ros2.py --deg 0 -90 0 -90 0 0 --controller mpc --wait

SAFETY: this moves a real robot arm. Make sure the workspace is clear and an
e-stop is within reach before running.
"""

import argparse
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS, HOME_LINUXCNC_DEG


class Mover(Node):
    def __init__(self):
        super().__init__("mycobot_mover")
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self._last_status = None
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)

    def _on_status(self, msg: String):
        try:
            self._last_status = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def send(self, target_deg, duration, controller):
        cmd = {"target_deg": [float(v) for v in target_deg], "duration": float(duration)}
        if controller:
            cmd["controller"] = controller
        msg = String()
        msg.data = json.dumps(cmd)
        # publishers need a moment for discovery before the first message lands
        for _ in range(20):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(msg)
        self.get_logger().info(f"sent move: {cmd}")

    def wait_done(self, timeout=30.0):
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._last_status and self._last_status.get("state") == "done":
                self.get_logger().info(
                    f"done: err={self._last_status.get('error_norm')} "
                    f"reason={self._last_status.get('done_reason')}"
                )
                return True
        self.get_logger().warn("timed out waiting for 'done' status")
        return False


def main():
    p = argparse.ArgumentParser(description="Move the myCobot Pro 630 over ROS 2.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--deg", nargs=MAX_JOINTS, type=float, metavar="J",
                   help=f"{MAX_JOINTS} target joint angles in LinuxCNC degrees")
    g.add_argument("--home", action="store_true", help="move to the home pose")
    p.add_argument("--duration", type=float, default=2.0, help="move duration (s)")
    p.add_argument("--controller", default=None, choices=["pid", "invdyn", "pd_velff", "mpc"])
    p.add_argument("--wait", action="store_true", help="block until the move reports done")
    args = p.parse_args()

    target = list(HOME_LINUXCNC_DEG) if args.home else args.deg

    rclpy.init()
    node = Mover()
    rc = 0
    try:
        node.send(target, args.duration, args.controller)
        if args.wait:
            rc = 0 if node.wait_done() else 2
        else:
            # let the message flush before exiting
            for _ in range(5):
                rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
