#!/usr/bin/env python3
"""
sinusoidal_controller :: a ROS 2 controller node that generates a SINUSOIDAL
joint trajectory bounded to +/-20 deg of the default base pose, publishes the
waypoints, and streams the trajectory to the robot (via the bridge).

Each joint j follows:   q_j(t) = base_j + A_j * sin(2*pi*f*t)
with A_j clamped to <= 20 deg, so every waypoint stays within 20 deg of the
default pose. The trajectory starts and ends at base (integer periods).

- Publishes each waypoint on /controller/waypoint (sensor_msgs/JointState,
  position in LinuxCNC degrees), streamed at the trajectory rate.
- Sends the whole trajectory to /mycobot/cmd/move as a robot_hal trajectory
  command (target_deg + trajectory + traj_dt), which robot_hal tracks in one
  realtime control loop.

Usage:
    python3 sinusoidal_controller.py --amplitude 15 --freq 0.15 --periods 2
    python3 sinusoidal_controller.py --joints 0 3 5 --amplitude 20   # subset
    python3 sinusoidal_controller.py --no-send       # publish waypoints only

SAFETY: moves the arm in a gentle bounded oscillation. Keep the workspace clear.
"""
import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS, JOINT_NAMES

DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]
PERTURB_LIMIT_DEG = 20.0  # waypoints must stay within this of base, per joint


class SinusoidalController(Node):
    def __init__(self, base, amp, freq, periods, traj_dt, joints):
        super().__init__("sinusoidal_controller")
        self.base = [float(v) for v in base]
        # clamp every amplitude to <= 20 deg so |waypoint - base| <= 20
        self.amp = [min(PERTURB_LIMIT_DEG, abs(a)) for a in amp]
        self.freq = float(freq)
        self.traj_dt = float(traj_dt)
        self.joints = joints

        self.pub_wp = self.create_publisher(JointState, "/controller/waypoint", 10)
        self.pub_cmd = self.create_publisher(String, "/mycobot/cmd/move", 10)

        # build trajectory: integer number of periods -> starts and ends at base
        n = max(1, int(round(periods / (self.freq * self.traj_dt))))
        self.traj = []
        for k in range(n + 1):
            t = k * self.traj_dt
            pose = list(self.base)
            for j in self.joints:
                pose[j] = self.base[j] + self.amp[j] * math.sin(2 * math.pi * self.freq * t)
            self.traj.append([round(p, 3) for p in pose])

        max_dev = max(abs(p[j] - self.base[j]) for p in self.traj for j in self.joints) if self.joints else 0.0
        v_max = max(self.amp[j] for j in self.joints) * 2 * math.pi * self.freq if self.joints else 0.0
        self.get_logger().info(
            f"sinusoid: joints={self.joints} amp={[self.amp[j] for j in self.joints]} deg "
            f"freq={self.freq} Hz periods={periods} -> {len(self.traj)} pts @ {self.traj_dt}s "
            f"(max dev {max_dev:.1f} deg <= {PERTURB_LIMIT_DEG}, peak speed ~{v_max:.1f} deg/s)")

    def send_to_robot(self):
        for _ in range(30):
            if self.pub_cmd.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        cmd = {"target_deg": self.base, "trajectory": self.traj,
               "traj_dt": self.traj_dt, "controller": "pid"}
        self.pub_cmd.publish(String(data=json.dumps(cmd)))
        self.get_logger().info(f"sent trajectory ({len(self.traj)} pts) to /mycobot/cmd/move")

    def stream_waypoints(self):
        """Publish each waypoint at the trajectory rate (in sync with execution)."""
        for pose in self.traj:
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(JOINT_NAMES)
            js.position = [float(p) for p in pose]  # LinuxCNC degrees
            self.pub_wp.publish(js)
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < self.traj_dt:
                rclpy.spin_once(self, timeout_sec=0.005)


def main():
    ap = argparse.ArgumentParser(description="Sinusoidal joint-trajectory controller node (+/-20 deg of base).")
    ap.add_argument("--amplitude", type=float, default=15.0, help="amplitude per joint (deg, clamped <=20)")
    ap.add_argument("--freq", type=float, default=0.15, help="frequency (Hz)")
    ap.add_argument("--periods", type=float, default=2.0)
    ap.add_argument("--traj-dt", type=float, default=0.05, help="trajectory sample interval (s)")
    ap.add_argument("--joints", type=int, nargs="+", default=list(range(MAX_JOINTS)),
                    help="which joints oscillate (default all)")
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    ap.add_argument("--no-send", action="store_true", help="publish waypoints only; do not move the robot")
    args = ap.parse_args()

    amp = [args.amplitude if j in args.joints else 0.0 for j in range(MAX_JOINTS)]

    rclpy.init()
    node = SinusoidalController(args.base, amp, args.freq, args.periods, args.traj_dt, args.joints)
    try:
        if not args.no_send:
            node.send_to_robot()
        node.stream_waypoints()
        node.get_logger().info("trajectory streaming complete (arm returns to base)")
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
