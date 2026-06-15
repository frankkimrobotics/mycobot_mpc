#!/usr/bin/env python3
"""
Desktop-side script: launch rviz2 + stream robot joint poses from the robot controller.

Runs everything in a single command:
  1. Launches robot_state_publisher + rviz2 (via display.launch.py with use_gui:=false)
  2. Connects to the robot's TCP streaming server (integrated in robot_hal.py)
  3. Publishes received joint angles to /joint_states for rviz2 visualization

Usage:
    python3 robot_pose_stream_ros2.py --host 10.0.0.27
    python3 robot_pose_stream_ros2.py --host 10.0.0.27 --port 9999
    python3 robot_pose_stream_ros2.py --host 127.0.0.1 --no-rviz   # client only (rviz2 already running)

Requires ROS2 environment to be sourced before running:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_env && \\
    source $CONDA_PREFIX/setup.zsh && source ~/ros2_ws/install/setup.zsh && \\
    python3 robot_pose_stream_ros2.py --host 10.0.0.27
"""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time

# Joint conventions (counts, names, LinuxCNC↔URDF calibration) are shared.
from joint_conventions import (
    MAX_JOINTS,
    JOINT_NAMES,
    JOINT_SIGNS,
    JOINT_OFFSETS_DEG,
)


def launch_rviz2():
    """Launch display.launch.py as a subprocess (robot_state_publisher + rviz2).

    Returns the Popen object. The subprocess inherits the current ROS2 environment.
    """
    cmd = [
        "ros2", "launch", "mycobot_description", "display.launch.py",
        "use_gui:=false",
    ]
    print(f"[rviz2] Launching: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"[rviz2] Started (PID {proc.pid}), waiting for initialization...")
    time.sleep(3.0)
    return proc


def run_client(host: str, port: int):
    """Connect to robot streaming server, publish joint angles to /joint_states."""
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("robot_pose_streamer")
    pub = node.create_publisher(JointState, "/joint_states", 10)
    clock = node.get_clock()

    node.get_logger().info(f"Connecting to robot at {host}:{port}")
    node.get_logger().info(f"Calibration offsets: {JOINT_OFFSETS_DEG}, signs: {JOINT_SIGNS}")

    msg_count = 0

    while rclpy.ok():
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            node.get_logger().info(f"Connected to {host}:{port}")

            buffer = ""
            sock.settimeout(2.0)

            while rclpy.ok():
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    node.get_logger().warn("Server closed connection.")
                    break

                buffer += data.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as e:
                        node.get_logger().warn(f"JSON decode error: {e}")
                        continue

                    joints_deg = payload.get("joints_deg")
                    if joints_deg is None or len(joints_deg) != MAX_JOINTS:
                        continue

                    joints_rad = [
                        math.radians(JOINT_SIGNS[i] * (d + JOINT_OFFSETS_DEG[i]))
                        for i, d in enumerate(joints_deg)
                    ]

                    js = JointState()
                    js.header.stamp = clock.now().to_msg()
                    js.name = list(JOINT_NAMES)
                    js.position = joints_rad

                    pub.publish(js)
                    msg_count += 1

                    if msg_count % 200 == 1:
                        deg_str = [round(d, 1) for d in joints_deg]
                        cal_str = [
                            round(JOINT_SIGNS[i] * (joints_deg[i] + JOINT_OFFSETS_DEG[i]), 1)
                            for i in range(MAX_JOINTS)
                        ]
                        node.get_logger().info(
                            f"#{msg_count} raw={deg_str} → urdf={cal_str} deg"
                        )

        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            node.get_logger().warn(f"Connection failed: {e}")
        finally:
            if sock:
                sock.close()

        node.get_logger().info("Reconnecting in 2 seconds...")
        time.sleep(2.0)

    node.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Desktop launcher: rviz2 + robot pose streaming client (single command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full launch (rviz2 + streaming client):
  python3 robot_pose_stream_ros2.py --host 10.0.0.27

  # Client only (rviz2 already running separately):
  python3 robot_pose_stream_ros2.py --host 10.0.0.27 --no-rviz

  # Custom port:
  python3 robot_pose_stream_ros2.py --host 10.0.0.27 --port 8888
""",
    )
    parser.add_argument("--host", default=os.environ.get("ROBOT_IP"),
        help="Robot IP (default: ROBOT_IP env, e.g. 10.0.0.27)")
    parser.add_argument("--port", type=int, default=9999,
        help="TCP port for streaming (default: 9999)")
    parser.add_argument("--no-rviz", action="store_true",
        help="Don't launch rviz2 (use if it's already running)")

    args = parser.parse_args()
    if not args.host:
        parser.error("Robot host not set. Set ROBOT_IP (e.g. run ./setup_robot_ip.sh) or pass --host <ip>.")

    rviz_proc = None

    def cleanup(signum=None, frame=None):
        """Clean shutdown: kill rviz2 process group on exit."""
        if rviz_proc and rviz_proc.poll() is None:
            print("\n[rviz2] Shutting down...")
            try:
                os.killpg(os.getpgid(rviz_proc.pid), signal.SIGTERM)
                rviz_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                os.killpg(os.getpgid(rviz_proc.pid), signal.SIGKILL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        if not args.no_rviz:
            rviz_proc = launch_rviz2()

        run_client(args.host, args.port)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
