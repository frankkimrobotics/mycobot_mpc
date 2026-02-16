#!/usr/bin/env python3
"""
Stream myCobot Pro 630 joint positions from LinuxCNC (robot controller)
to ROS2 /joint_states so rviz2 reflects the real robot pose.

Architecture:
  [Robot Controller]                        [Desktop (macOS)]
   LinuxCNC HAL  ──TCP socket──>  ROS2 JointState publisher ──> rviz2
   (server mode)                  (client mode)

The two machines are connected via Ethernet cable.

Server (run on the robot controller that has LinuxCNC):
    python3 robot_pose_stream_ros2.py server --host 0.0.0.0 --port 9999

Client (run on the desktop with ROS2 + rviz2):
    python3 robot_pose_stream_ros2.py client --host <robot-ip> --port 9999

Then launch rviz2 with the joint_state_publisher_gui disabled:
    ros2 launch mycobot_description display.launch.py use_gui:=false

Mock server (for testing without LinuxCNC — publishes sine-wave motion):
    python3 robot_pose_stream_ros2.py mock-server --host 0.0.0.0 --port 9999
"""

import argparse
import json
import math
import socket
import sys
import threading
import time

MAX_JOINTS = 6
JOINT_NAMES = [f"joint{i+1}" for i in range(MAX_JOINTS)]

# Default home pose in degrees (matches mpc_hal.py init)
HOME_DEG = [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0]

# ═══════════════════════════════════════════════════════════════════════════════
#  Joint calibration: LinuxCNC → URDF mapping
# ═══════════════════════════════════════════════════════════════════════════════
#
#  urdf_angle = JOINT_SIGNS[i] * (linuxcnc_angle + JOINT_OFFSETS_DEG[i])
#
#  CALIBRATION PROCEDURE:
#    1. Power on robot, let it home (all joints report ~0° in LinuxCNC)
#    2. Run server + client, observe rviz2 vs physical robot
#    3. For each joint that looks wrong:
#       - If the joint moves the OPPOSITE direction in rviz2 → flip sign: -1
#       - If the joint has a constant angular offset → add offset in degrees
#    4. Update the arrays below and re-run
#
#  QUICK TEST: manually jog one joint at a time on the robot and watch rviz2:
#    - Joint moves same direction in rviz2? → sign is correct (+1)
#    - Joint moves opposite direction?       → flip to -1
#    - Joint resting position offset?        → adjust JOINT_OFFSETS_DEG
#
#  URDF axis directions (from mycobot_pro_630.urdf):
#    joint1: axis z=+1    joint2: axis z=-1    joint3: axis z=-1
#    joint4: axis z=-1    joint5: axis z=+1    joint6: axis x=-1
#
#  LinuxCNC encoder scale signs (from elerob_mpc.hal):
#    joint0: +    joint1: +    joint2: -
#    joint3: -    joint4: -    joint5: -

JOINT_SIGNS       = [+1, +1, +1, +1, +1, +1]  # per-joint sign: +1 or -1
JOINT_OFFSETS_DEG = [0.0, 90.0, 0.0, 90.0, 0.0, 0.0]  # per-joint offset (degrees)


# ═══════════════════════════════════════════════════════════════════════════════
#  Server mode — runs on the robot controller (LinuxCNC machine)
# ═══════════════════════════════════════════════════════════════════════════════

def run_server(host: str, port: int, rate_hz: float):
    """Read joint positions from LinuxCNC and stream them over TCP as JSON lines.

    Each line sent to connected clients:
        {"joints_deg": [j1, j2, j3, j4, j5, j6], "timestamp": <epoch>}\n
    """
    try:
        import linuxcnc
    except ImportError:
        print("ERROR: 'linuxcnc' Python module not found.")
        print("       This mode must run on the robot controller where LinuxCNC is installed.")
        sys.exit(1)

    stat = linuxcnc.stat()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    print(f"[server] Listening on {host}:{port} at {rate_hz} Hz")  # noqa: F541
    print(f"[server] Joint names: {JOINT_NAMES}")  # noqa: F541
    print("[server] Press Ctrl+C to stop.\n")

    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            try:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with clients_lock:
                    clients.append(conn)
                print(f"[server] Client connected: {addr} (total: {len(clients)})")
            except OSError:
                break

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()

    period = 1.0 / rate_hz
    loop_count = 0

    try:
        while True:
            t0 = time.time()

            # Poll LinuxCNC for current joint positions (degrees)
            try:
                stat.poll()
                joints_deg = [round(stat.joint_actual_position[i], 4)
                              for i in range(MAX_JOINTS)]
            except (RuntimeError, OSError) as e:
                print(f"[server] LinuxCNC poll error: {e}")
                time.sleep(1.0)
                continue

            msg = json.dumps({
                "joints_deg": joints_deg,
                "timestamp": time.time(),
            }) + "\n"
            msg_bytes = msg.encode("utf-8")

            # Broadcast to all connected clients
            dead = []
            with clients_lock:
                for conn in clients:
                    try:
                        conn.sendall(msg_bytes)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        dead.append(conn)
                for conn in dead:
                    clients.remove(conn)
                    conn.close()

            if dead:
                print(f"[server] {len(dead)} client(s) disconnected (remaining: {len(clients)})")

            loop_count += 1
            if loop_count % (int(rate_hz) * 5) == 0:
                print(f"[server] loop={loop_count} joints_deg={[round(j, 1) for j in joints_deg]} clients={len(clients)}")

            # Sleep for remainder of period
            elapsed = time.time() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[server] Shutting down.")
    finally:
        with clients_lock:
            for conn in clients:
                conn.close()
        server_sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Mock server — for testing without LinuxCNC (sine-wave motion)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mock_server(host: str, port: int, rate_hz: float):
    """Simulate joint motion with sinusoidal oscillation around the home pose.

    Useful for testing the client + rviz2 pipeline on the desktop without
    needing the real robot controller.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    print(f"[mock-server] Listening on {host}:{port} at {rate_hz} Hz")  # noqa: F541
    print(f"[mock-server] Generating sine-wave motion around home={HOME_DEG}")  # noqa: F541
    print("[mock-server] Press Ctrl+C to stop.\n")

    clients = []
    clients_lock = threading.Lock()

    def accept_loop():
        while True:
            try:
                conn, addr = server_sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with clients_lock:
                    clients.append(conn)
                print(f"[mock-server] Client connected: {addr}")
            except OSError:
                break

    accept_thread = threading.Thread(target=accept_loop, daemon=True)
    accept_thread.start()

    period = 1.0 / rate_hz
    t_start = time.time()

    try:
        while True:
            t0 = time.time()
            t_elapsed = t0 - t_start

            # Generate sinusoidal motion: each joint oscillates with different frequency/amplitude
            joints_deg = []
            for i in range(MAX_JOINTS):
                amplitude = 15.0  # degrees
                frequency = 0.2 + i * 0.05  # Hz (slightly different per joint)
                offset = HOME_DEG[i]
                angle = offset + amplitude * math.sin(2.0 * math.pi * frequency * t_elapsed)
                joints_deg.append(round(angle, 4))

            msg = json.dumps({
                "joints_deg": joints_deg,
                "timestamp": time.time(),
            }) + "\n"
            msg_bytes = msg.encode("utf-8")

            dead = []
            with clients_lock:
                for conn in clients:
                    try:
                        conn.sendall(msg_bytes)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        dead.append(conn)
                for conn in dead:
                    clients.remove(conn)
                    conn.close()

            elapsed = time.time() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[mock-server] Shutting down.")
    finally:
        with clients_lock:
            for conn in clients:
                conn.close()
        server_sock.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Client mode — runs on the desktop (ROS2 + rviz2 machine)
# ═══════════════════════════════════════════════════════════════════════════════

def run_client(host: str, port: int, reconnect: bool = True):
    """Connect to the robot server, receive joint angles, publish to /joint_states.

    Publishes sensor_msgs/JointState with positions in radians (URDF convention).
    Auto-reconnects if the connection drops.
    """
    import rclpy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("robot_pose_streamer")
    pub = node.create_publisher(JointState, "/joint_states", 10)
    clock = node.get_clock()

    node.get_logger().info(f"Connecting to robot server at {host}:{port}")
    node.get_logger().info(f"Publishing to /joint_states with joints: {JOINT_NAMES}")
    node.get_logger().info(
        "Make sure rviz2 is launched with: "
        "ros2 launch mycobot_description display.launch.py use_gui:=false"
    )

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

                # Process complete JSON lines
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

                    # Apply calibration: sign flip + offset, then deg → rad
                    joints_rad = [
                        math.radians(JOINT_SIGNS[i] * (d + JOINT_OFFSETS_DEG[i]))
                        for i, d in enumerate(joints_deg)
                    ]

                    # Build and publish JointState
                    js = JointState()
                    js.header.stamp = clock.now().to_msg()
                    js.name = list(JOINT_NAMES)
                    js.position = joints_rad

                    pub.publish(js)
                    msg_count += 1

                    if msg_count % 100 == 1:
                        deg_str = [round(d, 1) for d in joints_deg]
                        node.get_logger().info(
                            f"Published #{msg_count}: {deg_str} deg"
                        )

        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            node.get_logger().warn(f"Connection failed: {e}")
        finally:
            if sock:
                sock.close()

        if not reconnect:
            break

        node.get_logger().info("Reconnecting in 2 seconds...")
        time.sleep(2.0)

    node.destroy_node()
    rclpy.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Stream myCobot Pro 630 joint poses between LinuxCNC and ROS2 rviz2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # On robot controller (LinuxCNC machine):
  python3 robot_pose_stream_ros2.py server --host 0.0.0.0 --port 9999

  # On desktop (ROS2 machine), with rviz2 running:
  python3 robot_pose_stream_ros2.py client --host 192.168.1.100 --port 9999

  # Test locally without the robot (mock sine-wave motion):
  python3 robot_pose_stream_ros2.py mock-server --port 9999   # terminal 1
  python3 robot_pose_stream_ros2.py client --host 127.0.0.1   # terminal 2
""",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Server
    sp_server = subparsers.add_parser("server",
        help="Run on robot controller: read LinuxCNC joints, stream over TCP")
    sp_server.add_argument("--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)")
    sp_server.add_argument("--port", type=int, default=9999,
        help="TCP port (default: 9999)")
    sp_server.add_argument("--rate", type=float, default=50.0,
        help="Streaming rate in Hz (default: 50)")

    # Mock server
    sp_mock = subparsers.add_parser("mock-server",
        help="Simulated server with sine-wave joint motion (no LinuxCNC needed)")
    sp_mock.add_argument("--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)")
    sp_mock.add_argument("--port", type=int, default=9999,
        help="TCP port (default: 9999)")
    sp_mock.add_argument("--rate", type=float, default=50.0,
        help="Streaming rate in Hz (default: 50)")

    # Client
    sp_client = subparsers.add_parser("client",
        help="Run on desktop: receive joint poses, publish ROS2 /joint_states")
    sp_client.add_argument("--host", required=True,
        help="Robot server IP address (e.g., 192.168.1.100)")
    sp_client.add_argument("--port", type=int, default=9999,
        help="TCP port (default: 9999)")
    sp_client.add_argument("--no-reconnect", action="store_true",
        help="Don't auto-reconnect on connection loss")

    args = parser.parse_args()

    if args.mode == "server":
        run_server(args.host, args.port, args.rate)
    elif args.mode == "mock-server":
        run_mock_server(args.host, args.port, args.rate)
    elif args.mode == "client":
        run_client(args.host, args.port, reconnect=not args.no_reconnect)


if __name__ == "__main__":
    main()
