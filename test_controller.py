#!/usr/bin/env python3
"""
End-to-end test: robot in RViz, controller receives target and runs, RViz reflects motion.

Steps:
  1. Launch RViz + robot pose stream (connects to robot stream port 9999).
  2. Connect to robot command server (port 9998); controller is already running on the Raspi.
  3. Send target joint angles; controller outputs control; stream feeds /joint_states → RViz.

Prerequisites:
  - On Raspi: LinuxCNC running with mpc_hal or invdyn_hal (elerob_mpc.ini or elerob_invdyn.ini).
  - On desktop: ROS2 env sourced if using RViz (see Usage).

Usage:
  python3 test_controller.py --host 10.0.0.27
  python3 test_controller.py --host 10.0.0.27 --no-rviz
  python3 test_controller.py --local              # no robot: local mock for debugging
  python3 test_controller.py --local --no-rviz     # mock only, no RViz
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

# Same dir as this script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Prevent conda PYTHONPATH from injecting incompatible packages
if "CONDA_PREFIX" in os.environ:
    os.environ.pop("PYTHONPATH", None)
    sys.path[:] = [p for p in sys.path if "conda" not in p and "envs" not in p]

import numpy as np

from control_robot import RobotConnection, launch_rviz_streamer, move_to_joints
from ik_pyroki import HOME_LINUXCNC_DEG

MAX_JOINTS = 6
# Slight offset from home (deg) for a visible move in RViz
TEST_OFFSET_DEG = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])


# -----------------------------------------------------------------------------
# Local mock robot (command server + stream server) for debugging without hardware
# -----------------------------------------------------------------------------

def _run_mock_stream_server(port: int, joint_state: list, lock: threading.Lock, rate_hz: float = 50.0):
    """Background: broadcast joint_state to stream clients (RViz)."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(5)
    server.settimeout(1.0)
    print(f"[mock] Stream server on 127.0.0.1:{port} ({rate_hz} Hz)")
    clients = []
    dt = 1.0 / rate_hz
    next_t = time.time()
    while True:
        try:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            clients.append(conn)
            print(f"[mock] Stream client connected: {addr}")
        except socket.timeout:
            pass
        with lock:
            j = list(joint_state)
        msg = json.dumps({"joints_deg": j, "timestamp": time.time()}) + "\n"
        dead = [c for c in clients if _send_all(c, msg) is False]
        for c in dead:
            clients.remove(c)
        next_t += dt
        time.sleep(max(0, next_t - time.time()))


def _send_all(sock: socket.socket, data: bytes | str) -> bool:
    try:
        if isinstance(data, str):
            data = data.encode("utf-8")
        sock.sendall(data)
        return True
    except (BrokenPipeError, ConnectionResetError, OSError):
        return False


def _run_mock_command_server(
    port: int,
    joint_state: list,
    lock: threading.Lock,
    stream_rate_hz: float,
):
    """Background: accept one client, read target_deg commands, simulate move, send status."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(2)
    print(f"[mock] Command server on 127.0.0.1:{port}")

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(0.5)
        print(f"[mock] Command client: {addr}")
        buffer = ""
        try:
            while True:
                # Send periodic status (same as real robot)
                with lock:
                    current = list(joint_state)
                status = {
                    "state": "idle",
                    "current_deg": [round(x, 3) for x in current],
                    "target_deg": [round(x, 3) for x in current],
                    "error_norm": 0.0,
                }
                if not _send_all(conn, json.dumps(status) + "\n"):
                    break
                # Read incoming
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "get_log" in cmd:
                        _send_all(conn, json.dumps({"state": "log_error", "error": "no_log_mock"}) + "\n")
                        continue
                    target = cmd.get("target_deg")
                    if target is None or len(target) != MAX_JOINTS:
                        continue
                    duration = max(0.1, float(cmd.get("duration", 2.0)))
                    _send_all(conn, json.dumps({"state": "ack", "target_deg": target}) + "\n")
                    # Simulate move: linear interpolation, update joint_state and stream
                    start = None
                    with lock:
                        start = list(joint_state)
                    n_steps = max(1, int(duration * stream_rate_hz))
                    for k in range(1, n_steps + 1):
                        t = k / n_steps
                        with lock:
                            for i in range(MAX_JOINTS):
                                joint_state[i] = start[i] + t * (target[i] - start[i])
                        err = (sum((target[i] - joint_state[i]) ** 2 for i in range(MAX_JOINTS))) ** 0.5
                        status = {
                            "state": "moving",
                            "current_deg": [round(joint_state[i], 3) for i in range(MAX_JOINTS)],
                            "target_deg": [round(x, 3) for x in target],
                            "error_norm": round(err, 4),
                        }
                        if not _send_all(conn, json.dumps(status) + "\n"):
                            break
                        time.sleep(duration / n_steps)
                    with lock:
                        for i in range(MAX_JOINTS):
                            joint_state[i] = target[i]
                    status = {
                        "state": "done",
                        "current_deg": [round(x, 3) for x in target],
                        "target_deg": [round(x, 3) for x in target],
                        "error_norm": 0.0,
                    }
                    _send_all(conn, json.dumps(status) + "\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            conn.close()
        print(f"[mock] Command client disconnected: {addr}")


def start_mock_robot(cmd_port: int = 9998, stream_port: int = 9999, stream_rate_hz: float = 50.0):
    """Start local mock robot (command + stream servers). Returns (joint_state, lock)."""
    joint_state = list(HOME_LINUXCNC_DEG)
    lock = threading.Lock()
    t_stream = threading.Thread(
        target=_run_mock_stream_server,
        args=(stream_port, joint_state, lock, stream_rate_hz),
        daemon=True,
    )
    t_cmd = threading.Thread(
        target=_run_mock_command_server,
        args=(cmd_port, joint_state, lock, stream_rate_hz),
        daemon=True,
    )
    t_stream.start()
    t_cmd.start()
    time.sleep(0.5)
    return joint_state, lock


def main():
    parser = argparse.ArgumentParser(
        description="Test: RViz + controller + joint target → motion in RViz.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host",
                        help="Robot IP (e.g., 10.0.0.27). Omit with --local to use mock.")
    parser.add_argument("--local", action="store_true",
                        help="Run without robot: start local mock server (command + stream) for debugging.")
    parser.add_argument("--cmd-port", type=int, default=9998,
                        help="Robot command port (default: 9998)")
    parser.add_argument("--stream-port", type=int, default=9999,
                        help="Robot stream port for RViz (default: 9999)")
    parser.add_argument("--controller", choices=["pd", "mpc", "invdyn"], default="pd",
                        help="Controller type (default: pd)")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Move duration per segment (default: 2.0)")
    parser.add_argument("--no-rviz", action="store_true",
                        help="Do not launch RViz (stream already running)")
    args = parser.parse_args()

    if args.local:
        args.host = "127.0.0.1"
        print("[mock] Starting local mock robot (no hardware)...")
        start_mock_robot(cmd_port=args.cmd_port, stream_port=args.stream_port)
        print("[mock] Mock robot ready.")
    elif not args.host:
        parser.error("Either --host <ip> or --local is required.")

    rviz_proc = None

    def cleanup(signum=None, frame=None):
        if rviz_proc and rviz_proc.poll() is None:
            print("\n[rviz2] Shutting down...")
            try:
                os.killpg(os.getpgid(rviz_proc.pid), signal.SIGTERM)
                rviz_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(rviz_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 1) Robot initialized in RViz: launch stream client + rviz2
    if not args.no_rviz:
        print("[1/4] Launching RViz + pose stream...")
        rviz_proc = launch_rviz_streamer(args.host, args.stream_port)
        time.sleep(1.0)
        print("      RViz and stream client running (robot should appear when connected).")
    else:
        print("[1/4] Skipping RViz (--no-rviz). Ensure stream is running for visualization.")

    # 2) Controller initialized and receives target
    print("[2/4] Connecting to robot command server...")
    conn = RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (ConnectionRefusedError, OSError) as e:
        print(f"ERROR: Cannot connect to robot at {args.host}:{args.cmd_port}: {e}")
        print("       Ensure LinuxCNC + mpc_hal or invdyn_hal is running on the robot.")
        cleanup()
        sys.exit(1)

    status = conn.get_status()
    if status:
        current = status.get("current_deg", [])
        print(f"      Controller ready. Current joints (deg): {[round(x, 1) for x in current]}")
    else:
        print("      Controller connected (no status line yet).")

    # 3) Send targets; controller outputs control; RViz reflects motion
    home = np.array(HOME_LINUXCNC_DEG, dtype=float)
    target_mid = home + TEST_OFFSET_DEG

    print("[3/4] Move to home (RViz should show motion)...")
    move_to_joints(
        conn, home,
        duration=args.duration, controller=args.controller,
    )
    print("      Home reached.")

    print("[4/4] Move to offset pose and back (visible motion in RViz)...")
    move_to_joints(
        conn, target_mid,
        duration=args.duration, controller=args.controller,
    )
    move_to_joints(
        conn, home,
        duration=args.duration, controller=args.controller,
    )
    print("      Test sequence complete.")

    conn.close()
    print("\nDone. RViz should have shown the robot moving to home, then offset, then home.")
    if rviz_proc and rviz_proc.poll() is None:
        print("RViz is still running; close the window or Ctrl+C to exit.")
        try:
            rviz_proc.wait()
        except KeyboardInterrupt:
            cleanup()


if __name__ == "__main__":
    main()
