#!/usr/bin/env python3
"""
Desktop-side robot controller: solve IK for target eef pose, send to robot.

Pipeline:
  Target (R, t) → IK (pyroki) → joint angles (deg) → TCP → robot (mpc_hal.py)

The robot must be running mpc_hal.py (via linuxcnc elerob_mpc.ini) which
listens for commands on port 9998 and streams joint feedback on port 9999.

Usage:
    # Move to a Cartesian position (uses home orientation):
    python3.10 control_robot.py --host 10.0.0.27 --xyz 0.3 0.0 0.5

    # Move to specific joint angles (LinuxCNC degrees):
    python3.10 control_robot.py --host 10.0.0.27 --joints -80 -85 5 -85 5 5

    # Interactive mode (enter poses interactively):
    python3.10 control_robot.py --host 10.0.0.27 --interactive

    # Move home:
    python3.10 control_robot.py --host 10.0.0.27 --home
"""

import argparse
import json
import socket
import sys
import time
import numpy as np

from ik_pyroki import MyCobotIK, HOME_LINUXCNC_DEG


class RobotConnection:
    """TCP connection to the robot's command server (mpc_hal.py port 9998).

    Sends target joint angles and monitors execution status.
    """

    def __init__(self, host: str, cmd_port: int = 9998):
        self.host = host
        self.cmd_port = cmd_port
        self.sock = None

    def connect(self):
        """Connect to the robot's command server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((self.host, self.cmd_port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[conn] Connected to robot at {self.host}:{self.cmd_port}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_target(
        self,
        target_deg: list[float],
        duration: float = 5.0,
        controller: str = "pd",
    ) -> dict:
        """Send a move command and return the ack response.

        Args:
            target_deg: 6 joint angles in LinuxCNC degrees
            duration: Max duration for the move (seconds)
            controller: "pd" or "mpc"

        Returns:
            Ack dict from robot, or empty dict on failure
        """
        cmd = {
            "target_deg": [round(float(v), 4) for v in target_deg],
            "duration": duration,
            "controller": controller,
        }
        msg = json.dumps(cmd) + "\n"
        self.sock.sendall(msg.encode("utf-8"))
        print(f"[conn] Sent target: {[round(v, 1) for v in target_deg]} (duration={duration}s, ctrl={controller})")

        # Read ack
        return self._read_status("ack", timeout=2.0)

    def wait_for_done(self, timeout: float = 30.0, poll_interval: float = 0.5) -> dict:
        """Wait until the robot reports 'done' or 'idle' state.

        Returns the final status dict.
        """
        t0 = time.time()
        last_print = 0
        while (time.time() - t0) < timeout:
            status = self._read_status(None, timeout=poll_interval)
            if not status:
                continue

            state = status.get("state", "")
            err = status.get("error_norm", 0)
            now = time.time()

            if now - last_print > 1.0:
                current = status.get("current_deg", [])
                if current:
                    print(f"  [{state}] err={err:.2f}° q={[round(v, 1) for v in current]}")
                last_print = now

            if state in ("done", "idle"):
                return status

        print(f"[conn] Timeout after {timeout}s")
        return {"state": "timeout"}

    def get_status(self) -> dict:
        """Read one status message from the robot."""
        return self._read_status(None, timeout=2.0)

    def _read_status(self, wait_for_state: str | None, timeout: float = 2.0) -> dict:
        """Read status messages, optionally waiting for a specific state."""
        self.sock.settimeout(timeout)
        buffer = ""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            try:
                data = self.sock.recv(4096)
                if not data:
                    return {}
                buffer += data.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if wait_for_state is None or msg.get("state") == wait_for_state:
                        return msg
            except socket.timeout:
                break
        return {}


def move_to_joints(
    conn: RobotConnection,
    target_deg: np.ndarray,
    duration: float = 5.0,
    controller: str = "pd",
) -> dict:
    """Send joint angle target and wait for completion.

    Args:
        conn: Robot connection
        target_deg: 6 joint angles in LinuxCNC degrees
        duration: Max duration for the move
        controller: "pd" or "mpc"

    Returns:
        Final status dict from robot
    """
    conn.send_target(list(target_deg), duration=duration, controller=controller)
    return conn.wait_for_done(timeout=duration + 5.0)


def move_to_pose(
    conn: RobotConnection,
    ik: MyCobotIK,
    R: np.ndarray,
    t: np.ndarray,
    duration: float = 5.0,
    controller: str = "pd",
) -> tuple[np.ndarray, dict]:
    """Solve IK for (R, t) and move the robot there.

    Args:
        conn: Robot connection
        ik: IK solver instance
        R: (3,3) rotation matrix for eef
        t: (3,) translation vector for eef (meters)
        duration: Max duration for the move
        controller: "pd" or "mpc"

    Returns:
        (solved_joints_deg, final_status)
    """
    t0 = time.time()
    joints_deg = ik.solve(R=R, t=t)
    solve_ms = (time.time() - t0) * 1000
    print(f"[ik] Solved in {solve_ms:.1f}ms → {[round(v, 1) for v in joints_deg]} deg")

    # Verify FK matches target
    _, t_check = ik.forward_kinematics(joints_deg)
    pos_err = np.linalg.norm(t - t_check)
    if pos_err > 0.01:
        print(f"[ik] WARNING: FK verification error = {pos_err*1000:.2f}mm (>10mm)")

    status = move_to_joints(conn, joints_deg, duration=duration, controller=controller)
    return joints_deg, status


def interactive_mode(conn: RobotConnection, ik: MyCobotIK, controller: str, duration: float):
    """Interactive control loop: enter poses from the terminal."""
    # Get home orientation for Cartesian commands
    R_home, _ = ik.forward_kinematics(HOME_LINUXCNC_DEG)

    print("\n" + "=" * 60)
    print("Interactive mode. Enter commands:")
    print("  xyz <x> <y> <z>        - Move eef to position (meters), home orientation")
    print("  joints <j1> ... <j6>   - Move to joint angles (LinuxCNC degrees)")
    print("  home                   - Move to home pose")
    print("  fk                     - Print current eef pose (from last known joints)")
    print("  quit                   - Exit")
    print("=" * 60)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd == "quit" or cmd == "q":
                break

            elif cmd == "home":
                print(f"Moving to home: {HOME_LINUXCNC_DEG}")
                move_to_joints(conn, np.array(HOME_LINUXCNC_DEG),
                               duration=duration, controller=controller)

            elif cmd == "xyz" and len(parts) == 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                t_target = np.array([x, y, z])
                print(f"Target position: {t_target.tolist()} m (home orientation)")
                move_to_pose(conn, ik, R_home, t_target,
                             duration=duration, controller=controller)

            elif cmd == "joints" and len(parts) == 7:
                joints = [float(v) for v in parts[1:7]]
                print(f"Target joints: {joints} deg")
                move_to_joints(conn, np.array(joints),
                               duration=duration, controller=controller)

            elif cmd == "fk":
                status = conn.get_status()
                current = status.get("current_deg")
                if current:
                    R, t_vec = ik.forward_kinematics(current)
                    print(f"  Joints (LinuxCNC): {[round(v, 1) for v in current]} deg")
                    print(f"  EEF position:      {np.round(t_vec, 4).tolist()} m")
                    print(f"  EEF rotation:\n{np.round(R, 4)}")
                else:
                    print("  No status available from robot.")

            else:
                print(f"  Unknown command: {line}")
                print("  Try: xyz 0.3 0.0 0.5 | joints -80 -85 5 -85 5 5 | home | fk | quit")

        except (ValueError, OSError) as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Desktop robot controller: IK + send to robot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.10 control_robot.py --host 10.0.0.27 --home
  python3.10 control_robot.py --host 10.0.0.27 --xyz 0.3 0.0 0.5
  python3.10 control_robot.py --host 10.0.0.27 --joints -80 -85 5 -85 5 5
  python3.10 control_robot.py --host 10.0.0.27 --interactive
""",
    )
    parser.add_argument("--host", required=True,
        help="Robot controller IP (e.g., 10.0.0.27)")
    parser.add_argument("--cmd-port", type=int, default=9998,
        help="Robot command port (default: 9998)")
    parser.add_argument("--controller", choices=["pd", "mpc"], default="pd",
        help="Controller type (default: pd)")
    parser.add_argument("--duration", type=float, default=5.0,
        help="Move duration in seconds (default: 5.0)")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--xyz", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="Target eef position in meters (uses home orientation)")
    group.add_argument("--joints", nargs=6, type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Target joint angles in LinuxCNC degrees")
    group.add_argument("--home", action="store_true",
        help="Move to home pose [-90, -90, 0, -90, 0, 0]")
    group.add_argument("--interactive", action="store_true",
        help="Interactive control mode")

    args = parser.parse_args()

    # Initialize IK solver
    print("Initializing IK solver...")
    ik = MyCobotIK()

    # Connect to robot
    conn = RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"ERROR: Cannot connect to robot at {args.host}:{args.cmd_port}: {e}")
        print("Make sure mpc_hal.py is running on the robot (via linuxcnc elerob_mpc.ini).")
        sys.exit(1)

    try:
        if args.interactive:
            interactive_mode(conn, ik, controller=args.controller, duration=args.duration)

        elif args.home:
            print(f"Moving to home: {HOME_LINUXCNC_DEG}")
            move_to_joints(conn, np.array(HOME_LINUXCNC_DEG),
                           duration=args.duration, controller=args.controller)

        elif args.xyz:
            R_home, _ = ik.forward_kinematics(HOME_LINUXCNC_DEG)
            t_target = np.array(args.xyz)
            print(f"Target position: {t_target.tolist()} m")
            move_to_pose(conn, ik, R_home, t_target,
                         duration=args.duration, controller=args.controller)

        elif args.joints:
            target_deg = np.array(args.joints)
            print(f"Target joints: {target_deg.tolist()} deg")
            move_to_joints(conn, target_deg,
                           duration=args.duration, controller=args.controller)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()
