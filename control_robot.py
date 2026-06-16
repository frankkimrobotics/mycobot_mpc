#!/opt/homebrew/bin/python3.10
"""
Desktop-side robot controller: solve IK for target eef pose, send to robot.

Pipeline:
  Target (R, t) → IK (pyroki) → joint angles (deg) → TCP → robot (robot_hal.py)

The robot runs a single HAL component (robot_hal.py) via LinuxCNC config elerob.ini.
The desktop sends the control law per move; use --controller pid, invdyn, or pd_velff.

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

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ik_pyroki import MyCobotIK

# When conda is active but we're running a different Python (e.g. homebrew), avoid
# conda's PYTHONPATH so it doesn't inject incompatible packages. If we're running
# the active conda env's Python, leave sys.path alone so its site-packages (numpy, etc.) work.
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix and not sys.executable.startswith(_conda_prefix):
    os.environ.pop("PYTHONPATH", None)
    sys.path[:] = [p for p in sys.path if "conda" not in p and "envs" not in p]

import numpy as np

# Home pose in LinuxCNC degrees (same as ik_pyroki). IK is imported only when needed (--xyz / --interactive).
HOME_LINUXCNC_DEG = [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0]

MAX_JOINTS = 6
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


class PipelineTimer:
    """Hierarchical stopwatch for measuring pipeline stages."""

    def __init__(self):
        self._timings: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timer and return elapsed milliseconds."""
        elapsed_ms = (time.perf_counter() - self._starts.pop(name)) * 1000
        self._timings[name] = elapsed_ms
        return elapsed_ms

    def get(self, name: str, default: float = 0.0) -> float:
        return self._timings.get(name, default)

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self._timings.items()}

    def summary(self) -> str:
        parts = [f"{k}={v:.1f}ms" for k, v in self._timings.items()]
        return " | ".join(parts)

    def reset(self):
        self._timings.clear()
        self._starts.clear()


class MoveLogger:
    """Accumulates per-move timing data and writes to CSV."""

    HEADER = [
        "timestamp", "move_id", "target_type", "controller",
        # Desktop-side timing (ms)
        "ik_solve_ms", "fk_verify_ms", "cmd_send_ms", "ack_rtt_ms",
        "wait_done_ms", "total_ms",
        # Robot-side timing (from status message)
        "robot_exec_ms", "robot_n_loops",
        "robot_avg_poll_ms", "robot_avg_solve_ms",
        "robot_avg_hal_write_ms", "robot_avg_sleep_ms",
        # Accuracy
        "pos_error_mm", "joint_error_deg",
    ] + [f"target_j{i}" for i in range(MAX_JOINTS)] \
      + [f"solved_j{i}" for i in range(MAX_JOINTS)] \
      + [f"final_j{i}" for i in range(MAX_JOINTS)]

    def __init__(self):
        self._rows: list[list] = []
        self._move_id = 0

    def log_move(
        self,
        timer: PipelineTimer,
        target_type: str,
        controller: str,
        target_deg: list[float],
        solved_deg: list[float] | None,
        final_status: dict,
        pos_error_mm: float = 0.0,
    ):
        self._move_id += 1
        final_deg = final_status.get("current_deg", [0.0] * MAX_JOINTS)
        joint_err = final_status.get("error_norm", 0.0)

        row = [
            datetime.now().strftime("%H:%M:%S.%f")[:-3],
            self._move_id,
            target_type,
            controller,
            # Desktop timings
            round(timer.get("ik_solve"), 3),
            round(timer.get("fk_verify"), 3),
            round(timer.get("cmd_send"), 3),
            round(timer.get("ack_rtt"), 3),
            round(timer.get("wait_done"), 3),
            round(timer.get("total"), 3),
            # Robot timings (from done status)
            final_status.get("robot_exec_ms", 0.0),
            final_status.get("n_loops", 0),
            final_status.get("avg_poll_ms", 0.0),
            final_status.get("avg_solve_ms", 0.0),
            final_status.get("avg_hal_write_ms", 0.0),
            final_status.get("avg_sleep_ms", 0.0),
            # Accuracy
            round(pos_error_mm, 3),
            round(joint_err, 3),
        ]
        row += [round(v, 4) for v in target_deg]
        row += [round(v, 4) for v in (solved_deg or [0.0] * MAX_JOINTS)]
        row += [round(v, 4) for v in final_deg]
        self._rows.append(row)

    def save(self, tag: str = ""):
        if not self._rows:
            return None
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"control_{tag}_{stamp}.csv" if tag else f"control_{stamp}.csv"
        path = os.path.join(LOG_DIR, name)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            writer.writerows(self._rows)
        print(f"[log] Saved {len(self._rows)} moves → {path}")
        return path


# Path to the streaming script (same directory as this file)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_STREAM_SCRIPT = os.path.join(_THIS_DIR, "robot_pose_stream_ros2.py")


def launch_rviz_streamer(host: str, stream_port: int = 9999) -> subprocess.Popen:
    """Launch robot_pose_stream_ros2.py in a subprocess with the ROS2 conda env.

    This starts rviz2 + the streaming client that subscribes to the robot's
    joint angle stream and publishes to /joint_states for rviz2 visualization.
    """
    shell_cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate ros_env && "
        "source $CONDA_PREFIX/setup.zsh && "
        "source ~/ros2_ws/install/setup.zsh && "
        f"python3 {_STREAM_SCRIPT} --host {host} --port {stream_port}"
    )
    print(f"[rviz2] Launching rviz2 + streaming client...")
    proc = subprocess.Popen(
        shell_cmd,
        shell=True,
        executable="/bin/zsh",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    print(f"[rviz2] Started (PID {proc.pid}), waiting for rviz2 to initialize...")
    time.sleep(5.0)
    return proc


class RobotConnection:
    """TCP connection to the robot's command server (robot_hal.py port 9998).

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
        duration: float = 2.0,
        controller: str = "pid",
        pos_tol: float = 0.5,
        settle_steps: int = 10,
        trajectory: list[list[float]] | None = None,
        traj_dt: float | None = None,
    ) -> dict:
        """Send a move command and return the ack response.

        Args:
            target_deg: 6 joint angles in LinuxCNC degrees (final pose)
            duration: Max duration for the move (seconds)
            controller: "pid", "invdyn", or "pd_velff"
            pos_tol: Position tolerance for early stop (degrees)
            settle_steps: Consecutive converged loops before early stop
            trajectory: optional list of N x 6 per-sample joint poses; the robot
                        tracks this time-varying setpoint (smooth B-spline/quintic
                        motion) in a single control loop and a single log.
            traj_dt: seconds between trajectory samples (required with trajectory)

        Returns:
            Ack dict from robot, or empty dict on failure
        """
        # Robot HAL (robot_hal.py) accepts pid, invdyn, pd_velff as-is
        robot_controller = controller
        # Send desktop timestamp so Raspi uses it for CSV filename (Raspi clock may be wrong)
        log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cmd = {
            "target_deg": [round(float(v), 4) for v in target_deg],
            "duration": duration,
            "controller": robot_controller,
            "pos_tol": pos_tol,
            "settle_steps": settle_steps,
            "log_stamp": log_stamp,
        }
        if trajectory is not None and traj_dt is not None:
            cmd["trajectory"] = [[round(float(v), 4) for v in row] for row in trajectory]
            cmd["traj_dt"] = float(traj_dt)
        msg = json.dumps(cmd) + "\n"
        self.sock.sendall(msg.encode("utf-8"))
        print(f"[conn] Sent target: {[round(float(v), 1) for v in target_deg]} "
              f"(dur={duration}s, ctrl={controller}, tol={pos_tol}°, settle={settle_steps})")

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

    def fetch_log(self, filename: str) -> str | None:
        """Request a log file from the robot (saved on Raspi) and save it to local logs/.
        Returns the local path if successful, else None.
        """
        self.sock.sendall((json.dumps({"get_log": filename}) + "\n").encode("utf-8"))
        self.sock.settimeout(10.0)
        buffer = ""
        t0 = time.time()
        while time.time() - t0 < 10.0:
            try:
                data = self.sock.recv(65536)
                if not data:
                    return None
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
                    if msg.get("state") == "log" and "log_content_base64" in msg:
                        raw = base64.b64decode(msg["log_content_base64"])
                        os.makedirs(LOG_DIR, exist_ok=True)
                        local_path = os.path.join(LOG_DIR, msg.get("filename", filename))
                        with open(local_path, "wb") as f:
                            f.write(raw)
                        print(f"[log] Saved robot log to {local_path}")
                        return local_path
                    if msg.get("state") == "log_error":
                        print(f"[log] Robot error: {msg.get('error', 'unknown')}")
                        return None
            except (socket.timeout, OSError):
                break
        return None

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


def _maybe_fetch_robot_log(conn: RobotConnection, status: dict, fetch_logs: bool) -> None:
    """If fetch_logs and robot reported last_log_name, fetch CSV and save to local logs/."""
    if not fetch_logs or not status:
        return
    name = status.get("last_log_name")
    if name:
        conn.fetch_log(name)


def move_to_joints(
    conn: RobotConnection,
    target_deg: np.ndarray,
    duration: float = 2.0,
    controller: str = "pid",
    pos_tol: float = 0.5,
    settle_steps: int = 10,
    timer: PipelineTimer | None = None,
    logger: MoveLogger | None = None,
) -> dict:
    """Send joint angle target and wait for completion.

    Args:
        conn: Robot connection
        target_deg: 6 joint angles in LinuxCNC degrees
        duration: Max duration for the move
        controller: "pid", "invdyn", or "pd_velff"
        pos_tol: Position tolerance for early stop (degrees)
        settle_steps: Consecutive converged loops before early stop
        timer: Optional PipelineTimer (will be created if None)
        logger: Optional MoveLogger to record the move

    Returns:
        Final status dict from robot
    """
    if timer is None:
        timer = PipelineTimer()

    timer.start("total")
    timer.start("cmd_send")
    timer.start("ack_rtt")
    ack = conn.send_target(list(target_deg), duration=duration, controller=controller,
                           pos_tol=pos_tol, settle_steps=settle_steps)
    timer.stop("ack_rtt")
    timer.stop("cmd_send")

    timer.start("wait_done")
    status = conn.wait_for_done(timeout=duration + 5.0)
    timer.stop("wait_done")
    timer.stop("total")

    # Fill in zero for stages that didn't apply
    for key in ("ik_solve", "fk_verify"):
        if key not in timer.as_dict():
            timer._timings[key] = 0.0

    print(f"[timer] {timer.summary()}")
    if status.get("done_reason"):
        print(f"  exit_reason={status['done_reason']}")

    if logger:
        logger.log_move(
            timer=timer,
            target_type="joints",
            controller=controller,
            target_deg=list(target_deg),
            solved_deg=None,
            final_status=status,
        )

    return status


def move_smooth(
    conn: RobotConnection,
    current_deg: np.ndarray,
    target_deg: np.ndarray,
    via: list | None = None,
    kind: str = "quintic",
    controller: str = "pid",
    rate_hz: float = 50.0,
    vel_frac: float = 0.6,
    acc_frac: float = 0.6,
    pos_tol: float = 0.5,
    settle_steps: int = 10,
    timer: PipelineTimer | None = None,
    logger: MoveLogger | None = None,
) -> dict:
    """Move from current_deg to target_deg along a smooth, time-scaled trajectory.

    Builds a min-jerk quintic (kind="quintic") or quintic B-spline through
    via-points (kind="bspline") with trajectory.plan(), samples it at rate_hz,
    and sends it as one trajectory command. The robot tracks the moving setpoint
    in a single control loop / single log. Returns the final status dict.
    """
    import trajectory as traj

    tr = traj.plan(np.asarray(current_deg, float), np.asarray(target_deg, float),
                   via=via, kind=kind, rate_hz=rate_hz, vel_frac=vel_frac, acc_frac=acc_frac)
    samples = tr["q"]              # (N, 6) per-sample joint poses
    T = tr["T"]
    traj_dt = 1.0 / rate_hz
    duration = T + 2.0             # play the trajectory, then a settle margin

    if timer is None:
        timer = PipelineTimer()
    timer.start("total"); timer.start("cmd_send"); timer.start("ack_rtt")
    ack = conn.send_target(
        list(samples[-1]), duration=duration, controller=controller,
        pos_tol=pos_tol, settle_steps=settle_steps,
        trajectory=[list(row) for row in samples], traj_dt=traj_dt,
    )
    timer.stop("ack_rtt"); timer.stop("cmd_send")
    print(f"[smooth] {kind} traj: {len(samples)} samples, T={T:.2f}s "
          f"({vel_frac*100:.0f}% vel / {acc_frac*100:.0f}% acc limits)")

    timer.start("wait_done")
    status = conn.wait_for_done(timeout=duration + 5.0)
    timer.stop("wait_done"); timer.stop("total")
    for key in ("ik_solve", "fk_verify"):
        if key not in timer.as_dict():
            timer._timings[key] = 0.0
    print(f"[timer] {timer.summary()}")
    if status.get("done_reason"):
        print(f"  exit_reason={status['done_reason']}")

    if logger:
        logger.log_move(
            timer=timer, target_type=f"smooth_{kind}", controller=controller,
            target_deg=list(target_deg), solved_deg=None, final_status=status,
        )
    return status


def move_to_pose(
    conn: RobotConnection,
    ik: MyCobotIK,
    R: np.ndarray,
    t: np.ndarray,
    duration: float = 2.0,
    controller: str = "pid",
    pos_tol: float = 0.5,
    settle_steps: int = 10,
    logger: MoveLogger | None = None,
) -> tuple[np.ndarray, dict]:
    """Solve IK for (R, t) and move the robot there.

    Args:
        conn: Robot connection
        ik: IK solver instance
        R: (3,3) rotation matrix for eef
        t: (3,) translation vector for eef (meters)
        duration: Max duration for the move
        controller: "pid", "invdyn", or "pd_velff"
        pos_tol: Position tolerance for early stop (degrees)
        settle_steps: Consecutive converged loops before early stop
        logger: Optional MoveLogger to record the move

    Returns:
        (solved_joints_deg, final_status)
    """
    timer = PipelineTimer()

    # IK solve
    timer.start("ik_solve")
    joints_deg = ik.solve(R=R, t=t)
    timer.stop("ik_solve")
    print(f"[ik] Solved in {timer.get('ik_solve'):.1f}ms → {[round(v, 1) for v in joints_deg]} deg")

    # FK verification
    timer.start("fk_verify")
    _, t_check = ik.forward_kinematics(joints_deg)
    timer.stop("fk_verify")
    pos_err = np.linalg.norm(t - t_check)
    pos_err_mm = pos_err * 1000
    if pos_err > 0.01:
        print(f"[ik] WARNING: FK verification error = {pos_err_mm:.2f}mm (>10mm)")

    # Send + wait
    timer.start("total")
    timer.start("cmd_send")
    timer.start("ack_rtt")
    conn.send_target(list(joints_deg), duration=duration, controller=controller,
                     pos_tol=pos_tol, settle_steps=settle_steps)
    timer.stop("ack_rtt")
    timer.stop("cmd_send")

    timer.start("wait_done")
    status = conn.wait_for_done(timeout=duration + 5.0)
    timer.stop("wait_done")
    timer.stop("total")

    # Add IK + FK time to total
    timer._timings["total"] += timer.get("ik_solve") + timer.get("fk_verify")

    print(f"[timer] {timer.summary()}")

    if logger:
        logger.log_move(
            timer=timer,
            target_type="pose",
            controller=controller,
            target_deg=list(joints_deg),
            solved_deg=list(joints_deg),
            final_status=status,
            pos_error_mm=pos_err_mm,
        )

    return joints_deg, status


def interactive_mode(
    conn: RobotConnection,
    ik: MyCobotIK,
    controller: str,
    duration: float,
    pos_tol: float = 0.5,
    settle_steps: int = 10,
    logger: MoveLogger | None = None,
    fetch_logs: bool = True,
):
    """Interactive control loop: enter poses from the terminal."""
    R_home, _ = ik.forward_kinematics(HOME_LINUXCNC_DEG)

    print("\n" + "=" * 60)
    print("Interactive mode. Enter commands:")
    print("  xyz <x> <y> <z>        - Move eef to position (meters), home orientation")
    print("  joints <j1> ... <j6>   - Move to joint angles (LinuxCNC degrees)")
    print("  home                   - Move to home pose")
    print("  fk                     - Print current eef pose (from last known joints)")
    print("  timing                 - Print last move timing breakdown")
    print("  quit                   - Exit")
    print("=" * 60)

    last_timer: PipelineTimer | None = None

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
                status = move_to_joints(conn, np.array(HOME_LINUXCNC_DEG),
                                        duration=duration, controller=controller,
                                        pos_tol=pos_tol, settle_steps=settle_steps,
                                        logger=logger)
                _maybe_fetch_robot_log(conn, status, fetch_logs)

            elif cmd == "xyz" and len(parts) == 4:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                t_target = np.array([x, y, z])
                print(f"Target position: {t_target.tolist()} m (home orientation)")
                _, status = move_to_pose(conn, ik, R_home, t_target,
                                         duration=duration, controller=controller,
                                         pos_tol=pos_tol, settle_steps=settle_steps,
                                         logger=logger)
                _maybe_fetch_robot_log(conn, status, fetch_logs)

            elif cmd == "joints" and len(parts) == 7:
                joints = [float(v) for v in parts[1:7]]
                print(f"Target joints: {joints} deg")
                status = move_to_joints(conn, np.array(joints),
                                        duration=duration, controller=controller,
                                        pos_tol=pos_tol, settle_steps=settle_steps,
                                        logger=logger)
                _maybe_fetch_robot_log(conn, status, fetch_logs)

            elif cmd == "fk":
                timer = PipelineTimer()
                timer.start("fk")
                status = conn.get_status()
                current = status.get("current_deg")
                if current:
                    R, t_vec = ik.forward_kinematics(current)
                    timer.stop("fk")
                    print(f"  Joints (LinuxCNC): {[round(v, 1) for v in current]} deg")
                    print(f"  EEF position:      {np.round(t_vec, 4).tolist()} m")
                    print(f"  EEF rotation:\n{np.round(R, 4)}")
                    print(f"  FK compute: {timer.get('fk'):.2f}ms")
                else:
                    print("  No status available from robot.")

            elif cmd == "timing":
                if logger and logger._rows:
                    last = logger._rows[-1]
                    header = MoveLogger.HEADER
                    print("  Last move timing:")
                    for i, h in enumerate(header):
                        if h.endswith("_ms") or h.endswith("_mm") or h in ("move_id", "target_type", "controller", "robot_n_loops", "joint_error_deg"):
                            print(f"    {h:>25s} = {last[i]}")
                else:
                    print("  No moves recorded yet.")

            else:
                print(f"  Unknown command: {line}")
                print("  Try: xyz 0.3 0.0 0.5 | joints -80 -85 5 -85 5 5 | home | fk | timing | quit")

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
    default_host = os.environ.get("ROBOT_IP")
    parser.add_argument("--host", default=default_host,
        help="Robot controller IP (default: ROBOT_IP env, e.g. 10.0.0.27)")
    parser.add_argument("--cmd-port", type=int, default=9998,
        help="Robot command port (default: 9998)")
    parser.add_argument("--stream-port", type=int, default=9999,
        help="Robot streaming port for rviz2 (default: 9999)")
    parser.add_argument("--controller", choices=["pid", "invdyn", "pd_velff"], default="pid",
        help="Controller type: pid, invdyn, or pd_velff (default: pid)")
    parser.add_argument("--duration", type=float, default=2.0,
        help="Move duration in seconds (default: 2.0)")
    parser.add_argument("--pos-tol", type=float, default=0.5,
        help="Position tolerance for early stop in degrees (default: 0.5)")
    parser.add_argument("--settle-steps", type=int, default=10,
        help="Consecutive converged loops before early stop (default: 10)")
    parser.add_argument("--rviz", action="store_true",
                        help="Launch rviz2 + pose stream (default: do not launch)")
    parser.add_argument("--no-fetch-logs", action="store_true",
                        help="Don't fetch robot CSV logs to local logs/ after each move")

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

    # If --host $ROBOT_IP was used and ROBOT_IP is unset, host can be "" or the next arg (e.g. "--controller")
    host = (args.host or "").strip()
    if not host or host.startswith("-"):
        host = os.environ.get("ROBOT_IP") or ""
    if not host:
        parser.error(
            "Robot host not set. Use either:\n"
            "  (1) ./setup_robot_ip.sh 10.0.0.27  then  conda activate ros_env  (no --host needed)\n"
            "  (2)  python control_robot.py --host 10.0.0.27 ...  (use your Raspi IP)"
        )
    args.host = host

    rviz_proc = None

    def cleanup(signum=None, frame=None):
        """Clean shutdown: kill rviz2 subprocess on exit."""
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

    # Launch rviz2 + streaming client only if requested
    if args.rviz:
        rviz_proc = launch_rviz_streamer(args.host, args.stream_port)

    # IK solver only needed for --xyz and --interactive (uses JAX/pyroki)
    ik = None
    if args.xyz or args.interactive:
        print("Initializing IK solver...")
        from ik_pyroki import MyCobotIK
        t_ik_init = time.perf_counter()
        ik = MyCobotIK()
        ik_init_ms = (time.perf_counter() - t_ik_init) * 1000
        print(f"IK solver ready ({ik_init_ms:.0f}ms)")

    # Per-session move logger
    logger = MoveLogger()

    # Connect to robot
    conn = RobotConnection(host=args.host, cmd_port=args.cmd_port)
    try:
        conn.connect()
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"ERROR: Cannot connect to robot at {args.host}:{args.cmd_port}: {e}")
        print("Make sure robot_hal.py is running on the robot (via linuxcnc elerob.ini).")
        cleanup()
        sys.exit(1)

    try:
        if args.interactive:
            interactive_mode(conn, ik, controller=args.controller,
                             duration=args.duration,
                             pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                             logger=logger, fetch_logs=not args.no_fetch_logs)

        elif args.home:
            print(f"Moving to home: {HOME_LINUXCNC_DEG}")
            status = move_to_joints(conn, np.array(HOME_LINUXCNC_DEG),
                                    duration=args.duration, controller=args.controller,
                                    pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                                    logger=logger)
            _maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)

        elif args.xyz:
            R_home, _ = ik.forward_kinematics(HOME_LINUXCNC_DEG)
            t_target = np.array(args.xyz)
            print(f"Target position: {t_target.tolist()} m")
            status = move_to_pose(conn, ik, R_home, t_target,
                                  duration=args.duration, controller=args.controller,
                                  pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                                  logger=logger)
            _maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)

        elif args.joints:
            target_deg = np.array(args.joints)
            print(f"Target joints: {target_deg.tolist()} deg")
            status = move_to_joints(conn, target_deg,
                                    duration=args.duration, controller=args.controller,
                                    pos_tol=args.pos_tol, settle_steps=args.settle_steps,
                                    logger=logger)
            _maybe_fetch_robot_log(conn, status, fetch_logs=not args.no_fetch_logs)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        logger.save(tag=args.controller)
        conn.close()
        cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
