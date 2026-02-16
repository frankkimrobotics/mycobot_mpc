#!/opt/homebrew/bin/python3.10
"""
Joystick teleoperation for myCobot Pro 630 using a DualShock 4 (PS4) controller.

Streams delta EEF poses from the joystick to the robot in real time.
Left stick controls position, right stick controls orientation.
L2 and R2 act as deadman switches — sticks only take effect while held.

Control mapping (DualShock 4):
  ┌──────────────────────────────────────────────────────────────┐
  │  L2 (deadman) + Left Stick   →  Position (translation)      │
  │    Stick X  →  EEF ΔY  (left / right)                       │
  │    Stick Y  →  EEF ΔX  (forward / backward)                 │
  │    D-pad ↑↓ →  EEF ΔZ  (up / down)                          │
  │                                                              │
  │  R2 (deadman) + Right Stick  →  Orientation (rotation)       │
  │    Stick X  →  EEF Δyaw   (rotate around world Z)           │
  │    Stick Y  →  EEF Δpitch (rotate around world Y)           │
  │    L1 / R1  →  EEF Δroll  (rotate around world X)           │
  │                                                              │
  │  Buttons:                                                    │
  │    ✕ (Cross)    →  Move to home pose                         │
  │    ○ (Circle)   →  Cycle speed (slow / medium / fast)        │
  │    △ (Triangle) →  Print current EEF pose                    │
  │    □ (Square)   →  Emergency stop (disconnect & exit)        │
  └──────────────────────────────────────────────────────────────┘

Usage:
    # Connect DualShock 4 via Bluetooth first (System Preferences → Bluetooth)
    python3.10 joystick_controller.py --host 10.0.0.27
    python3.10 joystick_controller.py --host 10.0.0.27 --speed medium
    python3.10 joystick_controller.py --host 10.0.0.27 --no-rviz
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

# Prevent conda PYTHONPATH from injecting incompatible packages (e.g. numpy 3.11
# into python 3.10). Safe to clear because Homebrew python3.10 has all we need.
if "CONDA_PREFIX" in os.environ:
    os.environ.pop("PYTHONPATH", None)
    sys.path[:] = [p for p in sys.path if "conda" not in p and "envs" not in p]

import numpy as np

try:
    import pygame
except ImportError:
    print("ERROR: pygame is required. Install with: pip install pygame")
    sys.exit(1)

from ik_pyroki import MyCobotIK, HOME_LINUXCNC_DEG

# ═══════════════════════════════════════════════════════════════════════════════
#  DualShock 4 button/axis mapping (SDL2 on macOS)
# ═══════════════════════════════════════════════════════════════════════════════
# Axes (pygame axis indices for DS4 via Bluetooth on macOS/SDL2):
AX_LEFT_X = 0    # Left stick horizontal  (-1 left, +1 right)
AX_LEFT_Y = 1    # Left stick vertical    (-1 up,   +1 down)
AX_RIGHT_X = 2   # Right stick horizontal (-1 left, +1 right)
AX_RIGHT_Y = 3   # Right stick vertical   (-1 up,   +1 down)
AX_L2 = 4        # L2 trigger             (-1 released, +1 fully pressed)
AX_R2 = 5        # R2 trigger             (-1 released, +1 fully pressed)

# Buttons:
BTN_CROSS = 0     # ✕
BTN_CIRCLE = 1    # ○
BTN_TRIANGLE = 2  # △
BTN_SQUARE = 3    # □
BTN_L1 = 4        # L1
BTN_R1 = 5        # R1
# D-pad is mapped as a hat (hat index 0)

# ═══════════════════════════════════════════════════════════════════════════════
#  Speed presets (meters/sec for position, rad/sec for orientation)
# ═══════════════════════════════════════════════════════════════════════════════
SPEED_PRESETS = {
    "slow":   {"pos": 0.03, "ori": 0.15, "label": "SLOW"},
    "medium": {"pos": 0.08, "ori": 0.40, "label": "MEDIUM"},
    "fast":   {"pos": 0.15, "ori": 0.80, "label": "FAST"},
}
SPEED_ORDER = ["slow", "medium", "fast"]

DEADZONE = 0.12          # Stick deadzone (ignore small drift)
TRIGGER_THRESHOLD = -0.5  # L2/R2 considered "pressed" above this value
DEFAULT_LOOP_HZ = 10     # Control loop rate (Hz)
MOVE_DURATION = 0.15     # Duration per micro-move sent to robot (seconds)


def apply_deadzone(value: float, dz: float = DEADZONE) -> float:
    """Apply deadzone and rescale to [0, 1] range."""
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from roll (X), pitch (Y), yaw (Z) in radians."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  Lightweight robot connection (non-blocking send, no wait-for-done)
# ═══════════════════════════════════════════════════════════════════════════════

class RobotStream:
    """Sends joint targets to the robot and reads status without blocking."""

    def __init__(self, host: str, cmd_port: int = 9998):
        self.host = host
        self.cmd_port = cmd_port
        self.sock: socket.socket | None = None
        self._buf = ""

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.host, self.cmd_port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.setblocking(False)
        print(f"[conn] Connected to robot at {self.host}:{self.cmd_port}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_target(self, target_deg: list[float], duration: float = MOVE_DURATION,
                    controller: str = "pd", pos_tol: float = 2.0, settle_steps: int = 1):
        """Send a move command (non-blocking)."""
        cmd = {
            "target_deg": [round(float(v), 4) for v in target_deg],
            "duration": duration,
            "controller": controller,
            "pos_tol": pos_tol,
            "settle_steps": settle_steps,
        }
        try:
            self.sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            print(f"[conn] Send error: {e}")

    def read_status(self) -> dict | None:
        """Non-blocking read of the latest status from robot."""
        try:
            data = self.sock.recv(8192)
            if data:
                self._buf += data.decode("utf-8", errors="replace")
        except BlockingIOError:
            pass
        except (ConnectionResetError, OSError):
            return None

        last_msg = None
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                try:
                    last_msg = json.loads(line)
                except json.JSONDecodeError:
                    pass
        return last_msg

    def drain_status(self) -> dict | None:
        """Read all pending status messages, return the most recent."""
        latest = None
        for _ in range(50):
            msg = self.read_status()
            if msg is None:
                break
            latest = msg
        return latest


# ═══════════════════════════════════════════════════════════════════════════════
#  Main controller
# ═══════════════════════════════════════════════════════════════════════════════

def init_joystick() -> pygame.joystick.JoystickType:
    """Initialize pygame and find a DualShock 4 controller.

    macOS requires a display surface for SDL2 to detect Bluetooth controllers,
    so we create a small status window.
    """
    pygame.init()

    # macOS Bluetooth controllers require a display surface for SDL2 detection
    screen = pygame.display.set_mode((420, 120))
    pygame.display.set_caption("myCobot Joystick Teleop")
    screen.fill((30, 30, 30))
    font = pygame.font.SysFont("menlo", 14)
    screen.blit(font.render("Detecting controller...", True, (200, 200, 200)), (10, 10))
    pygame.display.flip()

    # Pump events to let SDL2 discover Bluetooth devices
    pygame.joystick.init()
    for _ in range(30):
        pygame.event.pump()
        if pygame.joystick.get_count() > 0:
            break
        time.sleep(0.1)

    count = pygame.joystick.get_count()
    if count == 0:
        print("ERROR: No joystick detected.")
        print("  1. Put DualShock 4 in pairing mode (hold Share + PS until light bar flashes)")
        print("  2. Connect via System Preferences → Bluetooth")
        print("  3. Re-run this script")
        pygame.quit()
        sys.exit(1)

    print(f"[joy] Found {count} joystick(s):")
    js = None
    for i in range(count):
        j = pygame.joystick.Joystick(i)
        j.init()
        name = j.get_name()
        print(f"  [{i}] {name}  (axes={j.get_numaxes()}, buttons={j.get_numbuttons()}, hats={j.get_numhats()})")
        if js is None:
            js = j

    print(f"[joy] Using: {js.get_name()}")

    # Update window with controller info
    screen.fill((30, 30, 30))
    screen.blit(font.render(f"Controller: {js.get_name()}", True, (100, 255, 100)), (10, 10))
    screen.blit(font.render("L2 + Left Stick = position    D-pad = Z", True, (180, 180, 180)), (10, 35))
    screen.blit(font.render("R2 + Right Stick = orient     L1/R1 = roll", True, (180, 180, 180)), (10, 55))
    screen.blit(font.render("X=home  O=speed  tri=pose  sq=quit", True, (180, 180, 180)), (10, 80))
    pygame.display.flip()

    return js


def print_eef_pose(ik: MyCobotIK, current_deg: list[float]):
    """Pretty-print the current EEF pose."""
    R, t = ik.forward_kinematics(current_deg)
    print(f"\n  EEF position:  [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    print(f"  EEF rotation:\n{np.array2string(R, precision=4, suppress_small=True)}")
    print(f"  Joints (LCNC): {[round(v, 1) for v in current_deg]} deg\n")


def run_teleop(
    host: str,
    cmd_port: int,
    controller: str,
    initial_speed: str,
    no_rviz: bool,
    stream_port: int,
    loop_hz: int = DEFAULT_LOOP_HZ,
):
    """Main teleoperation loop."""

    # ── Initialize joystick ───────────────────────────────────────────────
    js = init_joystick()

    # ── Launch rviz2 ──────────────────────────────────────────────────────
    rviz_proc = None
    if not no_rviz:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _stream_script = os.path.join(_this_dir, "robot_pose_stream_ros2.py")
        shell_cmd = (
            "source ~/miniconda3/etc/profile.d/conda.sh && "
            "conda activate ros_env && "
            "source $CONDA_PREFIX/setup.zsh && "
            "source ~/ros2_ws/install/setup.zsh && "
            f"python3 {_stream_script} --host {host} --port {stream_port}"
        )
        print("[rviz2] Launching rviz2 + streaming client...")
        rviz_proc = subprocess.Popen(
            shell_cmd, shell=True, executable="/bin/zsh",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        time.sleep(3.0)

    def cleanup(_signum=None, _frame=None):
        if rviz_proc and rviz_proc.poll() is None:
            try:
                os.killpg(os.getpgid(rviz_proc.pid), signal.SIGTERM)
                rviz_proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(rviz_proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, lambda s, f: None)  # handle in loop

    # ── Initialize IK solver ──────────────────────────────────────────────
    print("Initializing IK solver...")
    ik = MyCobotIK()

    # ── Connect to robot ──────────────────────────────────────────────────
    robot = RobotStream(host, cmd_port)
    try:
        robot.connect()
    except (ConnectionRefusedError, socket.timeout, OSError) as e:
        print(f"ERROR: Cannot connect to robot at {host}:{cmd_port}: {e}")
        cleanup()
        sys.exit(1)

    # ── Get initial pose ──────────────────────────────────────────────────
    time.sleep(0.3)
    status = robot.drain_status()
    if status and "current_deg" in status:
        current_deg = np.array(status["current_deg"])
    else:
        print("[warn] No status from robot, assuming home pose")
        current_deg = np.array(HOME_LINUXCNC_DEG, dtype=float)

    R_current, t_current = ik.forward_kinematics(current_deg)
    print_eef_pose(ik, current_deg.tolist())

    # ── Warmup IK solver for real-time use ────────────────────────────────
    # The first solve after __init__ warmup triggers jaxls recompilation (~7s).
    # Do it here so the teleop loop is never blocked.
    print("[ik] Priming IK solver for real-time (one-time ~7s)...")
    t0 = time.perf_counter()
    ik.solve(R=R_current, t=t_current)
    prime_ms = (time.perf_counter() - t0) * 1000
    # Second solve should be fast — verify
    t0 = time.perf_counter()
    ik.solve(R=R_current, t=t_current)
    fast_ms = (time.perf_counter() - t0) * 1000
    print(f"[ik] Prime: {prime_ms:.0f}ms → subsequent: {fast_ms:.1f}ms  ✓")

    # ── Speed state ───────────────────────────────────────────────────────
    speed_idx = SPEED_ORDER.index(initial_speed)
    speed = SPEED_PRESETS[SPEED_ORDER[speed_idx]]

    # ── Control banner ────────────────────────────────────────────────────
    print("=" * 64)
    print("  JOYSTICK TELEOPERATION ACTIVE")
    print("=" * 64)
    print(f"  Speed: {speed['label']}  (○ to cycle)")
    print("  L2 + Left Stick  → position   |  D-pad ↑↓ → Z up/down")
    print("  R2 + Right Stick → orientation |  L1/R1 → roll")
    print("  ✕ = home  |  △ = print pose  |  □ = quit")
    print(f"  Loop rate: {loop_hz} Hz  |  Controller: {controller}")
    print("=" * 64)

    dt = 1.0 / loop_hz
    running = True
    circle_was_pressed = False
    cross_was_pressed = False
    triangle_was_pressed = False

    try:
        while running:
            loop_start = time.time()

            # ── Pump pygame events ────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # ── Read joystick state ───────────────────────────────────────
            lx = apply_deadzone(js.get_axis(AX_LEFT_X))
            ly = apply_deadzone(js.get_axis(AX_LEFT_Y))
            rx = apply_deadzone(js.get_axis(AX_RIGHT_X))
            ry = apply_deadzone(js.get_axis(AX_RIGHT_Y))

            # Triggers: SDL2 maps L2/R2 as axes from -1 (released) to +1 (pressed)
            l2_val = js.get_axis(AX_L2) if js.get_numaxes() > AX_L2 else -1.0
            r2_val = js.get_axis(AX_R2) if js.get_numaxes() > AX_R2 else -1.0
            l2_pressed = l2_val > TRIGGER_THRESHOLD
            r2_pressed = r2_val > TRIGGER_THRESHOLD

            # Buttons
            l1 = js.get_button(BTN_L1) if js.get_numbuttons() > BTN_L1 else 0
            r1 = js.get_button(BTN_R1) if js.get_numbuttons() > BTN_R1 else 0
            cross = js.get_button(BTN_CROSS) if js.get_numbuttons() > BTN_CROSS else 0
            circle = js.get_button(BTN_CIRCLE) if js.get_numbuttons() > BTN_CIRCLE else 0
            triangle = js.get_button(BTN_TRIANGLE) if js.get_numbuttons() > BTN_TRIANGLE else 0
            square = js.get_button(BTN_SQUARE) if js.get_numbuttons() > BTN_SQUARE else 0

            # D-pad (hat)
            hat_y = 0
            if js.get_numhats() > 0:
                _, hat_y = js.get_hat(0)

            # ── Button actions (edge-triggered) ───────────────────────────

            # □ = quit
            if square:
                print("\n[joy] □ pressed — stopping.")
                running = False
                continue

            # ○ = cycle speed (on press edge)
            if circle and not circle_was_pressed:
                speed_idx = (speed_idx + 1) % len(SPEED_ORDER)
                speed = SPEED_PRESETS[SPEED_ORDER[speed_idx]]
                print(f"[joy] Speed: {speed['label']}  (pos={speed['pos']}m/s, ori={speed['ori']}rad/s)")
            circle_was_pressed = circle

            # △ = print pose (on press edge)
            if triangle and not triangle_was_pressed:
                status = robot.drain_status()
                if status and "current_deg" in status:
                    print_eef_pose(ik, status["current_deg"])
            triangle_was_pressed = triangle

            # ✕ = home (on press edge)
            if cross and not cross_was_pressed:
                print("[joy] ✕ pressed — moving home...")
                robot.send_target(
                    list(HOME_LINUXCNC_DEG), duration=3.0,
                    controller=controller, pos_tol=0.5, settle_steps=10,
                )
                time.sleep(3.5)
                status = robot.drain_status()
                if status and "current_deg" in status:
                    current_deg = np.array(status["current_deg"])
                    R_current, t_current = ik.forward_kinematics(current_deg)
                print("[joy] Home reached. Resuming teleoperation.")
            cross_was_pressed = cross

            # ── Compute deltas ────────────────────────────────────────────
            dx = dy = dz = 0.0
            droll = dpitch = dyaw = 0.0

            # Position: L2 (deadman) + left stick + D-pad
            if l2_pressed:
                dx = -ly * speed["pos"] * dt   # stick Y → world X (forward/back)
                dy = lx * speed["pos"] * dt    # stick X → world Y (left/right)
                dz = hat_y * speed["pos"] * dt  # D-pad ↑↓ → world Z

            # Orientation: R2 (deadman) + right stick + L1/R1
            if r2_pressed:
                dyaw = -rx * speed["ori"] * dt    # stick X → yaw
                dpitch = ry * speed["ori"] * dt   # stick Y → pitch
                droll = (r1 - l1) * speed["ori"] * dt  # R1/L1 → roll

            has_input = (abs(dx) + abs(dy) + abs(dz) +
                         abs(droll) + abs(dpitch) + abs(dyaw)) > 1e-6

            # ── Apply delta and send ──────────────────────────────────────
            if has_input:
                # Update position
                t_new = t_current + np.array([dx, dy, dz])

                # Update orientation: apply small rotation to current
                dR = rotation_matrix_from_euler(droll, dpitch, dyaw)
                R_new = dR @ R_current

                # Solve IK
                t0 = time.perf_counter()
                joints_deg = ik.solve(R=R_new, t=t_new)
                ik_ms = (time.perf_counter() - t0) * 1000

                # FK verification
                R_verify, t_verify = ik.forward_kinematics(joints_deg)
                pos_err_mm = np.linalg.norm(t_new - t_verify) * 1000

                if pos_err_mm > 20.0:
                    # IK couldn't reach — don't update pose, skip this frame
                    print(f"\r[joy] IK unreachable (err={pos_err_mm:.0f}mm) — holding position  ", end="")
                else:
                    # Accept the new pose
                    R_current = R_verify
                    t_current = t_verify

                    # Send to robot
                    robot.send_target(
                        joints_deg.tolist(),
                        duration=MOVE_DURATION,
                        controller=controller,
                        pos_tol=2.0,
                        settle_steps=1,
                    )

                    dl2 = "L2" if l2_pressed else "  "
                    dr2 = "R2" if r2_pressed else "  "
                    print(f"\r[{dl2}|{dr2}] pos=[{t_current[0]:.3f},{t_current[1]:.3f},{t_current[2]:.3f}]"
                          f"  ik={ik_ms:.1f}ms  err={pos_err_mm:.1f}mm"
                          f"  Δ=[{dx:+.4f},{dy:+.4f},{dz:+.4f}]"
                          f"  spd={speed['label']:6s}", end="", flush=True)

            # ── Drain robot status to stay in sync ────────────────────────
            status = robot.drain_status()
            if status and "current_deg" in status:
                current_deg = np.array(status["current_deg"])
                # Update R_current, t_current from actual robot feedback
                # when sticks are idle (no input) to prevent drift
                if not has_input:
                    R_current, t_current = ik.forward_kinematics(current_deg)

            # ── Rate limiting ─────────────────────────────────────────────
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[joy] Interrupted.")
    finally:
        print("\n[joy] Shutting down...")
        robot.close()
        cleanup()
        pygame.quit()
        print("[joy] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="DualShock 4 joystick teleoperation for myCobot Pro 630",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  L2 + Left Stick   → position (X/Y), D-pad ↑↓ → Z
  R2 + Right Stick   → orientation (yaw/pitch), L1/R1 → roll
  ✕ = home  |  ○ = cycle speed  |  △ = print pose  |  □ = quit

Examples:
  python3.10 joystick_controller.py --host 10.0.0.27
  python3.10 joystick_controller.py --host 10.0.0.27 --speed fast
  python3.10 joystick_controller.py --host 10.0.0.27 --no-rviz --hz 20
""",
    )
    parser.add_argument("--host", required=True,
                        help="Robot controller IP (e.g., 10.0.0.27)")
    parser.add_argument("--cmd-port", type=int, default=9998,
                        help="Robot command port (default: 9998)")
    parser.add_argument("--stream-port", type=int, default=9999,
                        help="Robot streaming port for rviz2 (default: 9999)")
    parser.add_argument("--controller", choices=["pd", "mpc"], default="pd",
                        help="Robot-side controller (default: pd)")
    parser.add_argument("--speed", choices=SPEED_ORDER, default="slow",
                        help="Initial speed preset (default: slow)")
    parser.add_argument("--hz", type=int, default=DEFAULT_LOOP_HZ,
                        help=f"Control loop rate in Hz (default: {DEFAULT_LOOP_HZ})")
    parser.add_argument("--no-rviz", action="store_true",
                        help="Don't launch rviz2")
    args = parser.parse_args()

    run_teleop(
        host=args.host,
        cmd_port=args.cmd_port,
        controller=args.controller,
        initial_speed=args.speed,
        no_rviz=args.no_rviz,
        stream_port=args.stream_port,
        loop_hz=args.hz,
    )


if __name__ == "__main__":
    main()
