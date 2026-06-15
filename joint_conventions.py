"""
Single source of truth for the myCobot Pro 630 joint conventions.

Both a LinuxCNC joint space (degrees, used by HAL/robot_hal and the desktop
clients) and a URDF joint space (radians, used by PyRoKi IK and the dynamics
model) describe the same arm. They differ by a per-joint sign and offset:

    urdf_rad[i] = JOINT_SIGNS[i] * deg2rad(linuxcnc_deg[i] + JOINT_OFFSETS_DEG[i])

    Upright pose:  LinuxCNC = [-90, -90, 0, -90, 0, 0] deg
                   URDF     = [-90,   0, 0,   0, 0, 0] deg

These constants and conversions were previously duplicated across
invdyn_model.py, identify_invdyn_from_log.py, ik_pyroki.py, control_robot.py,
and robot_pose_stream_ros2.py. Import them from here instead.
"""

import math
import os

import numpy as np

# ── Counts / names ──────────────────────────────────────────────────────────
MAX_JOINTS = 6
NUM_JOINTS = MAX_JOINTS  # alias used by the dynamics/identification modules
JOINT_NAMES = [f"joint{i + 1}" for i in range(MAX_JOINTS)]

# ── LinuxCNC ↔ URDF calibration ─────────────────────────────────────────────
JOINT_SIGNS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
JOINT_OFFSETS_DEG = [0.0, 90.0, 0.0, 90.0, 0.0, 0.0]

# Home / rest pose in LinuxCNC degrees
HOME_LINUXCNC_DEG = [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0]

# LinuxCNC per-joint soft limits (degrees), from the machine [JOINT_N] sections.
# These can differ from URDF limits because of the calibration offset above.
LINUXCNC_SOFT_LIMITS_DEG = [
    (-360.0, 360.0),   # Joint 0
    (-360.0, 360.0),   # Joint 1
    (-160.0, 160.0),   # Joint 2
    (-180.0, 180.0),   # Joint 3
    (-180.0, 180.0),   # Joint 4
    (-180.0, 180.0),   # Joint 5
]

# ── URDF / IK ───────────────────────────────────────────────────────────────
DEFAULT_URDF_PATH = os.path.join(
    os.path.expanduser("~"),
    "ros2_ws/src/mycobot_description/urdf/mycobot_pro_630.urdf",
)
TARGET_LINK = "eef"


# ── Conversions ─────────────────────────────────────────────────────────────
def linuxcnc_deg_to_rad(deg) -> np.ndarray:
    """LinuxCNC joint angles (deg) → URDF joint angles (rad)."""
    deg = np.asarray(deg, dtype=float)
    return np.array([
        JOINT_SIGNS[i] * math.radians(deg[i] + JOINT_OFFSETS_DEG[i])
        for i in range(MAX_JOINTS)
    ])


def rad_to_linuxcnc_deg(rad) -> np.ndarray:
    """URDF joint angles (rad) → LinuxCNC joint angles (deg)."""
    rad = np.asarray(rad, dtype=float)
    return np.array([
        math.degrees(rad[i]) / JOINT_SIGNS[i] - JOINT_OFFSETS_DEG[i]
        for i in range(MAX_JOINTS)
    ])


# Names used by ik_pyroki.py — identical functions, kept for clarity at call sites.
linuxcnc_deg_to_urdf_rad = linuxcnc_deg_to_rad
urdf_rad_to_linuxcnc_deg = rad_to_linuxcnc_deg
