#!/usr/bin/env python3
"""
Inverse Kinematics solver for myCobot Pro 630 using PyRoKi.

Solves IK for the 'eef' frame defined in the URDF, given a target (R, t) pose.
Handles calibration between URDF joint space and LinuxCNC joint space.

Usage:
    # As a module:
    from ik_pyroki import MyCobotIK
    ik = MyCobotIK()
    joints_linuxcnc_deg = ik.solve(R=R_3x3, t=t_3)

    # Demo (FK + IK round-trip verification):
    python3 ik_pyroki.py
    python3 ik_pyroki.py --target-linuxcnc -80 -85 5 -85 5 5
"""

import math
import os
import time
from typing import Optional  # noqa: F401 – used in type hints

# Force JAX to use CPU (Apple Metal GPU has incomplete support)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

# ═══════════════════════════════════════════════════════════════════════════════
#  Calibration: LinuxCNC ↔ URDF joint angle mapping
# ═══════════════════════════════════════════════════════════════════════════════
#
#  urdf_rad[i] = JOINT_SIGNS[i] * deg2rad(linuxcnc_deg[i] + JOINT_OFFSETS_DEG[i])
#
#  Robot upright pose:  LinuxCNC = [-90, -90,  0, -90, 0, 0] deg
#                       URDF     = [-90,   0,  0,   0, 0, 0] deg

MAX_JOINTS = 6
JOINT_NAMES = [f"joint{i+1}" for i in range(MAX_JOINTS)]
JOINT_SIGNS = [+1, +1, +1, +1, +1, +1]
JOINT_OFFSETS_DEG = [0.0, 90.0, 0.0, 90.0, 0.0, 0.0]

# Home pose in LinuxCNC degrees
HOME_LINUXCNC_DEG = [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0]

# LinuxCNC per-joint soft limits (degrees) from elerob_mpc.ini [JOINT_N] sections.
# These can differ from URDF limits because of the calibration offset.
LINUXCNC_SOFT_LIMITS_DEG = [
    (-360.0, 360.0),   # Joint 0
    (-360.0, 360.0),   # Joint 1
    (-160.0, 160.0),   # Joint 2
    (-180.0, 180.0),   # Joint 3
    (-180.0, 180.0),   # Joint 4
    (-180.0, 180.0),   # Joint 5
]

# URDF path (relative to this file or absolute)
DEFAULT_URDF_PATH = os.path.join(
    os.path.expanduser("~"),
    "ros2_ws/src/mycobot_description/urdf/mycobot_pro_630.urdf",
)

# Target link for IK
TARGET_LINK = "eef"


def linuxcnc_deg_to_urdf_rad(linuxcnc_deg: np.ndarray) -> np.ndarray:
    """Convert LinuxCNC joint angles (degrees) → URDF joint angles (radians)."""
    linuxcnc_deg = np.asarray(linuxcnc_deg, dtype=np.float64)
    return np.array([
        JOINT_SIGNS[i] * math.radians(linuxcnc_deg[i] + JOINT_OFFSETS_DEG[i])
        for i in range(MAX_JOINTS)
    ])


def urdf_rad_to_linuxcnc_deg(urdf_rad: np.ndarray) -> np.ndarray:
    """Convert URDF joint angles (radians) → LinuxCNC joint angles (degrees)."""
    urdf_rad = np.asarray(urdf_rad, dtype=np.float64)
    return np.array([
        math.degrees(urdf_rad[i]) / JOINT_SIGNS[i] - JOINT_OFFSETS_DEG[i]
        for i in range(MAX_JOINTS)
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  URDF loading helper
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_effective_joint_limits(robot: pk.Robot,
                                  margin_deg: float = 1.0) -> None:
    """Override robot joint limits with the intersection of URDF and LinuxCNC limits.

    The URDF and LinuxCNC controller can have different limits, and the
    calibration offset (JOINT_OFFSETS_DEG) shifts the mapping between them.
    The IK solver must respect BOTH, so we compute the tighter envelope.

    A safety margin (default 1°) is applied inward to avoid hitting the
    exact boundary due to floating-point rounding.
    """
    n_act = robot.joints.num_actuated_joints
    assert n_act == MAX_JOINTS

    margin_rad = math.radians(margin_deg)

    urdf_lower = np.array(robot.joints.lower_limits)
    urdf_upper = np.array(robot.joints.upper_limits)

    eff_lower = np.copy(urdf_lower)
    eff_upper = np.copy(urdf_upper)

    for i in range(n_act):
        lcnc_lo, lcnc_hi = LINUXCNC_SOFT_LIMITS_DEG[i]
        # Map LinuxCNC limits into URDF space:
        #   urdf_rad = sign * deg2rad(lcnc_deg + offset)
        mapped_lo = JOINT_SIGNS[i] * math.radians(lcnc_lo + JOINT_OFFSETS_DEG[i])
        mapped_hi = JOINT_SIGNS[i] * math.radians(lcnc_hi + JOINT_OFFSETS_DEG[i])
        if mapped_lo > mapped_hi:
            mapped_lo, mapped_hi = mapped_hi, mapped_lo

        old_lo, old_hi = eff_lower[i], eff_upper[i]
        eff_lower[i] = max(float(urdf_lower[i]), mapped_lo + margin_rad)
        eff_upper[i] = min(float(urdf_upper[i]), mapped_hi - margin_rad)

        if eff_lower[i] != old_lo or eff_upper[i] != old_hi:
            print(f"  [limits] J{i+1}: URDF [{math.degrees(old_lo):.1f}, {math.degrees(old_hi):.1f}]° "
                  f"→ effective [{math.degrees(eff_lower[i]):.1f}, {math.degrees(eff_upper[i]):.1f}]° "
                  f"(LinuxCNC offset {JOINT_OFFSETS_DEG[i]}°, margin {margin_deg}°)")

    # Replace actuated-joint limits
    new_lower_act = jnp.array(eff_lower, dtype=jnp.float32)
    new_upper_act = jnp.array(eff_upper, dtype=jnp.float32)
    object.__setattr__(robot.joints, "lower_limits", new_lower_act)
    object.__setattr__(robot.joints, "upper_limits", new_upper_act)

    # Replace all-joint limits (includes fixed joints at indices from actuated_indices)
    new_lower_all = np.array(robot.joints.lower_limits_all)
    new_upper_all = np.array(robot.joints.upper_limits_all)
    for j_all in range(robot.joints.num_joints):
        act_idx = robot.joints.actuated_indices[j_all]
        if act_idx >= 0:
            new_lower_all[j_all] = eff_lower[act_idx]
            new_upper_all[j_all] = eff_upper[act_idx]
    object.__setattr__(robot.joints, "lower_limits_all",
                       jnp.array(new_lower_all, dtype=jnp.float32))
    object.__setattr__(robot.joints, "upper_limits_all",
                       jnp.array(new_upper_all, dtype=jnp.float32))


def _scale_robot_collision(robot_coll: pk.collision.RobotCollision,
                           scale: float) -> pk.collision.RobotCollision:
    """Scale collision capsule geometry (e.g. 0.001 to convert mm → m).

    Scales both the capsule dimensions (radius, height) and the pose
    translation (offset from link frame). Needed because myCobot DAE
    meshes are authored in millimeters but the URDF uses meters.
    """
    old_coll = robot_coll.coll
    old_pose = old_coll.pose

    # Scale translation while preserving rotation
    scaled_pose = jaxlie.SE3.from_rotation_and_translation(
        old_pose.rotation(),
        old_pose.translation() * scale,
    )
    scaled_coll = pk.collision.Capsule(
        pose=scaled_pose,
        size=old_coll.size * scale,
    )
    return pk.collision.RobotCollision(
        num_links=robot_coll.num_links,
        link_names=robot_coll.link_names,
        coll=scaled_coll,
        active_idx_i=robot_coll.active_idx_i,
        active_idx_j=robot_coll.active_idx_j,
    )


def _load_urdf(urdf_path: str) -> yourdfpy.URDF:
    """Load the myCobot URDF, resolving package:// mesh paths.

    build_scene_graph must be True so yourdfpy computes base_link,
    which pyroki's topological sort requires.
    """
    pkg_dir = os.path.dirname(os.path.dirname(urdf_path))  # mycobot_description/

    def filename_handler(fname: str) -> str:
        prefix = "package://mycobot_description/"
        if fname.startswith(prefix):
            return os.path.join(pkg_dir, fname[len(prefix):])
        return fname

    return yourdfpy.URDF.load(
        urdf_path,
        filename_handler=filename_handler,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Collision geometry — ground plane and wall
# ═══════════════════════════════════════════════════════════════════════════════

# Ground at z=0 (outward normal = +z → keeps robot above ground)
GROUND_PLANE = pk.collision.HalfSpace.from_point_and_normal(
    point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0]
)
# Wall at x=-0.1 (outward normal = +x → keeps robot on x > -0.1 side)
WALL_X = pk.collision.HalfSpace.from_point_and_normal(
    point=[-0.1, 0.0, 0.0], normal=[1.0, 0.0, 0.0]
)

# Collision margins (meters)
SELF_COLLISION_MARGIN = 0.02   # 2 cm buffer between links
WORLD_COLLISION_MARGIN = 0.01  # 1 cm buffer from ground / wall


# ═══════════════════════════════════════════════════════════════════════════════
#  JIT-compiled IK solver (inner loop)
# ═══════════════════════════════════════════════════════════════════════════════

@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    ground: pk.collision.HalfSpace,
    wall: pk.collision.HalfSpace,
    pos_weight: jdc.Static[float] = 50.0,
    ori_weight: jdc.Static[float] = 10.0,
    self_coll_weight: jdc.Static[float] = 10.0,
) -> jax.Array:
    """Solve IK with pose, joint limits, self-collision, and world collision.

    Returns URDF joint configuration in radians, shape (num_actuated_joints,).
    """
    joint_var = robot.joint_var_cls(0)
    target_se3 = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(target_wxyz), target_position
    )
    costs = [
        # Primary: reach the target pose
        pk.costs.pose_cost_analytic_jac(
            robot, joint_var, target_se3, target_link_index,
            pos_weight=pos_weight, ori_weight=ori_weight,
        ),
        # Joint limits (hard constraint — stay within URDF limits)
        pk.costs.limit_constraint(robot, joint_var),
        # Self-collision avoidance (soft cost — pushes links apart)
        pk.costs.self_collision_cost(
            robot, robot_coll, joint_var,
            margin=SELF_COLLISION_MARGIN, weight=self_coll_weight,
        ),
        # World collision: ground plane (soft cost)
        pk.costs.world_collision_cost(
            robot, robot_coll, joint_var, ground,
            margin=WORLD_COLLISION_MARGIN, weight=self_coll_weight,
        ),
        # World collision: wall at x=-0.1 (soft cost)
        pk.costs.world_collision_cost(
            robot, robot_coll, joint_var, wall,
            margin=WORLD_COLLISION_MARGIN, weight=self_coll_weight,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(),
            termination=jaxls.TerminationConfig(
                max_iterations=200,
                cost_tolerance=1e-6,
                gradient_tolerance=1e-5,
                parameter_tolerance=1e-7,
            ),
        )
    )
    return sol[joint_var]


# ═══════════════════════════════════════════════════════════════════════════════
#  MyCobotIK class
# ═══════════════════════════════════════════════════════════════════════════════

class MyCobotIK:
    """Inverse Kinematics solver for myCobot Pro 630 using PyRoKi.

    Handles:
      - Loading the URDF and building the pyroki Robot
      - Forward kinematics (joint angles → eef pose)
      - Inverse kinematics (eef pose → joint angles)
      - Calibration between LinuxCNC and URDF joint conventions

    Example:
        ik = MyCobotIK()

        # FK: get eef pose from LinuxCNC joint angles
        R, t = ik.forward_kinematics([-90, -90, 0, -90, 0, 0])

        # IK: solve for joint angles given eef pose
        joints_deg = ik.solve(R=R, t=t)
    """

    def __init__(self, urdf_path: str = DEFAULT_URDF_PATH):
        print(f"[MyCobotIK] Loading URDF: {urdf_path}")
        self.urdf = _load_urdf(urdf_path)
        self.robot = pk.Robot.from_urdf(
            self.urdf,
            default_joint_cfg=linuxcnc_deg_to_urdf_rad(HOME_LINUXCNC_DEG),
        )

        # Tighten joint limits to respect LinuxCNC soft limits + calibration offset
        _apply_effective_joint_limits(self.robot)

        # Build collision model from URDF meshes
        # Scale by 0.001: myCobot DAE meshes are authored in mm, URDF uses meters
        self.robot_coll = _scale_robot_collision(
            pk.collision.RobotCollision.from_urdf(self.urdf), scale=0.001
        )

        # Resolve target link index
        self.target_link_name = TARGET_LINK
        self.target_link_index = self.robot.links.names.index(self.target_link_name)
        print(f"[MyCobotIK] Robot loaded: {self.robot.joints.num_actuated_joints} actuated joints")
        print(f"[MyCobotIK] Target link: '{self.target_link_name}' (index {self.target_link_index})")
        print(f"[MyCobotIK] Links: {self.robot.links.names}")
        print(f"[MyCobotIK] Actuated joints: {self.robot.joints.num_actuated_joints}")
        print(f"[MyCobotIK] Collision: self-coll margin={SELF_COLLISION_MARGIN*100:.0f}cm, "
              f"world margin={WORLD_COLLISION_MARGIN*100:.0f}cm")

        # Default seed for IK (home pose in URDF radians)
        self._home_urdf_rad = linuxcnc_deg_to_urdf_rad(HOME_LINUXCNC_DEG)

        # Warm up JIT (first call is slow due to compilation + collision tracing)
        print("[MyCobotIK] Warming up JIT (first IK solve with collision)...")
        t0 = time.time()
        R, t_vec = self.forward_kinematics_urdf(self._home_urdf_rad)
        print(f"[MyCobotIK] Home EEF position: {np.round(t_vec, 4).tolist()} m")
        wxyz = rotation_matrix_to_wxyz(R)
        _solve_ik_jax(
            self.robot,
            self.robot_coll,
            jnp.array(self.target_link_index),
            jnp.array(wxyz, dtype=jnp.float32),
            jnp.array(t_vec, dtype=jnp.float32),
            GROUND_PLANE,
            WALL_X,
        )
        elapsed = time.time() - t0
        print(f"[MyCobotIK] JIT warmup done ({elapsed:.2f}s). Subsequent solves are fast.\n")

    # ── Forward Kinematics ────────────────────────────────────────────────

    def forward_kinematics_urdf(self, urdf_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK: URDF joint angles (radians) → eef pose (R, t).

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector (meters)
        """
        cfg = jnp.array(urdf_rad, dtype=jnp.float32)
        all_poses = self.robot.forward_kinematics(cfg)
        eef_pose_wxyz_xyz = all_poses[self.target_link_index]
        eef_se3 = jaxlie.SE3(eef_pose_wxyz_xyz)

        R = np.array(eef_se3.rotation().as_matrix())
        t = np.array(eef_se3.translation())
        return R, t

    def forward_kinematics(self, linuxcnc_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK: LinuxCNC joint angles (degrees) → eef pose (R, t).

        Returns:
            R: (3, 3) rotation matrix
            t: (3,) translation vector (meters)
        """
        urdf_rad = linuxcnc_deg_to_urdf_rad(linuxcnc_deg)
        return self.forward_kinematics_urdf(urdf_rad)

    # ── Inverse Kinematics ────────────────────────────────────────────────

    def solve_urdf(
        self,
        R: np.ndarray,
        t: np.ndarray,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
    ) -> np.ndarray:
        """IK: eef pose (R, t) → URDF joint angles (radians).

        Args:
            R: (3, 3) rotation matrix for the eef frame
            t: (3,) translation vector (meters) for the eef frame
            pos_weight: Position error weight (default: 50.0)
            ori_weight: Orientation error weight (default: 10.0)

        Returns:
            urdf_rad: (6,) URDF joint angles in radians
        """
        wxyz = rotation_matrix_to_wxyz(R)
        cfg = _solve_ik_jax(
            self.robot,
            self.robot_coll,
            jnp.array(self.target_link_index),
            jnp.array(wxyz, dtype=jnp.float32),
            jnp.array(t, dtype=jnp.float32),
            GROUND_PLANE,
            WALL_X,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
        )
        return np.array(cfg)

    def solve(
        self,
        R: np.ndarray,
        t: np.ndarray,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
    ) -> np.ndarray:
        """IK: eef pose (R, t) → LinuxCNC joint angles (degrees).

        Args:
            R: (3, 3) rotation matrix for the eef frame
            t: (3,) translation vector (meters) for the eef frame
            pos_weight: Position error weight (default: 50.0)
            ori_weight: Orientation error weight (default: 10.0)

        Returns:
            linuxcnc_deg: (6,) joint angles in LinuxCNC degrees
        """
        urdf_rad = self.solve_urdf(R, t, pos_weight=pos_weight,
                                   ori_weight=ori_weight)
        return urdf_rad_to_linuxcnc_deg(urdf_rad)

    def solve_from_matrix(
        self,
        T: np.ndarray,
        pos_weight: float = 50.0,
        ori_weight: float = 10.0,
    ) -> np.ndarray:
        """IK: 4x4 homogeneous transform → LinuxCNC joint angles (degrees).

        Args:
            T: (4, 4) homogeneous transformation matrix for the eef frame
            pos_weight: Position error weight
            ori_weight: Orientation error weight

        Returns:
            linuxcnc_deg: (6,) joint angles in LinuxCNC degrees
        """
        R = T[:3, :3]
        t = T[:3, 3]
        return self.solve(R, t, pos_weight, ori_weight)


# ═══════════════════════════════════════════════════════════════════════════════
#  Rotation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def wxyz_to_rotation_matrix(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo / CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="myCobot Pro 630 IK solver using PyRoKi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 ik_pyroki.py                                         # FK+IK round-trip at home
  python3 ik_pyroki.py --target-linuxcnc -80 -85 5 -85 5 5    # FK+IK for specific pose
  python3 ik_pyroki.py --target-xyz 0.3 0.0 0.4                # IK for a Cartesian position
""",
    )
    parser.add_argument("--urdf", default=DEFAULT_URDF_PATH,
        help="Path to URDF file")
    parser.add_argument("--target-linuxcnc", nargs=6, type=float, default=None,
        help="Target pose in LinuxCNC degrees (6 values). Runs FK→IK round-trip.")
    parser.add_argument("--target-xyz", nargs=3, type=float, default=None,
        help="Target eef position [x, y, z] in meters (uses home orientation)")
    args = parser.parse_args()

    ik = MyCobotIK(urdf_path=args.urdf)

    # ── Demo 1: FK at home pose ──────────────────────────────────────────
    print("=" * 70)
    print("Demo 1: Forward Kinematics at home pose")
    print("=" * 70)
    home_deg = np.array(HOME_LINUXCNC_DEG)
    R_home, t_home = ik.forward_kinematics(home_deg)
    print(f"  LinuxCNC angles: {home_deg.tolist()} deg")
    print(f"  URDF angles:     {np.round(np.degrees(linuxcnc_deg_to_urdf_rad(home_deg)), 2).tolist()} deg")
    print(f"  EEF position:    {np.round(t_home, 5).tolist()} m")
    print(f"  EEF rotation:\n{np.round(R_home, 5)}")

    # ── Demo 2: FK→IK round-trip ─────────────────────────────────────────
    target_linuxcnc = args.target_linuxcnc or [-80.0, -85.0, 5.0, -85.0, 5.0, 5.0]
    print(f"\n{'=' * 70}")
    print("Demo 2: FK → IK round-trip")
    print("=" * 70)
    target_deg = np.array(target_linuxcnc)
    R_target, t_target = ik.forward_kinematics(target_deg)
    print(f"  Input LinuxCNC:  {target_deg.tolist()} deg")
    print(f"  FK → EEF pos:    {np.round(t_target, 5).tolist()} m")

    t0 = time.time()
    solved_deg = ik.solve(R=R_target, t=t_target)
    solve_ms = (time.time() - t0) * 1000
    print(f"  IK solve time:   {solve_ms:.2f} ms")
    print(f"  IK → LinuxCNC:   {np.round(solved_deg, 3).tolist()} deg")

    error_deg = np.abs(target_deg - solved_deg)
    print(f"  Round-trip error: {np.round(error_deg, 4).tolist()} deg")
    print(f"  Max error:        {np.max(error_deg):.4f} deg")

    # Verify FK of solved angles matches target
    R_verify, t_verify = ik.forward_kinematics(solved_deg)
    pos_error = np.linalg.norm(t_target - t_verify)
    print(f"  Position error:   {pos_error * 1000:.4f} mm")

    # ── Demo 3: IK from Cartesian target ──────────────────────────────────
    if args.target_xyz:
        print(f"\n{'=' * 70}")
        print("Demo 3: IK from Cartesian position (with home orientation)")
        print("=" * 70)
        t_cart = np.array(args.target_xyz)
        print(f"  Target position: {t_cart.tolist()} m")
        print(f"  Target orientation: (using home rotation)")

        t0 = time.time()
        solved_cart = ik.solve(R=R_home, t=t_cart)
        solve_ms = (time.time() - t0) * 1000
        print(f"  IK solve time:   {solve_ms:.2f} ms")
        print(f"  LinuxCNC angles: {np.round(solved_cart, 3).tolist()} deg")

        R_v, t_v = ik.forward_kinematics(solved_cart)
        pos_err = np.linalg.norm(t_cart - t_v)
        print(f"  Achieved pos:    {np.round(t_v, 5).tolist()} m")
        print(f"  Position error:  {pos_err * 1000:.4f} mm")

    # ── Demo 4: Timing benchmark ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Demo 4: IK solve timing (100 iterations)")
    print("=" * 70)
    times = []
    for _ in range(100):
        t0 = time.time()
        ik.solve(R=R_target, t=t_target)
        times.append((time.time() - t0) * 1000)
    times = np.array(times)
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Min:    {np.min(times):.2f} ms")
    print(f"  Max:    {np.max(times):.2f} ms")


if __name__ == "__main__":
    main()
