#!/usr/bin/env python3
"""curobo_controller_node :: cuRobo-based ROS 2 motion controller for the myCobot Pro 630.

Pipeline
--------
    /joint_states (current q, URDF rad)
          +  goal  (Cartesian pose OR joint target)
          v
    curobo_planner_server.py   (GPU MotionGen, separate `curobo` conda env)
          v
    collision-free trajectory (URDF rad + dt)
          v   rad -> LinuxCNC deg  (joint_conventions)
    /mycobot/cmd/move  {"trajectory":[[..deg..],...], "traj_dt":dt, ...}
          v
    mycobot_ros2_bridge.py -> robot_hal.py   (tracks the trajectory)

cuRobo (Py3.10 + CUDA) and rclpy (system Humble, Py3.8) cannot share a process,
so the GPU planning lives in ``curobo_planner_server.py`` and this node talks to
it over a newline-JSON TCP socket -- the same pattern robot_hal.py already uses
for the 9998/9999 servers.

Subscribed
----------
    /joint_states                sensor_msgs/JointState   current q (URDF rad)
    /mycobot/curobo/goal_pose    geometry_msgs/PoseStamped  EE goal in base_link
    /mycobot/curobo/goal_joint   sensor_msgs/JointState     joint goal (URDF rad)

Published
---------
    /mycobot/cmd/move            std_msgs/String   trajectory command for the bridge
    /mycobot/curobo/status       std_msgs/String   plan result JSON

Safety
------
Planning always runs, but the trajectory is sent to the real robot ONLY when
``--execute`` is given. Without it the node does a dry run: it plans, logs, and
publishes status, but never moves the arm.

Run (system ROS env, after `source ros2node/config/ros2node.env`):
    # 1) in the curobo env, on the same machine:
    #      python curobo_planner_server.py
    # 2) the bridge must be up (python3 mycobot_ros2_bridge.py --robot-host ...)
    # 3) then:
    python3 curobo_controller_node.py --execute
"""
import argparse
import json
import socket
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS, JOINT_NAMES, rad_to_linuxcnc_deg


class CuroboController(Node):
    def __init__(self, planner_host, planner_port, controller, execute, max_attempts):
        super().__init__("curobo_controller")
        self._phost = planner_host
        self._pport = planner_port
        self._controller = controller
        self._execute = execute
        self._max_attempts = max_attempts
        self._latest_q = None             # dict: joint name -> rad
        self._plan_lock = threading.Lock()

        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(PoseStamped, "/mycobot/curobo/goal_pose",
                                 self._on_goal_pose, 10)
        self.create_subscription(JointState, "/mycobot/curobo/goal_joint",
                                 self._on_goal_joint, 10)
        self._pub_cmd = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self._pub_status = self.create_publisher(String, "/mycobot/curobo/status", 10)

        self.get_logger().info(
            f"curobo_controller up. planner={planner_host}:{planner_port} "
            f"controller={controller} execute={execute}")
        if not execute:
            self.get_logger().warn("DRY RUN: plans will NOT be sent to the robot "
                                   "(pass --execute to move the arm)")
        self._check_planner()

    # ---- planner socket round-trip (one JSON line each way) ----
    def _planner_request(self, req, timeout=30.0):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect((self._phost, self._pport))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = ""
            while "\n" not in buf:
                data = s.recv(1 << 20)
                if not data:
                    break
                buf += data.decode("utf-8", errors="replace")
            line = buf.split("\n", 1)[0].strip()
            return json.loads(line) if line else None
        finally:
            s.close()

    def _check_planner(self):
        try:
            resp = self._planner_request({"type": "ping"}, timeout=5.0)
            if resp and resp.get("ok"):
                self.get_logger().info(
                    f"planner ready: dof={resp.get('dof')} ee={resp.get('ee_link')}")
            else:
                self.get_logger().warn(f"planner ping returned: {resp}")
        except OSError as e:
            self.get_logger().warn(
                f"planner not reachable at {self._phost}:{self._pport} ({e}); "
                f"start curobo_planner_server.py in the curobo env")

    # ---- current joint state ----
    def _on_js(self, msg: JointState):
        if msg.name:
            self._latest_q = {n: p for n, p in zip(msg.name, msg.position)}
        elif len(msg.position) >= MAX_JOINTS:
            self._latest_q = {JOINT_NAMES[i]: msg.position[i] for i in range(MAX_JOINTS)}

    def _current_q(self):
        if self._latest_q is None:
            return None
        try:
            return [float(self._latest_q[n]) for n in JOINT_NAMES]
        except KeyError:
            return None

    # ---- goal callbacks (plan off the executor thread) ----
    def _on_goal_pose(self, msg: PoseStamped):
        q = self._current_q()
        if q is None:
            self.get_logger().warn("no /joint_states yet; ignoring pose goal")
            return
        p, o = msg.pose.position, msg.pose.orientation
        goal = [p.x, p.y, p.z, o.w, o.x, o.y, o.z]   # cuRobo wants [xyz, qw qx qy qz]
        req = {"type": "plan_pose", "start_q": q, "goal_pose": goal,
               "max_attempts": self._max_attempts}
        label = f"pose [{p.x:.3f}, {p.y:.3f}, {p.z:.3f}]"
        threading.Thread(target=self._plan_and_exec, args=(req, label),
                         daemon=True).start()

    def _on_goal_joint(self, msg: JointState):
        q = self._current_q()
        if q is None:
            self.get_logger().warn("no /joint_states yet; ignoring joint goal")
            return
        if msg.name:
            lut = dict(zip(msg.name, msg.position))
            try:
                goal_q = [float(lut[n]) for n in JOINT_NAMES]
            except KeyError:
                self.get_logger().warn("joint goal missing a joint name; ignoring")
                return
        elif len(msg.position) >= MAX_JOINTS:
            goal_q = [float(msg.position[i]) for i in range(MAX_JOINTS)]
        else:
            self.get_logger().warn("joint goal has too few positions; ignoring")
            return
        req = {"type": "plan_joint", "start_q": q, "goal_q": goal_q,
               "max_attempts": self._max_attempts}
        threading.Thread(target=self._plan_and_exec, args=(req, "joint goal"),
                         daemon=True).start()

    def _plan_and_exec(self, req, label):
        if not self._plan_lock.acquire(blocking=False):
            self.get_logger().warn("a plan is already running; ignoring new goal")
            return
        try:
            self.get_logger().info(f"planning to {label} ...")
            try:
                resp = self._planner_request(req)
            except OSError as e:
                self.get_logger().error(f"planner request failed: {e}")
                self._publish_status({"success": False,
                                      "status": f"planner unreachable: {e}"})
                return
            if not resp:
                self.get_logger().error("empty planner response")
                self._publish_status({"success": False, "status": "empty response"})
                return

            ok = bool(resp.get("success"))
            traj = resp.get("trajectory") or []
            n = len(traj)
            self.get_logger().info(
                f"plan success={ok} status={resp.get('status')} pts={n} "
                f"solve={resp.get('solve_time', 0.0):.3f}s "
                f"motion_time={resp.get('motion_time', 0.0):.2f}s")

            status = {k: resp.get(k) for k in
                      ("success", "status", "solve_time", "motion_time")}
            status["n_points"] = n
            status["executed"] = False
            if ok and n > 0:
                if self._execute:
                    self._send_trajectory(traj, resp.get("dt"))
                    status["executed"] = True
                    self.get_logger().info(
                        f"-> sent trajectory to /mycobot/cmd/move ({n} pts, "
                        f"dt={resp.get('dt')}s, controller={self._controller})")
                else:
                    self.get_logger().info("dry run: trajectory NOT sent "
                                           "(pass --execute to move the arm)")
            self._publish_status(status)
        finally:
            self._plan_lock.release()

    def _send_trajectory(self, traj_rad, dt):
        # cuRobo plans in URDF radians; the bridge's trajectory field is LinuxCNC degrees.
        traj_deg = [rad_to_linuxcnc_deg(wp).tolist() for wp in traj_rad]
        cmd = {
            "trajectory": traj_deg,
            "traj_dt": float(dt),
            "target_deg": traj_deg[-1],
            "controller": self._controller,
        }
        m = String()
        m.data = json.dumps(cmd)
        self._pub_cmd.publish(m)

    def _publish_status(self, d):
        m = String()
        m.data = json.dumps(d)
        self._pub_status.publish(m)


def main():
    p = argparse.ArgumentParser(
        description="cuRobo-based ROS 2 motion controller for the myCobot Pro 630.")
    p.add_argument("--planner-host", default="127.0.0.1",
                   help="host of curobo_planner_server.py")
    p.add_argument("--planner-port", type=int, default=9997)
    p.add_argument("--controller", default="pid",
                   choices=["pid", "invdyn", "pd_velff", "mpc"],
                   help="low-level controller the robot uses to track the trajectory")
    p.add_argument("--max-attempts", type=int, default=5,
                   help="cuRobo plan attempts per goal")
    p.add_argument("--execute", action="store_true",
                   help="actually send the planned trajectory to the robot "
                        "(default: dry run, no motion)")
    args, _ = p.parse_known_args()

    rclpy.init()
    node = CuroboController(args.planner_host, args.planner_port,
                            args.controller, args.execute, args.max_attempts)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
