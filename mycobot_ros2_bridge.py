#!/usr/bin/env python3
"""
mycobot_ros2_bridge :: expose the myCobot Pro 630 robot controller as a ROS 2 node.

Architecture
------------
The real-time controller ``robot_hal.py`` runs *inside* LinuxCNC (loaded by
``elerob.hal``) and owns the HAL pins. It already exposes two TCP servers:

    * STREAM_PORT 9999 - emits ``{"joints_deg":[...], "timestamp":...}`` @ 50 Hz
    * CMD_PORT    9998 - accepts ``{"target_deg":[...], ...}`` and streams status

HAL can only be touched from within that LinuxCNC process, so this ROS 2 node
does NOT re-implement control - it *bridges* those TCP servers to ROS 2 topics.
Run it on the Pi (``--robot-host 127.0.0.1``) so the Pi becomes a first-class
ROS 2 fleet node, or anywhere on the network (``--robot-host 10.0.0.27``).

Published topics
----------------
    /joint_states              sensor_msgs/JointState   URDF radians (rviz-ready)
    /mycobot/joint_states_deg  sensor_msgs/JointState   raw LinuxCNC degrees
    /mycobot/status            std_msgs/String          controller status JSON

Subscribed topics
-----------------
    /mycobot/cmd/joint_deg     std_msgs/Float64MultiArray  6 target angles, deg
    /mycobot/cmd/joint_rad     sensor_msgs/JointState      6 targets, URDF rad
    /mycobot/cmd/move          std_msgs/String             raw JSON passthrough
                                 e.g. {"target_deg":[...],"duration":3,"controller":"pid"}

Services
--------
    /mycobot/home              std_srvs/Trigger            move to HOME pose

Usage
-----
    source <ros2 env>
    python3 mycobot_ros2_bridge.py --robot-host 10.0.0.27
"""

import argparse
import json
import math
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray
from std_srvs.srv import Trigger

from joint_conventions import (
    MAX_JOINTS,
    JOINT_NAMES,
    JOINT_SIGNS,
    JOINT_OFFSETS_DEG,
    HOME_LINUXCNC_DEG,
    LINUXCNC_SOFT_LIMITS_DEG,
    rad_to_linuxcnc_deg,
)

DEFAULT_STREAM_PORT = 9999
DEFAULT_CMD_PORT = 9998


def _epoch_to_stamp(ts):
    """Unix epoch seconds (float) -> builtin_interfaces/Time.

    robot_hal.py stamps each stream packet with time.time() on the Pi. Using
    that source time as the ROS header stamp (instead of the desktop's receive
    time) keeps the timestamp tied to when the joints were actually sampled, so
    it is consistent across the fleet -- PROVIDED the Pi and desktop clocks are
    NTP/chrony-synced (see /mycobot/clock_offset_ms logged below).
    """
    sec = int(ts)
    nanosec = int(round((ts - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    return TimeMsg(sec=sec, nanosec=nanosec)


def _deg_to_urdf_rad(joints_deg):
    """LinuxCNC degrees -> URDF radians (matches robot_pose_stream_ros2.py)."""
    return [
        math.radians(JOINT_SIGNS[i] * (joints_deg[i] + JOINT_OFFSETS_DEG[i]))
        for i in range(MAX_JOINTS)
    ]


def _clamp_to_limits(target_deg):
    """Clamp each joint to the LinuxCNC soft limits; return (clamped, was_clamped)."""
    out = list(target_deg)
    clamped = False
    for i in range(MAX_JOINTS):
        lo, hi = LINUXCNC_SOFT_LIMITS_DEG[i]
        if out[i] < lo:
            out[i], clamped = lo, True
        elif out[i] > hi:
            out[i], clamped = hi, True
    return out, clamped


class _LineSocketClient(threading.Thread):
    """Persistent, auto-reconnecting newline-JSON TCP client.

    Calls ``on_line(dict)`` for each received JSON object. Thread-safe
    ``send(dict)`` queues an outbound JSON line (used for the command socket).
    """

    def __init__(self, node, host, port, on_line, name):
        super().__init__(daemon=True)
        self._node = node
        self._host = host
        self._port = port
        self._on_line = on_line
        self._name = name
        self._sock = None
        self._sock_lock = threading.Lock()
        self._running = True

    def stop(self):
        self._running = False
        with self._sock_lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass

    def send(self, obj):
        """Send a JSON object as a line. Returns True if written."""
        line = (json.dumps(obj) + "\n").encode("utf-8")
        with self._sock_lock:
            if not self._sock:
                self._node.get_logger().warn(
                    f"[{self._name}] not connected; dropping command {obj}"
                )
                return False
            try:
                self._sock.sendall(line)
                return True
            except OSError as e:
                self._node.get_logger().warn(f"[{self._name}] send failed: {e}")
                return False

    def run(self):
        while self._running and rclpy.ok():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect((self._host, self._port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(2.0)
                with self._sock_lock:
                    self._sock = s
                self._node.get_logger().info(
                    f"[{self._name}] connected to {self._host}:{self._port}"
                )
                buffer = ""
                while self._running and rclpy.ok():
                    try:
                        data = s.recv(4096)
                    except socket.timeout:
                        continue
                    if not data:
                        self._node.get_logger().warn(f"[{self._name}] server closed")
                        break
                    buffer += data.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        try:
                            self._on_line(obj)
                        except Exception as e:  # never let a callback kill the loop
                            self._node.get_logger().error(
                                f"[{self._name}] callback error: {e}"
                            )
            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                if self._running:
                    self._node.get_logger().warn(
                        f"[{self._name}] connect failed ({e}); retrying in 2s"
                    )
            finally:
                with self._sock_lock:
                    if self._sock:
                        try:
                            self._sock.close()
                        except OSError:
                            pass
                    self._sock = None
            if self._running:
                time.sleep(2.0)


class MyCobotBridge(Node):
    def __init__(self, host, stream_port, cmd_port, default_duration, default_controller,
                 stamp_source="robot"):
        super().__init__("mycobot_bridge")
        self._default_duration = default_duration
        self._default_controller = default_controller
        self._stamp_source = stamp_source  # "robot" (synced source time) or "local"
        self._stream_count = 0

        # publishers
        self._pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self._pub_js_deg = self.create_publisher(JointState, "/mycobot/joint_states_deg", 10)
        self._pub_status = self.create_publisher(String, "/mycobot/status", 10)
        # live clock-sync quality: receive_time - robot_sample_time (ms)
        self._pub_offset = self.create_publisher(Float64MultiArray, "/mycobot/clock_offset_ms", 10)
        # low-level drive feedback (one JointState: position=posfb deg,
        # velocity=velfb deg/s, effort=torqfb) -> all HAL feedback pins, timestamped
        self._pub_drive = self.create_publisher(JointState, "/mycobot/drive_feedback", 50)

        # subscribers
        self.create_subscription(Float64MultiArray, "/mycobot/cmd/joint_deg", self._on_cmd_deg, 10)
        self.create_subscription(JointState, "/mycobot/cmd/joint_rad", self._on_cmd_rad, 10)
        self.create_subscription(String, "/mycobot/cmd/move", self._on_cmd_json, 10)

        # service
        self.create_service(Trigger, "/mycobot/home", self._on_home)

        # TCP clients
        self._stream = _LineSocketClient(self, host, stream_port, self._on_stream, "stream")
        self._cmd = _LineSocketClient(self, host, cmd_port, self._on_status, "cmd")
        self._stream.start()
        self._cmd.start()

        self.get_logger().info(
            f"mycobot_bridge up. robot={host} stream:{stream_port} cmd:{cmd_port} "
            f"default_controller={default_controller} default_duration={default_duration}s"
        )

    # ---- inbound from robot (TCP -> ROS topics) ----
    def _on_stream(self, obj):
        joints_deg = obj.get("joints_deg")
        if not joints_deg or len(joints_deg) != MAX_JOINTS:
            return

        now = self.get_clock().now()
        robot_ts = obj.get("timestamp")
        # Stamp with the robot's sample time (synced) unless asked for local time
        # or the packet lacks a timestamp.
        if self._stamp_source == "robot" and robot_ts is not None:
            stamp = _epoch_to_stamp(float(robot_ts))
        else:
            stamp = now.to_msg()

        # Publish/track the offset between robot stamp and local receive time.
        # When clocks are synced this is ~network latency; a large/growing value
        # means the Pi and desktop clocks are NOT in sync.
        if robot_ts is not None:
            offset_ms = (now.nanoseconds * 1e-9 - float(robot_ts)) * 1e3
            self._pub_offset.publish(Float64MultiArray(data=[offset_ms]))
            self._stream_count += 1
            if self._stream_count % 250 == 0:  # steady-state, ~every 5 s at 50 Hz
                self.get_logger().info(
                    f"clock offset (recv - robot_stamp) = {offset_ms:.2f} ms "
                    f"[stamp_source={self._stamp_source}]"
                )

        js = JointState()
        js.header.stamp = stamp
        js.name = list(JOINT_NAMES)
        js.position = _deg_to_urdf_rad(joints_deg)
        self._pub_js.publish(js)

        js_deg = JointState()
        js_deg.header.stamp = stamp
        js_deg.name = list(JOINT_NAMES)
        js_deg.position = [float(d) for d in joints_deg]  # degrees on this topic
        self._pub_js_deg.publish(js_deg)

        # low-level drive feedback pins (posfb/velfb/torqfb) in one timestamped msg
        posfb = obj.get("posfb")
        if posfb and len(posfb) == MAX_JOINTS:
            df = JointState()
            df.header.stamp = stamp
            df.name = list(JOINT_NAMES)
            df.position = [float(v) for v in posfb]                      # deg
            velfb = obj.get("velfb"); torqfb = obj.get("torqfb")
            df.velocity = [float(v) for v in velfb] if velfb else []     # deg/s
            df.effort = [float(v) for v in torqfb] if torqfb else []     # torque
            self._pub_drive.publish(df)

    def _on_status(self, obj):
        msg = String()
        msg.data = json.dumps(obj)
        self._pub_status.publish(msg)

    # ---- outbound commands (ROS topics -> TCP) ----
    def _send_target(self, target_deg, duration=None, controller=None, gains=None, source=""):
        if len(target_deg) != MAX_JOINTS:
            self.get_logger().warn(f"{source}: expected {MAX_JOINTS} joints, got {len(target_deg)}")
            return
        clamped, was_clamped = _clamp_to_limits([float(v) for v in target_deg])
        if was_clamped:
            self.get_logger().warn(f"{source}: target clamped to soft limits -> {clamped}")
        cmd = {
            "target_deg": clamped,
            "duration": float(duration) if duration is not None else self._default_duration,
            "controller": controller or self._default_controller,
        }
        if gains:  # optional runtime PID gain override for tuning
            cmd["gains"] = gains
        if self._cmd.send(cmd):
            extra = f" gains={gains}" if gains else ""
            self.get_logger().info(f"{source}: -> {[round(v, 1) for v in clamped]} ({cmd['controller']}){extra}")

    def _on_cmd_deg(self, msg: Float64MultiArray):
        self._send_target(list(msg.data), source="cmd/joint_deg")

    def _on_cmd_rad(self, msg: JointState):
        target_deg = rad_to_linuxcnc_deg(list(msg.position)).tolist()
        self._send_target(target_deg, source="cmd/joint_rad")

    def _on_cmd_json(self, msg: String):
        try:
            obj = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"cmd/move: bad JSON: {e}")
            return
        if "raw_step" in obj:  # low-level servo characterization (PID-bypass step)
            if self._cmd.send({"raw_step": obj["raw_step"]}):
                self.get_logger().info(f"cmd/move: raw_step {obj['raw_step']}")
            return
        target = obj.get("target_deg")
        if target is None:
            self.get_logger().warn("cmd/move: missing target_deg")
            return
        self._send_target(target, obj.get("duration"), obj.get("controller"),
                          gains=obj.get("gains"), source="cmd/move")

    def _on_home(self, request, response):
        self._send_target(list(HOME_LINUXCNC_DEG), source="home")
        response.success = True
        response.message = f"home command sent: {HOME_LINUXCNC_DEG}"
        return response

    def destroy_node(self):
        self._stream.stop()
        self._cmd.stop()
        super().destroy_node()


def main():
    p = argparse.ArgumentParser(description="ROS 2 bridge for the myCobot Pro 630 (robot_hal.py).")
    p.add_argument("--robot-host", default="127.0.0.1",
                   help="Robot controller IP (127.0.0.1 on the Pi, else e.g. 10.0.0.27)")
    p.add_argument("--stream-port", type=int, default=DEFAULT_STREAM_PORT)
    p.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    p.add_argument("--duration", type=float, default=2.0, help="Default move duration (s)")
    p.add_argument("--controller", default="pid", choices=["pid", "invdyn", "pd_velff", "mpc"])
    p.add_argument("--stamp", default="robot", choices=["robot", "local"],
                   help="Header stamp source: 'robot' = synced sample time from the "
                        "controller (default), 'local' = desktop receive time")
    args, _ = p.parse_known_args()

    rclpy.init()
    node = MyCobotBridge(args.robot_host, args.stream_port, args.cmd_port,
                         args.duration, args.controller, stamp_source=args.stamp)
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
