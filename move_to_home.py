#!/usr/bin/env python3
"""move_to_home :: plan (cuRobo) + velocity-limited track to the robot HOME/default pose.

HOME = joint_conventions.HOME_LINUXCNC_DEG. Uses the cuRobo planner (plan_joint)
for a collision-free trajectory and the bridge trajectory path (time-scaled to a
safe peak speed) so the large move doesn't trip the firmware following-error.
After settling, reads the ACTUAL reached joint state from the real robot.
"""
import argparse
import numpy as np
import rclpy
from std_msgs.msg import String

from perturb_loop import PlannerClient, RobotState, execute
from joint_conventions import (linuxcnc_deg_to_rad, rad_to_linuxcnc_deg,
                               HOME_LINUXCNC_DEG, JOINT_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-vel-deg", type=float, default=18.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9997)
    args = ap.parse_args()

    home_q = np.array(linuxcnc_deg_to_rad(HOME_LINUXCNC_DEG))
    print(f"HOME_LINUXCNC_DEG = {HOME_LINUXCNC_DEG}")
    print(f"home_q (URDF rad) = {[round(float(v),5) for v in home_q]}")

    pc = PlannerClient(args.host, args.port)
    print(f"[home] planner: {pc.rpc({'type':'ping'}).get('backend')}")

    rclpy.init()
    node = rclpy.create_node("move_to_home")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    q = state.get_q()
    if q is None:
        print("[home] ABORT: no /joint_states"); rclpy.shutdown(); return
    print(f"[home] current q (URDF deg): {np.round(np.rad2deg(q),1)}")
    print(f"[home] max |delta| to home: {np.abs(np.rad2deg(q-home_q)).max():.1f} deg")

    r = pc.plan_joint(q, home_q)
    if not r.get("success"):
        print(f"[home] ABORT: plan_joint failed ({r.get('status')})"); rclpy.shutdown(); return

    track = {"ramp_time": 0.15, "pos_gain": 1.0, "vff_scale": 1.0}
    ex = execute(state, pub, np.array(r["trajectory"]), r["dt"], "pid",
                 args.max_vel_deg, args.duration, "home", track=track)

    qf = state.get_q()
    print(f"\n[home] settled={ex['ok']} reach_err={ex['reach_err']:.2f} deg")
    print(f"[home] ACTUAL reached q (URDF deg):    {np.round(np.rad2deg(qf),2)}")
    print(f"[home] ACTUAL reached q (LinuxCNC deg):{np.round(rad_to_linuxcnc_deg(qf),2)}")
    print(f"[home] ACTUAL reached q (URDF rad):    {[round(float(v),5) for v in qf]}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
