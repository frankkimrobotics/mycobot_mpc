#!/usr/bin/env python3
"""
max_velocity_probe :: discriminate a VELOCITY clamp from an ACCELERATION/
controller limit by holding frequency fixed and increasing amplitude.

For a fixed frequency f and a list of amplitudes A, command q=base+A*sin(2 pi f t)
and measure the achieved peak joint velocity from /mycobot/drive_feedback.

  * VELOCITY clamp  -> achieved peak velocity plateaus at the same ceiling for
                       every amplitude (bigger swing can't go faster).
  * ACCEL/ctrl limit-> achieved peak velocity keeps RISING with amplitude
                       (more distance -> more room to build up speed).

Usage: python3 max_velocity_probe.py --joint 0 --freq 0.4 --amps 10 20 40 60
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from joint_conventions import MAX_JOINTS

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE_DEG = [0.0, -110.3, 111.4, -90.1, -90.3, 0.0]


class MaxVelProbe(Node):
    def __init__(self, joint):
        super().__init__("max_velocity_probe")
        self._j = joint
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 50)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._collect = False
        self._fb = []
        self._status = None

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, m):
        try:
            self._status = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def _on_fb(self, m):
        if self._collect and len(m.position) >= MAX_JOINTS:
            self._fb.append((self._now(), m.position[self._j],
                             m.velocity[self._j] if m.velocity else 0.0,
                             m.effort[self._j] if m.effort else 0.0))

    def _spin(self, s):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _home(self, base, dur=6):
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(String(data=json.dumps(
            {"target_deg": base, "duration": dur, "controller": "pid", "gains": {"u_max": 8}})))
        self._spin(dur + 1.0)

    def run(self, base, amp, freq, traj_dt=0.04, periods=2.0):
        n = max(1, int(round(periods / (freq * traj_dt))))
        traj = []
        for k in range(n + 1):
            t = k * traj_dt
            pose = list(base); pose[self._j] = base[self._j] + amp * math.sin(2 * math.pi * freq * t)
            traj.append([round(p, 3) for p in pose])
        self._home(base)
        self._fb = []; self._collect = True
        t_send = self._now()
        self._pub.publish(String(data=json.dumps(
            {"target_deg": traj[-1], "trajectory": traj, "traj_dt": traj_dt,
             "controller": "pid", "vff_scale": 1.0, "vel_clamp": 1500.0})))
        self._spin(n * traj_dt + 0.5)
        self._collect = False
        self._home(base)
        if len(self._fb) < 20:
            return None
        a = np.array(self._fb); t = a[:, 0] - t_send
        m = t >= 0.7
        pos, vel = a[m, 1], a[m, 2]
        vpk = float(np.percentile(np.abs(vel), 98))
        amp_ach = float((pos.max() - pos.min()) / 2)
        v_cmd = amp * 2 * math.pi * freq
        return dict(amp=amp, v_cmd=v_cmd, vpk_ach=vpk, amp_ach=amp_ach,
                    samples=[[round(float(x[0] - t_send), 4), round(float(x[1]), 4),
                              round(float(x[2]), 4), round(float(x[3]), 5)] for x in self._fb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--freq", type=float, default=0.4)
    ap.add_argument("--amps", type=float, nargs="+", default=[10, 20, 40, 60])
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()
    base = [float(v) for v in args.base]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"maxvel_probe_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = MaxVelProbe(args.joint)
    rows = []
    try:
        print(f"\n{'amp[deg]':>8}{'v_cmd[deg/s]':>13}{'vpk_ach[deg/s]':>15}{'amp_ach[deg]':>13}")
        for amp in args.amps:
            r = node.run(base, amp, args.freq)
            if r is None:
                node.get_logger().warn(f"amp {amp}: no data - ABORT"); break
            s = r.pop("samples")
            rows.append(r)
            open(out, "a").write(json.dumps({**r, "freq": args.freq, "joint": args.joint,
                                             "samples": s}) + "\n")
            print(f"{amp:>8.0f}{r['v_cmd']:>13.0f}{r['vpk_ach']:>15.1f}{r['amp_ach']:>13.1f}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if len(rows) >= 2:
        rising = rows[-1]["vpk_ach"] - rows[0]["vpk_ach"]
        print(f"\nachieved peak velocity {'RISES' if rising > 8 else 'PLATEAUS'} "
              f"with amplitude ({rows[0]['vpk_ach']:.0f} -> {rows[-1]['vpk_ach']:.0f} deg/s) => "
              f"{'NOT a velocity clamp (accel/ctrl-limited)' if rising > 8 else 'VELOCITY clamp'}")
    print(f"log: {out}")


if __name__ == "__main__":
    main()
