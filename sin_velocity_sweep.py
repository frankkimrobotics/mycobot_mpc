#!/usr/bin/env python3
"""
sin_velocity_sweep :: run the sinusoidal controller at increasing peak velocity
(10..100 % of the joint limit) and plot how well the robot tracks (amplitude &
phase) vs commanded velocity - the closed-loop tracking bandwidth.

For each %: q_j(t)=base_j+A*sin(2*pi*f*t) with f chosen so the peak speed = % of
the limit (A*2*pi*f). Sends the trajectory (velocity-FF tracker, vff_scale),
captures /mycobot/joint_states_deg, fits amplitude + phase, returns to base.
Aborts on fault. Plots amplitude(%) and phase lag vs commanded velocity %.

Usage: python3 sin_velocity_sweep.py --joint 0 --amplitude 15 --vff-scale 14
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
JOINT_LIMIT = [180.0, 180.0, 180.0, 200.0, 200.0, 200.0]


class SinSweep(Node):
    def __init__(self, joint):
        super().__init__("sin_velocity_sweep")
        self._j = joint
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/joint_states_deg", self._on_js, 50)
        self.create_subscription(String, "/mycobot/status", self._on_status, 10)
        self._status = None
        self._collect = False
        self._buf = []

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_status(self, m):
        try:
            self._status = json.loads(m.data)
        except json.JSONDecodeError:
            pass

    def _on_js(self, m):
        if m.position and self._collect:
            self._buf.append((self._now(), m.position[self._j]))

    def _spin(self, s):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _pub_wait(self, obj, dur):
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._status = None
        self._pub.publish(String(data=json.dumps(obj)))
        t0 = time.time()
        while rclpy.ok() and time.time() < t0 + dur + 3:
            rclpy.spin_once(self, timeout_sec=0.03)
            if self._status and self._status.get("state") == "done":
                return

    def run_pct(self, base, amp, freq, periods, traj_dt, vff, vel_clamp):
        n = max(1, int(round(periods / (freq * traj_dt))))
        traj = []
        for k in range(n + 1):
            t = k * traj_dt
            pose = list(base); pose[self._j] = base[self._j] + amp * math.sin(2 * math.pi * freq * t)
            traj.append([round(p, 3) for p in pose])
        # to base
        self._pub_wait({"target_deg": base, "duration": 6, "controller": "pid", "gains": {"u_max": 8}}, 6)
        self._spin(0.4)
        dur = n * traj_dt
        # publish the trajectory, then collect for the FULL execution window.
        # (do NOT early-return on status=="done" - the bridge acks immediately,
        #  which would truncate both the capture and the trajectory itself.)
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._buf = []; self._collect = True
        t_send = self._now()
        self._pub.publish(String(data=json.dumps(
            {"target_deg": traj[-1], "trajectory": traj, "traj_dt": traj_dt,
             "controller": "pid", "vff_scale": vff, "vel_clamp": vel_clamp})))
        self._spin(dur + 0.5)
        self._collect = False
        self._pub_wait({"target_deg": base, "duration": 6, "controller": "pid", "gains": {"u_max": 8}}, 6)
        if len(self._buf) < 20:
            return None
        # time relative to when the trajectory command was sent (not first sample)
        arr = np.array(self._buf); t = arr[:, 0] - t_send; y = arr[:, 1]
        # skip robot_hal's 0.7 s velocity ramp-in, keep the steady oscillation window
        m = (t >= 0.7) & (t <= dur)
        if m.sum() < 20:
            m = t <= dur
        t, y = t[m], y[m]
        # robust amplitude: half peak-to-peak about the base (matches the raw range)
        y_rel = y - base[self._j]
        amp_pp = float((np.max(y_rel) - np.min(y_rel)) / 2.0)
        rng_lo, rng_hi = float(np.min(y_rel)), float(np.max(y_rel))
        # least-squares fit (detrended) for amplitude + phase at the commanded freq
        w = 2 * math.pi * freq
        M = np.c_[np.sin(w * t), np.cos(w * t), np.ones_like(t)]
        a, b, _ = np.linalg.lstsq(M, y, rcond=None)[0]
        amp_fit = float(np.hypot(a, b)); phase = float(np.degrees(np.arctan2(-b, a)))
        # use the peak-to-peak amplitude (robust); fall back to fit if pp looks degenerate
        amp_act = amp_pp if amp_pp > 0.1 else amp_fit
        return dict(amp_act=amp_act, amp_pct=100 * amp_act / amp,
                    amp_fit=amp_fit, amp_pp=amp_pp,
                    rng_lo=rng_lo, rng_hi=rng_hi, nfit=int(m.sum()),
                    phase_lag=-phase)


def main():
    ap = argparse.ArgumentParser(description="Sinusoid tracking vs commanded velocity (10-100%).")
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--amplitude", type=float, default=15.0)
    ap.add_argument("--pcts", type=float, nargs="+", default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument("--periods", type=float, default=1.5)
    ap.add_argument("--traj-dt", type=float, default=0.04)
    ap.add_argument("--vff-scale", type=float, default=1.0)
    ap.add_argument("--vel-clamp", type=float, default=900.0)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    args = ap.parse_args()

    base = [float(v) for v in args.base]
    lim = JOINT_LIMIT[args.joint]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"sin_velsweep_j{args.joint}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    rclpy.init()
    node = SinSweep(args.joint)
    rows = []
    try:
        for pct in args.pcts:
            vpk = pct / 100.0 * lim                 # desired peak speed (deg/s)
            freq = vpk / (args.amplitude * 2 * math.pi)
            node.get_logger().info(f"J{args.joint} {pct:.0f}%: peak {vpk:.0f} deg/s, freq {freq:.3f} Hz")
            r = node.run_pct(base, args.amplitude, freq, args.periods, args.traj_dt,
                             args.vff_scale, args.vel_clamp)
            if r is None:
                node.get_logger().warn(f"{pct:.0f}%: no data (fault?) - ABORTING"); break
            r["pct"] = pct; r["vpk"] = vpk
            rows.append(r)
            node.get_logger().info(f"  amplitude {r['amp_act']:.1f} deg ({r['amp_pct']:.0f}%), "
                                   f"raw range [{r['rng_lo']:+.1f},{r['rng_hi']:+.1f}] deg, "
                                   f"fit {r['amp_fit']:.1f} deg, phase lag {r['phase_lag']:.0f} deg "
                                   f"(n={r['nfit']})")
            open(out, "a").write(json.dumps(r) + "\n")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not rows:
        print("no data"); return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pct = [r["pct"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(pct, [r["amp_pct"] for r in rows], "o-", color="C0")
    ax[0].axhline(100, ls=":", color="0.6"); ax[0].set_ylim(0, 110)
    ax[0].set_xlabel("commanded peak velocity [% of limit]"); ax[0].set_ylabel("tracked amplitude [% of commanded]")
    ax[0].set_title("amplitude vs velocity"); ax[0].grid(alpha=.3)
    ax[1].plot(pct, [r["phase_lag"] for r in rows], "o-", color="C3")
    ax[1].set_xlabel("commanded peak velocity [% of limit]"); ax[1].set_ylabel("phase lag [deg]")
    ax[1].set_title("phase lag vs velocity"); ax[1].grid(alpha=.3)
    fig.suptitle(f"Sinusoid tracking vs velocity - J{args.joint}, +/-{args.amplitude:.0f} deg, "
                 f"velocity-FF (vff_scale={args.vff_scale:.0f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = out.replace(".jsonl", ".png"); fig.savefig(png, dpi=110)
    print(f"\n{'%limit':>7}{'peak v':>8}{'amp%':>8}{'phase':>8}")
    for r in rows:
        print(f"{r['pct']:>6.0f}%{r['vpk']:>8.0f}{r['amp_pct']:>8.0f}{r['phase_lag']:>8.0f}")
    print(f"saved {png}\nlog: {out}")


if __name__ == "__main__":
    main()
