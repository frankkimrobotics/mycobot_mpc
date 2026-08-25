#!/usr/bin/env python3
"""
bspline_vs_sine :: head-to-head comparison of a SINUSOID move vs a jerk-limited
B-spline / S-curve (smooth trapezoid) move, at the SAME peak velocity.

Why: the per-joint limit is a --max-vel-deg cap (verified fault-free to 50+ deg/s; real cap unmeasured) - the
drive faults when pos_cmd outruns posfb. The question is whether a different
trajectory shape, run at a higher *useful* velocity, covers a move better.

Both profiles move joint J by the same displacement D with the same peak
velocity Vmax (kept < ceiling). We measure, from /mycobot/drive_feedback:
  * completion time to reach the target (how well the speed budget is used)
  * RMS position tracking error
  * peak |pos_cmd - posfb|  (the fault-relevant gap)

  - SINE   : q = base + D*(0.5 - 0.5*cos(pi t/T)),  T = pi*D/(2*Vmax)
             smooth but only momentarily at Vmax -> slow for a given peak speed.
  - SCURVE : smoothstep accel ramp -> CRUISE at Vmax -> smoothstep decel.
             jerk-limited (bounded), cruises at Vmax -> covers D faster, and the
             gentle onset keeps pos_cmd from outrunning posfb.

Usage: python3 bspline_vs_sine.py --joint 0 --dist 30 --vmax 25
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


def sine_profile(D, vmax, dt):
    """Half-cosine position profile 0->D, peak velocity = vmax."""
    T = math.pi * D / (2 * vmax)
    n = max(2, int(round(T / dt)))
    return [D * (0.5 - 0.5 * math.cos(math.pi * k / n)) for k in range(n + 1)], T


def scurve_profile(D, vmax, dt, t_acc=0.4):
    """Smoothstep accel -> cruise at vmax -> smoothstep decel. Jerk-limited.
    Ramp uses smoothstep s(x)=3x^2-2x^3 (zero slope at both ends -> bounded jerk).
    Distance covered during each ramp = vmax * t_acc / 2."""
    ramp_dist = vmax * t_acc / 2.0
    cruise_dist = D - 2 * ramp_dist
    if cruise_dist < 0:                      # too short to reach vmax: shrink t_acc
        t_acc = math.sqrt(D / vmax) if vmax > 0 else 0.2
        ramp_dist = vmax * t_acc / 2.0
        cruise_dist = max(0.0, D - 2 * ramp_dist)
    t_cruise = cruise_dist / vmax
    T = 2 * t_acc + t_cruise
    n = max(2, int(round(T / dt)))
    pos = []
    for k in range(n + 1):
        t = min(T, k * dt)
        if t < t_acc:                        # accel ramp
            x = t / t_acc
            v_int = t_acc * (x ** 3 - 0.5 * x ** 4)      # integral of vmax*s(x) dx*t_acc
            p = vmax * v_int / t_acc * t_acc              # = vmax * t_acc * (x^3 - x^4/2)
            p = vmax * t_acc * (x ** 3 - 0.5 * x ** 4)
        elif t < t_acc + t_cruise:           # cruise
            p = ramp_dist + vmax * (t - t_acc)
        else:                                # decel ramp (mirror)
            td = t - t_acc - t_cruise
            x = td / t_acc
            p = ramp_dist + cruise_dist + (vmax * t_acc * (x - (x ** 3 - 0.5 * x ** 4)) - vmax * t_acc * (x - 1) if False else 0)
            # closed form of mirrored smoothstep decel:
            p = D - vmax * t_acc * ((1 - x) ** 3 - 0.5 * (1 - x) ** 4)
        pos.append(p)
    pos[-1] = D
    return pos, T


class TrajCompare(Node):
    def __init__(self, joint):
        super().__init__("bspline_vs_sine")
        self._j = joint
        self._pub = self.create_publisher(String, "/mycobot/cmd/move", 10)
        self.create_subscription(JointState, "/mycobot/drive_feedback", self._on_fb, 50)
        self._collect = False
        self._fb = []

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_fb(self, m):
        if self._collect and len(m.position) >= MAX_JOINTS:
            self._fb.append((self._now(), m.position[self._j],
                             m.velocity[self._j] if m.velocity else 0.0))

    def _spin(self, s):
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.02)

    def _home(self, base, dur=8):
        for _ in range(30):
            if self._pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self._pub.publish(String(data=json.dumps(
            {"target_deg": base, "duration": dur, "controller": "pid", "gains": {"u_max": 6}})))
        self._spin(dur + 2.0)

    def run_profile(self, base, prof_pos, dt, label, vff=1.0, vel_clamp=1000.0):
        target = list(base); target[self._j] = base[self._j] + prof_pos[-1]
        traj = []
        for p in prof_pos:
            pose = list(base); pose[self._j] = base[self._j] + p
            traj.append([round(v, 3) for v in pose])
        self._home(base)
        self._fb = []; self._collect = True
        t_send = self._now()
        self._pub.publish(String(data=json.dumps(
            {"target_deg": target, "trajectory": traj, "traj_dt": dt,
             "controller": "pid", "vff_scale": vff, "vel_clamp": vel_clamp})))
        self._spin(len(prof_pos) * dt + 3.5)        # collect through settle
        self._collect = False
        if len(self._fb) < 10:
            return None
        a = np.array(self._fb); t = a[:, 0] - t_send
        pos = a[:, 1] - base[self._j]; vel = a[:, 2]
        cmd_end = prof_pos[-1]
        # commanded position vs time (held at cmd_end after the profile ends)
        tp = np.arange(len(prof_pos)) * dt
        cmd_at = np.interp(t, tp, prof_pos, right=cmd_end)
        # tracking error only over the profile window (lag during the move)
        mv = t <= len(prof_pos) * dt
        gap = np.abs(cmd_at[mv] - pos[mv])
        rms = float(np.sqrt(np.mean((cmd_at[mv] - pos[mv]) ** 2)))
        # completion: first time actual is within 1 deg of target AND stays there
        tol = 1.0
        t_complete = float("nan")
        for i in range(len(pos)):
            if np.all(np.abs(pos[i:] - cmd_end) <= tol):
                t_complete = float(t[i]); break
        return dict(label=label, t_complete=t_complete, rms=rms, peak_gap=float(gap.max()),
                    vpk=float(np.percentile(np.abs(vel), 98)),
                    t=t.tolist(), pos=pos.tolist(), cmd=cmd_at.tolist(), vel=vel.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--dist", type=float, default=30.0, help="move displacement (deg)")
    ap.add_argument("--vmax", type=float, default=25.0, help="peak velocity (deg/s, keep < ~32)")
    ap.add_argument("--dt", type=float, default=0.04)
    ap.add_argument("--base", nargs=MAX_JOINTS, type=float, default=DEFAULT_BASE_DEG)
    ap.add_argument("--vff", type=float, default=1.0, help="velocity-feedforward gain (0=off)")
    ap.add_argument("--vel-clamp", type=float, default=1000.0, help="vel_cmd clamp (0=position-only)")
    args = ap.parse_args()
    base = [float(v) for v in args.base]

    sine_pos, T_sine = sine_profile(args.dist, args.vmax, args.dt)
    sc_pos, T_sc = scurve_profile(args.dist, args.vmax, args.dt)
    print(f"profiles: D={args.dist} deg, Vmax={args.vmax} deg/s  |  "
          f"sine T={T_sine:.2f}s ({len(sine_pos)} pts), scurve T={T_sc:.2f}s ({len(sc_pos)} pts)")

    rclpy.init()
    node = TrajCompare(args.joint)
    rows = []
    try:
        for pos, lbl in [(sine_pos, "sine"), (sc_pos, "scurve")]:
            r = node.run_profile(base, pos, args.dt, lbl, vff=args.vff, vel_clamp=args.vel_clamp)
            if r is None:
                node.get_logger().warn(f"{lbl}: no data"); continue
            rows.append(r)
            print(f"  {lbl:>7}: complete={r['t_complete']:.2f}s  "
                  f"RMSerr={r['rms']:.2f} deg  peak cmd-fb gap={r['peak_gap']:.2f} deg  "
                  f"Vpk={r['vpk']:.1f} deg/s")
        node._home(base)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not rows:
        print("no data"); return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, "logs", f"bspline_vs_sine_j{args.joint}_{stamp}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump([{k: v for k, v in r.items()} for r in rows], open(out + ".json", "w"))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    col = {"sine": "C0", "scurve": "C1"}
    for r in rows:
        c = col[r["label"]]
        t = np.array(r["t"])
        ax[0].plot(t, r["cmd"], "--", color=c, lw=1, alpha=.7, label=f"{r['label']} cmd")
        ax[0].plot(t, r["pos"], "-", color=c, lw=1.8, label=f"{r['label']} actual")
        ax[1].plot(t, r["vel"], "-", color=c, lw=1.5, label=r["label"])
        ax[2].plot(t, np.abs(np.array(r["cmd"]) - np.array(r["pos"])), "-", color=c, lw=1.5, label=r["label"])
    ax[0].axhline(args.dist, ls=":", color="0.6")
    ax[0].set_title("position: cmd (dashed) vs actual"); ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("deg from base"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].axhline(args.vmax, ls=":", color="0.6"); ax[1].text(0.1, args.vmax + 1, f"Vmax {args.vmax}", fontsize=8)
    ax[1].set_title("velocity"); ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("deg/s"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].set_title("tracking gap |cmd - actual| (fault-relevant)"); ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("deg"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle(f"B-spline/S-curve vs Sinusoid - J{args.joint}, D={args.dist} deg @ Vmax={args.vmax} deg/s "
                 f"(same peak velocity)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out + ".png", dpi=110)
    print(f"saved {out}.png")


if __name__ == "__main__":
    main()
