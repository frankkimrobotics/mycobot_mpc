#!/usr/bin/env python3
"""
plot_traj_responses :: per-joint overlay of velocity + torque response for the
trajectory-speed sweep, aligned to the command instant (t=0).

For each joint, overlays the drive response at every commanded velocity on a
common time axis: 0 = command sent, then the motion. Left column = velfb
(velocity response), right column = torqfb (torque the arm reports). Also prints
the command vs response table.

Usage: python3 plot_traj_responses.py [traj_speed_alljoints_*.jsonl]
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 1.1  # seconds after command to show (covers the trajectory move)


def newest():
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "traj_speed_alljoints_*.jsonl")))
    if not logs:
        raise SystemExit("no traj_speed_alljoints logs")
    return logs[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows = [json.loads(l) for l in open(path)]
    cmds = {r["tag"]: r for r in rows if r.get("type") == "command"}
    fb = {}
    for r in rows:
        if r.get("type") == "drive_feedback":
            fb.setdefault(r["tag"], []).append((r["t_sync"], r["velfb"], r.get("torqfb", 0.0)))

    joints = sorted({c["joint"] for c in cmds.values()})
    fig, axes = plt.subplots(len(joints), 2, figsize=(13, 2.6 * len(joints)), squeeze=False)

    print(f"log: {os.path.basename(path)}\n")
    print(f"{'joint':>6}{'cmd V':>8}{'peak velfb':>12}{'peak |torq|':>13}")
    for ji, j in enumerate(joints):
        axv, axt = axes[ji][0], axes[ji][1]
        tags = sorted((t for t, c in cmds.items() if c["joint"] == j),
                      key=lambda t: cmds[t]["cmd_V"])
        for t in tags:
            c = cmds[t]; t0 = c["t_sync"]; V = c["cmd_V"]
            seg = [(ts - t0, abs(vv), tq) for ts, vv, tq in fb.get(t, []) if 0 <= ts - t0 <= WINDOW]
            if len(seg) < 3:
                continue
            arr = np.array(seg)
            lab = f"{V:.0f}"
            axv.plot(arr[:, 0], arr[:, 1], lw=1.3, label=lab)
            axt.plot(arr[:, 0], arr[:, 2], lw=1.3, label=lab)
            print(f"{('J'+str(j)):>6}{V:>8.0f}{arr[:,1].max():>12.1f}{np.abs(arr[:,2]).max():>13.4f}")
        axv.set_ylabel(f"J{j}\nvelfb [deg/s]")
        axt.set_ylabel("torqfb")
        axv.grid(alpha=0.3); axt.grid(alpha=0.3)
        if ji == 0:
            axv.set_title("velocity response (overlaid by commanded deg/s)")
            axt.set_title("torque response (arm feedback)")
            axv.legend(fontsize=7, title="cmd deg/s", ncol=2)
    axes[-1][0].set_xlabel("time since command [s]   (0 = command sent → motion)")
    axes[-1][1].set_xlabel("time since command [s]")
    fig.suptitle("myCobot Pro 630 — per-joint trajectory response: velocity & torque, "
                 "overlaid by commanded velocity", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = path.replace(".jsonl", "_responses.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
