#!/usr/bin/env python3
"""
analyze_response :: step-response metrics from a calibrate_ros2 log.

For every step command that moves a single joint, compute against the synced
response (/joint_states -> LinuxCNC deg):
  * overshoot  [%]   - peak excursion past target / step size
  * settling   [s]   - time to enter and stay within a +/-2% (>=0.2 deg) band
  * steady-err [deg] - target - mean(last 0.5 s)

Prints per-step rows and a per-joint summary. Used to drive PID tuning.

Usage: python3 analyze_response.py [log.jsonl]   (default: newest)
"""
import glob
import json
import os
import re
import sys

import numpy as np

from joint_conventions import MAX_JOINTS, rad_to_linuxcnc_deg

HERE = os.path.dirname(os.path.abspath(__file__))
SETTLE_FRAC = 0.02     # 2% band
SETTLE_MIN_DEG = 0.2   # but at least +/-0.2 deg
TAIL_SEC = 0.5         # window for steady-state average


def newest():
    logs = sorted(glob.glob(os.path.join(HERE, "logs", "calibrate_ros2_*.jsonl")))
    if not logs:
        raise SystemExit("no logs")
    return logs[-1]


def step_metrics(t, y, y0, target):
    """t (rel s), y (deg) response; y0 start value; target setpoint."""
    step = target - y0
    if abs(step) < 1e-6:
        return None
    band = max(SETTLE_FRAC * abs(step), SETTLE_MIN_DEG)
    # overshoot in the direction of motion
    if step > 0:
        osh = max(0.0, float(np.max(y)) - target)
    else:
        osh = max(0.0, target - float(np.min(y)))
    osh_pct = 100.0 * osh / abs(step)
    # settling: last time |y-target| leaves the band
    outside = np.abs(y - target) > band
    settle = float(t[np.where(outside)[0][-1]]) if outside.any() else 0.0
    # steady-state error: mean over last TAIL_SEC
    tail = y[t >= (t[-1] - TAIL_SEC)]
    sse = target - float(np.mean(tail)) if len(tail) else float("nan")
    return dict(step=step, overshoot_pct=osh_pct, settling_s=settle, sse_deg=sse)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest()
    rows = [json.loads(l) for l in open(path)]
    cmds = [r for r in rows if r.get("type") == "command"]
    resp = [r for r in rows if r.get("source") == "joint_states" and r.get("joints_rad")]
    rt = np.array([r["t_sync"] for r in resp])
    rdeg = np.array([rad_to_linuxcnc_deg(r["joints_rad"]) for r in resp])

    print(f"log: {os.path.basename(path)}   ({len(cmds)} commands, {len(resp)} responses)\n")
    print(f"{'label':<12}{'joint':>6}{'step':>8}{'overshoot%':>12}{'settle[s]':>11}{'sse[deg]':>10}")
    per_joint = {j: [] for j in range(MAX_JOINTS)}
    for k, c in enumerate(cmds):
        m = re.match(r"J(\d)", c.get("label", ""))
        if not m:
            continue
        j = int(m.group(1))
        t_start = c["t_sync"]
        t_end = cmds[k + 1]["t_sync"] if k + 1 < len(cmds) else rt[-1]
        sel = (rt >= t_start) & (rt < t_end)
        if sel.sum() < 5:
            continue
        t = rt[sel] - t_start
        y = rdeg[sel, j]
        met = step_metrics(t, y, y[0], c["target_deg"][j])
        if not met:
            continue
        per_joint[j].append(met)
        print(f"{c['label']:<12}{j:>6}{met['step']:>8.1f}{met['overshoot_pct']:>12.1f}"
              f"{met['settling_s']:>11.2f}{met['sse_deg']:>10.2f}")

    print(f"\n{'JOINT SUMMARY (mean)':<20}{'overshoot%':>12}{'settle[s]':>11}{'|sse|[deg]':>11}")
    for j in range(MAX_JOINTS):
        ms = per_joint[j]
        if not ms:
            continue
        osh = np.mean([m["overshoot_pct"] for m in ms])
        st = np.mean([m["settling_s"] for m in ms])
        ss = np.mean([abs(m["sse_deg"]) for m in ms])
        print(f"  J{j}{'':<16}{osh:>12.1f}{st:>11.2f}{ss:>11.2f}")


if __name__ == "__main__":
    main()
