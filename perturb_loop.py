#!/usr/bin/env python3
"""perturb_loop :: repeat [return-to-initial -> cuRobo-plan SE3-perturbed pose ->
move -> capture] K times.

Python 3.8 (ROS 2 + libusb pyrealsense2). Plans by talking to a persistent cuRobo
planner server (curobo2 env, default 127.0.0.1:9997) so the planner stays warm.

Each iteration:
  1. plan_joint(current_q -> initial_q)            -> move back to base
  2. sample base-EE pose perturbed by `--perturb` m (random dir) + `--rot-deg`
     (random axis); plan_pose(settled_q -> goal)   -> move to perturbed pose
  3. capture color frames from D405/D435 during the perturbed move

Robustness/safety:
  * waits until joints actually SETTLE after each move (polls /joint_states to
    stability) instead of trusting the status topic; plans the next move from the
    settled pose, so the start always matches.
  * time-scales every trajectory to peak <= --max-vel-deg (default 30 for safety;
    drives SATURATE at ~60 deg/s/joint = the max — above ~60 they fault).
  * aborts the whole loop if moves repeatedly fail to settle/reach (likely a
    hardware/CAN fault) instead of hammering a faulting bus.
  * bridge clamps all targets to soft limits.

Usage (ros2node.env sourced, PYTHONPATH=~/librealsense/build/release):
  python3 perturb_loop.py --iters 10 --perturb 0.10 --rot-deg 30
  python3 perturb_loop.py --iters 1            # single test iteration
"""
import argparse
import json
import os
import socket
import threading
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyrealsense2 as rs

from joint_conventions import rad_to_linuxcnc_deg, JOINT_NAMES

CAMERAS = {"d405": "218622271300", "d435": "043422070101"}


def quat_mul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1])


def sample_se3_goal(rng, pos0, quat0, trans_m, rot_deg):
    d = rng.normal(size=3); d = d / (np.linalg.norm(d) + 1e-9) * trans_m
    ax = rng.normal(size=3); ax = ax / (np.linalg.norm(ax) + 1e-9)
    a = np.deg2rad(rot_deg)
    dq = np.array([np.cos(a/2), *(np.sin(a/2) * ax)])
    gq = quat_mul(dq, quat0); gq = gq / np.linalg.norm(gq)
    return pos0 + d, gq, d


def closest_reachable_plan(pc, q, pos0, quat0, trans_vec, axis, rot_deg, bisect=6):
    """cuRobo IK + collision-free B-spline plan to the desired SE3 perturbation;
    if the full target is unreachable, return the CLOSEST reachable pose along the
    SAME direction.

    The base pose (scale 0) is reachable by construction, so we bisect the
    perturbation magnitude in [0, 1]: the largest scale whose pose cuRobo can plan
    is the closest reachable pose to the desired one.

    Returns (plan|None, goal_pos, goal_quat, scale), scale in (0, 1] = fraction of
    the desired translation + rotation actually achieved (1.0 = full target).
    """
    def goal_at(s):
        gp = pos0 + trans_vec * s
        a = np.deg2rad(rot_deg * s)
        dq = np.array([np.cos(a / 2.0), *(np.sin(a / 2.0) * axis)])
        gq = quat_mul(dq, quat0); gq = gq / np.linalg.norm(gq)
        return gp, gq

    gp, gq = goal_at(1.0)                       # try the full perturbation first
    r = pc.plan_pose(q, [float(v) for v in (*gp, *gq)])
    if r.get("success"):
        return r, gp, gq, 1.0
    lo, hi = 0.0, 1.0                           # lo reachable, hi not (yet)
    best = None; best_s = 0.0; bgp = bgq = None
    for _ in range(bisect):
        mid = 0.5 * (lo + hi)
        gp, gq = goal_at(mid)
        r = pc.plan_pose(q, [float(v) for v in (*gp, *gq)])
        if r.get("success"):
            best, best_s, bgp, bgq = r, mid, gp, gq; lo = mid
        else:
            hi = mid
    return best, bgp, bgq, best_s


# ----------------------------- cuRobo planner client -----------------------
class PlannerClient:
    def __init__(self, host="127.0.0.1", port=9997):
        self.sock = socket.create_connection((host, port), timeout=120)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = ""

    def rpc(self, req):
        self.sock.sendall((json.dumps(req) + "\n").encode())
        while "\n" not in self.buf:
            data = self.sock.recv(1 << 20)
            if not data:
                raise RuntimeError("planner closed connection")
            self.buf += data.decode()
        line, self.buf = self.buf.split("\n", 1)
        return json.loads(line)

    def plan_joint(self, start_q, goal_q, max_attempts=8):
        return self.rpc({"type": "plan_joint", "start_q": list(start_q),
                         "goal_q": list(goal_q), "max_attempts": max_attempts})

    def plan_pose(self, start_q, goal_pose, max_attempts=8):
        return self.rpc({"type": "plan_pose", "start_q": list(start_q),
                         "goal_pose": list(goal_pose), "max_attempts": max_attempts})


# ----------------------------- robot state ---------------------------------
class RobotState:
    """Persistent /joint_states subscriber with settle detection."""
    def __init__(self, node):
        self.node = node
        self.q = None
        for rel in (ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT):
            node.create_subscription(
                JointState, "/joint_states", self._cb,
                QoSProfile(reliability=rel, history=HistoryPolicy.KEEP_LAST, depth=10))

    def _cb(self, m):
        self.q = np.array(m.position)

    def get_q(self, timeout=8.0):
        t0 = time.time()
        while self.q is None and time.time() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return self.q

    def wait_settled(self, timeout, eps_deg=0.3, need_cycles=8):
        """Spin until joint positions stop changing; returns (q, settled?)."""
        prev = None; stable = 0; t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            q = self.q
            if q is None:
                continue
            if prev is not None and np.rad2deg(np.abs(q - prev)).max() < eps_deg:
                stable += 1
                if stable >= need_cycles:
                    return q, True
            else:
                stable = 0
            prev = q
        return self.q, False


# ----------------------------- camera capture ------------------------------
class CameraCapture(threading.Thread):
    def __init__(self, name, serial, w=848, h=480, fps=30):
        super().__init__(daemon=True)
        self.name, self.serial = name, serial
        self.w, self.h, self.fps = w, h, fps
        self.stop_evt = threading.Event()
        self.saving = False
        self.cur_dir = "/tmp"
        self.frames = 0
        self.ok = False

    def run(self):
        pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, self.fps)
        try:
            pipe.start(cfg)
        except Exception as e:
            print(f"[cap:{self.name}] start failed: {e}"); return
        self.ok = True
        try:
            while not self.stop_evt.is_set():
                try:
                    fr = pipe.wait_for_frames(1000)
                except RuntimeError:
                    continue
                c = fr.get_color_frame()
                if not c or not self.saving:
                    continue
                cv2.imwrite(os.path.join(self.cur_dir,
                            f"{self.name}_{self.frames:04d}_{time.time():.3f}.png"),
                            np.asanyarray(c.get_data()))
                self.frames += 1
        finally:
            pipe.stop()


def scale_traj(traj, base_dt, max_vel_deg, min_duration):
    N = traj.shape[0]
    step_max = np.abs(np.diff(traj, axis=0)).max() if N > 1 else 0.0
    peak0 = np.rad2deg(step_max) / base_dt if base_dt > 0 else 0.0
    s = max(1.0, peak0 / max_vel_deg if peak0 > 0 else 1.0,
            min_duration / (base_dt * (N - 1)) if N > 1 else 1.0)
    return base_dt * s, peak0 / s


def execute(state, pub, traj, dt, controller, max_vel_deg, min_dur, label, track=None):
    """Time-scale + send; record actual joint trace until settle.

    `track` (optional dict) merges B-spline tracking gains into the move command
    (ramp_time, pos_gain, vff_scale) so robot_hal follows the trajectory with
    less lag instead of using its conservative defaults.

    Returns dict: ok, reach_err, q_end, sdt, peak, cmd_traj (rad), trace_t, trace_q.
    """
    sdt, peak = scale_traj(traj, dt, max_vel_deg, min_dur)
    q = state.get_q()
    if q is None:
        print(f"    [{label}] ABORT: no joint_states")
        return {"ok": False, "reach_err": 999.0, "q_end": None}
    start_err = np.rad2deg(np.abs(q - traj[0])).max()
    if start_err > 8.0:
        print(f"    [{label}] ABORT: live q {start_err:.1f} deg from plan start")
        return {"ok": False, "reach_err": 999.0, "q_end": q}
    traj_deg = [list(map(float, rad_to_linuxcnc_deg(wp))) for wp in traj]
    cmd = {"trajectory": traj_deg, "traj_dt": sdt,
           "target_deg": traj_deg[-1], "controller": controller}
    if track:
        cmd.update(track)
    total = sdt * (traj.shape[0] - 1)
    t0 = time.time()
    pub.publish(String(data=json.dumps(cmd)))
    # record the actual joint trajectory while the move runs. Only start looking
    # for "settled" AFTER the commanded motion time has elapsed -- otherwise the
    # brief stationary window between publishing the command and the robot
    # actually starting to move is mistaken for the move being already finished
    # (settle fires in ~10 ms, reach_err is read before any motion happens).
    trace_t, trace_q = [], []
    settled = False
    while time.time() - t0 < total + 8.0:
        rclpy.spin_once(state.node, timeout_sec=0.02)
        cq = state.q
        if cq is None:
            continue
        trace_t.append(time.time() - t0); trace_q.append([float(x) for x in cq])
        # Declare settled only after the commanded motion time AND once the recent
        # trajectory window has truly stopped. The robot lags the command (~0.8 s
        # dead-time) and keeps converging to the held final target after `total`,
        # so a windowed range test (vs consecutive-sample deltas) avoids calling
        # slow creep "settled".
        if (time.time() - t0) >= total and len(trace_q) >= 15:
            win = np.array(trace_q[-15:])
            if np.rad2deg(win.max(0) - win.min(0)).max() < 0.4:
                settled = True; break
    q_end = state.q
    reached = float(np.rad2deg(np.abs(q_end - traj[-1])).max()) if q_end is not None else 999.0
    print(f"    [{label}] {traj.shape[0]} wpts, peak {peak:.1f} deg/s, ~{total:.1f}s | "
          f"settled={settled} reach_err={reached:.1f} deg")
    return {"ok": settled, "reach_err": reached, "q_end": q_end, "sdt": sdt, "peak": peak,
            "cmd_traj": traj.tolist(), "trace_t": trace_t, "trace_q": trace_q}


def plot_cmd_vs_actual(res, joint_names, path, title):
    """6-subplot figure: commanded vs actual joint angle (deg) over time."""
    cmd = np.array(res["cmd_traj"]); ct = np.arange(cmd.shape[0]) * res["sdt"]
    at = np.array(res["trace_t"]); aq = np.array(res["trace_q"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for j in range(6):
        ax = axes[j // 3][j % 3]
        ax.plot(ct, np.rad2deg(cmd[:, j]), "--", color="C1", lw=2, label="cmd")
        if aq.size:
            ax.plot(at, np.rad2deg(aq[:, j]), "-", color="C0", lw=1.5, label="actual")
        ax.set_title(joint_names[j]); ax.set_ylabel("angle (deg)"); ax.grid(alpha=0.3)
        if j >= 3:
            ax.set_xlabel("time (s)")
    axes[0][0].legend(loc="best")
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/tmp/perturb_plan.json")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--perturb", type=float, default=0.10)
    ap.add_argument("--rot-deg", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--max-vel-deg", type=float, default=30.0,
                    help="peak joint speed (deg/s); default 30 for safety, max ~60 (drive saturation)")
    ap.add_argument("--controller", default="pid")
    ap.add_argument("--reach-tol-deg", type=float, default=3.0)
    ap.add_argument("--ramp-time", type=float, default=0.15,
                    help="robot_hal soft-start ramp (s); lower = less startup lag (was hardcoded 0.7)")
    ap.add_argument("--pos-gain", type=float, default=1.0,
                    help="robot_hal reactive position-error gain for traj tracking (was hardcoded 0.5)")
    ap.add_argument("--vff-scale", type=float, default=1.0,
                    help="robot_hal trajectory velocity feedforward scale")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9997)
    ap.add_argument("--cameras", nargs="+", default=["d405", "d435"])
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--outbase", default=None)
    args = ap.parse_args()

    # B-spline tracking gains sent to robot_hal (override its conservative defaults
    # of ramp 0.7s / pos_gain 0.5 that caused the ~0.55s following lag).
    track = {"ramp_time": args.ramp_time, "pos_gain": args.pos_gain,
             "vff_scale": args.vff_scale}

    plan = json.load(open(args.plan))
    initial_q = np.array(plan["start_q"])
    ee_pos0 = np.array(plan["ee_current"]["pos"])
    ee_quat0 = np.array(plan["ee_current"]["quat_wxyz"])
    rng = np.random.default_rng(args.seed)

    pc = PlannerClient(args.host, args.port)
    print(f"[loop] planner: {pc.rpc({'type':'ping'}).get('backend')}")

    rclpy.init()
    node = rclpy.create_node("perturb_loop")
    pub = node.create_publisher(String, "/mycobot/cmd/move", 10)
    state = RobotState(node)
    if state.get_q() is None:
        print("[loop] ABORT: no /joint_states"); rclpy.shutdown(); return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outbase = args.outbase or os.path.expanduser(f"~/Desktop/2026/mycobot_mpc/captures/loop_{stamp}")
    os.makedirs(outbase, exist_ok=True)
    caps = [CameraCapture(n, CAMERAS[n]) for n in args.cameras if n in CAMERAS]
    for c in caps:
        c.start()
    time.sleep(1.5)

    log = []; consecutive_fail = 0
    for k in range(args.iters):
        print(f"[loop] === iteration {k+1}/{args.iters} ===")
        # 1) return to initial
        q = state.get_q()
        r = pc.plan_joint(q, initial_q)
        if not r.get("success"):
            print(f"    return plan failed: {r.get('status')}"); consecutive_fail += 1
            if consecutive_fail >= 2:
                print("[loop] ABORT loop: repeated failures (check robot/CAN bus)"); break
            continue
        ex_ret = execute(state, pub, np.array(r["trajectory"]), r["dt"],
                         args.controller, args.max_vel_deg, args.duration, "return", track=track)
        err_ret = ex_ret["reach_err"]
        time.sleep(0.4)
        q = state.get_q()
        if err_ret > args.reach_tol_deg:
            print(f"    return did not reach initial ({err_ret:.1f} deg); skipping perturb")
            consecutive_fail += 1
            if consecutive_fail >= 2:
                print("[loop] ABORT loop: robot not reaching targets (likely CAN fault)"); break
            continue

        # 2) sample desired SE3 perturbation (translation + rotation about a random
        #    axis), then cuRobo IK + collision-free B-spline plan from the actual
        #    settled q. If the full target is unreachable, fall back to the CLOSEST
        #    reachable pose along that direction; only resample a fresh direction if
        #    even that degenerates to nothing.
        pl = None; gpos = gquat = perturb = None; ach_rot = args.rot_deg; scale = 0.0
        for _ in range(20):
            d = rng.normal(size=3); d = d / (np.linalg.norm(d) + 1e-9) * args.perturb
            axis = rng.normal(size=3); axis = axis / (np.linalg.norm(axis) + 1e-9)
            rr, gp, gq, scale = closest_reachable_plan(
                pc, q, ee_pos0, ee_quat0, d, axis, args.rot_deg)
            if rr is not None and rr.get("success"):
                pl, gpos, gquat = rr, gp, gq
                perturb = d * scale; ach_rot = args.rot_deg * scale
                if scale < 1.0:
                    print(f"    full {args.perturb*100:.0f}cm/{args.rot_deg:.0f}deg target "
                          f"unreachable -> closest reachable {scale*100:.0f}% "
                          f"(|t|={np.linalg.norm(perturb)*100:.1f}cm, rot={ach_rot:.1f}deg)")
                break
        if pl is None:
            print("    perturbed plan failed after retries"); continue

        # 3) execute + capture frames + record joint trace
        itdir = os.path.join(outbase, f"iter_{k:02d}")
        os.makedirs(itdir, exist_ok=True)
        for c in caps:
            c.cur_dir = itdir; c.saving = True
        f0 = {c.name: c.frames for c in caps}
        ex = execute(state, pub, np.array(pl["trajectory"]), pl["dt"],
                     args.controller, args.max_vel_deg, args.duration, "perturb", track=track)
        time.sleep(0.6)
        for c in caps:
            c.saving = False
        nf = {c.name: c.frames - f0[c.name] for c in caps}
        err = ex["reach_err"]; reached_ok = err <= args.reach_tol_deg
        q = ex.get("q_end", q)

        # save move log (cmd + actual joint trajectory) and cmd-vs-actual plot
        move_log = {"iter": k, "joint_names": JOINT_NAMES,
                    "goal_pos": gpos.tolist(), "goal_quat_wxyz": gquat.tolist(),
                    "perturb_trans_m": perturb.tolist(), "perturb_rot_deg": ach_rot,
                    "desired_trans_m": args.perturb, "desired_rot_deg": args.rot_deg,
                    "reachable_scale": scale,
                    "cmd_dt": ex.get("sdt"), "cmd_traj_rad": ex.get("cmd_traj"),
                    "actual_t": ex.get("trace_t"), "actual_q_rad": ex.get("trace_q"),
                    "reach_err_deg": err, "reached": bool(reached_ok)}
        with open(os.path.join(itdir, "move_log.json"), "w") as f:
            json.dump(move_log, f)
        if ex.get("cmd_traj") is not None:
            plot_cmd_vs_actual(ex, JOINT_NAMES, os.path.join(itdir, "joint_cmd_vs_actual.png"),
                               f"iter {k}: perturb {np.linalg.norm(perturb)*100:.1f}cm/"
                               f"{ach_rot:.0f}deg | reach_err {err:.1f} deg")
        print(f"    perturb |t|={np.linalg.norm(perturb)*100:.1f}cm rot={ach_rot:.1f}deg "
              f"| reached={reached_ok} | frames {nf} -> {itdir}")
        log.append({"iter": k, "goal_pos": gpos.tolist(), "perturb_vec_m": perturb.tolist(),
                    "reached": bool(reached_ok), "reach_err_deg": err, "frames": nf, "dir": itdir})
        consecutive_fail = 0 if (ex["ok"] and reached_ok) else consecutive_fail + 1
        if consecutive_fail >= 2:
            print("[loop] ABORT loop: repeated non-settle/non-reach (likely CAN fault)"); break

    for c in caps:
        c.stop_evt.set()
    for c in caps:
        c.join(timeout=3.0)
    with open(os.path.join(outbase, "loop_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    good = sum(1 for e in log if e["reached"])
    print(f"[loop] done. {good}/{args.iters} clean perturbed captures -> {outbase}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
