#!/usr/bin/env python3
"""
MuJoCo simulation of the myCobot Pro 630 — a drop-in replacement for robot_hal.py.

Same command server (:9998) and joint stream (:9999) and the same control
architecture as robot_hal.py (outer law from controller_solvers -> pos_cmd/vel_cmd;
inner loop tracks it), but the plant is a MuJoCo model of the URDF instead of
LinuxCNC/HAL + real motors. So the desktop pipeline drives the SIM unchanged:

    # terminal 1 (this computer): start the simulated robot
    python sim_hal.py --controller pid              # headless
    python sim_hal.py --controller pid --viewer     # with a MuJoCo window

    # terminal 2: drive it exactly like the real robot, but localhost
    python control_robot.py    --host 127.0.0.1 --joints -90 -90 0 -90 0 0
    python calibrate_perturb.py --host 127.0.0.1 --smooth
    # (optional) visualize separately: python ../mujoco_robot_mirror.py --source robot --host 127.0.0.1

Control mirrors robot_hal.run_control_loop:
  outer: pid|invdyn|pd_velff|mpc  (controller_solvers)  -> next_pos/vel_cmd (deg)
  inner: motor-PID emulation tau = Kp*(next_pos - q) + Kd*(vel_cmd - qd) + G(q)
         (Kp/Kd = virtual_driver gains from controller_params.yaml; G from qfrc_bias),
         clamped to the per-joint rated output torque, then mj_step.

The URDF/meshes come from the ROS2 description used everywhere else; the model
build (mesh convert, URDF patch, DH inertia/friction/actuators) is reused from
the parent mujoco_viewer.py.
"""

import argparse
import base64
import csv
import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np
import mujoco
import mujoco.viewer

# Reuse the model build + dynamics + torque limits from the parent mujoco_viewer.
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
from mujoco_viewer import (
    CACHE_DIR, MESHES_DIR, URDF_PATH, JOINT_NAMES,
    convert_meshes, patch_urdf, enhance_mjcf, RATED_TORQUE_OUTPUT,
)

# Outer control laws — identical to robot_hal.py.
from controller_solvers import (
    MAX_JOINTS, PERIOD_SEC,
    pid_solve, invdyn_solve, pd_velff_solve, mpc_solve,
)
from controller_params import get_controller_params
# Interfaces speak LinuxCNC degrees (like robot_hal); MuJoCo is URDF radians.
from joint_conventions import linuxcnc_deg_to_rad, rad_to_linuxcnc_deg, HOME_LINUXCNC_DEG

CMD_PORT = 9998
STREAM_PORT = 9999
STREAM_RATE_HZ = 50.0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


# ── Inner-loop (motor-PID emulation) gains from controller_params virtual_driver ──
def _load_inner_gains():
    p = get_controller_params()
    vd = p.get("virtual_driver", {})

    def arr(scalar_key, per_key, default):
        per = vd.get(per_key)
        if per is not None and len(per) >= MAX_JOINTS:
            return np.asarray(per[:MAX_JOINTS], float)
        return np.full(MAX_JOINTS, float(vd.get(scalar_key, default)))

    return {
        "kp": arr("kp_v", "kp_v_per_joint", 200.0),
        "kd": arr("kd_v", "kd_v_per_joint", 25.0),
        "ki": arr("ki_v", "ki_v_per_joint", 0.0),
        "integral_clamp": float(vd.get("integral_clamp_v", 5.0)),
        "gravity_comp": bool(vd.get("gravity_comp", True)),
        "torque_limit": float(vd.get("torque_limit_nm", 186.0)),
        "vel_cmd_limit": float(vd.get("vel_cmd_limit_deg_s", 720.0)),
        "rest_pose_deg": list(p.get("rest_pose_deg", [-90, -90, 0, -90, 0, 0])),
    }


# ── Shared sim state (control loop writes q; stream thread reads it) ──
_state_lock = threading.Lock()
_sim_q_deg = [0.0] * MAX_JOINTS

# ── Command protocol (mirrors robot_hal.py; no HAL/LinuxCNC) ──
_cmd_queue = queue.Queue()
_cmd_status = {"state": "idle", "current_deg": [0.0] * MAX_JOINTS,
               "target_deg": [0.0] * MAX_JOINTS, "error_norm": 0.0, "last_log_name": None}
_cmd_status_lock = threading.Lock()
_last_log_filename = None
_desktop_log_stamp = None


def _update_cmd_status(state, current_deg, target_deg, error_norm, **extra):
    with _cmd_status_lock:
        _cmd_status["state"] = state
        _cmd_status["current_deg"] = [round(v, 3) for v in current_deg]
        _cmd_status["target_deg"] = [round(v, 3) for v in target_deg]
        _cmd_status["error_norm"] = round(error_norm, 4)
        for k, v in extra.items():
            _cmd_status[k] = v


def _handle_cmd_client(conn, addr, log_dir):
    print(f"[cmd] Client connected: {addr}")
    buffer = ""
    conn.settimeout(1.0)
    try:
        while True:
            with _cmd_status_lock:
                status = dict(_cmd_status)
            try:
                conn.sendall((json.dumps(status) + "\n").encode("utf-8"))
            except (BrokenPipeError, ConnectionResetError, OSError):
                break
            try:
                data = conn.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8", errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cmd = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "target_deg" in cmd:
                        _cmd_queue.put(cmd)
                        conn.sendall((json.dumps({"state": "ack", "target_deg": cmd["target_deg"]}) + "\n").encode("utf-8"))
                    elif "get_log" in cmd:
                        name = cmd.get("get_log")
                        if name and isinstance(name, str):
                            name = os.path.basename(name)
                            path = os.path.join(log_dir, name)
                            if name.endswith(".csv") and os.path.isfile(path):
                                with open(path, "rb") as f:
                                    payload = base64.b64encode(f.read()).decode("ascii")
                                conn.sendall((json.dumps({"state": "log", "filename": name, "log_content_base64": payload}) + "\n").encode("utf-8"))
                            else:
                                conn.sendall((json.dumps({"state": "log_error", "error": "file_not_found"}) + "\n").encode("utf-8"))
            except socket.timeout:
                pass
            time.sleep(0.01)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        conn.close()


def _command_server_thread(port, log_dir):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(2)
    print(f"[cmd] Command server listening on 0.0.0.0:{port}")
    while True:
        try:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=_handle_cmd_client, args=(conn, addr, log_dir), daemon=True).start()
        except OSError:
            break


def _stream_server_thread(port, rate_hz):
    """Same as robot_hal's, but joints come from the MuJoCo sim state."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(5)
    print(f"[stream] Listening on 0.0.0.0:{port} at {rate_hz} Hz")
    clients, clients_lock = [], threading.Lock()

    def accept_loop():
        while True:
            try:
                conn, _ = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with clients_lock:
                    clients.append(conn)
            except OSError:
                break
    threading.Thread(target=accept_loop, daemon=True).start()

    period = 1.0 / rate_hz
    while True:
        t0 = time.time()
        with _state_lock:
            joints_deg = [round(v, 4) for v in _sim_q_deg]
        msg = json.dumps({"joints_deg": joints_deg, "timestamp": time.time()}) + "\n"
        dead = []
        with clients_lock:
            for conn in clients:
                try:
                    conn.sendall(msg.encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    dead.append(conn)
            for c in dead:
                clients.remove(c); c.close()
        rem = period - (time.time() - t0)
        if rem > 0:
            time.sleep(rem)


def _save_log(log_rows, target_angles, controller, log_dir, enabled):
    global _last_log_filename, _desktop_log_stamp
    if not enabled or not log_rows:
        return
    os.makedirs(log_dir, exist_ok=True)
    stamp = _desktop_log_stamp if _desktop_log_stamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    _desktop_log_stamp = None
    target_str = "_".join(str(int(a)) for a in target_angles)
    filename = os.path.join(log_dir, f"sim_hal_{stamp}_t{target_str}.csv")
    header = (
        ["controller", "timestamp", "loop"]
        + [f"q{i}" for i in range(MAX_JOINTS)] + [f"qvel{i}" for i in range(MAX_JOINTS)]
        + [f"target{i}" for i in range(MAX_JOINTS)] + [f"cmd_pos{i}" for i in range(MAX_JOINTS)]
        + [f"cmd_vel{i}" for i in range(MAX_JOINTS)] + [f"u{i}" for i in range(MAX_JOINTS)]
        + [f"hal_vel{i}" for i in range(MAX_JOINTS)] + [f"hal_torq{i}" for i in range(MAX_JOINTS)]
        + ["err_norm", "poll_ms", "solve_ms", "hal_write_ms", "sleep_ms"]
    )
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(log_rows)
    _last_log_filename = os.path.basename(filename)
    print(f"  Log saved: {filename}")


# ── Control loop (mirrors robot_hal.run_control_loop; plant = MuJoCo) ──
def run_control_loop_sim(model, data, data_lock, gains, target_angles, duration_sec,
                         controller, pos_tol=0.5, vel_tol=1.0, settle_steps=10,
                         log_dir=None, log_enabled=True, trajectory=None, traj_dt=None):
    nj = MAX_JOINTS
    use_traj = bool(trajectory) and bool(traj_dt)
    final_target = list(target_angles)
    n_traj = len(trajectory) if use_traj else 0
    kp, kd, ki = gains["kp"], gains["kd"], gains["ki"]
    iclamp, grav, tlim = gains["integral_clamp"], gains["gravity_comp"], gains["torque_limit"]
    vlim = gains["vel_cmd_limit"]
    out_lim = np.asarray(RATED_TORQUE_OUTPUT[:nj], float)
    sim_dt = model.opt.timestep
    # Two-rate: outer law at the control period (like robot_hal), inner motor-PID at
    # the physics rate. Solving the outer law every physics step over-drives it.
    control_dt = max(PERIOD_SEC, sim_dt)
    n_sub = max(1, int(round(control_dt / sim_dt)))

    integral = [0.0] * nj
    integral_v = np.zeros(nj)
    prev_q_deg = None
    prev_target_deg = None
    t_start = time.time()
    loop_count = 0
    converged_count = 0
    done_reason = "duration"
    traj_done = not use_traj
    log_rows = []

    while (time.time() - t_start) < duration_sec:
        loop_t0 = time.perf_counter()
        elapsed = time.time() - t_start
        if use_traj:
            idx = int(elapsed / traj_dt)
            if idx >= n_traj - 1:
                idx = n_traj - 1
                traj_done = True
            target_angles = trajectory[idx]
        target_deg = list(target_angles)

        with data_lock:
            q_rad = data.qpos[:nj].copy()
            qd_rad = data.qvel[:nj].copy()
        q_deg = rad_to_linuxcnc_deg(q_rad)   # LinuxCNC degrees (solver convention, like robot_hal)
        qd_deg = np.rad2deg(qd_rad)          # deg/s (signs +1, offset-free)

        # --- OUTER law at the control period (same calls as robot_hal) ---
        t_solve = time.perf_counter()
        if controller == "pid":
            next_pos, vel_cmd, integral = pid_solve(list(q_deg), target_deg, integral, q_vel=list(qd_deg), prev_q=prev_q_deg, dt=control_dt)
        elif controller == "invdyn":
            res = invdyn_solve(list(q_deg), target_deg, None, q_vel=list(qd_deg), prev_q=prev_q_deg, dt=control_dt, prev_target=prev_target_deg)
            next_pos, vel_cmd = res[0], res[1]
        elif controller == "mpc":
            next_pos, vel_cmd = mpc_solve(list(q_deg), target_deg, dt=control_dt)
        else:  # pd_velff
            next_pos, vel_cmd = pd_velff_solve(list(q_deg), target_deg, prev_target_deg, q_vel=list(qd_deg), prev_q=prev_q_deg, dt=control_dt)
        solve_ms = (time.perf_counter() - t_solve) * 1000

        # Clamp commanded velocity (as mujoco_viewer does); convert LinuxCNC deg -> URDF rad.
        vel_cmd = np.clip(np.asarray(vel_cmd, float), -vlim, vlim)
        next_pos_rad = linuxcnc_deg_to_rad(np.asarray(next_pos, float))
        vel_cmd_rad = np.deg2rad(vel_cmd)

        # --- INNER motor-PID at the physics rate: tau = Kp*e + Ki*∫e - Kd*qd + G(q) ---
        # next_pos held for the control period; tau recomputed each substep. Damping
        # is on the ACTUAL velocity (-Kd*qd), not chasing the outer vel_cmd — with the
        # large virtual-driver Kd, tracking a big vel_cmd would saturate and destabilize
        # coupled joints (the real motor PID's velocity feedforward is far gentler).
        with data_lock:
            for _ in range(n_sub):
                qc = data.qpos[:nj]
                qdc = data.qvel[:nj]
                pos_err = next_pos_rad - qc
                integral_v = np.clip(integral_v + pos_err * sim_dt, -iclamp, iclamp)
                tau = kp * pos_err + ki * integral_v - kd * qdc
                if grav:
                    tau = tau + data.qfrc_bias[:nj]
                tau = np.clip(np.clip(tau, -tlim, tlim), -out_lim, out_lim)
                data.qfrc_applied[:nj] = tau
                mujoco.mj_step(model, data)
            new_q_deg = rad_to_linuxcnc_deg(data.qpos[:nj]).tolist()
        with _state_lock:
            _sim_q_deg[:] = new_q_deg

        loop_count += 1
        u = [float(next_pos[i] - q_deg[i]) for i in range(nj)]
        err = float(np.linalg.norm(np.asarray(final_target) - q_deg))
        vel_norm = float(np.linalg.norm(vel_cmd))
        if traj_done and err < pos_tol and vel_norm < vel_tol:
            converged_count += 1
            if converged_count >= settle_steps:
                done_reason = "converged"
                break
        else:
            converged_count = 0

        if log_enabled:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            log_rows.append([
                controller, ts, loop_count,
                *q_deg.tolist(), *qd_deg.tolist(), *target_deg, *list(next_pos),
                *list(vel_cmd), *u, *qd_deg.tolist(), *tau.tolist(),
                err, 0.0, round(solve_ms, 3), 0.0, 0.0,
            ])
        prev_q_deg = q_deg.tolist()
        prev_target_deg = target_deg

        # pace the outer loop to wall clock at the control period
        rem = control_dt - (time.perf_counter() - loop_t0)
        if rem > 0:
            time.sleep(rem)

    _save_log(log_rows, final_target, controller, log_dir or LOG_DIR, log_enabled)
    return done_reason


def build_model_and_data():
    convert_meshes(MESHES_DIR, CACHE_DIR)
    patched = patch_urdf(URDF_PATH, CACHE_DIR)
    murdf = os.path.join(CACHE_DIR, "mycobot_pro_630.urdf")
    with open(murdf, "w", encoding="utf-8") as f:
        f.write(patched)
    tmp = mujoco.MjModel.from_xml_path(murdf)
    mjcf = os.path.join(CACHE_DIR, "mycobot_pro_630.xml")
    mujoco.mj_saveLastXML(mjcf, tmp)
    enhance_mjcf(mjcf)
    model = mujoco.MjModel.from_xml_path(mjcf)
    return model, mujoco.MjData(model)


def consume_commands(model, data, data_lock, gains, default_controller, log_dir, log_enabled):
    """Dequeue desktop commands and run each as a control loop (like robot_hal.main)."""
    print("\n" + "=" * 60)
    print("Simulated robot ready. Controller: pid | invdyn | pd_velff | mpc")
    print("Drive it with control_robot.py / calibrate_perturb.py --host 127.0.0.1")
    print("=" * 60 + "\n")
    global _desktop_log_stamp
    while True:
        try:
            cmd = _cmd_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        target = cmd.get("target_deg")
        if target is None or len(target) != MAX_JOINTS:
            continue
        duration = cmd.get("duration", 2.0)
        controller = cmd.get("controller", default_controller)
        if controller not in ("pid", "invdyn", "pd_velff", "mpc"):
            controller = default_controller
        _desktop_log_stamp = cmd.get("log_stamp") if isinstance(cmd.get("log_stamp"), str) else None
        pos_tol = cmd.get("pos_tol", 0.5)
        settle = cmd.get("settle_steps", 10)
        trajectory = cmd.get("trajectory")
        traj_dt = cmd.get("traj_dt")
        if trajectory and traj_dt:
            duration = max(duration, len(trajectory) * traj_dt + 1.0)

        with data_lock:
            current = rad_to_linuxcnc_deg(data.qpos[:MAX_JOINTS]).tolist()
        err = float(np.linalg.norm(np.asarray(target) - np.asarray(current)))
        kind = "trajectory" if (trajectory and traj_dt) else "target"
        print(f"[cmd] Moving ({kind}) -> {[round(v,1) for v in target]} controller={controller}")
        _update_cmd_status("moving", current, target, err)

        t0 = time.perf_counter()
        done = run_control_loop_sim(model, data, data_lock, gains, target, duration, controller,
                                    pos_tol=pos_tol, settle_steps=settle, log_dir=log_dir,
                                    log_enabled=log_enabled, trajectory=trajectory, traj_dt=traj_dt)
        with data_lock:
            final = rad_to_linuxcnc_deg(data.qpos[:MAX_JOINTS]).tolist()
        ferr = float(np.linalg.norm(np.asarray(target) - np.asarray(final)))
        _update_cmd_status("done", final, target, ferr, done_reason=done,
                           robot_exec_ms=round((time.perf_counter() - t0) * 1000, 2),
                           last_log_name=_last_log_filename)
        print(f"[cmd] Done. exit_reason={done} err={ferr:.3f}°")


def main():
    ap = argparse.ArgumentParser(description="MuJoCo simulated myCobot Pro 630 (drop-in for robot_hal.py)")
    ap.add_argument("--controller", choices=["pid", "invdyn", "pd_velff", "mpc"], default="pid",
                    help="Default control law (overridable per command from the desktop)")
    ap.add_argument("--cmd-port", type=int, default=CMD_PORT)
    ap.add_argument("--stream-port", type=int, default=STREAM_PORT)
    ap.add_argument("--stream-rate", type=float, default=STREAM_RATE_HZ)
    ap.add_argument("--no-log", action="store_true", help="Disable per-move CSV logging")
    ap.add_argument("--viewer", action="store_true", help="Open a MuJoCo window (control loop runs in a thread)")
    args = ap.parse_args()

    print("Building MuJoCo model from URDF...")
    model, data = build_model_and_data()
    gains = _load_inner_gains()

    # Start at the LinuxCNC home pose [-90,-90,0,-90,0,0] (= URDF [-90,0,0,0,0,0]).
    data.qpos[:MAX_JOINTS] = linuxcnc_deg_to_rad(HOME_LINUXCNC_DEG)
    data.qvel[:MAX_JOINTS] = 0.0
    mujoco.mj_forward(model, data)
    with _state_lock:
        _sim_q_deg[:] = rad_to_linuxcnc_deg(data.qpos[:MAX_JOINTS]).tolist()
    print(f"  Start pose (LinuxCNC deg): {HOME_LINUXCNC_DEG}")

    data_lock = threading.Lock()
    log_enabled = not args.no_log
    threading.Thread(target=_stream_server_thread, args=(args.stream_port, args.stream_rate), daemon=True).start()
    threading.Thread(target=_command_server_thread, args=(args.cmd_port, LOG_DIR), daemon=True).start()

    if args.viewer:
        threading.Thread(target=consume_commands,
                         args=(model, data, data_lock, gains, args.controller, LOG_DIR, log_enabled),
                         daemon=True).start()
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                with data_lock:
                    viewer.sync()
                time.sleep(0.02)
    else:
        consume_commands(model, data, data_lock, gains, args.controller, LOG_DIR, log_enabled)


if __name__ == "__main__":
    main()
