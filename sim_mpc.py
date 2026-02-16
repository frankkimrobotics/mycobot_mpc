#!/usr/bin/env python3
"""
Offline simulation of the PD and MPC controllers from mpc_hal.py.

Simulates the closed-loop response to a target joint pose using the same
control law and constants as the real controller. No LinuxCNC/HAL needed.

Usage:
    python sim_mpc.py                           # default (PD) demo
    python sim_mpc.py --mode mpc                # MPC with QP solver
    python sim_mpc.py --mode both               # compare PD vs MPC
    python sim_mpc.py --target -85 -85 5 -85 5 5
    python sim_mpc.py --duration 5 --period 2   # 5s run, 2ms period
"""

import argparse
import sys
import types
import numpy as np
import matplotlib.pyplot as plt

# Stub out linuxcnc and hal so mpc_hal can be imported without the real hardware
for mod_name in ("linuxcnc", "hal"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

from mpc_hal import (  # noqa: E402
    pd_solve as _pd_solve,
    mpc_solve as _mpc_solve,
    MAX_JOINTS, MPC_PERIOD_MS, U_MAX_PER_STEP, KP, KD,
    MPC_HORIZON, MPC_Q, MPC_R, MPC_Q_TERMINAL,
)


def simulate(q0, target, duration_sec, period_ms, mode="pd"):
    """Run PD or MPC controller in a simulated loop.

    The "plant" is a simple integrator: q(k+1) = cmd_pos(k).
    This matches the real system where HAL writes the commanded position
    directly and the low-level PID + motor tracks it within one period.
    """
    dt = period_ms / 1000.0
    n_steps = int(duration_sec / dt)

    # Storage
    t_arr = np.zeros(n_steps)
    q_arr = np.zeros((n_steps, MAX_JOINTS))
    qvel_arr = np.zeros((n_steps, MAX_JOINTS))
    cmd_pos_arr = np.zeros((n_steps, MAX_JOINTS))
    cmd_vel_arr = np.zeros((n_steps, MAX_JOINTS))
    err_arr = np.zeros(n_steps)

    q = np.asarray(q0, dtype=float).copy()
    prev_q = None

    for k in range(n_steps):
        t_arr[k] = k * dt
        q_arr[k] = q
        qvel_arr[k] = (q - prev_q) / dt if prev_q is not None else 0.0

        # Controller
        if mode == "mpc":
            next_pos, vel_cmd = _mpc_solve(list(q), list(target), dt)
        else:
            next_pos, vel_cmd = _pd_solve(
                list(q), list(target), q_vel=None,
                prev_q=list(prev_q) if prev_q is not None else None, dt=dt,
            )
        cmd_pos_arr[k] = next_pos
        cmd_vel_arr[k] = vel_cmd
        err_arr[k] = np.linalg.norm(np.asarray(target) - q)

        # Plant update: position tracks command
        prev_q = q.copy()
        q = np.asarray(next_pos, dtype=float)

    return {
        "t": t_arr,
        "q": q_arr,
        "qvel": qvel_arr,
        "cmd_pos": cmd_pos_arr,
        "cmd_vel": cmd_vel_arr,
        "err": err_arr,
        "target": np.asarray(target),
        "mode": mode,
    }


def plot_results(results_list):
    """Plot results for one or more controllers (PD, MPC, or both)."""
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    n_modes = len(results_list)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_pos, ax_vel = axes

    # Build title
    if n_modes == 1:
        res = results_list[0]
        mode = res["mode"].upper()
        if mode == "PD":
            title = (f"PD Simulation — Kp={KP}, Kd={KD}, "
                     f"period={MPC_PERIOD_MS}ms, U_max={U_MAX_PER_STEP}°/step")
        else:
            title = (f"MPC Simulation — N={MPC_HORIZON}, Q={MPC_Q}, R={MPC_R}, "
                     f"Q_term={MPC_Q_TERMINAL}, U_max={U_MAX_PER_STEP}°/step")
    else:
        title = (f"PD vs MPC — period={MPC_PERIOD_MS}ms, U_max={U_MAX_PER_STEP}°/step")

    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Plot target lines (same for all modes)
    target = results_list[0]["target"]
    for j in range(MAX_JOINTS):
        ax_pos.plot(results_list[0]["t"], np.full_like(results_list[0]["t"], target[j]),
                    color=colors[j], linewidth=1.5, alpha=0.4)

    linestyles = {"pd": "--", "mpc": "-"}

    for res in results_list:
        t = res["t"]
        mode = res["mode"]
        ls = linestyles.get(mode, "-")

        for j in range(MAX_JOINTS):
            lbl = f"J{j} ({mode.upper()})" if n_modes > 1 else f"J{j}"
            ax_pos.plot(t, res["q"][:, j], color=colors[j], linestyle=ls,
                        linewidth=1.2, label=lbl if j == 0 or n_modes > 1 else None)
            ax_vel.plot(t, res["qvel"][:, j], color=colors[j], linestyle=ls,
                        linewidth=1, label=lbl if j == 0 or n_modes > 1 else None)

    ax_pos.set_ylabel("Joint angle (deg)")
    ax_pos.set_title("Joint Angles — faded: desired")
    ax_pos.legend(fontsize=7, ncol=6, loc="upper right")
    ax_pos.grid(True, alpha=0.3)

    ax_vel.set_ylabel("Velocity (deg/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_title("Joint Velocities")
    ax_vel.legend(fontsize=7, ncol=6, loc="upper right")
    ax_vel.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate PD / MPC controller")
    parser.add_argument("--q0", nargs=6, type=float, default=[-90, -90, 0, -90, 0, 0],
                        help="Initial joint angles (deg), 6 values")
    parser.add_argument("--target", nargs=6, type=float, default=[-85, -85, 5, -85, 5, 5],
                        help="Target joint angles (deg), 6 values")
    parser.add_argument("--duration", type=float, default=0.1,
                        help="Simulation duration (seconds)")
    parser.add_argument("--period", type=float, default=MPC_PERIOD_MS,
                        help="Control period (ms)")
    parser.add_argument("--mode", type=str, default="pd", choices=["pd", "mpc", "both"],
                        help="Controller mode: pd, mpc, or both (comparison)")
    args = parser.parse_args()

    modes = ["pd", "mpc"] if args.mode == "both" else [args.mode]

    print(f"Simulation (params from mpc_hal.py):")
    print(f"  q0     = {args.q0}")
    print(f"  target = {args.target}")
    print(f"  period = {args.period}ms, U_max={U_MAX_PER_STEP} deg/step")
    print(f"  duration = {args.duration}s")
    print(f"  mode(s) = {modes}")

    results = []
    for mode in modes:
        print(f"\n--- Running {mode.upper()} ---")
        if mode == "pd":
            print(f"  Kp={KP}, Kd={KD}")
        else:
            print(f"  Horizon={MPC_HORIZON}, Q={MPC_Q}, R={MPC_R}, Q_term={MPC_Q_TERMINAL}")

        res = simulate(args.q0, args.target, args.duration, args.period, mode=mode)
        results.append(res)

        final_err = res["err"][-1]
        threshold = 0.5
        settled = np.where(res["err"] < threshold)[0]
        t_settle = res["t"][settled[0]] if len(settled) > 0 else float("inf")
        print(f"  Final error norm:  {final_err:.4f} deg")
        print(f"  Settling time (<{threshold} deg): {t_settle:.3f}s")
        print(f"  Total steps:       {len(res['t'])}")

    plot_results(results)


if __name__ == "__main__":
    main()
