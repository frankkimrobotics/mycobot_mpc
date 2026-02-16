#!/usr/bin/env python3
"""
Offline simulation of the MPC (PD) controller from mpc_hal.py.

Simulates the closed-loop response to a target joint pose using the same
PD law and constants as the real controller. No LinuxCNC/HAL needed.

Usage:
    python sim_mpc.py                           # default demo
    python sim_mpc.py --target -85 -85 5 -85 5 5
    python sim_mpc.py --target -85 -85 5 -85 5 5 --q0 -90 -90 0 -90 0 0
    python sim_mpc.py --kp 0.5 --kd 0.05        # tune gains
    python sim_mpc.py --duration 5 --period 2    # 5s run, 2ms period
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
    mpc_solve_qp as _mpc_solve_qp,
    MAX_JOINTS, MPC_PERIOD_MS, U_MAX_PER_STEP, KP, KD,
)


def simulate(q0, target, duration_sec, period_ms):
    """Run the PD controller in a simulated loop.

    The "plant" is a simple integrator: q(k+1) = cmd_pos(k).
    This matches the real system where HAL writes the commanded position
    directly and the low-level PID + motor tracks it within one period.

    Uses mpc_solve_qp from mpc_hal.py directly (same KP, KD, U_MAX).
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

        # Controller — same function as the real robot
        next_pos, vel_cmd = _mpc_solve_qp(
            list(q), list(target), q_vel=None, prev_q=list(prev_q) if prev_q is not None else None, dt=dt,
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
    }


def plot_results(res):
    """Plot all 6 joint angles and velocities on two subplots."""
    t = res["t"]
    target = res["target"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"MPC (PD) Simulation — Kp={KP}, Kd={KD}, "
        f"period={MPC_PERIOD_MS}ms, U_max={U_MAX_PER_STEP}°/step",
        fontsize=12, fontweight="bold",
    )

    # Joint angles: desired (solid) + actual (dashed)
    for j in range(MAX_JOINTS):
        ax_pos.plot(t, np.full_like(t, target[j]), color=colors[j], linewidth=1.5)
        ax_pos.plot(t, res["q"][:, j], color=colors[j], linestyle="--", linewidth=1.2,
                    label=f"J{j}")
    ax_pos.set_ylabel("Joint angle (deg)")
    ax_pos.set_title("Joint Angles — solid: desired, dashed: actual")
    ax_pos.legend(fontsize=8, ncol=6, loc="upper right")
    ax_pos.grid(True, alpha=0.3)

    # Joint velocities
    for j in range(MAX_JOINTS):
        ax_vel.plot(t, res["qvel"][:, j], color=colors[j], linewidth=1, label=f"J{j}")
    ax_vel.set_ylabel("Velocity (deg/s)")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_title("Joint Velocities")
    ax_vel.legend(fontsize=8, ncol=6, loc="upper right")
    ax_vel.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate MPC (PD) controller")
    parser.add_argument("--q0", nargs=6, type=float, default=[-90, -90, 0, -90, 0, 0],
                        help="Initial joint angles (deg), 6 values")
    parser.add_argument("--target", nargs=6, type=float, default=[-85, -85, 5, -85, 5, 5],
                        help="Target joint angles (deg), 6 values")
    parser.add_argument("--duration", type=float, default=0.1,
                        help="Simulation duration (seconds)")
    parser.add_argument("--period", type=float, default=MPC_PERIOD_MS,
                        help="Control period (ms)")
    args = parser.parse_args()

    print(f"Simulation (params from mpc_hal.py):")
    print(f"  q0     = {args.q0}")
    print(f"  target = {args.target}")
    print(f"  Kp={KP}, Kd={KD}, period={args.period}ms, U_max={U_MAX_PER_STEP} deg/step")
    print(f"  duration = {args.duration}s")

    res = simulate(args.q0, args.target, args.duration, args.period)

    # Print summary
    final_err = res["err"][-1]
    threshold = 0.5
    settled = np.where(res["err"] < threshold)[0]
    t_settle = res["t"][settled[0]] if len(settled) > 0 else float("inf")
    print(f"\nResults:")
    print(f"  Final error norm:  {final_err:.4f} deg")
    print(f"  Settling time (<{threshold} deg): {t_settle:.3f}s")
    print(f"  Total steps:       {len(res['t'])}")

    plot_results(res)


if __name__ == "__main__":
    main()
