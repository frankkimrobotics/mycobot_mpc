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
import numpy as np
import matplotlib.pyplot as plt

# ── Controller constants (mirrored from mpc_hal.py) ─────────────────────────
MAX_JOINTS = 6
MPC_PERIOD_MS = 2       # control period (ms)
U_MAX_PER_STEP = 5.0    # max position increment per step (deg)
KP = 0.3
KD = 0.1


def mpc_solve_qp(q, target, q_vel, dt, kp=KP, kd=KD):
    """PD controller — identical logic to mpc_hal.py:mpc_solve_qp.

    Returns (next_pos, vel_cmd, u_opt).
    """
    current = np.asarray(q, dtype=float)
    tgt = np.asarray(target, dtype=float)
    vel = np.asarray(q_vel, dtype=float)

    error = tgt - current
    u = kp * error - kd * vel
    u = np.clip(u, -U_MAX_PER_STEP, U_MAX_PER_STEP)

    next_pos = current + u
    vel_cmd = u / dt if dt > 0 else np.zeros(MAX_JOINTS)
    return next_pos, vel_cmd, u


def simulate(q0, target, duration_sec, period_ms, kp, kd):
    """Run the PD controller in a simulated loop.

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
    u_arr = np.zeros((n_steps, MAX_JOINTS))

    q = np.asarray(q0, dtype=float).copy()
    q_vel = np.zeros(MAX_JOINTS)

    for k in range(n_steps):
        t_arr[k] = k * dt
        q_arr[k] = q
        qvel_arr[k] = q_vel

        # Controller
        next_pos, vel_cmd, u = mpc_solve_qp(q, target, q_vel, dt, kp, kd)
        cmd_pos_arr[k] = next_pos
        cmd_vel_arr[k] = vel_cmd
        u_arr[k] = u
        err_arr[k] = np.linalg.norm(np.asarray(target) - q)

        # Plant update: position tracks command, velocity = position change / dt
        q_prev = q.copy()
        q = next_pos.copy()
        q_vel = (q - q_prev) / dt

    return {
        "t": t_arr,
        "q": q_arr,
        "qvel": qvel_arr,
        "cmd_pos": cmd_pos_arr,
        "cmd_vel": cmd_vel_arr,
        "u": u_arr,
        "err": err_arr,
        "target": np.asarray(target),
    }


def plot_results(res, kp, kd):
    """Plot all 6 joint angles and velocities on two subplots."""
    t = res["t"]
    target = res["target"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"MPC (PD) Simulation — Kp={kp}, Kd={kd}, "
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
    parser.add_argument("--kp", type=float, default=KP, help="Proportional gain")
    parser.add_argument("--kd", type=float, default=KD, help="Derivative gain")
    args = parser.parse_args()

    period_ms = args.period

    print(f"Simulation: q0={args.q0}")
    print(f"            target={args.target}")
    print(f"            Kp={args.kp}, Kd={args.kd}, period={period_ms}ms, duration={args.duration}s")
    print(f"            U_max={U_MAX_PER_STEP} deg/step")

    res = simulate(args.q0, args.target, args.duration, period_ms, args.kp, args.kd)

    # Print summary
    final_err = res["err"][-1]
    threshold = 0.5
    settled = np.where(res["err"] < threshold)[0]
    t_settle = res["t"][settled[0]] if len(settled) > 0 else float("inf")
    print(f"\nResults:")
    print(f"  Final error norm:  {final_err:.4f}°")
    print(f"  Settling time (<{threshold}°): {t_settle:.3f}s")
    print(f"  Total steps:       {len(res['t'])}")

    plot_results(res, args.kp, args.kd)


if __name__ == "__main__":
    main()
