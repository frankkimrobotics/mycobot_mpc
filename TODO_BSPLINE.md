# TODO — B-spline trajectory transport, interpolation and welding

Status 2026-09-17: **not implemented.** What we have today:

| piece | where | what it does | spline-aware? |
|---|---|---|---|
| `TrajectoryWelder` | `traj_weld.py` (desktop, `online_servo.py`) | welds **sampled** chunks on a 10 ms grid, linear interp, smoothstep blend over 60 ms | no |
| chunk mode | `robot_hal.py` on the Pi (`_weld_chunk` / `_sample_ref`) | welds **sampled** chunks (`{"chunk":[[6 deg]…],"traj_dt","t_anchor"}`), linear interp, no blend | no |
| policy action encoding | `pick_and_place/policy/convert_litdata.py`, `real_rollout.py` | 16 control points of a **clamped cubic B-spline**, fixed knots (`dataset_spec.json`), evaluated on the desktop, sent as targets | desktop only |
| cuRobo plans | `curobo_planner_server_v2.py` | dense samples at 25 ms; streamed via `ctrl_tuner /api/stream_traj` after time-scaling | no |

So a B-spline (control points + knots) is always **flattened to samples on the desktop** before it
reaches the robot, and welds are done on those samples. Nothing preserves the spline's C² structure
across chunk boundaries, and velocity feedforward is a finite difference of the samples.

## Goal

Send the policy's (or planner's) spline **as control points**, evaluate it **on the Pi at the control
rate**, weld spline-to-spline with exact C¹/C² continuity, and feed the controller analytic
velocity (and acceleration) instead of finite differences.

## Plan / TODO

1. **Spline chunk message** (protocol, both ends)
   `{"spline": {"ctrl": [[6]×n], "knots": [m], "degree": 3, "t_anchor": epoch, "t_span": s,
   "time_scale": k}, "seq", "gains", "tag"}` — 16×6 floats instead of 100×6 samples per 1.5 s.
   `t_span` maps spline parameter `s ∈ [0, s_max]` to wall time from `t_anchor`; `time_scale`
   is the `--slow` factor (spline phase advances at `1/k` real time). Ack like `ack_chunk`.
2. **Pi-side evaluator** (`robot_hal.py`, pure numpy, Python 3.7)
   De Boor evaluation of position **and analytic 1st/2nd derivatives** for a clamped cubic
   (derivative splines from control-point differences). Budget: < 0.2 ms for 6 DOF at 100 Hz.
   Unit test against `scipy.interpolate.BSpline` on the desktop (same knots as `dataset_spec.json`).
3. **Spline-to-spline weld with exact continuity**
   On arrival of a new spline at `t_anchor`: evaluate the *live* reference `q, q̇, q̈` at `t_anchor`,
   then **overwrite the new spline's first three control points** so its `s=0` position, velocity and
   acceleration match (for a clamped cubic these are determined by `P0, P1, P2`). No blend window,
   no linear interpolation. Keep the old spline active until `t_anchor`. Fallback if the policy's
   first point is far from the live state (> tol): insert a short quintic bridge instead of
   trusting the policy's start.
4. **Feedforward from derivatives**
   `u = vff·q̇_ref + kaff·q̈_ref − K0·(q − q_ref) − K1·(q̇ − q̇_ref)`. Measured tracking residuals are
   acceleration-shaped (12° 0.5 Hz sine: rms 0.10°), so the `kaff` term is the expected win.
   Scale `q̇_ref`, `q̈_ref` by `1/time_scale`, `1/time_scale²`.
5. **Limits and auto time-scaling on the Pi**
   Bound peak velocity/acceleration from control-point differences (convex-hull bound), stretch
   `t_span` so peak `q̇ ≤ vmax` (≈ 40–50 °/s) and peak `q̈ ≤` drive cap (≈ 800 °/s² at accel 4×);
   refuse the chunk (nack) if it would still exceed the following-error budget.
6. **Late / missing chunk policy**
   If no new spline has arrived when the live one ends: hold the end position (current behavior) —
   or, configurable, extrapolate for ≤ 100 ms with decaying velocity. Log the gap in the status.
   Chunks must be sent ≥ 100 ms before their anchor (measured Pi-side jitter ≤ 6 ms on the cable).
7. **Desktop side**
   `ctrl_tuner.py`: `/api/stream_spline` (+ GUI card: knots preset from `dataset_spec.json`,
   `time_scale`, `kaff`); `real_rollout.py`: send `ctrl_pts` per inference as a spline chunk
   anchored at `t_inference + margin` instead of evaluating locally; cuRobo path: optional
   least-squares fit of the dense plan to the same 16-point spline so both sources use one pipeline.
8. **Tests**
   a. offline: evaluator vs scipy; weld continuity (`|Δq̇|`, `|Δq̈|` at the junction = 0 numerically);
   b. MuJoCo twin;
   c. hardware, same joint-0 rig as 2026-09-17: sine encoded as spline vs sampled-chunk baseline
      (S17: rms 0.117°, max 0.217°); then receding-horizon overlapping splines at 10 Hz (policy-like);
      then a real policy rollout.
9. **Docs**: protocol in README (next to the chunk protocol), `dataset_spec.json` knots referenced
   as the canonical knot vector.

## Notes

* Fixed knots make the basis matrix constant per `s`; a precomputed basis table at the 10 ms grid
  is an alternative to De Boor if the evaluator is too slow on the Pi.
* Welding by control-point overwrite changes the first ~3 spans of the policy's spline; that is the
  intended behavior (the live state is the truth), same idea as `retime_to_velocity` in `traj_weld.py`.
* Everything above is joint-space; a Cartesian (relative-EEF `ctrl_pts_eef`) variant needs IK on the
  Pi and is out of scope.
