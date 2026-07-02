# Real-time model-based tracking with a REDUCED model — plan

Goal: shrink the ~2 s tracking rise (following lag) on the MyCobot Pro 630 streaming
controller by adding **feed-forward** — reference-velocity FF (no model) plus a **reduced
dynamics model** (gravity + friction, closed-form) — while keeping the control compute cheap
enough to run inside the real-time loop, with no heavy dependencies.

## First, separate two things the phrase "real-time communication" bundles

1. **Comm/loop RATE** (4 ms / 10 ms): set by the RTAPI base clock `SERVO_PERIOD` (10 ms floor).
   Gains/models do NOT change it. 10 ms is already active; 4 ms is a **real-time-KERNEL** task
   (PREEMPT_RT + isolcpus + thread priorities on the Pi), tracked separately in Phase 4. This is
   the remaining wall once compute is cheap.
2. **Real-time model-based CONTROL**: running the feed-forward law every loop tick without
   blowing the budget. The **reduced model is what makes this feasible** — the full Pinocchio
   M/C/G can't run deterministically at 100–250 Hz on the Pi (dependency + per-call cost); a
   closed-form gravity+friction model runs in microseconds. That's the point of "reduced."

## Where the lag actually is (measured this session)

- `q_ref -> pos_cmd` (online_servo follower): **~2 s rise**, dq/dt = 0.33·clip(q_ref-q, ±max_step).
  Purely reactive (kp·e); the **reference velocity q̇_ref is discarded**. THIS is the target.
- `pos_cmd -> posfb` (drive): **~153 ms, 0.04°** — already tight. Not the problem.
- So: fix online_servo's command generation. The drive is velocity-commanded (`ctrl.vel_cmd`),
  no torque port, so feed-forward enters as **velocity**, not torque.

## The reduced model (final definition)

Per-joint, closed-form, pure numpy (NO Pinocchio), ~µs to evaluate:

    vel_cmd_i = q̇_ref_i                      # (A) reference-velocity feed-forward  <-- biggest lever, no model
              + k_trim · (q_ref_i - q_i)      # (B) light position feedback trim
              + bias_i(q, q̇)                  # (C) reduced-dynamics feed-forward:

    bias_i(q,q̇) = Ĝ_i(q) + F̂c_i·sat(q̇_i) + F̂v_i·q̇_i     [+ optional M̂_ii·q̈_ref_i]

- Ĝ(q): gravity, closed-form 6-DOF (link mass × moment-arm; ~10–12 params). Dominant static term.
- F̂c, F̂v: per-joint Coulomb + viscous friction (12 params). Fixes low-speed stiction/deadband.
- M̂_ii: diagonal inertia only (6 params), for accel FF; optional.
- **Dropped:** coupled M(q) off-diagonals and full C(q,q̇) (Coriolis ∝ q̇², negligible ≤60°/s,
  and the expensive Pinocchio part). ~30 params total vs the full regressor.
- Units caveat: (C) is a torque-equivalent; the drive is velocity-commanded, so the bias must be
  mapped/scaled into `ctrl.vel_cmd` units and calibrated (Phase 2). The drive's own inner loop
  already handles most gravity droop (posfb tracks pos_cmd to 0.04°), so (C) is a TRIM — (A) is
  the win. Lead with (A), add (C) only if residual droop remains.

## Phases

### Phase 0 — Instrument + baseline (desktop + arm when up; no risk)
- `from __future__ import annotations` in invdyn_model.py / controller_params.py / controller_solvers.py
  so imports work on the Pi's Python 3.7 (already partly done; verify).
- Capture clean baselines with the HAL pin sampler: `q_ref` (welder), `ctrl.jointN_pos_cmd`,
  `ctrl.jointN_vel_cmd`, `pro600.jointN_posfb`, during (i) a step and (ii) a slow ramp — per joint.
  Quantify following error vs commanded velocity → the curve we're flattening.

### Phase 1 — Reference-velocity feed-forward (the 80/20 win, NO model)
- In `online_servo._servo_loop`: sample `q̇_ref` from the welder (finite-diff of q_ref(t±dt) or a
  welder velocity method) and set `vel_cmd = q̇_ref + k_trim·(q_ref - q)`; keep `pos_cmd = q_ref`.
- Calibrate the `q̇_ref -> ctrl.vel_cmd` scale against the drive (we saw vel_cmd≈50 ↔ ~4°/s at the
  pin; find the true units so q̇_ref maps to the intended deg/s). Start k_trim small.
- Expect the rise to collapse from ~2 s toward the drive floor (~150–250 ms) with SMALL following
  error at speed — because the arm is now told the velocity, not left to react to position error.
- This also dissolves the max_step speed/accuracy tradeoff (max_step becomes a pure safety clamp).

### Phase 2 — Reduced-dynamics trim (gravity + friction)
- Implement closed-form Ĝ(q) + friction in a small `reduced_dyn.py` (no Pinocchio). Seed Ĝ from the
  URDF mesh-derived inertials (see memory: inertials are mesh-derived) as nominal.
- Identify F̂c, F̂v (and refine Ĝ) from logged data: reuse `identify_invdyn_from_log.py`'s regressor
  but restrict to the reduced parameter set. ID trajectories = slow per-joint sweeps + static
  gravity poses (log `pro600.jointN_torqfb`, which we confirmed is real, ~2 mNm noise).
- Add `bias_i` to vel_cmd (Phase-1 law), calibrated into velocity units. Validate the model
  predicts torqfb within a target (%), then check the residual following error drops further.

### Phase 3 — RT-loop hardening (make the compute deterministic)
- Measure the per-tick compute of (A)+(B)+(C): must be << budget (should be tens of µs).
- Preallocate all arrays, no per-tick allocation/GC, no Pinocchio import in the hot path.
- Confirm a clean, jitter-free 10 ms (100 Hz drive) loop; watch RT-latency warnings.

### Phase 4 — Toward 4 ms (separate REAL-TIME-KERNEL track, optional)
- Lower `SERVO_PERIOD` needs deterministic sub-10 ms scheduling: PREEMPT_RT kernel, isolcpus for
  the servo thread, SCHED_FIFO priority, disable CPU freq scaling. Only attempt once Phase 3 shows
  compute headroom. Expect diminishing returns: the drive's own ~150 ms + firmware dominates; 4 ms
  mostly reduces command-delivery jitter, not the drive's physical response.

### Phase 5 — Hardware validation
- Step + ramp traces before/after (Phase-0 protocol). Target: following error at 15 °/s from
  ~36–45° down to a few degrees; rise ~2 s → ~0.2 s.
- Re-run the pick-and-place: transit fast AND accurate (no more max_step tradeoff); watch for
  following-error/CAN faults and oscillation. Stage gains incrementally.

## Risks / notes
- Velocity-command scaling is the crux of Phase 1 — get the `q̇_ref -> vel_cmd` units right, or the
  FF over/under-drives. Bench it with small moves first.
- Drive is deliberately detuned; over-aggressive FF/trim can trip following-error faults (FERROR is
  huge, so it tolerates a lot, but the ~60°/s velocity ceiling is hard).
- No identified model exists yet (no .npz) and Pinocchio may be absent on the Pi — the reduced,
  URDF-seeded closed-form model sidesteps both.
- Biggest bang-for-buck is Phase 1 (velocity FF); Phases 2–4 are refinements. Do Phase 1 first and
  re-measure before investing in ID / RT-kernel work.
