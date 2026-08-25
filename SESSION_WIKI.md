# MyCobot Pro 630 — Robot Control Session Wiki

Quick reference for the perception + cuRobo motion-control work (2026-06-21/22).
Helper scripts live in `mycobot_mpc/session_tools/`.

---

## 1. Infrastructure & what must be running

| Component | Where | Detail |
|---|---|---|
| **Robot** | Pi `10.0.0.27` | `linuxcnc elerob.ini` → runs `robot_hal.py` (ports **9998** cmd / **9999** stream). **Must be started from the Pi's desktop** (vendor GUI `elephantmonitor` on `DISPLAY=:0`). |
| **ROS bridge** | desktop | `python3 mycobot_ros2_bridge.py --robot-host 10.0.0.27` → `/joint_states`, `/mycobot/cmd/move`, `/mycobot/status`, `/mycobot/home` |
| **cuRobo planner** | desktop, `curobo2` env | `curobo_planner_server_v2.py --ground-z -0.1` on **127.0.0.1:9997** (newline-JSON: `ping`/`plan_pose`/`plan_joint`). Config at `frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo/`. |
| **Cameras** | desktop USB | D405 `218622271300`, D435 `043422070101`. libusb `pyrealsense2` at `~/librealsense/build/release`. |
| **SAM 3 server** | desktop, `sam3` env | `ros2node/scripts/run_sam3_server.sh` → ZMQ `tcp://127.0.0.1:5599` (perception only). |

### Standard env setup (before any move/FK script)
```bash
cd ~/Desktop/2026/mycobot_mpc
source ~/Desktop/2026/ros2node/config/ros2node.env            # ROS 2 Humble, DOMAIN_ID=42
export PYTHONPATH=~/Desktop/2026/mycobot_mpc:~/librealsense/build/release:$PYTHONPATH
```
FK scripts run in the **curobo2** env: `~/miniconda3/envs/curobo2/bin/python ...`

---

## 2. Command cheat-sheet

```bash
# --- Move to a joint config (URDF rad), velocity-limited cuRobo trajectory ---
python3 session_tools/move_to_q.py --target "[0,-0.349,1.92,0,-1.571,0]" --max-vel-deg 18 --duration 4

# --- Move TCP (suction-cup tip) to a Cartesian pose (tries orientations) ---
python3 session_tools/move_cart.py        # edit TARGET=[x,y,z] inside

# --- Move FLANGE center to a Cartesian pose, keep/choose orientation ---
python3 session_tools/move_flange.py --pos 0.45,0.08,0.15 --quat w,x,y,z   # --quat optional

# --- base <-> target cycle, timed, plots cmd-vs-actual ---
python3 session_tools/cycle_demo.py --cycles 3 --max-vel-deg 28 --duration 1.0

# --- FK current pose (curobo2 env): reads /tmp/cur_q.json, prints tcp + flange ---
~/miniconda3/envs/curobo2/bin/python session_tools/fk_flange.py

# --- Perception capture (image + depth + SAM3 mask overlay + point cloud) ---
PYTHONPATH=~/librealsense/build/release python3 ~/Desktop/2026/ros2node/perception/capture_and_plot.py --cameras d405=218622271300 --prompt object
PYTHONPATH=~/librealsense/build/release python3 ~/Desktop/2026/ros2node/perception/object_pointclouds.py --serial 218622271300 --name d405 --prompt object --render-color rgb

# --- Perturbation loop (50x random SE3 perturb, capture, log, plot) ---
python3 perturb_loop.py --iters 50 --perturb 0.10 --rot-deg 10 --reach-tol-deg 8 --duration 2.5 --cameras d405 d435
```

---

## 3. Key concepts & gotchas

### Frames: tcp vs flange
- **tcp** = **tip of the suction cup** (same thing) = cuRobo `tool_frame`. **Every "move EE to [x,y,z]" targets the tcp by default.**
- **flange** = `link6` = joint6 output face = **0.06 m behind tcp** along the tool axis, rotated `RotY(90°)`.
- Chain: `link6 ──RotY(90°),0──▶ suction_cup ──+0.06m──▶ tcp`.
- To target the flange, convert flange goal → tcp goal: `p_tcp = p_f + R_f·[0.06,0,0]`, `R_tcp = R_f·RotY(90°)` (done in `move_flange.py`).

### ⚠️ Tool-orientation model/hardware MISMATCH
The URDF tool axis is **inverted vs the real hardware**: commanding model **"tool-down" makes the real tool point UP**, and vice-versa. To get the **real tool pointing down**, command the model's **tool-up (keep-current)** orientation. Flange *positions* (pure arm kinematics) are reliable; the tool-frame orientation is what's flipped. (Not yet fixed in URDF.)

### The ~0.58 s lag (dominant motion feature)
- A **fixed ~0.58 s command-transport dead-time** (publish → DDS → bridge → TCP → `robot_hal` queue → motion start), **uniform across all joints**.
- Dynamic (mid-move) error ≈ **velocity × 0.58 s** → fast joints show big apparent error (e.g. j2 at 32°/s ≈ 18° mid-move) but it's pure time-shift, **not** path error.
- Final **reach error ~3.6°** at settle (actual catches up once setpoint stops).
- This lag — not the planner/gains — is the real limiter on both tracking and safe speed.

### Stop / settle tolerances
- **Robot-side** (`robot_hal.run_control_loop`): "converged" when, *after the trajectory finishes*, pos error **< 0.5° (L2 over 6 joints)** AND velocity **< 1.0°/s**, held **10 loops**. Else runs `len(traj)·traj_dt + 1.0 s`.
- **Desktop-side** (`perturb_loop.execute`): settled when last **15 `/joint_states` samples span < 0.4°**.
- `reach_tol_deg` (3–8°) is a **quality flag / abort**, NOT a stop gate.

### Speed limits
- `scale_traj` time-scales every trajectory so **peak ≤ `--max-vel-deg`**, floored by `--duration` (only ever *slows*, never speeds beyond native).
- The move is **velocity-limited** at our cap → raising planner **accel/jerk gave 0 speedup** (proved empirically). The earlier low "following-error ceiling" estimate was wrong, and the real ceiling is now **MEASURED: the drives saturate at ~60 °/s on every joint** (`traj_speed_test.py`, 2026-06-22 — commanding 54→140 °/s all achieve ~60). So: ≤50 °/s tracks cleanly; **above ~60 °/s the drive saturates and accumulates following error → firmware fault** (80 °/s faulted the j2 corner). Configured joint limits (180–200°/s) and cuRobo native (~86°/s, URDF 1.5 rad/s) are *far above* the real ~60. **Practical `--max-vel-deg` ≈ 55** (just under saturation).
- Speedups achieved: 18→28→32°/s ⇒ per-leg ~5.0→3.5→3.0 s.

---

## 4. Changes made this session

| File | Change | Notes |
|---|---|---|
| `robot_hal.py` (on Pi) | Made soft-start `ramp_time` (was hard 0.7 s) and `pos_gain` (was 0.5) **tunable via cmd**; `vff_scale` already existed. | Backup `robot_hal.py.bak_*` on Pi. Deployed + LinuxCNC restarted. **This file already had full B-spline tracking** (the repo copy is an older version that lacks it). |
| `perturb_loop.py` | (1) `closest_reachable_plan()` — bisect perturbation toward base if full target unreachable. (2) Fixed **premature-settle** bug (windowed-stop only after commanded motion time). (3) Sends `track` gains (`ramp_time/pos_gain/vff_scale`) to robot_hal. (4) Logs achieved (scaled) rotation. | |
| `mycobot_pro_630.yml` (planner cfg) | `max_acceleration 12→24`, `max_jerk 500→2000`. | **No speedup — consider reverting.** Backup `.bak_*` saved. v2 cfg auto-ported on planner restart. |
| `ros2node/perception/capture_and_plot.py` | `colorize_depth` now uses a **Tukey-fence** auto-range (rejects z16-saturation far-noise that blew out the depth colormap). | D405 depth panel now renders correctly without `--depth-max`. |
| `session_tools/*` | New helper move/FK scripts (this session). | |

---

## 5. ⚠️ Hard-won lessons / warnings

- **DO NOT restart LinuxCNC over SSH.** The vendor GUI (`elephantmonitor`) needs the Pi's logged-in X session; SSH gets **"Can't open display"**. A headless restart **took the robot down** and required the user to restart from the Pi desktop. → For any `robot_hal.py` change: deploy the file, then **ask the user to restart LinuxCNC on the Pi**.
- The **cuRobo planner restart IS safe** (desktop process, port 9997) — kill + relaunch in `curobo2`, ~36 s warmup.
- `pkill -f "<pattern>"` **self-matches the SSH command** → kills your own shell. Use explicit PIDs / `pidof <exactname>`.
- A stale **`/tmp/linuxcnc.lock`** on the Pi blocks LinuxCNC startup — remove it.
- Move scripts read live `/joint_states`; they abort safely if the live start is >8° from the plan start. Big moves (e.g. → home, ~110°) are velocity-limited (`--max-vel-deg 18`) to avoid following-error faults.

---

## 6. Key poses (URDF deg unless noted)

| Pose | Joint config | EE (tcp) | Flange (link6) |
|---|---|---|---|
| **cuRobo default** (`retract`) | `[0,0,0,0,0,0]` | pos `[0, 0.067, 0.861]` | — |
| **Robot HOME** (`HOME_LINUXCNC_DEG`) | LinuxCNC `[-90,-90,0,-90,0,0]` = URDF `[-90,0,0,0,0,0]` | pos `[0.067, 0, 0.861]` | — |
| **Operating base** (perturb loop) | URDF `[0,-20.3,111.4,0,-90.3,-1.8]` | `[0.268, 0.073, 0.481]` | `[0.266, 0.073, 0.422]` |
| **Fine-tuned base** (set this session) | URDF `[0,-20,110,0,-90,0]` | — | `[0.270, 0.073, 0.429]`, quat `[0.500,0.500,-0.500,0.500]` |

Conventions (`joint_conventions.py`): `urdf_rad = sign·(linuxcnc_deg + offset)`, `JOINT_OFFSETS_DEG=[0,90,0,90,0,0]`, all signs +1.

---

## 7. Output locations
- Perception: `ros2node/captures/` (`d405_panel.png`, `_overlay.png`), `ros2node/captures/clouds/` (`.ply`, renders).
- Perturbation: `mycobot_mpc/captures/loop_<stamp>/iter_NN/` (frames, `move_log.json`, `joint_cmd_vs_actual.png`) + `summary_50runs.png`.
- Cycle plots: `mycobot_mpc/captures/cycle_cmd_vs_actual_<stamp>.png`.

---

## 8. Open items / TODO
- [ ] Investigate where the **0.58 s transport lag** is spent (bridge forward path vs `robot_hal` cmd-server `recv`/`sleep(0.01)` cadence) — biggest lever for tracking + speed.
- [ ] **Measure** the real per-joint following-error velocity ceiling (probe scripts). an earlier low estimate was wrong (40 & 50 °/s run fault-free); find the true cap to push `--max-vel-deg` further (up to the ~86°/s cuRobo native limit).
- [ ] **Revert** planner `accel 24 / jerk 2000` (gave no benefit).
- [ ] Fix the **tool-orientation URDF/hardware mismatch** (or document the flip).
- [ ] Persist the **fine-tuned base** `[0,-20,110,0,-90,0]` into `/tmp/perturb_plan.json` (regen `ee_current` FK) if it should be the new perturbation base.
