This is the custom MPC (placeholder) for mycobot pro 630 robot from Elephant Robotics


The robot is the 6DOF manipulator controlled by STM32 and connected to Raspberry Pi(4) for control
Existing python api has slow communication (>20ms) and not responding while the command is done.

- By directly using the HAL, communication between the STM32 <==> Raspi becomes around 2.1 ms (safely use 3ms)
- But if computing MPC or trajectory optimization in a separate computed, this can be increased


Connect to the raspi via SSH with pi@10.0.0.27


In raspi, run the command with 

```linuxcnc /home/pi/Desktop/mpc/elerob_mpc.ini```

RVIZ2 (ros2) command to show mycobot pro 630 urdf with single suction cup gripper 

```source ~/miniconda3/etc/profile.d/conda.sh && conda activate ros_env && source ~/ros2_ws/install/setup.zsh && cd ~/ros2_ws && colcon build --packages-select mycobot_description --symlink-install && source ~/ros2_ws/install/setup.zsh && ros2 launch mycobot_description display.launch.py```


joystick controller 
```PYTHONPATH= python3.10 joystick_controller.py --host 10.0.0.27 --speed medium --hz 20```

## ROS 2 bridge (robot as a ROS 2 node)

`robot_hal.py` runs inside LinuxCNC and exposes two TCP servers (port **9999**
joint-state stream, port **9998** command/status). `mycobot_ros2_bridge.py`
wraps those into a ROS 2 node so the arm is a first-class fleet peer — it does
**not** touch HAL directly (only the LinuxCNC process can).

**Published topics**

| Topic | Type | Meaning |
|-------|------|---------|
| `/joint_states` | `sensor_msgs/JointState` | joint angles in URDF **radians** (rviz-ready) |
| `/mycobot/joint_states_deg` | `sensor_msgs/JointState` | raw LinuxCNC **degrees** |
| `/mycobot/status` | `std_msgs/String` | controller status JSON (state, error_norm, …) |
| `/mycobot/clock_offset_ms` | `std_msgs/Float64MultiArray` | live `recv - robot_stamp` (sync quality) |

### Timestamp synchronization

`robot_hal.py` stamps every stream packet with `time.time()` on the Pi. The
bridge uses **that source time** as the ROS header stamp (`--stamp robot`,
default) instead of the desktop's receive time, so `/joint_states` reflects when
the joints were actually sampled — consistent across the whole fleet **as long
as the Pi and desktop clocks are NTP/chrony-synced**. The bridge publishes and
logs `recv - robot_stamp` on `/mycobot/clock_offset_ms` so you can watch sync
quality live (≈ network latency when synced; large/drifting ⇒ clocks not
synced). Use `--stamp local` to fall back to desktop receive time.

**Subscribed topics / service**

| Topic | Type | Meaning |
|-------|------|---------|
| `/mycobot/cmd/joint_deg` | `std_msgs/Float64MultiArray` | 6 target angles, LinuxCNC degrees |
| `/mycobot/cmd/joint_rad` | `sensor_msgs/JointState` | 6 targets, URDF radians |
| `/mycobot/cmd/move` | `std_msgs/String` | raw JSON: `{"target_deg":[...],"duration":3,"controller":"pid"}` |
| `/mycobot/home` (service) | `std_srvs/Trigger` | move to the home pose |

Targets are clamped to the LinuxCNC soft limits from `joint_conventions.py`.

**Run** (after starting `linuxcnc elerob.ini`, which runs `robot_hal.py`):

```bash
# on the Pi (robot is local):
python3 mycobot_ros2_bridge.py --robot-host 127.0.0.1
# from the desktop instead:
python3 mycobot_ros2_bridge.py --robot-host 10.0.0.27
```

**Move the arm** (⚠️ real motion — clear the workspace, keep e-stop in reach):

```bash
python3 move_arm_ros2.py --home
python3 move_arm_ros2.py --deg -90 -90 0 -90 0 0 --duration 3 --controller pid --wait
# or directly:
ros2 topic pub --once /mycobot/cmd/move std_msgs/String \
  '{data: "{\"target_deg\":[0,-90,0,-90,0,0],\"duration\":3}"}'
ros2 service call /mycobot/home std_srvs/srv/Trigger
```

### Calibration over ROS 2 (synced-timestamp logging)

`calibrate_ros2.py` runs the `calibrate_perturb.py` sequence **through the
bridge** and logs command input + robot response on one synchronized clock. For
each joint it perturbs +/-`step` deg about a measured base pose, returning to
base between moves: `base -> base[j]+step -> base -> base[j]-step -> base`.

```bash
python3 calibrate_ros2.py --dry-run                 # print the plan, no motion
python3 calibrate_ros2.py --quick                   # smoke test: joint 0, +/-5 deg
python3 calibrate_ros2.py --step-deg 10 --duration 6   # full perturbation set
```

Output: `logs/calibrate_ros2_<stamp>.jsonl`, one JSON object per line. Commands
are stamped with the desktop clock; responses (`/joint_states`,
`/mycobot/status`) with the robot's source time — both on the **same NTP/chrony
-synced timeline** (`t_sync`):

```json
{"t_sync": ..., "type": "command",  "label": "J0+5", "target_deg": [...]}
{"t_sync": ..., "type": "response", "source": "joint_states", "joints_rad": [...]}
{"t_sync": ..., "type": "response", "source": "status", "current_deg": [...], "error_norm": ...}
```

> Requires the Pi and desktop clocks to be synced (see `ros2node` chrony setup).
> The robot must be running `linuxcnc /home/pi/Desktop/mpc/elerob.ini` (the only
> ini whose HAL starts `robot_hal.py`'s 9998/9999 servers).

Plot command vs response from a log:

```bash
python3 plot_calibrate_ros2.py            # newest log -> logs/<stamp>.png
```

It overlays each joint's commanded target (step) on the measured response
(`/joint_states`, converted back to LinuxCNC degrees) against `t_sync`.

**Test without hardware** — `tests/mock_robot_server.py` emulates the 9999/9998
protocol so the bridge and move scripts can be exercised on any machine:

```bash
python3 tests/mock_robot_server.py &          # fake robot
python3 mycobot_ros2_bridge.py --robot-host 127.0.0.1 &
ros2 topic echo /joint_states                  # watch state
python3 move_arm_ros2.py --deg 45 -90 0 -90 0 0
```

## cuRobo motion-planning controller (collision-free)

`curobo_controller_node.py` turns a **goal** (Cartesian pose or joint target)
into a **collision-free trajectory** with NVIDIA cuRobo, then streams it to the
arm through the bridge's existing `trajectory` command path. cuRobo (Py3.10 +
CUDA) and rclpy (system Humble, Py3.8) can't share a process, so the GPU
planning runs in a sidecar — `curobo_planner_server.py`, in the `curobo` conda
env — and the node talks to it over a newline-JSON socket (the same pattern
`robot_hal.py` uses for 9998/9999).

```
goal ─▶ curobo_controller_node (rclpy) ─socket▶ curobo_planner_server (GPU)
        │                                          │  MotionGen, collision spheres
        │  rad → LinuxCNC deg                       ▼  → trajectory (URDF rad + dt)
        └─▶ /mycobot/cmd/move ─▶ bridge ─▶ robot_hal (tracks the trajectory)
```

`curobo_planner_server.py` lives with the cuRobo config at
`frankkimrobotics/ros2_mycobot/src/mycobot_description/curobo/`.

**Topics**

| Topic | Type | Direction | Meaning |
|-------|------|-----------|---------|
| `/joint_states` | `sensor_msgs/JointState` | sub | current q (URDF rad) → plan start |
| `/mycobot/curobo/goal_pose` | `geometry_msgs/PoseStamped` | sub | EE goal in `base_link` |
| `/mycobot/curobo/goal_joint` | `sensor_msgs/JointState` | sub | joint goal (URDF rad) |
| `/mycobot/cmd/move` | `std_msgs/String` | pub | trajectory command for the bridge |
| `/mycobot/curobo/status` | `std_msgs/String` | pub | plan result JSON (success, solve_time, …) |

**Run** (planner + bridge must be up; the bridge can run against
`tests/mock_robot_server.py` for a hardware-free dry run):

```bash
# 1) GPU planner (curobo env). --ground-z sets the table height in base_link.
conda activate curobo
python curobo_planner_server.py                      # 127.0.0.1:9997
#   add obstacles with a cuRobo world yaml:  --world world.yml

# 2) controller node (system ROS env). DRY RUN by default — add --execute to move.
python3 curobo_controller_node.py                    # plan + log only
python3 curobo_controller_node.py --execute --controller pid

# 3) send a goal
ros2 topic pub --once /mycobot/curobo/goal_pose geometry_msgs/PoseStamped \
  '{header: {frame_id: base_link}, pose: {position: {x: 0.30, y: 0.20, z: 0.35},
    orientation: {w: 0.0, x: 1.0, y: 0.0, z: 0.0}}}'
ros2 topic echo /mycobot/curobo/status
```

> ⚠️ `--execute` moves the real arm. Without it the node plans, logs, and
> publishes `/mycobot/curobo/status` but sends nothing. The planner enforces
> self-collision + the ground/obstacle world; targets are still clamped to the
> LinuxCNC soft limits by the bridge.

### Planner backend: v1 vs V2 (dynamics-aware)

The controller node is backend-agnostic (same socket protocol on port 9997).
Pick the planner server:

| Server | Env | Planner |
|--------|-----|---------|
| `curobo_planner_server.py` | `curobo` (0.7.7) | kinematic trajopt (vel/accel/jerk limits) |
| `curobo_planner_server_v2.py` | `curobo2` (0.8.0) | **dynamics-aware** trajopt: B-spline + torque limits + inverse dynamics |

cuRobo **V2** reads link mass/inertia/CoM from the URDF `<inertial>` tags and
joint torque limits from `<limit effort=...>` — so the mesh-derived inertials
(`compute_inertia.py`) actually feed its planning. The V2 server also returns
the **B-spline control points** in the response (`control_points`), alongside
the sampled trajectory. The V2 robot config is auto-ported from the v1 yaml at
startup to `mycobot_pro_630_v2.yml`.

```bash
# V2 backend (dynamics-aware); curobo2 env
conda activate curobo2
python curobo_planner_server_v2.py            # 127.0.0.1:9997
# then the SAME controller node, unchanged:
python3 curobo_controller_node.py --execute
```

> Note: the URDF `<limit effort>` values are still the placeholder `1000` Nm, so
> V2's torque limiting is effectively inactive until realistic joint torque
> limits are set. The inertials are real (mesh-derived); the effort limits are
> the remaining piece for meaningful torque-aware planning.

## online_servo — 4 ms streaming controller (vs robot_hal)

`robot_hal.py` converges to a target **per command** (~600 ms/cmd), so it can't track a
high-rate stream. `online_servo.py` replaces it when driving the arm from an **online /
streaming planner**: it **welds incoming trajectory chunks** into a continuous reference
`q_ref(t)` and **servos it at the HAL command rate** (`controller_params period_ms=4` →
**250 Hz**), so the desktop just streams chunks.

```
desktop planner ── weld chunks (TCP JSON, :9994) ──▶ online_servo  → q_ref(t)
  servo @250 Hz:  target = q_ref(now + lead)  → PID → HAL pos/vel cmd   (pure-PD, no windup)
  feedback on :9999  (joints_deg + per-joint torque pro600.joint{i}_torqfb)
```

- **Dead-time lead** — the actuator has a constant transport dead-time; sampling
  `q_ref(now + lead)` cancels it for known trajectories (only reactive events pay the delay).
- **Pure-PD** — the welded reference is the feed-forward, so the integral is dropped (a
  wound-up integral through the dead-time was the source of terminal overshoot).
- **Torque in the stream** — `:9999` now also carries `pro600.joint{i}_torqfb`, enabling
  torque-based contact detection on the desktop side.

Load it instead of `robot_hal.py` via LinuxCNC (the Pi runs `loadusr -Wn ctrl python
online_servo.py` from **`elerob_online.hal`**, started with `linuxcnc elerob_online.ini`).
**`elerob_online.hal`** also moves the PID + `pro_socketcan` off the 20 ms slow-thread onto a
fast thread (staged at 10 ms; 4 ms target) with `motor_time_interval` matched — otherwise the
4 ms loop is downsampled to 50 Hz at the drive (the HAL link itself is ~2.1 ms).

The cuRobo planner can feed it as **0.4 s sliding-window chunks at 10 Hz**; the welder bridges
the 10 Hz chunk rate to the 250 Hz control rate. (Desktop side: `../pick_and_place`
`online_planner_node.py` + `chunk_to_pi.py`.)

## MPC on the real robot — validated process (2026-08-24)

The `mpc` controller is now a **lag-aware LQR-clamp** (`controller_solvers.mpc_solve`):
`u = -K @ [pos_err, vel]`, `K = [53.48, 4.71]`, clamped ±40 °/s — the closed form of a
2-state drive-lag MPC, sim-tuned on the MuJoCo twin to **0.0% overshoot** at the
drive-limited rise time (the old integrator MPC overshot ~5% = the drive braking
distance; pure PID rings 17–21%). Solver-free, so it runs on the Pi (no osqp needed).
Gains assume drive lag kv≈40/s @ 4 ms; recompute via the DARE if stage-1 identification
disagrees.

### Bring-up ritual (the only sequence that reliably works)

1. Power the arm; press the base START button once (watch `pro600.svr_poweroned`).
2. Kill any stale stack: all `linuxcnc` pids + `rm /tmp/linuxcnc.lock`.
3. `linuxcnc ~/Desktop/mpc/elerob.ini` — robot_hal then self-initializes everything
   (drive power-on, motor init, machine-on, ctrl preload) and serves :9998/:9999.
4. Home if using the headless variant (`set home -1` via linuxcncrsh :5007).
5. **Probe before any motion**: command +1° on J1, verify the :9999 stream moves.
   Frozen drives + repeated commands wind the PID integral into a jump hazard.

### robot_hal changes to keep (in this repo's copy)

* **Idle hold mode**: between commands robot_hal now re-servos the last target in
  0.6 s bursts (0.4 s breather for the stream thread). Without it the 250 Hz loop
  stopped at command end → drives coasted → gravity sag ~0.4 °/s → STM32
  ferror-tripped the enable (the recurring "drift"/dropout).
* `mpc_solve` receives `q_vel` (the lag model is useless with v=0).

### Validation protocol

`pick_n_place/real_ctrl_validate.py --exec` (dry-run without `--exec`):
stage 1 fits the real drive lag from a +5° step (gate: within 30% of kv=40);
stage 2 replays the sim step-bench with `pid` vs `mpc` and prints metrics against
the sim predictions (mpc: 0% overshoot, 0.5–0.7 s settle); stage 3 steps all six
joints at once (coupling check). Probes before every case, homes between cases,
logs everything, writes a comparison plot.

Known hardware caveat: servo-enable hold-time degrades across soft restarts and
resets only with a full power cycle — suspected 48 V path issue, physical
inspection pending. Time-box on-robot sessions accordingly.
