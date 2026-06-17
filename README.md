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

**Test without hardware** — `tests/mock_robot_server.py` emulates the 9999/9998
protocol so the bridge and move scripts can be exercised on any machine:

```bash
python3 tests/mock_robot_server.py &          # fake robot
python3 mycobot_ros2_bridge.py --robot-host 127.0.0.1 &
ros2 topic echo /joint_states                  # watch state
python3 move_arm_ros2.py --deg 45 -90 0 -90 0 0
```
