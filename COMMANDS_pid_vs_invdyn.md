# Compare PID vs InvDyn performance

Use **elerob_invdyn.ini** on the robot so both `pid` and `invdyn` are available. Run the same target twice (once per controller), then plot the robot logs.

**Important:** `control_robot.py --controller invdyn` does **not** use MuJoCo on the desktop. It only sends the target to the robot; the robot runs **invdyn_hal.py** (NumPy + LinuxCNC/HAL). So for PID vs InvDyn comparison you only need the robot to run `elerob_invdyn.ini`. If you also want to run **mujoco_viewer.py** or **mycobot_pro630_streaming** invdyn streaming on the desktop, install MuJoCo in ros_env: `pip install -r requirements-ros_env.txt`.

---

## 1. Robot (Raspi)

```bash
cd /home/pi/Desktop/mpc   # or your repo path on the Raspi
linuxcnc elerob_invdyn.ini
```

Leave this running. It loads **invdyn_hal.py**, which accepts `controller=pd` (PID) or `controller=invdyn`. If you start **elerob_mpc.ini** by mistake, `--controller invdyn` on the desktop will be ignored (robot falls back to PD).

---

## 2. Desktop (laptop)

**One-time setup (optional):** set a default robot IP for `ros_env` so you can omit `--host`:

```bash
cd /path/to/mycobot_mpc
chmod +x setup_robot_ip.sh
./setup_robot_ip.sh 10.0.0.27   # your Raspi IP; default is 10.0.0.27
# then in new terminals: conda activate ros_env
```

Then for each session:

```bash
cd /path/to/mycobot_mpc
conda activate ros_env
# If you didn't run setup_robot_ip.sh, set: export ROBOT_IP=10.0.0.27
```

### Optional: RViz + stream (see robot move)

```bash
python robot_pose_stream_ros2.py --host $ROBOT_IP
```

(In another terminal, run the moves below.)

### Run 1: PID

```bash
python control_robot.py --controller pid --joints -85 -85 5 -85 5 5 --duration 3
```
(Add `--host $ROBOT_IP` or `--host 10.0.0.27` if you didn’t run `setup_robot_ip.sh`.)

Log will be fetched to `logs/` (e.g. `invdyn_YYYYMMDD_HHMMSS_t-85_-85_5_-85_5_5.csv` with `controller=pd`).

### Run 2: InvDyn (same target)

```bash
python control_robot.py --controller invdyn --joints -85 -85 5 -85 5 5 --duration 3
```

Another CSV will be saved with `controller=invdyn`.

### Plot robot logs (PID vs InvDyn)

```bash
python plot_log.py --robot
```

Or plot only the two most recent robot logs:

```bash
python plot_log.py --robot --latest 2
```

Figures are saved under `figures/`. Compare position error, command velocity, and per-loop timing (poll, solve, hal_write, sleep) between the two runs.

---

## Summary

| Where   | Command |
|--------|--------|
| Robot  | `linuxcnc elerob_invdyn.ini` |
| Desktop (one-time) | `./setup_robot_ip.sh 10.0.0.27` then `conda activate ros_env` in new terminals |
| Desktop | `python control_robot.py --controller pid --joints -85 -85 5 -85 5 5 --duration 3` |
| Desktop | `python control_robot.py --controller invdyn --joints -85 -85 5 -85 5 5 --duration 3` |
| Desktop | `python plot_log.py --robot --latest 2` |

Use the same `--joints` and `--duration` for both controllers so the comparison is fair.

---

## Troubleshooting: "invdyn didn't work"

1. **Robot must run elerob_invdyn.ini** (not elerob_mpc.ini). With mpc.ini, the robot only supports `pd` and `mpc`; `invdyn` is then treated as PD.
2. **No MuJoCo needed on desktop** for `control_robot.py --controller invdyn`. The inverse dynamics run on the Raspi in invdyn_hal.py (NumPy only).
3. **Optional – MuJoCo in ros_env** (for mujoco_viewer.py or mycobot_pro630_streaming invdyn mode):
   ```bash
   conda activate ros_env
   pip install -r requirements-ros_env.txt
   ```
4. **mycobot_controller.py** uses **PyBullet** (conda env `robot_sim`), not MuJoCo; it's a separate sim.
