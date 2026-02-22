# Compare PID vs InvDyn performance

Use **elerob_invdyn.ini** on the robot so both `pid` and `invdyn` are available. Run the same target twice (once per controller), then plot the robot logs.

---

## 1. Robot (Raspi)

```bash
cd /home/pi/Desktop/mpc   # or your repo path on the Raspi
linuxcnc elerob_invdyn.ini
```

Leave this running. It loads **invdyn_hal.py**, which accepts `controller=pd` (PID) or `controller=invdyn`.

---

## 2. Desktop (laptop)

Set the robot IP and use the same target for both runs.

```bash
cd /path/to/mycobot_mpc
conda activate ros_env
export ROBOT_IP=10.0.0.27   # your Raspi IP
```

### Optional: RViz + stream (see robot move)

```bash
python robot_pose_stream_ros2.py --host $ROBOT_IP
```

(In another terminal, run the moves below.)

### Run 1: PID

```bash
python control_robot.py --host $ROBOT_IP --controller pid --joints -85 -85 5 -85 5 5 --duration 3
```

Log will be fetched to `logs/` (e.g. `invdyn_YYYYMMDD_HHMMSS_t-85_-85_5_-85_5_5.csv` with `controller=pd`).

### Run 2: InvDyn (same target)

```bash
python control_robot.py --host $ROBOT_IP --controller invdyn --joints -85 -85 5 -85 5 5 --duration 3
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
| Desktop | `python control_robot.py --host $ROBOT_IP --controller pid --joints -85 -85 5 -85 5 5 --duration 3` |
| Desktop | `python control_robot.py --host $ROBOT_IP --controller invdyn --joints -85 -85 5 -85 5 5 --duration 3` |
| Desktop | `python plot_log.py --robot --latest 2` |

Use the same `--joints` and `--duration` for both controllers so the comparison is fair.
