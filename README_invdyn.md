# InvDyn control for MyCobot Pro 630

Inverse-dynamics control using the same Raspi/LinuxCNC + HAL architecture as MPC.

## Files

- **invdyn_hal.py** — HAL component `invdyn`: runs on the robot (Raspi). Control law: desired acceleration `q̈_d = Kp*e - Kd*q̇` (deg/s²), then `next_pos = q + q̇*dt + 0.5*q̈_d*dt²`, `vel_cmd = q̇ + q̈_d*dt`. Writes to `invdyn.joint{i}_pos_cmd` and `invdyn.joint{i}_vel_cmd`. Same command server (port 9998) and streaming server (port 9999) as `mpc_hal.py`.
- **invdyn_linuxcnc.py** — Optional: InvDyn loop via MDI (G-code waypoints), like `mpc_linuxcnc.py`. Use when not using HAL direct write.
- **elerob_invdyn.hal** — HAL config that loads `invdyn_hal.py` and wires `invdyn.*` pins to the mux/PID.
- **elerob_invdyn.ini** — LinuxCNC config that uses `elerob_invdyn.hal` (same as `elerob_mpc.ini` but with InvDyn HAL).

---

## How to run (control the robot via Raspi)

Same idea as MPC: start LinuxCNC on the **Raspi** with the InvDyn config; then from the **desktop** send targets with `control_robot.py`.

### 1. On the Raspi (robot)

- Copy or sync the repo to the Raspi (e.g. `/home/pi/Desktop/mpc/`).
- In **elerob_invdyn.hal**, set the path to `invdyn_hal.py` if needed (line 16):
  ```hal
  loadusr -Wn invdyn python /home/pi/Desktop/mpc/invdyn_hal.py
  ```
- From the repo directory, start LinuxCNC with the InvDyn config (same way you start the MPC config, but with the InvDyn INI):
  ```bash
  cd /home/pi/Desktop/mpc
  linuxcnc elerob_invdyn.ini
  ```
  Or, if you keep the config in a subdir:
  ```bash
  linuxcnc /home/pi/Desktop/mpc/config/elerob_invdyn.ini
  ```
  (ensure `elerob_invdyn.ini` and `elerob_invdyn.hal` are in that config dir or adjust paths in the INI).
- LinuxCNC will start the GUI and load the HAL file; **invdyn_hal.py** is started by `loadusr` and will enable the machine, then listen on **port 9998** (commands) and **port 9999** (streaming). You should see “Waiting for commands from desktop (control_robot.py)...”.

### 2. On the desktop (your laptop/PC)

- From the **mycobot_mpc** repo, run **control_robot.py** with the Raspi’s IP and ask for the **invdyn** controller:
  ```bash
  python3 control_robot.py --host 10.0.0.27 --joints 0 -90 0 -90 0 0 --controller invdyn
  ```
  Or use `--xyz`, `--home`, `--interactive` as with MPC; add a way to pass `--controller invdyn` (see below if your script doesn’t support it yet).
- The desktop connects to `host:9998` and sends JSON like:
  ```json
  {"target_deg": [0, -90, 0, -90, 0, 0], "duration": 5.0, "controller": "invdyn"}
  ```
  The Raspi runs the InvDyn loop and drives the robot to the target.

- **ROS2 rviz:** By default, `control_robot.py` does not launch rviz2. Use `--rviz` to launch **rviz2** and **robot_pose_stream_ros2.py**. The streamer connects to the robot’s **port 9999** (same as with MPC). `invdyn_hal.py` runs the same streaming server on 9999, so rviz2 will show the robot motion when using InvDyn.

### 3. Optional: InvDyn via MDI (like mpc_linuxcnc.py)

If you want to run an InvDyn loop **on the Raspi** without the desktop (e.g. a fixed target for testing):

1. Start LinuxCNC first (e.g. `linuxcnc elerob_mpc.ini` or your usual config; no need for InvDyn HAL for this).
2. In a terminal on the Raspi:
   ```bash
   cd /home/pi/Desktop/mpc
   python3 invdyn_linuxcnc.py
   ```
   That runs a fixed sequence (current+5° then home) via MDI. Edit `main()` in `invdyn_linuxcnc.py` to change targets.

---

## Desktop: passing `controller` to the robot

**control_robot.py** sends a `controller` field in the JSON command. If your version doesn’t expose it on the command line, you can add an option, e.g.:

```bash
python3 control_robot.py --host 10.0.0.27 --controller invdyn --joints 0 -90 0 -90 0 0
```

and in the code ensure the dict sent to the robot includes `"controller": args.controller` (or `"invdyn"` by default when using the InvDyn config).

---

## Gains

- **invdyn_hal.py**: `INVDYN_KP = 144`, `INVDYN_KD = 24`, `QDD_MAX_DEG = 150` (tune in the script).
- **invdyn_linuxcnc.py**: same Kp/Kd and `QDD_MAX_DEG`.
