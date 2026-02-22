# InvDyn control for MyCobot Pro 630

Inverse-dynamics control using the same Raspi/LinuxCNC + HAL architecture as MPC.

## Dependencies

- **Robot (invdyn_hal.py):** numpy, linuxcnc, hal (no MuJoCo). Runs on the Raspi under LinuxCNC.
- **Desktop (control_robot.py):** numpy, optional IK (pyroki/jax for --xyz and --interactive). **No MuJoCo needed** for `--controller invdyn`; the desktop only sends targets to the robot.
- **Optional:** To run **mujoco_viewer.py** or **mycobot_pro630_streaming** invdyn streaming on the desktop, install MuJoCo in ros_env: `pip install -r requirements-ros_env.txt`.

## Files

- **invdyn_hal.py** — HAL component `invdyn`: runs on the robot (Raspi). **It is a PD controller**: desired acceleration `q̈_d = Kp*e - Kd*q̇`, then integrated to `next_pos` and `vel_cmd`. No mass/inertia matrix. The only difference from the `pd_solve` path is that this one integrates acceleration (smaller steps per cycle); both output pos/vel setpoints to the PIDs. Same command server (port 9998) and streaming server (port 9999) as `mpc_hal.py`.
- **invdyn_linuxcnc.py** — Optional: InvDyn loop via MDI (G-code waypoints), like `mpc_linuxcnc.py`. Use when not using HAL direct write.
- **elerob_invdyn.hal** — HAL config that loads `invdyn_hal.py` and wires `invdyn.*` pins to the mux/PID.
- **elerob_invdyn.ini** — LinuxCNC config that uses `elerob_invdyn.hal` (same as `elerob_mpc.ini` but with InvDyn HAL).

---

## How elerob_invdyn.ini works (which script runs?)

**elerob_invdyn.ini uses invdyn_hal.py, not invdyn_linuxcnc.py.**

1. You run: `linuxcnc elerob_invdyn.ini`.
2. LinuxCNC reads `[HAL] HALFILE = elerob_invdyn.hal`.
3. **elerob_invdyn.hal** runs: `loadusr -Wn invdyn python /home/pi/Desktop/mpc/invdyn_hal.py`  
   So **invdyn_hal.py** is loaded as the HAL component named `invdyn`. It creates pins `invdyn.joint0_pos_cmd`, `invdyn.enable`, etc., and starts the command server (port 9998) and stream server (port 9999). When the desktop sends a target, invdyn_hal.py runs the InvDyn (or PD) control loop and writes to those pins; the mux passes them to the PIDs when `invdyn.enable` is TRUE.

**invdyn_linuxcnc.py** is a different, optional path: a standalone script you run manually (`python invdyn_linuxcnc.py`). It drives the robot by sending G-code waypoints via LinuxCNC MDI. It is **not** loaded by the INI. Use it only if you want InvDyn without the HAL component (e.g. with a config that doesn’t load invdyn_hal.py).

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

## Troubleshooting: "Robot didn't move" (status shows moving/done but joints don't change)

If the desktop shows `[moving]` then `[done]` but the robot does not move (and reported `q` stays near the initial pose):

1. **Confirm InvDyn HAL is active**  
   On the Raspi you must have started LinuxCNC with **elerob_invdyn.ini** (not elerob_mpc.ini). The HAL must load `invdyn_hal.py` and wire `invdyn.jointN_pos_cmd` / `invdyn.jointN_vel_cmd` into the mux (see elerob_invdyn.hal).

2. **Check invdyn.enable on the Raspi**  
   The mux passes InvDyn commands to the PIDs only when `invdyn.enable` is TRUE. In a shell on the Raspi:
   ```bash
   halcmd getp invdyn.enable
   ```
   During a move it should be TRUE. If it is FALSE, InvDyn output is not selected and the robot will not follow invdyn commands.

3. **Check that InvDyn commands are changing**  
   While sending a move from the desktop, on the Raspi run:
   ```bash
   watch -n 0.5 'halcmd getp invdyn.enable; halcmd getp invdyn.joint3_pos_cmd'
   ```
   For a target with joint 4 = -60°, `invdyn.joint3_pos_cmd` (joint index 3) should move from ~-90 toward -60. If it never changes, the Python component may not be writing (or may have exited).

4. **Compare with PID**  
   Try the same target with PD to see if the robot moves at all:
   ```bash
   python control_robot.py --host 10.0.0.27 --controller pid --joints -90 -90 0 -60 0 0 --duration 3
   ```
   If the robot moves with `pid` but not with `invdyn`, the issue is specific to the InvDyn path (mux selection, invdyn.enable, or wiring of invdyn pins).

5. **Raspi console output**  
   In the terminal where LinuxCNC was started, check for `[cmd] Moving → ... controller=invdyn` and any Python tracebacks. If invdyn_hal.py crashes or never enters the control loop, the robot will not move.

---

## Gains

- **invdyn_hal.py**: `INVDYN_KP = 144`, `INVDYN_KD = 24`, `QDD_MAX_DEG = 150`, `INVDYN_PERIOD_MS = 20` (50 Hz). The loop period must be large enough that the position step 0.5*qdd*dt² is above the motor deadband (~20 ms works; 2 ms is too small and the robot won’t move).
- **invdyn_linuxcnc.py**: same Kp/Kd and `QDD_MAX_DEG`.
