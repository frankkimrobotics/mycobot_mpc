#!/bin/bash
# Clean launch of the custom LinuxCNC + robot_hal stack (survives ssh logout).
pkill -x linuxcncsvr; pkill -x milltask; pkill -x rtapi_app; pkill -x halui; pkill -x io; pkill -f elephantmonitor
sleep 2; rm -f /tmp/linuxcnc.lock
systemctl --user stop lcnc-mpc 2>/dev/null; systemctl --user reset-failed lcnc-mpc 2>/dev/null
systemd-run --user --unit=lcnc-mpc --working-directory=/home/pi/Desktop/mpc -p StandardOutput=file:/home/pi/Desktop/mpc/logs/lcnc_stack.log -p StandardError=file:/home/pi/Desktop/mpc/logs/lcnc_stack.err linuxcnc /home/pi/Desktop/mpc/elerob.ini
