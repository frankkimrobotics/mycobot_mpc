This is the custom MPC (placeholder) for mycobot pro 630 robot from Elephant Robotics


The robot is the 6DOF manipulator controlled by STM32 and connected to Raspberry Pi(4) for control
Existing python api has slow communication (>20ms) and not responding while the command is done.

- By directly using the HAL, communication between the STM32 <==> Raspi becomes around 2.1 ms (safely use 3ms)
- But if computing MPC or trajectory optimization in a separate computed, this can be increased


Connect to the raspi via SSH with pi@10.0.0.27


In raspi, run the command with 

"linuxcnc /home/pi/Desktop/mpc/elerob_mpc.ini"
