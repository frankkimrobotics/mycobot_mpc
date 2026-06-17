#!/usr/bin/env python3
"""
mock_robot_server :: emulate robot_hal.py's TCP servers WITHOUT LinuxCNC/hardware.

Lets you test mycobot_ros2_bridge.py and move_arm_ros2.py on any machine:
  * STREAM_PORT 9999 - streams {"joints_deg":[...], "timestamp":...} @ 50 Hz
  * CMD_PORT    9998 - accepts {"target_deg":[...]} and streams status JSON,
                        slewing the simulated joints toward the last target.

Run:  python3 tests/mock_robot_server.py
"""

import json
import socket
import threading
import time

MAX_JOINTS = 6
STREAM_PORT = 9999
CMD_PORT = 9998
RATE_HZ = 50.0

_state_lock = threading.Lock()
_joints = [-90.0, -90.0, 0.0, -90.0, 0.0, 0.0]  # home-ish
_target = list(_joints)
_status = {"state": "idle", "current_deg": list(_joints), "target_deg": list(_target),
           "error_norm": 0.0}


def _slew_loop():
    """Move joints toward target at a fixed rate; update status."""
    global _joints
    step = 60.0 / RATE_HZ  # deg per tick (~60 deg/s)
    while True:
        with _state_lock:
            for i in range(MAX_JOINTS):
                d = _target[i] - _joints[i]
                _joints[i] += max(-step, min(step, d))
            err = sum((_target[i] - _joints[i]) ** 2 for i in range(MAX_JOINTS)) ** 0.5
            _status["current_deg"] = [round(v, 3) for v in _joints]
            _status["target_deg"] = list(_target)
            _status["error_norm"] = round(err, 4)
            _status["state"] = "moving" if err > 0.5 else "done"
        time.sleep(1.0 / RATE_HZ)


def _stream_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", STREAM_PORT))
    s.listen(5)
    print(f"[mock stream] :{STREAM_PORT}")
    conns = []
    def accept():
        while True:
            c, _ = s.accept()
            conns.append(c)
    threading.Thread(target=accept, daemon=True).start()
    while True:
        with _state_lock:
            msg = json.dumps({"joints_deg": [round(v, 4) for v in _joints],
                              "timestamp": time.time()}) + "\n"
        for c in list(conns):
            try:
                c.sendall(msg.encode())
            except OSError:
                conns.remove(c)
        time.sleep(1.0 / RATE_HZ)


def _cmd_server():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", CMD_PORT))
    s.listen(2)
    print(f"[mock cmd] :{CMD_PORT}")
    while True:
        conn, addr = s.accept()
        threading.Thread(target=_handle_cmd, args=(conn,), daemon=True).start()


def _handle_cmd(conn):
    global _target
    conn.settimeout(1.0)
    buf = ""
    while True:
        with _state_lock:
            status = dict(_status)
        try:
            conn.sendall((json.dumps(status) + "\n").encode())
        except OSError:
            return
        try:
            data = conn.recv(4096)
            if not data:
                return
            buf += data.decode("utf-8", "replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    cmd = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "target_deg" in cmd and len(cmd["target_deg"]) == MAX_JOINTS:
                    with _state_lock:
                        _target = [float(v) for v in cmd["target_deg"]]
                    print(f"[mock cmd] new target: {_target}")
        except socket.timeout:
            pass
        time.sleep(0.02)


if __name__ == "__main__":
    threading.Thread(target=_slew_loop, daemon=True).start()
    threading.Thread(target=_stream_server, daemon=True).start()
    threading.Thread(target=_cmd_server, daemon=True).start()
    print("mock robot server running (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
