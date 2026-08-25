#!/usr/bin/env python3
"""Reset RealSense cameras until they train at USB-3 (fixes the 'stuck at USB 2.1'
link-speed problem). Uses the SDK hardware_reset() — no sudo, no replug.

  export PYTHONPATH=~/librealsense/build/release:$PYTHONPATH
  python3 session_tools/rs_reset.py                 # reset any camera not at USB3
  python3 session_tools/rs_reset.py --serial 218622271300 --tries 4
"""
import argparse, time, sys
import pyrealsense2 as rs

ap = argparse.ArgumentParser()
ap.add_argument("--serial", default=None, help="only this serial (default: all)")
ap.add_argument("--tries", type=int, default=3, help="max reset attempts per camera")
ap.add_argument("--wait", type=float, default=6.0, help="seconds to wait after a reset")
a = ap.parse_args()

def speed(d):
    return d.get_info(rs.camera_info.usb_type_descriptor) if d.supports(rs.camera_info.usb_type_descriptor) else "?"

def find():
    return [d for d in rs.context().query_devices()
            if a.serial is None or d.get_info(rs.camera_info.serial_number) == a.serial]

ds = find()
if not ds:
    print("no matching camera connected"); sys.exit(1)
for d in ds:
    print(f"{d.get_info(rs.camera_info.name)} {d.get_info(rs.camera_info.serial_number)}: {speed(d)}")

ok = True
for sn in [d.get_info(rs.camera_info.serial_number) for d in ds]:
    for t in range(a.tries):
        d = next((x for x in find() if x.get_info(rs.camera_info.serial_number) == sn), None)
        if d is None:
            print(f"  {sn}: vanished, waiting..."); time.sleep(a.wait); continue
        if speed(d).startswith("3"):
            print(f"  {sn}: USB3 ({speed(d)}) OK"); break
        print(f"  {sn}: {speed(d)} -> hardware_reset (try {t+1}/{a.tries})")
        try: d.hardware_reset()
        except Exception as e: print("    reset error:", e)
        time.sleep(a.wait)
    else:
        d = next((x for x in find() if x.get_info(rs.camera_info.serial_number) == sn), None)
        sp = speed(d) if d else "gone"
        print(f"  {sn}: still {sp} after {a.tries} tries -> try controller reset / a USB3 data cable")
        ok = False

sys.exit(0 if ok else 2)
