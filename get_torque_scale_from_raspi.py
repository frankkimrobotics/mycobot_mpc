#!/usr/bin/env python3
"""
Read pro600 torque scale parameters from the Raspi's HAL (LinuxCNC must be running).

Runs halcmd getp on the Raspi via SSH and prints pro600.joint*_torque_enc_scale
and pro600.joint*_torque_scale. Use the reported scale with identify_invdyn_from_log.py
--torque-scale so that tau is in Nm and M is in kg·m².

Usage:
  # From laptop (SSHs to Raspi and runs halcmd):
  python get_torque_scale_from_raspi.py
  python get_torque_scale_from_raspi.py --host 10.0.0.27

  # Or on the Raspi directly (with LinuxCNC running):
  halcmd getp pro600.joint0_torque_enc_scale
"""

import argparse
import os
import subprocess
import sys

DEFAULT_SSH_USER = "pi"


def main():
    ap = argparse.ArgumentParser(
        description="Read pro600 torque scale from Raspi HAL (requires LinuxCNC running on robot)."
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("ROBOT_IP", "").strip(),
        help="Raspi IP (default: ROBOT_IP)",
    )
    ap.add_argument("--user", default=DEFAULT_SSH_USER, help="SSH user (default: pi)")
    args = ap.parse_args()

    if not args.host:
        print("Error: --host or ROBOT_IP required.", file=sys.stderr)
        sys.exit(1)

    # Single SSH run: execute all halcmd getp in one session (one password prompt)
    remote_script = (
        "for i in 0 1 2 3 4 5; do "
        "echo -n \"enc_$i \"; halcmd getp pro600.joint${i}_torque_enc_scale 2>/dev/null || echo ''; "
        "echo -n \"scale_$i \"; halcmd getp pro600.joint${i}_torque_scale 2>/dev/null || echo ''; "
        "done"
    )
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=no", f"{args.user}@{args.host}", remote_script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    # Parse "enc_0 1.7e-5" and "scale_0 -57676" lines
    enc_scales = [""] * 6
    scale_vals = [""] * 6
    for line in (r.stdout or "").strip().splitlines():
        line = line.strip()
        if line.startswith("enc_"):
            try:
                i = int(line.split()[0].replace("enc_", ""))
                enc_scales[i] = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
            except (ValueError, IndexError):
                pass
        elif line.startswith("scale_"):
            try:
                i = int(line.split()[0].replace("scale_", ""))
                scale_vals[i] = line.split(None, 1)[1] if len(line.split(None, 1)) > 1 else ""
            except (ValueError, IndexError):
                pass

    print(f"Reading torque scale from {args.user}@{args.host} (one SSH, halcmd)...")
    if r.returncode != 0 or not any(enc_scales):
        print("  SSH or halcmd failed. Is LinuxCNC running on the Raspi? Check stderr:", file=sys.stderr)
        if r.stderr:
            print(r.stderr, file=sys.stderr)
        print("  Tip: set up SSH keys so you don't need a password: ssh-copy-id pi@10.0.0.27")
        sys.exit(1)
    print("  (LinuxCNC must be running on the Raspi.)\n")

    print("pro600.joint*_torque_enc_scale (often: encoder/counts -> Nm):")
    for i in range(6):
        val = enc_scales[i]
        try:
            x = float(val)
            print(f"  joint{i}: {val}  ({x:.6e})")
        except (ValueError, TypeError):
            print(f"  joint{i}: {val or '(failed)'}")

    print("\npro600.joint*_torque_scale:")
    for i in range(6):
        val = scale_vals[i]
        try:
            x = float(val)
            print(f"  joint{i}: {val}  ({x:.4f})")
        except (ValueError, TypeError):
            print(f"  joint{i}: {val or '(failed)'}")

    # Suggest --torque-scale for identify_invdyn_from_log
    try:
        first_enc = float(enc_scales[0].strip())
        # hal_torq might be in counts; tau_Nm = counts * torque_enc_scale
        scale = abs(first_enc)
        print("\n--- For identify_invdyn_from_log.py ---")
        print("  If tau_Nm = hal_torq * torque_enc_scale, use:")
        print(f"    --torque-scale {scale:.10e}")
        print(f"  or (same): --torque-scale {scale}")
    except (ValueError, TypeError, IndexError):
        print("\n  (Could not suggest --torque-scale; check halcmd output above.)")


if __name__ == "__main__":
    main()
