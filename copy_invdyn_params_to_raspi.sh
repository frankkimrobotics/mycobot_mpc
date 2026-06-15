#!/usr/bin/env bash
# Copy logs/invdyn_params.npz to the Raspi so robot_hal can use it with --params.
# Usage: ./copy_invdyn_params_to_raspi.sh [RASPI_IP]
#   If RASPI_IP is omitted, uses ROBOT_IP env var or prompts.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPZ="${SCRIPT_DIR}/logs/invdyn_params.npz"
REMOTE_DIR="/home/pi/Desktop/mpc/logs"

if [ ! -f "$NPZ" ]; then
  echo "Error: $NPZ not found. Run: python identify_invdyn_from_log.py -o logs/invdyn_params.npz"
  exit 1
fi

HOST="${1:-${ROBOT_IP}}"
if [ -z "$HOST" ]; then
  echo "Usage: $0 <RASPI_IP>"
  echo "   or set ROBOT_IP and run: $0"
  exit 1
fi

echo "Copying $NPZ -> pi@${HOST}:${REMOTE_DIR}/"
ssh pi@"$HOST" "mkdir -p $REMOTE_DIR"
scp "$NPZ" "pi@${HOST}:${REMOTE_DIR}/invdyn_params.npz"
echo "Done. On Raspi: linuxcnc elerob.ini (robot_hal will load --params ${REMOTE_DIR}/invdyn_params.npz)"
