#!/usr/bin/env bash
# One-time setup: set ROBOT_IP for your current conda environment.
# Activate the env you use for control_robot (e.g. ros_env), then run this.
#
# Usage:
#   conda activate ros_env
#   ./setup_robot_ip.sh              # use default 10.0.0.27
#   ./setup_robot_ip.sh 192.168.1.5  # use custom IP

set -e
ROBOT_IP="${1:-10.0.0.27}"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Source conda first, e.g.:"
  echo "  source ~/miniconda3/etc/profile.d/conda.sh"
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No conda env active. Activate your env first, then run this script again:"
  echo "  conda activate ros_env"
  echo "  ./setup_robot_ip.sh $ROBOT_IP"
  exit 1
fi

conda env config vars set ROBOT_IP="$ROBOT_IP" -p "$CONDA_PREFIX"
echo "Set ROBOT_IP=$ROBOT_IP for current env: $CONDA_PREFIX"
echo "Open a new terminal or run:  conda activate $(basename "$CONDA_PREFIX")"
echo "Then you can run without --host:  python control_robot.py --joints -90 -90 0 -90 0 0"
