#!/usr/bin/env bash
# One-touch hand-eye calibration: poses -> capture -> FK -> solve.
# Hides the env split (cv2/realsense/ROS in python3 ; cuRobo FK in the curobo2 env).
#
#   ./session_tools/run_calib.sh                          # defaults below
#   ./session_tools/run_calib.sh --target 0.35,0.08,0.0 --square 0.035 --marker 0.026
#   ./session_tools/run_calib.sh --poses captures/calib_poses_XXXX.json   # reuse a pose set
#
# NB: no `-u` (nounset) — sourcing the ROS env references unbound vars and would abort.
set -eo pipefail

# ---- defaults (override via flags) ----
TARGET="0.35,0.08,0.0"
SQUARE="0.035"
MARKER="0.026"
D405="218622271300"
D435="043422070101"
POSES=""                       # empty -> generate fresh from TARGET
MAXVEL="18"

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGET="$2"; shift 2;;
    --square) SQUARE="$2"; shift 2;;
    --marker) MARKER="$2"; shift 2;;
    --d405)   D405="$2";   shift 2;;
    --d435)   D435="$2";   shift 2;;
    --poses)  POSES="$2";  shift 2;;
    --max-vel-deg) MAXVEL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

REPO="$HOME/Desktop/2026/mycobot_mpc"
CUROBO_PY="$HOME/miniconda3/envs/curobo2/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="$REPO/captures/calib_session_$STAMP"
cd "$REPO"

# shellcheck disable=SC1090
source "$HOME/Desktop/2026/ros2node/config/ros2node.env" 2>/dev/null || true
export PYTHONPATH="$REPO:$HOME/librealsense/build/release:${PYTHONPATH:-}"

hr() { printf '\n=== %s ===\n' "$1"; }

# ---- 0. planner reachable? ----
if ! timeout 2 bash -c 'cat < /dev/null > /dev/tcp/127.0.0.1/9997' 2>/dev/null; then
  echo "ERROR: cuRobo planner not listening on 127.0.0.1:9997 — start it first." >&2
  exit 1
fi

# ---- 1. poses (generate unless one was supplied) ----
if [ -z "$POSES" ]; then
  hr "1/4  generating hemisphere poses around $TARGET"
  python3 session_tools/gen_calib_poses.py --target "$TARGET" --out "$SESSION/poses.json"
  POSES="$SESSION/poses.json"
else
  hr "1/4  using supplied poses: $POSES"
fi

# ---- 2. capture (move + grab + detect) ----
hr "2/4  capturing (move arm, grab D405+D435, detect board)"
python3 session_tools/calib_capture.py \
    --poses "$POSES" --square "$SQUARE" --marker "$MARKER" \
    --d405 "$D405" --d435 "$D435" --max-vel-deg "$MAXVEL" --out "$SESSION"

SJ="$SESSION/session.json"
[ -f "$SJ" ] || { echo "ERROR: capture produced no session.json" >&2; exit 1; }

# ---- 3. FK in curobo2 env (TCP pose per shot) ----
hr "3/4  FK (curobo2 env): tool-frame pose per shot"
"$CUROBO_PY" session_tools/calib_fk.py --session "$SJ"

# ---- 4. solve ----
hr "4/4  solving hand-eye extrinsics"
python3 session_tools/calib_solve.py --session "$SJ"

hr "DONE"
echo "session    : $SESSION"
echo "extrinsics : $SESSION/extrinsics.json"
