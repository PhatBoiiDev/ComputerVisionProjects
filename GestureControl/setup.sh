#!/usr/bin/env bash
# Create the virtualenv, install dependencies and fetch the hand landmark model.
set -euo pipefail

cd "$(dirname "$0")"

MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH="models/hand_landmarker.task"

if [ ! -d .venv ]; then
  echo "==> creating virtualenv"
  python3 -m venv .venv
fi

echo "==> installing dependencies"
./.venv/bin/python -m pip install --upgrade pip -q
./.venv/bin/python -m pip install -q -r requirements.txt

if [ ! -f "$MODEL_PATH" ]; then
  echo "==> downloading hand landmark model (~7.5 MB)"
  mkdir -p models
  curl -sSL -o "$MODEL_PATH" "$MODEL_URL"
fi

echo "==> checking environment"
./.venv/bin/python -m gesturectl doctor || true

cat <<'EOF'

Setup finished.

If the check above reported a missing permission, grant it in System Settings,
then FULLY QUIT and reopen the terminal app before trying again -- macOS only
re-reads these permissions when the app starts.

Start it with:   ./run.sh
Test it safely:  ./run.sh --dry-run
EOF
