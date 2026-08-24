#!/usr/bin/env bash
# Launch the gesture controller. Any arguments are passed straight through.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "virtualenv missing -- run ./setup.sh first" >&2
  exit 1
fi

exec ./.venv/bin/python -m gesturectl "$@"
