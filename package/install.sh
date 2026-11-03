#!/usr/bin/env bash
set -euo pipefail

# usage: ./install.sh [-e] [--no-compat]
EDITABLE=""; COMPAT=1
for a in "$@"; do
  case "$a" in
    -e|--editable) EDITABLE="-e" ;;
    --no-compat)   COMPAT=0 ;;
  esac
done

export MAX_JOBS=5
export PYTORCH_ROCM_ARCH=gfx1030
# make sure ROCm's hipcc is on PATH (adjust if ROCm lives elsewhere)
export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
export PATH="$ROCM_HOME/bin:$PATH"

cd "$(dirname "$(readlink -f "$0")")"

python -m pip install --no-deps --no-build-isolation $EDITABLE .
if [ "$COMPAT" = 1 ]; then
  python -m pip install --no-deps --no-build-isolation $EDITABLE ../compat
fi
