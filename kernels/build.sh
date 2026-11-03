#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
FLAGS="--offload-arch=gfx1030 -O3 -ffast-math -std=c++17"

echo "== forward gate =="; hipcc fa_forward.hip -o fa_forward $FLAGS -DFA_STANDALONE
echo "== decode gate ==";  hipcc fa_decode.hip  -o fa_decode  $FLAGS -DFA_STANDALONE
echo "== running gates =="; ./fa_forward; ./fa_decode
