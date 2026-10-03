#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CUDA_HOME:?Set CUDA_HOME to a CUDA 12.1 toolkit}"
test -x "$CUDA_HOME/bin/nvcc"
export CUTLASS_ROOT="${CUTLASS_ROOT:-$PWD/.third_party/cutlass}"
if [[ ! -d "$CUTLASS_ROOT" ]]; then
    git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_ROOT"
fi
test -f "$CUTLASS_ROOT/include/cutlass/cutlass.h"
cutlass_commit=$(git -C "$CUTLASS_ROOT" rev-parse HEAD)
if [[ "$cutlass_commit" != f7b19de32c5d1f3cedfc735c2849f12b537522ee ]]; then
    echo "Expected CUTLASS v3.5.1 commit f7b19de32c5d1f3cedfc735c2849f12b537522ee; found $cutlass_commit" >&2
    exit 1
fi
uv_binary="${UV_BIN:-uv}"
"$uv_binary" sync --frozen
"$uv_binary" run --frozen python -c 'from kernel.ops import extension; print(extension().__file__)'
