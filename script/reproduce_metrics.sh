#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${UV_BIN:-uv}" run --frozen python -m quantdemoire.reproduce "$@"
