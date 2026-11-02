#!/usr/bin/env bash
# This branch's SM90 W4A8 integration uses the sgl-deep-gemm TVM-FFI API.
set -euo pipefail

script_dir=$(realpath "$(dirname "$0")")
exec bash "${script_dir}/build_sgl_deep_gemm.sh"
