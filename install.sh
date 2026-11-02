#!/usr/bin/env bash
set -euo pipefail

script_dir=$(realpath "$(dirname "$0")")
python_exe=${PYTHON:-python}
version=$(tr -d '[:space:]' < "${script_dir}/sgl_deep_gemm/VERSION")
shopt -s nullglob
wheels=("${script_dir}/dist/sgl_deep_gemm-${version}-"*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected one sgl-deep-gemm ${version} wheel; run bash develop.sh first." >&2
    exit 1
fi

# Use the active interpreter and preserve its existing Torch/TVM dependencies.
"$python_exe" -m pip install --no-deps --force-reinstall "${wheels[0]}"

# -I prevents the separate root pybind package from shadowing the installed API.
"$python_exe" -I - <<'PY'
import importlib.metadata
import deep_gemm

for name in (
    'fp8_fp4_mega_moe',
    'mega_moe_pre_dispatch_sm90',
    'transform_weights_for_mega_moe_sm90_fp4',
):
    if not hasattr(deep_gemm, name):
        raise RuntimeError(f'Installed deep_gemm is missing {name}')
if not hasattr(deep_gemm._C, 'fp8_fp4_mega_moe_sm90'):
    raise RuntimeError('Installed native module is missing SM90 W4A8 MegaMoE')
print('Installed sgl-deep-gemm:', importlib.metadata.version('sgl-deep-gemm'))
print('Loaded from:', deep_gemm.__file__)
PY
