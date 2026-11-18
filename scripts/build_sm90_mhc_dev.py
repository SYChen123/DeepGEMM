"""Build an isolated development package, without changing the installed wheel.

CUDA_VISIBLE_DEVICES=1 /path/to/csy/bin/python scripts/build_sm90_mhc_dev.py \
    --output build-sm90-mhc
PYTHONPATH=$PWD/build-sm90-mhc /path/to/csy/bin/python scripts/bench_sm90_megamhc.py
"""

import argparse
import os
from pathlib import Path
import shutil
import sysconfig

import torch
import tvm_ffi.cpp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = args.output.resolve() / "deep_gemm"
    package.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "cuda_helpers.py", "VERSION"):
        shutil.copy2(root / "sgl_deep_gemm" / name, package / name)
    for name in ("utils", "testing", "legacy", "mega"):
        shutil.copytree(root / "deep_gemm" / name, package / name, dirs_exist_ok=True)
    for source, target in (
        (root / "deep_gemm/include/deep_gemm", "deep_gemm"),
        (root / "third-party/cutlass/include/cute", "cute"),
        (root / "third-party/cutlass/include/cutlass", "cutlass"),
        (root / "third-party/deep_jit/include/deep_jit", "deep_jit"),
    ):
        shutil.copytree(source, package / "include" / target, dirs_exist_ok=True)
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    torch_dir = Path(torch.__file__).parent
    include = [cuda / "include", cuda / "include/cccl", Path(sysconfig.get_path("include")),
               torch_dir / "include", torch_dir / "include/torch/csrc/api/include",
               package / "include", root / "third-party/deep_jit/include"]
    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
    flags = ["-std=c++20", "-O3", "-fPIC", "-Wno-psabi", "-Wno-deprecated-declarations",
             f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}"]
    build = args.output.resolve() / "extension-build"
    build.mkdir(exist_ok=True)
    lib = tvm_ffi.cpp.build(
        name="_C", cpp_files=[str(root / "csrc/tvm_ffi_api.cpp")],
        extra_cflags=flags, extra_include_paths=[str(p) for p in include],
        extra_ldflags=[f"-L{cuda}/lib64", f"-L{torch_dir}/lib", "-lcudart", "-lnvrtc",
                       "-lcublasLt", "-lcublas", "-ltorch", "-ltorch_cpu", "-lc10",
                       "-lc10_cuda", "-ltorch_cuda"], build_directory=str(build))
    shutil.copy2(lib, package / "_C.so")
    print(f"Development package: {package}", flush=True)


if __name__ == "__main__":
    main()
