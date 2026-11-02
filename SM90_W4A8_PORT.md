# SM90 W4A8 MegaMoE port onto v0.2.2

## Pinned inputs

- Base: `v0.2.2`, commit `ec226bcf400c11d62755137b916fae9ee359c684`.
- Source: [DeepGEMM PR #53](https://github.com/sgl-project/DeepGEMM/pull/53),
  commit `b5675bf7c812a583d6eadabfa23ad1e00207c39d`.
- PR merge base used for the patch: `1ff3932f0f9d9187b92afb476da4e3ade10a6575`.
- Target branch: `0.2.2-sm90-megamoe`.
- Distribution: `sgl-deep-gemm` (TVM-FFI), built with
  `build_sgl_deep_gemm.sh`; this port does not extend the separate root pybind API.

## Design and conflict assessment (before applying the patch)

The PR's 12-file net patch applies textually to v0.2.2, but a clean patch is not
sufficient to establish feature compatibility. Apply that patch, then adapt the
new Python dispatcher to v0.2.2's public API. Keep the fused CUDA implementation
and its JIT launch implementation unchanged from the pinned PR.

| Area | Conflict or dependency | Resolution |
| --- | --- | --- |
| SM90 block32 FP8 GEMM | v0.2.2 includes native block32 scale handling and GEMM heuristics absent from the PR's base. | Preserve the GEMM API, layouts, kernels and regression tests from v0.2.2. W4A8 MegaMoE uses a separate implementation. |
| SM100 TRT-LLM layouts | v0.2.2 added the public `weight_layout` argument and native `use_trtllm_weights` argument. The PR's new top-level wrapper predates these. Copying it without adaptation would reject existing calls or lose the layout flag. | Preserve positional `weight_layout` and forward it unchanged to the existing SM100 implementation. Reject TRT-LLM layout explicitly on SM90, where the new kernel requires its own gran-8/K-major layout. Preserve `transform_scales_for_mega_moe` and the native SM100/BF16 FFI argument lists. |
| W4A8 versus W4A4/NVFP4 | SM100 typed MMA paths support different FP4 activation and weight-scale layouts. SM90 in this PR supports FP8 activations, E2M1 weights and K32 UE8M0 scales. | Dispatch SM90 through a separate native symbol, keep the SM100 implementation, and reject SM100-only outer scales/shared experts/layouts on the SM90 path. |
| Weight preparation | The old SM90 helper interleaves FP8 weights only. SM100 uses MN-major packed scales, incompatible with the new SM90 kernel. | Add `transform_weights_for_mega_moe_sm90_fp4`: keep packed FP4, interleave gate/up weights and scales together, pack four UE8M0 bytes per int32 along K. Do not expand whole expert tensors to FP8. |
| Symmetric workspace / precision | The new path uses FP32 activation scales, g128 for L1 and g64 for L2. Its TMA/barrier layout and buffer slicing must agree. | Import the PR's workspace constants, slicing, FP4 JIT/TMA descriptors and fused kernel together. Preserve the existing SM90 FP8 workspace contract and import its upstream regression changes too. |
| Kernel scheduling | FP4 decode needs packed-load/unpack staging, scale application and small-batch swapAB; prefill uses a different pipeline and warp partition. | Import the PR's complete heuristics and dispatch defaults instead of transplanting only the GEMM body. Retain its accompanying SM90 FP8 heuristic change and BLOCK_M safety assertion. |
| CP / graphs | CP can produce strided local inputs and ranks with no tokens; graph replay uses padded physical rows with invalid route ids. These are SGLang integration contracts, not new attention metadata in DeepGEMM. | Preserve native pre-dispatch and distributed completion protocols. SGLang makes inputs contiguous, keeps invalid ids, participates on empty ranks, caches the workspace, and checks graph capacity. No new allocation or host synchronization in the native forward is introduced by this adaptation. |
| Dependencies / packaging | A partial port can miss JIT headers, TVM exports or required CUTLASS/DeepJIT revisions. | Copy all four new native/test source files plus shared changes; retain the build script's existing header packaging. Base and PR submodule revisions are identical, so no dependency or submodule upgrade is required. |

## Scope and restrictions

The new top-level `fp8_fp4_mega_moe` retains v0.2.2 arguments and adds keyword-only
SiTU parameters supported by the PR. On SM90 it invokes
`_C.fp8_fp4_mega_moe_sm90`; other architectures delegate to the existing
`mega.fp8_fp4_mega_moe`. Native and typed-MMA validation remains in place.
Device selection uses the output tensor's CUDA device. The SGLang integration
was committed separately as `e4b4427ee0` in `v4.1/h20-dev-pd`.

The library kernel supports SwiGLU and SiTU. The accompanying SGLang adapter
currently selects bias-free SwiGLU for DSV4.1. Hidden and intermediate dimensions
must be multiples of 128; intermediate size must not exceed 4096. DSV4.1-Flash
H=5120, I=2304 meets these shape constraints. This is not a measured memory or
performance guarantee.

Keep `DG_USE_FP4_ACTS=0` and `DG_USE_FP8_COMBINE=0` (or leave them unset) for this
SM90 path. SM100 shared experts, NVFP4 outer scales and TRT-LLM weight layout are
not supported by the new SM90 entry point. SM100 functionality is preserved.

The PR is still open. The pinned revision includes its later scale/FP32
intermediate correctness fixes; importing only its initial kernel revision is
not equivalent to this port.

## Validation plan

This task is explicitly limited to static validation. Do not install packages,
build wheels, run CUDA kernels or claim GPU accuracy/performance acceptance.

Static checks:

1. Python AST parsing for changed Python files and shell syntax for the test runner.
2. `git diff --check` and merge-marker inspection.
3. Exact comparison of the new native implementations with the pinned PR.
4. Compare existing SM90 block32 GEMM and SM100 TRT-LLM implementation files
   against v0.2.2; inspect the additive TVM FFI changes and the adapted dispatcher.
5. Confirm the public wrapper preserves all v0.2.2 parameters and positional
   ordering, forwards `weight_layout`, and restricts SM90-only options.
6. Confirm CUTLASS/DeepJIT gitlinks remain at the v0.2.2 revisions.

Static results on 2026-11-02: all six checks passed. Ten of the twelve PR files
match the pinned PR byte-for-byte; only the Python wrapper and additive FFI file
differ because they retain v0.2.2 changes. Fifteen files changed between the PR's
base and v0.2.2, excluding those two shared API files, remain byte-for-byte equal
to v0.2.2. Existing FP8/FP4, BF16 and SM90 FP8 native FFI function bodies and
signatures are unchanged. The full old public positional parameter list and
defaults are retained, and `weight_layout` is forwarded to the SM100 path.

An offline-source follow-up replaces the CUTLASS/DeepJIT gitlinks with ordinary
tracked files from those exact commits, preserving both upstream Git trees.
See `third-party/vendor_manifest.json` and `third-party/README.md`. The earlier
gitlink-preservation check describes the kernel-port commit, before this
separately requested source-vendoring change.

After a separate GPU-validation task, build the TVM distribution on Linux CUDA
and run at least `test_mega_moe_fp4_hopper.py`, `test_mega_moe_hopper.py`, existing
block32 FP8 GEMM tests and the SM100 TRT-LLM regression tests on their respective
architectures. Then validate full-model CP/BCG/bounded replay in SGLang; upstream
unit scenarios alone do not establish end-to-end correctness.
