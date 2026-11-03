# H20 SM90 W4A8 MegaMoE small-shape path

The native SM90 W4A8 kernel supports the register-source WGMMA pipeline for
the measured H20-3e configuration. It uses a 16-row activation tile, two
resident CTAs per SM, an independent SFB producer, and a CTA shared-memory LUT
for FP4 decode. The host validates occupancy before launching the persistent
grid. The scheduler uses a 12-expert wave at 32 rows per rank and retains
its usual wave heuristic at 8 rows per rank.

The register path is explicitly selected by
`deep_gemm.pack_sm90_fp4_register_weights`. First call
`transform_weights_for_mega_moe_sm90_fp4` to interleave gate/up and prepare
UE8M0 scales. Then apply the register packer to both FP4 weight tensors.
The packer rearranges fragments, pairs adjacent output rows for the TMA
swizzle, and permutes sign bits. Its trailing singleton dimension is
the native layout tag. Keep the original UE8M0 scale tensors.

The tagged layout currently supports SwiGLU with H=5120, I=2304,
384 experts, top-k=6, EP4, and 8 or 32 rows per rank. The ordinary rank-3
layout still selects the existing kernel path and supports other shapes.

Build the TVM-FFI wheel with `bash build_sgl_deep_gemm.sh`. Select the
wheel's unpacked directory with `PYTHONPATH` for a standalone experiment.
The SGLang sibling repository provides
`scripts/h20_megamoe_small_opt_20261102/bench_small.py --native-register-pack`
for the four-rank comparison. The standalone benchmark prepares weights
before timing. SGLang serving continues to use the ordinary layout.
