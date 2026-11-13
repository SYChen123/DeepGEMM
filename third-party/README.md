# Vendored build dependencies

The `codex/0.2.2-sm90-megamoe-pr473` branch includes the full CUTLASS and
DeepJIT snapshots as tracked source files. Cloning the branch from AntCode
includes both dependencies without a GitHub fetch.

```bash
git clone --branch codex/0.2.2-sm90-megamoe-pr473 https://code.alipay.com/csy512356/DeepGEMM.git
cd DeepGEMM
```

| Directory | Upstream snapshot | Files |
| --- | --- | ---: |
| `cutlass/` | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` | 6617 |
| `deep_jit/` | `e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16` | 102 |

The full tracked snapshots occupy 140,619,988 uncompressed bytes (about 134 MiB).
Provenance, Git tree IDs and the upstream license-file inventory are recorded in
[vendor_manifest.json](vendor_manifest.json). All tracked source files, file modes,
symlinks and existing copyright/license notices are preserved. DeepJIT's pinned
snapshot has no separate LICENSE/COPYING/NOTICE file; no replacement license is
invented or added to its snapshot. Nested `.git` metadata is not committed.

`git status` reports these directories only when their ordinary files change.
`git ls-files third-party/cutlass` now lists individual source files rather than
a single `160000` gitlink entry. No `git submodule update` is required.

For SGLang, build with `bash build_sgl_deep_gemm.sh` on the Linux CUDA machine.
The script checks the local dependency headers and fails clearly if sources
are missing; it no longer fetches GitHub submodules. This change supplies GitHub
source dependencies only: Python packages, CUDA and compiler dependencies must
already be installed or be available through the machine's internal mirrors.
The separate root pybind build may try a GitHub prebuilt wheel unless
`DG_FORCE_BUILD=1`; it is not the TVM-FFI build used by this integration.
