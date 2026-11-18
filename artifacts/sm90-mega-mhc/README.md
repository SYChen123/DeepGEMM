# H200 Mega mHC 证据归档

2026-11-18，`workflow_62960518`，主机 `gpuxdn010192024126.idt2`，物理 GPU 1，NVIDIA H200，132 SM。第一轮使用空闲卡；后续机器出现新的 4-GPU SGLang 服务，候选性能复测停止。

| 文件 | 证据范围 |
|---|---|
| `benchmark-v1.json` | 首轮 V1 的真实权重精度、变输入 graph replay、4096/32 token 正式计时及 profile summary |
| `sm90_mega_mhc.v1.cuh` | 实测 kernel 原始快照；当前默认实现与它逐字节一致 |
| `sm90-megamhc-4096-cta0.json` | 首轮 4096-token 全网格 profile 的 CTA 0 片段，可用 Perfetto 打开 |
| `sm90-megamhc-32-cta0.json` | 首轮 32-token 全网格 profile 的 CTA 0 片段 |
| `unittest-pipeline-03.log` | 双 outstanding atom 候选的正确性回归；不是默认 V1 的完整回归日志 |

V1 kernel SHA256：`6982d0d8254161efe6477a752d1e74fcd903d63018fd87f14f10d9a056f314e3`。实现 commit：`3a7c6a1`。运行命令见 [commands.md](commands.md)。

SGLang revision：`7c5b581a8d521e3a2d80910d3cc72fbfbfbe982c`，与本地 `../sglang` 一致。Torch `2.14.1+cu130`，CUDA `13.0`。checkpoint 的 mHC/Norm 权重哈希、模型参数和输入种子都在 JSON metadata 中。

每种 shape：32 calls/CUDA Graph，12 轮计时，两 arm 轮换顺序，报告 median；profile 使用另一个带打点的 specialization，未混入 latency。decode baseline 使用 stats side stream 并在 boundary 末 join。结果只代表单卡 mHC boundary，不包含 MoE、模型 forward 或跨 GPU collective。

远端原始目录：

```text
/mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/experiments/sm90-megamhc-20261118/
```

完整 profile 是 `profiles-v1/sm90-megamhc-4096.json`（约 84 MB）和 `profiles-v1/sm90-megamhc-32.json`。两者 dropped_events=0。这里的 CTA 0 片段保留全网格 summary，绘图只展示选中的 warp，不代表所有 CTA 的总关键路径。

远端 `DeepGEMM/build-sm90-mhc` 在候选正确性测试时更新过，不能将该 package 的复测称为 V1。复现 V1 请从当前默认源码重新构建，或使用本归档的 V1 快照。停止时没有留下本任务 GPU benchmark 进程，原有 SGLang 服务保持运行。

构建缺少的 `libdw-dev/libelf-dev` 仅解包在远端 `build-deps/unpacked`；没有安装到系统。构建时为命令设置 `CPATH=$PWD/../build-deps/unpacked/usr/include`。
