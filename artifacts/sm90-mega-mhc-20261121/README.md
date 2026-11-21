# H200 MegaMHC：2026-11-21 证据索引

最终保留 SM90 host 的 split 启发式调整；device kernel 仍为 V1。4096 tokens：152.3390→147.1100 µs，降低 3.43%；8192：301.7930→291.0707 µs，降低 3.55%。decode 的 split 与计算路径未改，不将计时波动算为优化收益。

| 文件 | 内容 |
|---|---|
| `heuristic4-comparison.json` | 最终真实 SGLang baseline，五种 shape 的 ABBA median 与各进程 median |
| `optimization-summary.json` | 每个原型各自的配对 V1、热 fixture 原始 samples、最终正式比较、依赖和 extension 哈希 |
| `memory-profile-summary.json` | V1 / 最终版的单独 HBM counter capture，保留单位 |
| `final-regression-heuristic4.log` | 六组最终回归，含 1024/4096-token、变输入、跨 stream 并发 graph replay |
| `heuristic4-controller.log` | 正式回归、多波和 ABBA 的完成状态 |
| `double-comparison.json` | 双 outstanding 的正式复测，ABBA×2，每 arm 四个进程 |
| `double-profile-summary.json` | 双 outstanding 的带打点等待分类；interval 和不等于 latency |
| `source-hashes.json` | 实际 host/device/benchmark/test 和 SGLang baseline 的源码哈希 |

最终 API SHA256：`c629d46ca11aed3a1ba81d93c2eace2ef4fadbcf0bffa7051a12b76cae53d0e5`；测试 extension SHA256：`df048dbf0f6044f426d2c3782ca18e50fd7af3367f41d4ae4c5cca86895994a5`。V1 device kernel SHA256：`6982d0d8254161efe6477a752d1e74fcd903d63018fd87f14f10d9a056f314e3`。源码与本地待提交文件逐字节一致。

真实权重、H5120/hc4/Sinkhorn20、csy Torch 2.14.1+cu130/CUDA13.0、H200 物理 GPU1。SGLang HEAD 为 7c5b581，实际参与 baseline 的四份源码哈希另列，不仅依赖 HEAD。最终五 shape 正式计时为 V1/新/新/V1 四进程顺序，每 graph 64 次调用、每 shape 16 轮；正式时间不启用 kernel 打点。

两批用户指定的历史 profile 已完整同步并校验：

```text
~/LLMInfer/profile/ncu/sm90-megamhc-double-20261121/
~/LLMInfer/profile/ncu/sm90-megamhc-20261118-profiles-v1/
```

两目录分别校验 79 / 5 个 manifest 文件；前者包含两版×两 shape 的四份正式 NCU report 和完整 `%globaltimer` traces，后者包含原 `profiles-v1` 的四份原始文件。两者均有 `local-verification.json`。

新增优化实验的完整远端归档：

```text
/mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/experiments/sm90-megamhc-double-20261121/optimization/optimization-results.tar.gz
```

压缩包 13,814,953 bytes，SHA256 `c59c745c8842deede53f9b82923baabf8750992403176bea0fba6a4f9b1ef69d`，manifest 包含 198 个文件。保留全部隔离原型源码、原始计时、失败日志、真实 SGLang 源码快照及最终/IO4 profile，不含模型权重和大 fixture。

本地同步目标为 `~/LLMInfer/profile/ncu/sm90-megamhc-optimization-20261121/`。其完整性以该目录的 `local-verification.json` 为准；网络中断留下的 `.tar.gz` 部分文件不能当作完整归档。仓库中上述精简证据已从完整 tar members 提取并按远端 manifest 校验。

[优化机制、负收益原因、修正过程与复现命令](../../docs/experimental/sm90-mega-mhc-optimization-20261121.md)。这些是单 GPU mHC boundary 结果，不代表 MoE 共跑或服务端到端收益。
