# H200 MegaMHC：2026-11-21 优化实验

最终保留 SM90 的 split 启发式调整。4096-token boundary 从 V1 的 **152.34 µs 降到 147.11 µs，延迟降低 3.43%**；同轮 SGLang baseline 为 205.12 µs，加速约 **1.394×**。8192 tokens 降低 3.55%。decode 的 launch 和计算路径不变，不把其计时波动算作优化收益。

双 outstanding、轮询、WGMMA 合批、padding 和 SIMT decode 原型均保留实验记录，不进入默认路径。完整数据、源码快照、失败记录及 profile 的远端归档与本地同步位置见 [归档索引](../../artifacts/sm90-mega-mhc-20261121/README.md)。

## 配对结果

设备为 `workflow_62960518` 的物理 GPU 1，NVIDIA H200、132 SM。csy 环境：Torch 2.14.1+cu130、CUDA 13.0；Nsight Compute 2025.3。使用 V4.1-Flash layer 0 FFN 的真实 mHC/Norm 权重、固定种子随机激活，H=5120、hc=4、Sinkhorn=20。SGLang HEAD 为 `7c5b581`，实际参与 baseline 的四份源码另存 SHA256 和快照。

最终结果来自独立进程 **V1 → heuristic4 → heuristic4 → V1**：每进程测试全部五种 shape，每 graph 64 次调用、16 轮 event 计时。表中为合并 event samples 的 median；每 arm、每 shape 有 32 个 samples 和两次独立进程运行。原始进程 median 一并保留。

| tokens/rank | V1 µs | 新启发式 µs | 相对 V1 延迟降低 | 同轮新版本的 SGLang baseline µs |
|---:|---:|---:|---:|---:|
| 1024 | 41.7753 | 41.2365 | 1.29% | 62.6892 |
| 2048 | 78.8692 | 78.2213 | 0.82% | 117.9750 |
| 4096 | 152.3390 | 147.1100 | **3.43%** | 205.1232 |
| 8192 | 301.7930 | 291.0707 | **3.55%** | 407.2315 |
| 32 | 16.8260 | 16.6730 | 0.91%（未改路径，视为波动） | 17.6990 |

计时包含 barrier clear 和 MegaMHC kernel，不含 CPU、模型 forward、MoE 或跨 GPU collective。decode baseline 使用 stats side stream 并在 boundary 末 join。不同 harness 的缓存、输入和预热方式不同，不能把下文热 fixture 的时间与本表直接相减。这里也没有整层 stats/MoE overlap 或端到端服务收益结论。

## 保留的实现

V1 在 split 上使用至少 16 的目标上限；Hopper 上改为 4，仍按 K64 的完整 block 分配长短 split，并选出达到该上限所允许最短最长任务的最少 split：

```text
deterministic: num_splits = 16
otherwise:
  min_cap = SM90 ? 4 : 16
  cap = min(64, max(min_cap, num_sms / num_m_blocks))
  num_splits = ceil(num_k_blocks / ceil(num_k_blocks / cap))
```

对 H200 的 4096 tokens，producer tasks 从 `64×16=1024` 减到 `64×4=256`，132 CTA 的调度波数从 8 降到 2；每 task 的 K64 blocks 从 5 增到 20。GEMM/stat partial workspace 从 6.5 MiB 降至 1.625 MiB，Mix/Norm 的跨 split 汇总缩短，系数读取与任务切换得到摊销。grid=132、CTA=768 threads、IO/Fn/A=3/2/4 均保持原设计。

这是参数与任务粒度优化：WGMMA、Post FMA、BF16 舍入边界、shared proxy fence、A/Fn 生命周期保护均沿用 V1。32 tokens 仍为 40 splits；SM100 仍使用原启发式；deterministic 仍为 16 splits。

先用安全超配 workspace 的隔离原型扫描 2/4/8 splits，再重建实际 host extension 验证最终 4-split 调度。2 splits 的约 150–151 µs 明显慢于 4/8 splits 的约 147 µs，说明只减少任务数会损失并行度，不能继续单向减小 split。

## 未采用的实验及原因

以下每项均有自己的配对 V1。热 fixture 实验复用先前真实权重测试保存的输入与期望输出，计时不打点。

| 实验 | 结果 | 机制、后续处理 |
|---|---|---|
| 双 outstanding atom | 正式 4096：152.3598→151.7565 µs，降低 0.40%；32：16.3890→16.5035 µs，无稳定收益 | 下一 atom 的准备可重叠，但当前 A slot 持有更久，增加 Post 回压。收益很小，不合入；进一步测试了下面的合批方案。 |
| 两 atom 合一个 commit，fence 移到累加器生命周期边界 | prefill 约 152.6 µs，基本持平；decode 无稳定收益 | 减少 commit/wait 与寄存器 fence，也延迟两个 A slots 的归还。通过完整回归及多波测试，不能仅凭指令更少判定关键路径改善。停止该方向。 |
| relaxed poll，ready 后 acquire | prefill 约慢 1% | 原型在已经 ready 时也多读一次，tight polling 还可能增加竞争。修正为 acquire 快路径，未 ready 才 relaxed poll，并加短 backoff。 |
| acquire 快路径 + relaxed/backoff | prefill 持平，decode 两次独立运行结论不一致 | 修正了已知额外读取和轮询压力，仍无稳定收益。source-PC 的 acquire/CCTL 长等待可能包含消费者等待生产者的时间，热点不等于可回收 latency。停止该方向。 |
| decode grid 132→40 | 同 context 15.7094→15.2851 µs，降低约 2.7% | 减少空闲 producer CTA，同时减少 Norm 并行分区；32/16 CTA 更慢。收益有限，不增加模型侧全局 SM 切换。 |
| 跳过完整 padding warp 的 Post 算术 | 回归通过，decode 未改善 | 仍需完整 M64 WGMMA、128-thread arrivals 和 padded A 清零；分支/清零有成本。未单独剥离每项成本，不能断言某条指令是唯一原因。不合入。 |
| 一个 CTA/token，完整 FP32 SIMT projection，寄存器保留 X1 | V1 15.6917 µs；256 threads 42.3641，512 threads 27.0289 µs | 仅 32 个 producer CTA，Fn 跨 token 的复用变差，投影转到 FP32 CUDA cores。去掉跨 CTA barrier 和 X1 回读不足以补偿。下面继续处理并行度。 |
| SIMT token 再分 4/8 份，独立 Mix+Norm kernel | 最好 4 splits×256 threads：16.8415 µs，对照 15.7063 µs，仍慢约 7.2% | 增加到 128 producer CTA，确实回收大部分损失；仍有 FP32 projection/权重流量和第二 kernel、中间 X1 访存。验证了并行度原因，但未获得总收益。停止 SIMT 路线。 |
| IO/Fn/A 从 3/2/4 重分配为 4/1/2 | prefill 约 179.5 µs，对照约 152.5 µs，慢约 18% | 第四 IO stage 牺牲了 Fn 下一 tile 的预取与 A 解耦。打点中 WGMMA 等 Fn 的 warp interval 总和增至 **5.35×**，见下文。恢复 3/2/4。 |

SIMT 原型只验证 shifted、H5120、BF16 的目标 fixture，不代表具备生产接口的全部 layout 支持。编译器报告两种 full-token 原型均无 spill；失败不能归咎于寄存器溢出。更多 SM 的修正版已实际测试，并非只停留在解释。

WGMMA 合批中的 fence 移动依据 [PTX wgmma.fence 语义](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence)：同 shape 连续 MMA 对 accumulator 的访问有序；仍在初始化后和最终读取前维护编译器/硬件所需边界。Post 的 shared async-proxy fence 没有删除。

## profile 提供的约束

V1 的单独 memory-counter capture：HBM read=277.8624 MB，write=251.7020 MB，DRAM throughput=71.01%，NCU duration=155.008 µs。这约 530 MB 的读写是主要约束之一；不能期望仅减少少量 WGMMA wait 就带来很大收益。NCU duration 仅用于诊断，不替代上面的 CUDA Graph latency。

最终 4-split 版本同配方 capture 的 read=255.4609 MB、write=249.4193 MB，合计 504.8801 MB，比 V1 少 4.66%；DRAM throughput=70.79%，NCU duration=148.256 µs。带宽利用率相近，减少任务/partial 汇总相关流量及改善局部性更符合本轮观察。单次 counter capture 仍受缓存影响，不能把这一流量变化拆成已独立证明的各项因果贡献。最终两种 shape 的打点分别记录 1,441,984 / 29,056 个事件，均无丢弃。

4/1/2 与 V1 的 `%globaltimer` 对照使用相同事件结构、相同 4096-token fixture：WGMMA 等 Fn 都有 20,480 个 warp intervals、等 A 都有 163,840 个，无 dropped events。Fn interval 总和从 4,572.864 增至 24,488.480 µs；A 从 25,745.856 变为 24,330.592 µs。增加 IO 深度没有补偿 Fn 单 stage 丢失的重叠。**这些跨 warp 的和不是 kernel latency，也不能按差值直接解释 27 µs 的正常计时回退。**

H200 每 CTA opt-in shared 上限为 232448 B，V1 dynamic shared 为 220160 B。直接增加一个 IO stage 还需 40960 B，容量不足；保留全部 Fn/A 深度时需要重新设计 tile/存储生命周期，无法靠简单调 stage 数解决。后续若研究 X1 留片上或 decode 小 M tile，应保持 tensor-core projection 与 Fn 跨 token 复用；本轮 SIMT 原型表明只消除 global barrier 不够。

## 验证与无效数据

最终版本通过六组回归，包含 normal/shifted、tail、BF16/FP8 与 SF、one-hot projection、同 capture stream 的跨 stream graph replay、deterministic；新增 1024 与 4096 tokens 两个 graph、变输入的十轮并发 replay。4-SM 的 T129/T4096 normal/shifted 多波回归通过。五种真实权重 shape 均对照 SGLang 和独立 FP32 参考，并验证变输入 graph replay。

两项 harness 问题已修正且原始失败记录保留：

- SIMT 首轮固定了 capture 之前的 stream，得到 empty graph。这组异常小时间明确作废；修正为每次读取当前 capture stream，并把 empty-graph warning 设为错误，表中只使用修正后的数据。
- 扩展至 1024/2048 tokens 时，SGLang 会 fallback 到独立 RMSNorm，原来的 `SimpleNamespace` 不可调用。脚本改用真实 `RMSNorm` 并加载同一权重，整个最终 ABBA 重跑。独立复制测试目录时缺失 third-party 路径的 import 失败也保留，补齐 PYTHONPATH 后六组回归通过。

H200 上没有 SM100 runtime 验证。SGLang 接口不变，已有接入补丁可继续使用；整模型服务、CP、Engram 和 MoE 共跑仍需各自的集成验收。

## 复现

从本分支构建独立 package，不覆盖 csy 环境已有安装：

```bash
CUDA_VISIBLE_DEVICES=1 /path/to/csy/bin/python scripts/build_sm90_mhc_dev.py --output build-sm90-mhc
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/build-sm90-mhc /path/to/csy/bin/python tests/test_sm90_mega_mhc.py
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/build-sm90-mhc /path/to/csy/bin/python scripts/bench_sm90_megamhc.py \
  --model /path/to/DeepSeek-V4.1-Flash --tokens 1024 2048 4096 8192 32 \
  --repetitions 64 --rounds 16 --output results.json
```

在独立进程中采集打点，不将它混入正式 latency：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/build-sm90-mhc /path/to/csy/bin/python scripts/bench_sm90_megamhc.py \
  --model /path/to/DeepSeek-V4.1-Flash --tokens 4096 32 \
  --kernel-profile-dir profiles --output results-profile.json
```

远端隔离实验根目录为 `/mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/experiments/sm90-megamhc-double-20261121/optimization`。`run_heuristic.py` 保存最终 ABBA 的完整 argv 与 extension SHA；`build_heuristic.py` 保存隔离 extension 构建；`optimization-summary.json` 记录各原型、计时 samples 和依赖哈希。
