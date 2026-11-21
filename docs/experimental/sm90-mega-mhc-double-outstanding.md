# 双 outstanding atom：候选优化归档

2026-11-21 更新：复测已完成。4096-token latency 从 V1 的 152.3598 降至 151.7565 µs，仅降低 0.40%；32 tokens 无稳定收益，不采用该候选。后续合批、轮询、SIMT、stage 分配与最终 split 调整见 [完整优化记录](sm90-mega-mhc-optimization-20261121.md)。

以下为 **2026-11-18 停测时的历史归档**。候选代码保存在 [补丁](sm90-mega-mhc-double-outstanding.patch)。当时仅通过正确性回归，不能将首轮 V1 的 1.350× / 1.035× 结果算到候选版上。

## 原理

V1 对每个 K8 atom 发出四条 route WGMMA，commit 一个 group，立即 `wait_group<0>`，然后归还该 A slot。Post、TMA 与 WGMMA 已在不同 warpgroup 中重叠，但 WGMMA worker 自身每个 atom 都 drain。

候选在提交当前 atom 后执行 `wait_group<1>`：确认前一个 group 完成，归还前一个 A slot，当前 group 可以保持 outstanding。下一 atom 的 barrier/descriptor 准备可与当前 group 的执行重叠。每个 Fn tile 末尾必须 `wait_group<0>`，再归还最后一个 A slot 和 Fn slot。

```mermaid
sequenceDiagram
  participant P as Post WG1/2
  participant A as Shared A ring (4 slots)
  participant W as WGMMA WG3
  P->>A: publish atom 0
  W->>A: acquire slot 0
  W->>W: issue + commit group 0
  P->>A: publish atom 1 / 2 / 3
  W->>A: acquire slot 1
  W->>W: issue + commit group 1
  W->>W: wait_group 1: group 0 complete
  W->>A: release slot 0 (group 1 still live)
  P->>A: reuse slot 0 for atom 4
  W->>W: continue; at Fn tile end wait_group 0
  W->>A: release last slot, then release Fn
```

四个 A slots 给两个 Post worker 留出继续生产的空间。不能提前释放当前 A；不能在最后一个 WGMMA 完成前释放 Fn；所有 128 个 WGMMA threads 必须保持一致的 commit/wait 和 arrival 序列。

补丁还改进 profile：区分 Post 的 IO/A 等待，并标记每个 Fn tile 的最终 drain。它不修改模型数值语义、split 分配、grid 或 CTA 数。

## 已完成证据

- 候选 kernel SHA256：`295124cef48ef521b93993e554550247401d7553afd9be57c8537d0d6c1d17b8`。
- H200 正确性日志：[unittest-pipeline-03.log](../../artifacts/sm90-mega-mhc/unittest-pipeline-03.log)。5 组测试通过，其中模式测试展开为 8 个 shape/layout 组合。
- 覆盖 normal/shifted、M=1/32/65/129、H=4096/5120/7168、BF16/FP8、两类 SF 布局、FP8-only、one-hot projection swizzle、同 capture stream 两 graph 跨 stream replay、deterministic 重复运行。
- 正式性能复测在 GPU 阶段前停止；当时机器新出现 4 个驻留 SGLang scheduler。没有候选性能数字。

## 恢复后的开发顺序

1. 在新开发分支执行 `git apply docs/experimental/sm90-mega-mhc-double-outstanding.patch`，重新构建独立 package。
2. 增加小 SM grid 的多波 persistent 回归（例如 num_sms=4，T=129/4096）。现有候选回归不能代替多波与大 shape 的性能验证。
3. 在空闲 H200 上以同一权重、输入种子、32 calls/graph、12 replay rounds 做 V1/候选交替顺序比较。分别测试 4096 和 32 tokens。
4. profile 单独采集；比较 WGMMA completion gate、Post 等 A slot、最后一波 Mix/Norm 的完成时刻。打点有明显扰动，不能将 instrumented span 当作无打点 latency。
5. 若改动稳定改善目标 shape，再保留；否则继续使用 V1。预期只针对 WGMMA worker 的 drain 暴露部分，尚无可报告的增量收益百分比。

## 其他待验证优化

- **decode grid / split：** 扫描 32/40/64/80/132 SM，观察减少空闲 CTA 和 Norm 分区是否有利。最终应在 operator 内选 launch 配置，避免模型侧临时修改进程全局 SM 设置。
- **wave-aware split：** 对 4096 tokens，64 m-blocks × 16 splits = 1024 tasks，132 CTA 共 8 波。比较 4/8/16 splits 的 partial/barrier 成本；理论任务长度相同也可能有不同实际开销。
- **barrier polling：** 参考 SGL #141 的 relaxed poll + 最后 acquire，以及 backoff；需要证明降低轮询压力的收益超过唤醒延迟。
- **Post 指令与任务队列：** unroll 2 已移植；共享 Post 指令路径和动态任务队列需结合 Hopper 的指令缓存/尾波证据评估。
- **X1 留片上：** SM107 的 319 KB oversized SMEM 方案不能直接用于 H200。重新设计小 tile 或寄存器保留时需排除 split 等待与 buffer 复用之间的循环依赖。
- **SGLang：** BF16 boundary 先接入；decode 保留 stats/MoE overlap 的端到端 A/B。CP、Engram、collective、batch-invariant 和大 shape 并发 replay 均另作集成验收。
