# 2026-11-22：SM90 MegaMHC decode 证据

最终保留四 warp 协作 Mix，以及配套 80-split producer 调度。32-token 正式计时：原始 MegaMHC V1 **16.3878→11.7900 µs（−28.06%）**，同轮 SGLang 17.5375 µs。设计和负收益原因见 [实验报告](../../docs/experimental/sm90-mega-mhc-decode-20261122.md)。

实现 commit `a621931`；最终测试源码按 hash 对齐此实现。

- [正式累计阶段计时](staged-comparison.json)：ABCDDCBA，48 samples/stage/shape；各进程原始结果在 `timing/`。
- [阶段源码与 extension hash](stages/stages.json)：baseline → prefill_opt1 → decode_opt1 → decode_opt2。
- [打点阶段汇总](profile-summary.json)：8 个正式 capture 和 3 个诊断 capture，均 dropped=0。时间属于独立 instrumented launch。
- [NCU 资源与 local sectors](ncu-resources-summary.json)：8 份 report，132×768、dynamic shared=220160 B，local read/write sectors=0。
- [原始 profile 文件清单](profile-files.json)：完整 JSON、NCU report、CSV 与元数据的 SHA256/字节数；大文件保存在本地 profile 目录，不放入 Git。
- `trials/`：Mix/split 的平衡对照、精确 cycle、稀疏 cycle、Norm16 等有效及失败原型的原始结果。
- `logs/`：最终九组回归、4-SM 多波回归、SM100 两个实例的编译检查及采集控制日志。NCU 不支持 LocalMemoryUsage 的原始 launch 前失败也保留。

`manifest.json` 对应**完整原始归档**，不是本目录这个精简集合；[local-verification.json](local-verification.json) 记录本地解压后 310 份文件及 9 份最终源码的核对结果。

完整归档远端路径：

```text
/mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/experiments/sm90-megamhc-decode-20261122/
  decode-results.tar.xz
  export/
```

归档 14,800,040 B，SHA256：`fa99cd5937dd6ba6272fdadfd4678593928b0f53b1e220ba84dfbb7888003ae8`。

本地路径：

```text
~/LLMInfer/profile/ncu/sm90-megamhc-decode-20261122/
~/LLMInfer/profile/ncu/sm90-megamhc-stages/index.html
```

累计阶段目录通过相对 symlink 指向原始文件，同时包含 4096-token prefill 与 32-token decode。交互页面支持选择任意两个阶段、共同时间轴、CTA0 或各角色最晚完成的代表 warp；不能将来自不同 CTA 的代表行拼成已证明的因果链。旧的三份原始目录继续保留。

最终 `production_v3/_C.so` SHA256：`4cb61fd1659eb09048dc0192e9acf43b9a023f67adcf975419a2bdad247be866`。源码归档 SHA256：`64c70dd68872bced32dbd295cc522f201e0a4d49b9389b332a40eaa1dc1e5c76`。实际参与测试的文件与本地 kernel/API/bench/test 一致，见 `source-final-v3/source-manifest.json`。

基于真实 checkpoint 权重和固定激活；不包含模型权重、fixture tensor、完整 Python package、JIT cache 或 `_C.so` 二进制。没有整模型、CP/MoE overlap 或 SM100 runtime 性能结论。
