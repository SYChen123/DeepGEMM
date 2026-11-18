# 运行记录与复现入口

本记录不启动任何后续任务。V1 已测，双 outstanding 性能测试按要求停止。

实际实验目录：

```bash
cd /mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/xiaoyuan/experiments/sm90-megamhc-20261118/DeepGEMM
```

独立 package 构建（需先使用当前已测 V1 源码；远端遗留 package 是候选版）：

```bash
CPATH=$PWD/../build-deps/unpacked/usr/include CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda \
  /mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/abing/pack/conda/env/envs/csy/bin/python \
  scripts/build_sm90_mhc_dev.py --output build-sm90-mhc
```

首轮正式计时及单独 profile 的命令：

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda PYTHONPATH=$PWD/build-sm90-mhc \
  DG_JIT_CACHE_DIR=$PWD/../jit-cache \
  /mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/abing/pack/conda/env/envs/csy/bin/python -u \
  scripts/bench_sm90_megamhc.py \
  --model /mnt/a6intlalipayplus-a6/common/models/DeepSeek-V4.1-Flash \
  --tokens 4096 32 --repetitions 32 --rounds 12 \
  --kernel-profile-dir ../profiles-v1 --output ../benchmark-v1.json
```

脚本在无打点 CUDA Graph 计时结束后才编译、运行带 `%globaltimer` 的 specialization。若只需要 latency，省略 `--kernel-profile-dir`；若只导出一个 CTA，增加 `--kernel-profile-cta 0`。独立 profile 的时间会受打点影响。

候选正确性回归的实际命令：

```bash
CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/usr/local/cuda PYTHONPATH=$PWD/build-sm90-mhc \
  /mnt/a6intlalipayplus-idt2-aidc/common/alipayplus/infer/abing/pack/conda/env/envs/csy/bin/python -u \
  tests/test_sm90_mega_mhc.py
```

该脚本使用标准库 unittest，无需 pytest。候选补丁的恢复、额外多波验证和性能计划见 `docs/experimental/sm90-mega-mhc-double-outstanding.md`。

本地重绘已归档图表：

```bash
MPLCONFIGDIR=/tmp/csy-sm90-mhc-mpl XDG_CACHE_HOME=/tmp/csy-sm90-mhc-cache \
  python scripts/plot_sm90_megamhc.py artifacts/sm90-mega-mhc/benchmark-v1.json \
  --profiles artifacts/sm90-mega-mhc/sm90-megamhc-4096-cta0.json \
             artifacts/sm90-mega-mhc/sm90-megamhc-32-cta0.json
```
