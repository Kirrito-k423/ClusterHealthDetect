# A3 Clean CPU Core/NPU 绑核矩阵复测报告

生成日期：2026-07-07

## 结论摘要

- 本次在 `A3-AK-182` 上清掉自动占卡任务后，重新跑了 H2D 全量矩阵和固定两卡 D2D/allgather 全量矩阵。
- H2D clean run 覆盖 `640 CPU core x 16 NPU x 16/64/256MB`，共 30720 个点，全部 `ok`。
- D2D clean run 覆盖固定 NPU pair `0,5` 下的 `640 rank0 core x 640 rank1 core x 8MB`，共 409600 个 core-pair，全部 `ok`。
- 清理后 H2D 64MB median 为 `58.376 GB/s`，256MB median 为 `59.129 GB/s`；NPU `0` 仍是结构性低列，64MB 各 CPU 段 median 约 `11.205-13.946 GB/s`，同段其他 NPU 最高约 `58.841-59.012 GB/s`。
- 清理后固定两卡 D2D/allgather median 为 `13.714 GB/s`，max 为 `15.967 GB/s`，极端慢点为 `0.196 GB/s`。旧共享环境 run 的 median 是 `13.523 GB/s`，说明外部任务会带来噪声，但 core-pair 结构仍然存在。
- H2D 有效结果来自 `results/core-h2d-matrix-clean-20260707-1552`；D2D 有效结果来自 `results/core-pair-d2d-matrix-clean-20260707-1700`。中间几次 D2D 尝试因为外部任务重新占卡或被手动终止，已废弃，不参与统计。

## 清理的外部占用

复测前，`npu-smi info` 观察到 A3-AK-182 上存在自动占卡任务，工作目录包括：

- `/home/m00659926/VeOmni`
- `/home/l00889958/code/MindSpeed-MM`
- `/home/l00672371/mm/MindSpeed-MM`
- `/root/g00510989/xllm_th`

清理动作包括：

- 停止 `train_test_1` 容器。
- 终止 `/root/g00510989/xllm_th` 下反复拉起的 xllm eval/codex resume 进程树。
- 终止 MindSpeed-MM `longcat_config` / `finetune_longcat.sh` 训练进程树。

清理后，H2D clean run 开始前 `npu-smi` 显示 NPU `0-7` 均无 running process；D2D clean run 开始前和结束后也均显示无 running process。日志见：

- [A3-clean-h2d-retest-20260707-1552.log](A3-clean-h2d-retest-20260707-1552.log)
- [A3-clean-d2d-retest-20260707-1700.log](A3-clean-d2d-retest-20260707-1700.log)

注意：`A3-clean-h2d-retest-20260707-1552.log` 中 H2D 已在 `2026-07-07T16:15:13+08:00` 完成；其后同一脚本继续尝试 D2D 时被终止。正式 D2D 使用独立 clean run `1700`。

## 产物

- [A3-core-h2d-matrix-clean-20260707.xlsx](A3-core-h2d-matrix-clean-20260707.xlsx)
- [A3-core-h2d-matrix-clean-20260707-16MB.png](A3-core-h2d-matrix-clean-20260707-16MB.png)
- [A3-core-h2d-matrix-clean-20260707-64MB.png](A3-core-h2d-matrix-clean-20260707-64MB.png)
- [A3-core-h2d-matrix-clean-20260707-256MB.png](A3-core-h2d-matrix-clean-20260707-256MB.png)
- [A3-core-pair-d2d-matrix-clean-20260707-8MB.xlsx](A3-core-pair-d2d-matrix-clean-20260707-8MB.xlsx)
- [A3-core-pair-d2d-matrix-clean-20260707-8MB.png](A3-core-pair-d2d-matrix-clean-20260707-8MB.png)

## H2D Clean Run

命令：

```bash
OUT_DIR=results/core-h2d-matrix-clean-20260707-1552 \
CPUS=auto \
DEVICES=auto \
SIZES_MB=16,64,256 \
ITERS=3 \
WARMUP=1 \
REPEATS=1 \
CHECKPOINT_EVERY_CPUS=16 \
DEVICE_KIND=auto \
bash scripts/run_core_h2d_matrix.sh
```

范围：

- CPU：640 个可绑定 core。
- NPU：16 张可见 NPU，device id `0-15`。
- size：16MB、64MB、256MB。
- 指标：host-to-device copy GB/s。

总体统计：

| size | min GB/s | p10 GB/s | median GB/s | p90 GB/s | max GB/s | max/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16MB | 1.297 | 6.531 | 56.132 | 57.240 | 58.503 | 45.104 |
| 64MB | 4.739 | 37.252 | 58.376 | 58.848 | 59.356 | 12.526 |
| 256MB | 15.791 | 58.404 | 59.129 | 59.621 | 59.926 | 3.795 |

64MB 按 CPU core 段观察，NPU `0` 在所有段上都明显低于同段最高 NPU：

| CPU core 段 | NPU0 median GB/s | 同段最高 NPU median GB/s |
| --- | ---: | ---: |
| 0-79 | 12.205 | 59.012 |
| 80-159 | 12.322 | 58.945 |
| 160-239 | 12.764 | 58.873 |
| 240-319 | 11.388 | 58.866 |
| 320-399 | 12.634 | 58.842 |
| 400-479 | 11.205 | 58.899 |
| 480-559 | 13.612 | 58.841 |
| 560-639 | 13.946 | 58.861 |

极值点：

| metric | CPU core | NPU | size | GB/s |
| --- | ---: | ---: | ---: | ---: |
| min | 164 | 0 | 16MB | 1.297 |
| max | 542 | 14 | 256MB | 59.926 |

## 固定两卡 D2D/allgather Clean Run

命令：

```bash
OUT_DIR=results/core-pair-d2d-matrix-clean-20260707-1700 \
DEVICE_PAIR=0,5 \
RANK0_CPUS=auto \
RANK1_CPUS=auto \
SIZES_MB=8 \
ITERS=1 \
WARMUP=0 \
REPEATS=1 \
CHECKPOINT_EVERY_PAIRS=32768 \
RECORD_RANKS=rank0 \
DEVICE_KIND=auto \
bash scripts/run_core_pair_d2d_matrix.sh
```

范围：

- 固定 NPU pair：`0,5`。
- rank0 CPU：640 个可绑定 core。
- rank1 CPU：640 个可绑定 core。
- core-pair：409600 个。
- size：8MB。
- 指标：`torch.distributed.all_gather` algorithm GB/s。

总体统计：

| metric | value |
| --- | ---: |
| points | 409600 |
| min GB/s | 0.196 |
| p10 GB/s | 13.000 |
| median GB/s | 13.714 |
| p90 GB/s | 14.409 |
| max GB/s | 15.967 |
| max/min | 81.387 |

按单侧 core 聚合的慢点和快点：

| side | slow cores by median | fast cores by median |
| --- | --- | --- |
| rank0 | 108, 110, 106, 107, 104, 105, 46, 103, 49, 381 | 430, 494, 444, 365, 434, 463, 498, 483, 495, 389 |
| rank1 | 80, 0, 240, 400, 320, 160, 480, 228, 120, 560 | 486, 543, 487, 483, 519, 542, 549, 485, 518, 509 |

极值点：

| metric | rank0 core | rank1 core | NPU pair | size | alg GB/s |
| --- | ---: | ---: | --- | ---: | ---: |
| min | 93 | 504 | 0,5 | 8MB | 0.196 |
| max | 427 | 485 | 0,5 | 8MB | 15.967 |

## 与旧共享环境 run 的差异

旧报告 [A3-full-core-matrix-report.md](A3-full-core-matrix-report.md) 的结果是在共享环境下得到的，当时记录到外部 `xllm` 进程占用部分 NPU。它仍然有诊断价值，但不应作为清洁基线。

| 实验 | 旧共享环境 median | clean median | 主要变化 |
| --- | ---: | ---: | --- |
| H2D 64MB | 56.672 GB/s | 58.376 GB/s | 整体中位数上升；NPU0 低列仍存在 |
| H2D 256MB | 58.380 GB/s | 59.129 GB/s | 整体中位数上升，长包更接近上限 |
| D2D/allgather 8MB | 13.523 GB/s | 13.714 GB/s | 中位数小幅上升，core-pair 块状结构仍存在 |

## 解读注意

- `cpu:1` profile 不是绑核，只是每个 rank 旁边多起 1 个 CPU burner。绑核结论应看显式 `sched_setaffinity` 的 core 矩阵。
- Clean run 说明旧结果里确实混入了外部任务噪声，但没有推翻逐 core 绑核矩阵的必要性：H2D 的 NPU0 低列和 D2D 的 core-pair 差异仍然存在。
- NUMA 标签仍只作为解释线索。本实验从每个 CPU core 的实测值出发，不预设 NUMA 配置可信。
