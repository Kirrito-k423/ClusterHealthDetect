# A3 全量 CPU Core/NPU 绑核矩阵报告

生成日期：2026-07-06

## 结论摘要

- H2D 已完成全量 `640 CPU core x 16 NPU x 3 sizes`，共 30720 个点，全部 `ok`。
- 固定两卡 D2D/allgather 已完成全量 `640 rank0 core x 640 rank1 core`，共 409600 个 core-pair，全部 `ok`。
- H2D 64MB 热力图显示 NPU `0` 在 CPU core `320-639` 区间出现连续低带宽列，median 可降到约 `5.8-12.4 GB/s`，而其他 NPU 多数维持在 `55-58 GB/s`。
- D2D 8MB 热力图显示明显块状结构，说明两个 rank 绑到不同 CPU core 的组合会影响固定两卡 allgather；全局 median 为 `13.523 GB/s`，极端慢点低至 `0.175 GB/s`。
- 本轮 A3 上存在外部 `xllm` 进程占用部分 NPU，结果代表“当前共享环境下的实测”，不是空载裸机基线。

## 产物

- [A3-core-h2d-matrix-full.xlsx](A3-core-h2d-matrix-full.xlsx)
- [A3-core-h2d-matrix-full-16MB.png](A3-core-h2d-matrix-full-16MB.png)
- [A3-core-h2d-matrix-full-64MB.png](A3-core-h2d-matrix-full-64MB.png)
- [A3-core-h2d-matrix-full-256MB.png](A3-core-h2d-matrix-full-256MB.png)
- [A3-core-pair-d2d-matrix-full-8MB.xlsx](A3-core-pair-d2d-matrix-full-8MB.xlsx)
- [A3-core-pair-d2d-matrix-full-8MB.png](A3-core-pair-d2d-matrix-full-8MB.png)

## H2D 全量实验

命令：

```bash
OUT_DIR=results/core-h2d-matrix-full-20260706 \
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

| size | min GB/s | median GB/s | max GB/s | max/min |
| --- | ---: | ---: | ---: | ---: |
| 16MB | 0.545 | 52.549 | 56.490 | 103.736 |
| 64MB | 1.582 | 56.672 | 59.340 | 37.501 |
| 256MB | 7.748 | 58.380 | 59.952 | 7.738 |

64MB 按 NUMA 段观察，NPU `0` 在后半 CPU core 段出现结构性下降：

| CPU core 段 | NPU0 median GB/s | 同段最高 NPU median GB/s |
| --- | ---: | ---: |
| 0-79 | 55.674 | 57.975 |
| 80-159 | 55.505 | 57.684 |
| 160-239 | 55.082 | 57.888 |
| 240-319 | 55.282 | 57.801 |
| 320-399 | 12.362 | 57.555 |
| 400-479 | 5.819 | 57.491 |
| 480-559 | 6.189 | 57.652 |
| 560-639 | 5.875 | 57.524 |

## 固定两卡 D2D/allgather 全量实验

命令：

```bash
OUT_DIR=results/core-pair-d2d-matrix-full-20260706 \
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
| min GB/s | 0.175 |
| median GB/s | 13.523 |
| max GB/s | 15.864 |
| max/min | 90.646 |
| elapsed seconds | 900 |

按单侧 core 聚合的慢点：

| side | slow cores by median |
| --- | --- |
| rank0 | 65, 316, 95, 93, 81, 14, 72, 94, 24, 92 |
| rank1 | 80, 320, 436, 218, 400, 560, 0, 240, 238, 81 |

## 解释注意

- `cpu:1` profile 不是绑核，只是每个 rank 旁边多起 1 个 CPU burner。它偶尔高于 `idle`，更可能是调度、预热、采样窗口或 HCCL 状态差异，不应被解释成“绑到 CPU 1 更好”。
- 绑核结论应看显式 `sched_setaffinity` 的 core 矩阵。本次 H2D 和 D2D 都是逐 core 绑定后实测。
- NUMA 标签只作为解释线索。本次实验不信任 NUMA 配置本身，而是用每个 CPU core 的实测值反推结构。
- 共享环境中外部任务会影响结果，尤其是当前 A3 上部分 NPU 有 `xllm` 进程占用；对 pod4/pod8 做最终归因时，应在同等空载或同等背景负载下重跑。
