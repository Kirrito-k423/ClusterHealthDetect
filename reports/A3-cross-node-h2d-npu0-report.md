# A3 Cross-Node H2D NPU0 Check

生成日期：2026-07-08

## 结论摘要

- 为排查 `A3-AK-182` clean H2D 中 NPU `0` 异常低列是否是 A3 机器的普遍现象，额外在 `A3-AX-180` 上复测了 `64MB` H2D 全 CPU core x 全 NPU 矩阵。
- `A3-AX-180` 当前可绑定 CPU core 为 384 个，NPU 可见 16 张；本次覆盖 `384 CPU core x 16 NPU x 64MB`，共 6144 个点，全部 `ok`。
- `A3-AX-180` 没有复现 `A3-AK-182` 的 NPU0 垂直低列。180 上 NPU0 median 为 `24.814 GB/s`，16 张 NPU 的 median 范围仅为 `24.756-24.888 GB/s`。
- 但 `A3-AX-180` 的全机 H2D 上限明显低于 182 的非 NPU0：180 的全局 64MB median 为 `24.830 GB/s`，182 的全局 64MB median 为 `58.376 GB/s`，182 去掉 NPU0 后多数 NPU median 约 `57.746-58.863 GB/s`。
- 因此当前更合理的解释是：`A3-AK-182` 有单卡/单路径 NPU0 异常；`A3-AX-180` 没有 NPU0 特异异常，但存在整机 H2D 上限偏低或环境差异。二者不应混成同一种问题。

## 可用机器与覆盖范围

本轮可达 A3：

| node | host | result |
| --- | --- | --- |
| A3-AK-182 | `192.168.13.182` | 已有 clean H2D `16/64/256MB` 全量结果 |
| A3-AX-180 | `192.168.13.180` | 新增 clean H2D `64MB` 全量结果 |

尝试但未能用于本轮实验：

| host | status |
| --- | --- |
| `192.168.13.181` | 免密与保存的 A3 bootstrap 凭据均登录失败 |
| `192.168.13.183` | 免密与保存的 A3 bootstrap 凭据均登录失败 |
| `192.168.13.184` | SSH 超时 |
| `192.168.13.178/179/185/186` | 当前账号不可达 |

## A3-AX-180 H2D 64MB 实验

运行环境：

- 机器：`A3-AX-180`
- hostname：`admin123`
- CPU：384 个在线 CPU，AMD EPYC 9654，2 NUMA nodes
- 环境：conda `qxm_mm`
- torch：`2.7.1+cpu`
- torch_npu：`2.7.1.post5`
- NPU：16 张可见 Ascend NPU
- 起跑前和结束后 `npu-smi` 均显示无外部 running process

命令：

```bash
OUT_DIR=results/core-h2d-matrix-clean-180-64mb-20260708-020224 \
CPUS=auto \
DEVICES=auto \
SIZES_MB=64 \
ITERS=3 \
WARMUP=1 \
REPEATS=1 \
CHECKPOINT_EVERY_CPUS=4 \
DEVICE_KIND=auto \
bash scripts/run_core_h2d_matrix.sh
```

产物：

- [A3-180-core-h2d-matrix-clean-64MB-20260708.xlsx](A3-180-core-h2d-matrix-clean-64MB-20260708.xlsx)
- [A3-180-core-h2d-matrix-clean-64MB-20260708.png](A3-180-core-h2d-matrix-clean-64MB-20260708.png)
- [A3-180-clean-h2d-64MB-20260708.log](A3-180-clean-h2d-64MB-20260708.log)

总体统计：

| node | size | points | min GB/s | p10 GB/s | median GB/s | p90 GB/s | max GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3-AK-182 | 64MB | 10240 | 4.739 | 37.252 | 58.376 | 58.848 | 59.356 |
| A3-AX-180 | 64MB | 6144 | 7.158 | 23.030 | 24.830 | 26.369 | 27.178 |

## NPU0 对照

| node | NPU0 median GB/s | NPU0 rank by median | best NPU median GB/s | worst NPU median GB/s | interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| A3-AK-182 | 12.676 | 16 / 16 | 58.863 | 12.676 | NPU0 是全机最慢且形成稳定低列 |
| A3-AX-180 | 24.814 | 9 / 16 | 24.888 | 24.756 | NPU0 接近同机其它 NPU，不是特异慢卡 |

`A3-AK-182` 64MB 每张 NPU 的 median：

| NPU | median GB/s |
| ---: | ---: |
| 0 | 12.676 |
| 1 | 34.952 |
| 2 | 58.378 |
| 3 | 58.536 |
| 4 | 58.488 |
| 5 | 58.307 |
| 6 | 58.809 |
| 7 | 58.863 |
| 8 | 57.746 |
| 9 | 57.839 |
| 10 | 58.429 |
| 11 | 58.277 |
| 12 | 58.330 |
| 13 | 58.415 |
| 14 | 58.836 |
| 15 | 58.660 |

`A3-AX-180` 64MB 每张 NPU 的 median：

| NPU | median GB/s |
| ---: | ---: |
| 0 | 24.814 |
| 1 | 24.799 |
| 2 | 24.802 |
| 3 | 24.811 |
| 4 | 24.859 |
| 5 | 24.843 |
| 6 | 24.835 |
| 7 | 24.799 |
| 8 | 24.803 |
| 9 | 24.756 |
| 10 | 24.888 |
| 11 | 24.783 |
| 12 | 24.877 |
| 13 | 24.874 |
| 14 | 24.878 |
| 15 | 24.854 |

## CPU 段观察

`A3-AK-182` 使用 80-core 段观察，NPU0 在所有段都是最慢：

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

`A3-AX-180` 使用本机 NUMA/CPU 布局对应的 96-core 段观察，NPU0 不构成最慢稳定低列：

| CPU core 段 | NPU0 median GB/s | 同段最高 NPU median GB/s | 同段最低 NPU median GB/s |
| --- | ---: | ---: | ---: |
| 0-95 | 24.799 | 24.908 | 24.508 |
| 96-191 | 24.850 | 24.903 | 24.681 |
| 192-287 | 24.806 | 24.906 | 24.730 |
| 288-383 | 24.812 | 24.861 | 24.690 |

## 解读

- `A3-AK-182` 的 NPU0 慢并未在 `A3-AX-180` 上复现，因此不能先验地认为 “A3 的 NPU0 都慢”。
- `A3-AX-180` 的 16 张 NPU 很一致，但整体 H2D 64MB 上限只有约 `25 GB/s`；这提示 180 和 182 的 CPU/NPU 拓扑、CANN/torch_npu 版本、BIOS/IOMMU/PCIe/NUMA 配置、虚拟化策略或 host memory/pinned memory 路径可能不同。
- 对生产 pod4/pod8 排查时，应把“单卡低列”和“整机 H2D 上限低”拆成两个独立故障模式：前者影响特定 rank-to-card placement，后者会影响整机所有 rank 的 H2D 下界。
