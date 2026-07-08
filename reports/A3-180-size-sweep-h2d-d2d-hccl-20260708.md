# A3-AX-180 H2D / D2D / HCCL Size Sweep

生成日期：2026-07-08

## 结论摘要

- 本轮在 `A3-AX-180` 上补跑了从 `0.125MB` 逐次翻倍到 `1024MB` 的 H2D、同卡 D2D copy、以及 2-rank HCCL `all_gather` 尺寸曲线。
- 实验期间 `NPU 0-7` 存在 VeOmni 训练进程占用；本轮 copy 曲线使用空闲的 `NPU 8`，HCCL 使用空闲的 `NPU 8,10`。因此这组数据适合作为“共享负载环境下的空闲卡路径曲线”，不应写成整机空载峰值。
- 大尺寸 H2D 在 `128-1024MB` 稳定在约 `26.846-27.494 GB/s`，与此前 `A3-AX-180` 全 core x 全 NPU H2D 64MB 矩阵的 `max=27.178 GB/s` 量级一致。
- 同卡 D2D copy 大尺寸 `128-1024MB` 在约 `599.242-620.540 GB/s`，比 H2D 高一个数量级；这是因为数据源和目的都在 NPU/HBM 侧，不走 CPU DRAM 到 NPU HBM 的 host ingress 路径。
- 2-rank HCCL `all_gather` 在 `1024MB` 下为 `74.533-78.513 GB/s`。它也操作 device-resident tensor，不能替代或加速 H2D staging。
- `HCCL_BUFFSIZE` 的官方含义是 communicator 共享数据 buffer，单位 MB，默认值 `200`。本轮 `1GB` 点上 `1024MB` buffer 略高于默认和 `200MB`，但差异只有约 `1-2%`，需要多轮重复后才能当作调参结论。

## 硬件与链路快照

采集命令来自 `lscpu`、`npu-smi info` 与 `lspci -vv`。

| item | observed value |
| --- | --- |
| host | `admin123` / `A3-AX-180` |
| CPU | AMD EPYC 9654 96-Core Processor |
| online CPUs | 384 |
| NUMA | 2 nodes, `0-95,192-287` and `96-191,288-383` |
| NPU | 16 x `Ascend910` |
| PCI device | Huawei `19e5:d803 (rev 20)` |
| PCIe link status | `LnkSta: Speed 32GT/s (ok), Width x8 (ok)` on sampled Ascend devices |

`32GT/s x8` 对应 PCIe Gen5 x8 量级的链路状态。实际 H2D 带宽仍会受到 CPU/NUMA、pinned host memory、IOMMU/root complex、runtime copy path、共享负载等因素影响。

## 实验边界

`npu-smi info` 在实验前后均显示：

- `NPU 0-7` 有 VeOmni 训练进程，占用约 `41-47GB` HBM，并有明显 AICore 使用率。
- `NPU 8-15` 无 running process。

因此本轮实验选择：

- H2D 和同卡 D2D copy：`device_id=8`
- HCCL `all_gather`：`device_id=8,10`
- CPU affinity：`[0]`
- dtype：`float16`
- size：`0.125, 0.25, 0.5, 1, ... 1024 MB`

## 产物

- [A3-180-h2d-d2d-hccl-size-sweep-20260708.png](A3-180-h2d-d2d-hccl-size-sweep-20260708.png)
- [A3-180-h2d-d2d-hccl-size-sweep-20260708.csv](A3-180-h2d-d2d-hccl-size-sweep-20260708.csv)
- [A3-180-copy-sweep-20260708.json](A3-180-copy-sweep-20260708.json)
- [A3-180-hccl-default-sweep-20260708.json](A3-180-hccl-default-sweep-20260708.json)
- [A3-180-hccl-buff32-sweep-20260708.json](A3-180-hccl-buff32-sweep-20260708.json)
- [A3-180-hccl-buff200-sweep-20260708.json](A3-180-hccl-buff200-sweep-20260708.json)
- [A3-180-hccl-buff1024-sweep-20260708.json](A3-180-hccl-buff1024-sweep-20260708.json)

## 曲线读数

| operation | selected size | measured GB/s | note |
| --- | ---: | ---: | --- |
| H2D | 16MB | 27.953 | first stable high point |
| H2D | 128MB | 26.846 | large-payload plateau begins |
| H2D | 1024MB | 27.494 | large-payload plateau |
| D2D same-device copy | 8MB | 657.601 | device-resident copy |
| D2D same-device copy | 128MB | 599.242 | large-payload plateau |
| D2D same-device copy | 1024MB | 620.540 | large-payload plateau |
| HCCL all_gather default | 8MB | 46.245 | 2-rank NPU 8,10 |
| HCCL all_gather default | 32MB | 81.765 | curve peak in this run |
| HCCL all_gather default | 128MB | 71.645 | large payload |
| HCCL all_gather default | 1024MB | 76.987 | large payload |

H2D 的 `32MB` 和 `64MB` 点在本轮单次曲线中分别只有 `2.813` 与 `3.210 GB/s`。这与 `128MB+` 平台值、以及此前 64MB 全量矩阵不一致，应优先视为共享环境/单次采样/运行时抖动信号，而不是链路物理上限。若要用于 launch-map 决策，需要重复多轮并取 median/p10。

## HCCL_BUFFSIZE 对 1GB all_gather 的影响

| setting | 1GB alg GB/s |
| --- | ---: |
| default | 76.987 |
| `HCCL_BUFFSIZE=32` | 74.533 |
| `HCCL_BUFFSIZE=200` | 77.783 |
| `HCCL_BUFFSIZE=1024` | 78.513 |

官方 HCCL 文档说明 `HCCL_BUFFSIZE` 设置 communicator 使用的共享数据 buffer 大小，默认 `200MB`；每个 communicator 会占用发送和接收方向的 buffer，集群内 communicator 多时会增加总 buffer 占用；当 collective 数据大小超过 `HCCL_BUFFSIZE` 时，性能可能下降，并建议 `HCCL_BUFFSIZE` 大于数据大小。

本轮数据与这个方向一致：`32MB` buffer 在 1GB collective 上略低，`1024MB` buffer 略高。但差异很小，而且只跑了一轮，所以当前只能说明“需要把 HCCL buffer 当作可测变量”，不能说明 `1024MB` 在所有训练配置下最优。

## H2D、D2D、HCCL 口径

- H2D：CPU DRAM / pinned host buffer 到 NPU HBM，一般走 PCIe DMA。它会受到 CPU core、NUMA、pinned memory、PCIe/root complex 和 runtime 同步策略影响。
- D2D same-device copy：同一张 NPU 内 device tensor 到 device tensor 的 copy，主要是 NPU/HBM 侧搬运。它不是 CPU->NPU 入口带宽，所以可以比 H2D 高很多。
- HCCL all_gather：多 rank device-resident tensor 的集合通信。它测的是 HCCL 后端和设备间通信路径，不会自动优化 H2D staging。

因此，“D2D/HCCL 比 H2D 快”不是矛盾，而是三条路径的起点、终点和瓶颈不同。
