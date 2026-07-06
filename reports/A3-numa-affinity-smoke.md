# ClusterHealthDetect NUMA Affinity Report

- 生成时间：2026-07-06T17:06:50
- 输入 metrics：26 条
- 说明：这是 `SIZES_MB=1`、`ITERS=1`、`WARMUP=0` 的功能 smoke，只证明绑核矩阵与 Excel 输出链路可用，不代表稳定带宽上限。正式对比请使用 `SIZES_MB=16,64,256`、`REPEATS>=3` 和充分 warmup。

## 环境概览

| host | rank | device | torch_ok | torch | torch_npu_ok | torch_npu | npu_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| localhost.localdomain | 0 | npu:0 | True | 2.7.1+cpu | True | 2.7.1.post3 | 16 |
| localhost.localdomain | 1 | npu:1 | True | 2.7.1+cpu | True | 2.7.1.post3 | 16 |

## 关键上限指标

| profile | test | op | scope | size | dtype | metric | max | median | p10 | max/min | ranks | hosts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bind:none | collective | all_gather | global | 1 | float16 | alg_gbps | 1.371 | 1.364 | 1.358 | 1.01x | 2 | localhost.localdomain |
| bind:none | collective | all_gather | local_host | 1 | float16 | alg_gbps | 0.257 | 0.255 | 0.254 | 1.01x | 2 | localhost.localdomain |
| bind:none | d2d | copy |  | 1 | float16 | gbps | 0.909 | 0.884 | 0.858 | 1.06x | 2 | localhost.localdomain |
| bind:none | h2d | copy |  | 1 | float16 | gbps | 4.519 | 4.379 | 4.238 | 1.07x | 2 | localhost.localdomain |
| bind:numa:0:shard | collective | all_gather | global | 1 | float16 | alg_gbps | 1.569 | 1.566 | 1.563 | 1.00x | 2 | localhost.localdomain |
| bind:numa:0:shard | collective | all_gather | local_host | 1 | float16 | alg_gbps | 0.176 | 0.127 | 0.078 | 2.25x | 2 | localhost.localdomain |
| bind:numa:0:shard | d2d | copy |  | 1 | float16 | gbps | 5.446 | 5.287 | 5.129 | 1.06x | 2 | localhost.localdomain |
| bind:numa:0:shard | h2d | copy |  | 1 | float16 | gbps | 6.477 | 6.429 | 6.380 | 1.02x | 2 | localhost.localdomain |

## 候选瓶颈提示

- `('bind:numa:0:shard', 'collective', 'all_gather', 'local_host', 1)` 的 rank 间差异超过 10%：最低 rank=1 host=localhost.localdomain alg_gbps=0.078，最高 rank=0 host=localhost.localdomain alg_gbps=0.176。

## Error / Skip

| status | test | op | scope | count | example |
| --- | --- | --- | --- | --- | --- |
| skip | collective | sendrecv |  | 4 | p2p disabled; set --enable-p2p after validating backend support |

## 解读口径

- `cpu`/`device` 的 `tflops` 是 matmul 实测吞吐上限，适合比较 CPU 线程、NPU AICore 或虚拟化影响。
- `h2d` 是 host 到 device copy 带宽；`d2d` 是同 rank device 内 copy 带宽。
- `collective/all_gather/global` 反映训练中跨所有 rank 的 allgather；`local_host` 反映本机多卡；`sendrecv/local_host_pair` 与 `sendrecv/inter_host_pair` 需要显式开启 P2P。
- `max/min` 超过 `1.10x` 时，需要继续查对应 rank 的 CPU 绑核、NPU health、HCCL/NIC 绑定、容器 cpuset、NUMA 和后台负载。
