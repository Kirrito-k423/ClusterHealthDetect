# ClusterHealthDetect

`ClusterHealthDetect` 是一个基于 `torchrun` 的 NPU/GPU/CPU 集群健康基准仓，用来把真实训练性能劣化拆成可复现实测项：

- CPU 计算吞吐和当前进程可绑定 CPU map
- NPU/GPU matmul 吞吐
- H2D host-to-device copy 带宽
- D2D device-to-device copy 带宽，以及本机卡间 `sendrecv/local_host_pair`
- `all_gather` 本机多卡、全局多机、跨机点对点 `sendrecv/inter_host_pair` 通信带宽
- 空载、CPU 干扰、设备计算干扰等不同负载 profile
- CPU 绑核可行性、NUMA 绑核策略对 H2D/D2D/allgather 的影响矩阵
- 逐 CPU core 绑定后，对每张 NPU 卡的 H2D 实测矩阵
- 固定两张 NPU 卡时，两个 rank 绑定到不同 CPU core 对卡间 allgather/D2D 通信的影响矩阵
- Excel `.xlsx` 与 PNG 热力图，便于横向比较 pod、NUMA、rank 和负载 profile

它的目标不是替代生产 SFT，而是给类似 Qwen3.5 397B SFT 中 “pod4 比 pod8 慢 10%，定位到 allgather 慢，但单独打流没差异” 的问题提供更完整的对照证据。

## 快速开始

单机 8 卡：

```bash
NPROC_PER_NODE=8 bash scripts/run_single_node.sh
```

双机时，在两台机器同一路径部署仓库，rank 0 机器执行：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<rank0-ip> NPROC_PER_NODE=8 bash scripts/run_multinode.sh
```

rank 1 机器执行：

```bash
NNODES=2 NODE_RANK=1 MASTER_ADDR=<rank0-ip> NPROC_PER_NODE=8 bash scripts/run_multinode.sh
```

常用短跑 smoke：

```bash
NPROC_PER_NODE=2 \
PROFILES=idle \
TESTS=affinity,cpu,device,h2d,d2d,collective \
SIZES_MB=4,16 \
CPU_SIZES=256,512 \
DEVICE_SIZES=512,1024 \
SECONDS_PER_SIZE=0.5 \
ITERS=3 \
WARMUP=1 \
bash scripts/run_single_node.sh
```

绑核可行性检查，回答“是否所有核心都支持绑核”：

```bash
bash scripts/run_affinity_probe.sh
```

NUMA 绑核矩阵，按 `none`、`local_rank`、`numa:N`、`numa:N:shard` 等策略测 H2D、D2D、allgather，并输出 Excel 热力图：

```bash
NPROC_PER_NODE=8 \
SIZES_MB=16,64,256 \
REPEATS=3 \
bash scripts/run_numa_affinity_matrix.sh
```

不相信 NUMA 配置时，直接逐 core × NPU card 测 H2D。下面会把每个可绑定 CPU core 逐一绑定，然后分别测到每张 NPU 的 H2D 带宽，输出 `core-h2d-matrix.xlsx` 和 `png/h2d_cpu_core_by_npu_<size>MB.png`：

```bash
CPUS=auto \
DEVICES=auto \
SIZES_MB=16,64,256 \
REPEATS=3 \
bash scripts/run_core_h2d_matrix.sh
```

正式跑全量时，A3 这类 `640 CPU x 16 NPU x 3 size x 3 repeat` 会比较久；先小样本检查链路可以这样：

```bash
CPUS=0-15 \
DEVICES=0-3 \
SIZES_MB=16 \
ITERS=5 \
WARMUP=2 \
REPEATS=1 \
bash scripts/run_core_h2d_matrix.sh
```

固定两张卡，测试两个 rank 分别绑到不同 CPU core 对卡间 allgather/D2D 通信的影响。输出 `core-pair-d2d-matrix.xlsx` 和 `png/d2d_rank0_core_by_rank1_core_<size>MB.png`，行是 rank0 绑定 core，列是 rank1 绑定 core：

```bash
DEVICE_PAIR=0,1 \
RANK0_CPUS=auto \
RANK1_CPUS=auto \
SIZES_MB=16,64,256 \
REPEATS=3 \
bash scripts/run_core_pair_d2d_matrix.sh
```

`RANK0_CPUS=auto` 和 `RANK1_CPUS=auto` 是平方级扫描。A3 上 640 个 CPU core 的固定两卡全量矩阵是 `640 x 640 = 409600` 个 core-pair；长跑建议先用 `SIZES_MB=8 ITERS=1 WARMUP=0 REPEATS=1` 验证结构，再按需要提高 size/iters。D2D runner 默认 `RECORD_RANKS=rank0`，rank1 仍参与通信和绑核，只是不重复记录 collective 计时，避免全量矩阵在收尾时生成双倍 JSON。

如果只想从已有 JSON 生成 Excel：

```bash
bash scripts/build_excel_heatmap.sh results/single-node-20260703-121618
python3 -m cluster_health_detect.matrix_excel results/core-h2d-matrix-... --out h2d.xlsx
```

## 输出

每个 rank 会写出：

```text
results/.../rank_00000.json
```

rank 0 会额外写出：

```text
results/.../results.json
results/.../report.md
results/.../numa-affinity-heatmap.xlsx
results/.../core-h2d-matrix.xlsx
results/.../core-pair-d2d-matrix.xlsx
results/.../png/*.png
```

如果 `all_gather_object` 在某些后端不可用，至少每个 rank 的 JSON 仍会保留下来。

## 关键参数

- `BACKEND=auto|hccl|nccl|gloo`：默认自动选择。Ascend NPU 通常为 `hccl`。
- `DEVICE=auto|npu|cuda|cpu`：默认自动选择。
- `PROFILES=idle,cpu:2,device`：空载、每 rank 2 个 CPU burner、设备 matmul 背景负载。
- `TESTS=all`：可选 `affinity,cpu,device,h2d,d2d,collective`。
- `SIZES_MB=16,64,256`：copy/collective tensor 大小。
- `CPU_SIZES=512,1024,2048`：CPU matmul N。
- `DEVICE_SIZES=1024,2048,4096`：NPU/GPU matmul N。
- `ENABLE_P2P=1`：开启 `sendrecv/local_host_pair` 和 `sendrecv/inter_host_pair`。部分 HCCL 版本的 P2P 可能很慢或不可用，所以默认关闭。
- `BIND_POLICIES=auto`：NUMA 绑核矩阵策略。`auto` 会展开为 `none`、`local_rank`、每个 NUMA 的 `numa:N` 和 `numa:N:shard`。也可显式传 `none;local_rank;numa:0;numa:1:shard`。
- `CPUS_PER_RANK=0`：绑核策略里每个 rank 使用多少 CPU。`0` 表示保留该策略解析出的完整 CPU 集合。
- `CPUS=auto` / `DEVICES=auto`：逐 core H2D 矩阵的 CPU 和 NPU 范围。支持 `0-15`、`0,80,160` 这样的列表。
- `DEVICE_PAIR=0,1` / `RANK0_CPUS=auto` / `RANK1_CPUS=auto`：固定两卡通信矩阵的卡号与两个 rank 的 CPU core 扫描范围。
- `RECORD_RANKS=rank0|all`：固定两卡通信矩阵记录 rank0 计时，或记录所有 rank。全量 core-pair 建议保持默认 `rank0`。

## 背景负载是什么意思

背景负载是 benchmark 主测试旁边人为制造的共存压力，用来模拟训练时 CPU/NPU 不空闲的情况：

- `idle`：不额外制造压力，只跑目标测试。
- `cpu:N`：每个 rank 旁边启动 `N` 个 CPU burner 进程，持续做 CPU 矩阵/数值计算，制造调度、cache 和内存带宽压力。它不是绑核。
- `device`：每个 rank 在设备侧启动 matmul burner，制造 NPU/GPU 计算压力，观察 H2D、D2D、allgather 在设备繁忙时是否下降。

如果 `cpu:1` 比 `idle` 更高，不能直接解释为“绑核更好”。它通常说明调度、预热、HCCL 算法状态或采样噪声改变了测试窗口。绑核结论应该看显式 `sched_setaffinity` 的 NUMA/core 矩阵。

## A3 全量样例

本仓已在 A3 机器上跑过两组全量 core 矩阵。`2026-07-06` 的 `full` 结果是在共享环境下得到的；`2026-07-07` 的 `clean` 结果是在 A3-AK-182 清掉自动占卡任务后复测得到，建议优先作为当前 A3 的基线。报告和产物见 `reports/`：

- `A3-clean-core-matrix-report.md`：清理外部占用后的 H2D/D2D 复测报告。
- `A3-core-h2d-matrix-clean-20260707.xlsx`：640 CPU core x 16 NPU x 16/64/256MB H2D clean run。
- `A3-core-h2d-matrix-clean-20260707-16MB.png`、`A3-core-h2d-matrix-clean-20260707-64MB.png`、`A3-core-h2d-matrix-clean-20260707-256MB.png`。
- `A3-core-pair-d2d-matrix-clean-20260707-8MB.xlsx`：固定 NPU pair `0,5`，640 rank0 core x 640 rank1 core，8MB allgather clean run。
- `A3-core-pair-d2d-matrix-clean-20260707-8MB.png`。
- `A3-cross-node-h2d-npu0-report.md`：对比 A3-AK-182 与 A3-AX-180 的 H2D 64MB NPU0 现象。
- `A3-180-core-h2d-matrix-clean-64MB-20260708.xlsx`：A3-AX-180 上 384 CPU core x 16 NPU x 64MB H2D clean run。
- `A3-180-core-h2d-matrix-clean-64MB-20260708.png`。
- `A3-180-size-sweep-h2d-d2d-hccl-20260708.md`：A3-AX-180 上 H2D、同卡 D2D copy、2-rank HCCL allgather 从 0.125MB 到 1024MB 的尺寸曲线与 `HCCL_BUFFSIZE` 对比。
- `A3-180-h2d-d2d-hccl-size-sweep-20260708.png`、`A3-180-h2d-d2d-hccl-size-sweep-20260708.csv`。
- `A3-core-h2d-matrix-full.xlsx`：640 CPU core x 16 NPU x 16/64/256MB H2D。
- `A3-core-h2d-matrix-full-16MB.png`、`A3-core-h2d-matrix-full-64MB.png`、`A3-core-h2d-matrix-full-256MB.png`。
- `A3-core-pair-d2d-matrix-full-8MB.xlsx`：固定 NPU pair `0,5`，640 rank0 core x 640 rank1 core，8MB allgather。
- `A3-core-pair-d2d-matrix-full-8MB.png`。
- `A3-full-core-matrix-report.md`：实验命令、统计摘要和注意事项。

## 关于 `cpu:1` 为什么可能更快

`PROFILES=cpu:1` 不是绑核。它只是在每个 rank 旁边额外启动 1 个 CPU burner 进程，当前 rank 和 burner 都仍由操作系统调度。它偶尔比 `idle` 更高，常见原因包括：

- benchmark 采样时间太短，HCCL 算法选择、缓存、预热状态和系统调度造成波动。
- 轻量 CPU 活动改变了进程被调度的时机，让通信线程更少被迁移或更快被唤醒。
- `idle` 与 `cpu:1` 的测试顺序不同，后跑的一组可能吃到更充分的设备/HCCL warmup。
- 这不是 NUMA/绑核结论；只有 `run_numa_affinity_matrix.sh` 这种显式 `sched_setaffinity` 矩阵才能回答“绑到哪个 NUMA/CPU 是否影响 H2D、D2D、allgather”。
- 如果不相信系统 NUMA 配置，优先看 `run_core_h2d_matrix.sh` 和 `run_core_pair_d2d_matrix.sh` 的逐 core 实测热力图；NUMA 只作为解释标签，不作为前提假设。

## 解读建议

对比 pod4 和 pod8 时建议至少保留这些维度：

1. 环境：torch/torch_npu/HCCL/CANN 版本、容器 cpuset、`ASCEND_VISIBLE_DEVICES`。
2. 绑核：`affinity` 中 `allowed_cpus`、`bindable_cpus`、`unavailable_cpus`。
3. 单 rank 上限：CPU/NPU matmul、H2D、D2D。
4. 分层通信：`collective/all_gather/local_host`、`global`、`sendrecv/local_host_pair`、`sendrecv/inter_host_pair`。
5. 负载敏感性：`idle` 与 `cpu:N`、`device` profile 的下降幅度。
6. NUMA 绑核敏感性：`bind_policy=none/local_rank/numa:N/numa:N:shard` 下 H2D、D2D 和 allgather 的 Excel 热力图差异。
7. 逐 core 实测：`core-h2d-matrix.xlsx` 看 CPU core × NPU card，`core-pair-d2d-matrix.xlsx` 看固定两卡下 rank0 core × rank1 core。

如果维护打流只覆盖了裸 `all_gather`，但生产 SFT 在 CPU 绑核、H2D、设备计算、NUMA 或跨机 NIC 竞争下才退化，这个仓可以把“空载没差异”和“训练负载下有差异”分开记录。
