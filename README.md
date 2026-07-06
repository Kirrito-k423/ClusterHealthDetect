# ClusterHealthDetect

`ClusterHealthDetect` 是一个基于 `torchrun` 的 NPU/GPU/CPU 集群健康基准仓，用来把真实训练性能劣化拆成可复现实测项：

- CPU 计算吞吐和当前进程可绑定 CPU map
- NPU/GPU matmul 吞吐
- H2D host-to-device copy 带宽
- D2D device-to-device copy 带宽，以及本机卡间 `sendrecv/local_host_pair`
- `all_gather` 本机多卡、全局多机、跨机点对点 `sendrecv/inter_host_pair` 通信带宽
- 空载、CPU 干扰、设备计算干扰等不同负载 profile
- CPU 绑核可行性、NUMA 绑核策略对 H2D/D2D/allgather 的影响矩阵
- Excel `.xlsx` 热力图，便于横向比较 pod、NUMA、rank 和负载 profile

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

如果只想从已有 JSON 生成 Excel：

```bash
bash scripts/build_excel_heatmap.sh results/single-node-20260703-121618
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

## 关于 `cpu:1` 为什么可能更快

`PROFILES=cpu:1` 不是绑核。它只是在每个 rank 旁边额外启动 1 个 CPU burner 进程，当前 rank 和 burner 都仍由操作系统调度。它偶尔比 `idle` 更高，常见原因包括：

- benchmark 采样时间太短，HCCL 算法选择、缓存、预热状态和系统调度造成波动。
- 轻量 CPU 活动改变了进程被调度的时机，让通信线程更少被迁移或更快被唤醒。
- `idle` 与 `cpu:1` 的测试顺序不同，后跑的一组可能吃到更充分的设备/HCCL warmup。
- 这不是 NUMA/绑核结论；只有 `run_numa_affinity_matrix.sh` 这种显式 `sched_setaffinity` 矩阵才能回答“绑到哪个 NUMA/CPU 是否影响 H2D、D2D、allgather”。

## 解读建议

对比 pod4 和 pod8 时建议至少保留这些维度：

1. 环境：torch/torch_npu/HCCL/CANN 版本、容器 cpuset、`ASCEND_VISIBLE_DEVICES`。
2. 绑核：`affinity` 中 `allowed_cpus`、`bindable_cpus`、`unavailable_cpus`。
3. 单 rank 上限：CPU/NPU matmul、H2D、D2D。
4. 分层通信：`collective/all_gather/local_host`、`global`、`sendrecv/local_host_pair`、`sendrecv/inter_host_pair`。
5. 负载敏感性：`idle` 与 `cpu:N`、`device` profile 的下降幅度。
6. NUMA 绑核敏感性：`bind_policy=none/local_rank/numa:N/numa:N:shard` 下 H2D、D2D 和 allgather 的 Excel 热力图差异。

如果维护打流只覆盖了裸 `all_gather`，但生产 SFT 在 CPU 绑核、H2D、设备计算、NUMA 或跨机 NIC 竞争下才退化，这个仓可以把“空载没差异”和“训练负载下有差异”分开记录。
