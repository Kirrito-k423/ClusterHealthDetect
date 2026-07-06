# A3 Validation Report

生成时间：2026-07-03 12:20 CST

## 验证结论

- 代码仓已在 `A3-AK-182` 验证通过，远端路径：`/home/t00906153/ClusterHealthDetect`。
- 验证环境：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` + conda env `hy_qwen35_muon`。
- `torchrun` 2 rank、2 NPU 验证完成：`metrics=86`，`ok=80`，`error=0`，`skip=6`。
- `skip=6` 全部来自默认关闭的 `sendrecv` P2P 探针；P2P 可用 `ENABLE_P2P=1` 显式开启，但本轮发现 HCCL P2P 会明显拉长运行时间，不作为默认项。
- A3-182 系统 Python 若未 source CANN，会出现 `libhccl.so` 缺失；脚本已通过 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 支持在不完整环境里做 CPU/gloo 诊断。

## 运行命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate hy_qwen35_muon
cd /home/t00906153/ClusterHealthDetect

NPROC_PER_NODE=2 \
PROFILES=idle,cpu:1,device \
TESTS=affinity,cpu,device,h2d,d2d,collective \
SIZES_MB=1,8 \
CPU_SIZES=64,128 \
DEVICE_SIZES=256,512 \
SECONDS_PER_SIZE=0.2 \
ITERS=3 \
WARMUP=1 \
BACKEND=auto \
DEVICE=auto \
bash scripts/run_single_node.sh
```

结果目录：

- 远端：`/home/t00906153/ClusterHealthDetect/results/single-node-20260703-121618`
- 本地副本：`results/a3-182/single-node-20260703-121618`

## 关键数值

| profile | metric | size | max | min | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| idle | NPU matmul TFLOPS | 512 | 29.121 | 26.893 | 空载设备计算 |
| device | NPU matmul TFLOPS | 512 | 24.848 | 15.766 | 设备背景负载下 rank 离散变大 |
| idle | H2D GB/s | 8 MB | 54.087 | 54.039 | 空载 H2D |
| device | H2D GB/s | 8 MB | 47.236 | 46.234 | 设备背景负载下下降 |
| idle | D2D copy GB/s | 8 MB | 420.763 | 241.793 | 空载同 rank device copy，rank 间差异明显 |
| device | D2D copy GB/s | 8 MB | 257.846 | 226.128 | 设备背景负载下下降 |
| idle | all_gather local_host alg GB/s | 8 MB | 28.669 | 28.600 | 本机 2 NPU HCCL allgather |
| idle | all_gather global alg GB/s | 8 MB | 27.137 | 27.097 | 单机 2 rank 下 global≈local |
| device | all_gather local_host alg GB/s | 8 MB | 25.171 | 23.944 | 设备背景负载下下降 |
| device | all_gather global alg GB/s | 8 MB | 20.784 | 20.248 | 设备背景负载下下降 |
| cpu:1 | all_gather global alg GB/s | 8 MB | 30.929 | 30.903 | CPU 背景负载未导致下降 |

## 观察

- `cpu:1` 不是绑核；它只是每 rank 增加 1 个 CPU burner 进程，rank 本身仍由系统调度。因此 `cpu:1` 的 allgather 高于 `idle` 不能直接解释为“绑核有效”，更可能是短测波动、warmup/测试顺序或调度状态差异。绑核影响应看 NUMA affinity matrix。
- 只看空载 allgather 会漏掉负载敏感性：本轮 `idle` 下 8 MB global allgather 约 27.1 GB/s，但 `device` 背景负载下降到约 20.2-20.8 GB/s。
- 设备背景负载下，NPU matmul 512 的 rank 差异达到约 1.58x；这类离散和真实 SFT 训练中的计算/通信重叠更接近。
- H2D 在 8 MB 下从约 54.0 GB/s 降到约 46-47 GB/s，说明 host-device 路径也应纳入 pod4/pod8 对比。
- D2D copy 的 rank 离散较大，建议后续扩大到所有 16 rank 并结合 `npu-smi info`、容器 cpuset、NUMA、HCCL/NIC 绑定一起看。

## 未完成项

- 本轮没有产出可信的跨机 HCCL 数值。原因是两台 A3 的可用 torch/torch_npu/CANN 环境未对齐：
  - A3-182 conda `hy_qwen35_muon`：`torch 2.7.1+cpu`，`torch_npu 2.7.1.post3`，NPU 可见 16。
  - A3-180 容器 `ascend_AUTO_TEST`：`torch 2.7.1+cpu`，`torch_npu 2.7.1.post6.dev20260702`，NPU 可见 16。
  - A3-182 容器里也有 `torch_npu 2.9.0.post2` 或 `2.7.1.post2`，但未与 A3-180 统一镜像。
- 曾尝试两台 A3 的双机 CPU/Gloo `torchrun` smoke；rank1 到 rank0 rendezvous TCP 已建立，但 worker 初始化未在合理时间内完成，已终止清理。该结果只说明网络连接能建立，不作为带宽数值。

## 后续建议

1. 用同一个容器镜像在 pod4/pod8、A3-180/A3-182 上复测，避免 torch_npu/CANN 版本差异混入结论。
2. 先跑 `NPROC_PER_NODE=16 PROFILES=idle,cpu:2,device SIZES_MB=16,64,256` 单机全卡，再跑双机 `NNODES=2`。
3. 如果需要点对点跨机带宽，先单独小规模验证 `ENABLE_P2P=1`；否则用 `all_gather/global` 作为训练相关通信主指标。
4. 把 pod4 与 pod8 的报告放到同一目录后，用 `python3 -m cluster_health_detect.summarize <dir1> <dir2>` 汇总对比。
