# A3 Core Matrix Smoke

生成时间：2026-07-06 20:31 CST

## 目的

本报告验证两个逐 core 矩阵工具是否按预期工作：

- `core_h2d_matrix`：行是 CPU core，列是 NPU card，测当前进程绑定到该 core 后到该 NPU 的 H2D 带宽。
- `core_pair_d2d_matrix`：固定两张 NPU 卡，行是 rank0 绑定 core，列是 rank1 绑定 core，测两卡 allgather 通信带宽。

## 验证命令

H2D 小矩阵：

```bash
OUT_DIR=results/core-h2d-matrix-smoke-20260706 \
CPUS=0-3 \
DEVICES=0-1 \
SIZES_MB=1 \
ITERS=1 \
WARMUP=0 \
REPEATS=1 \
DEVICE_KIND=auto \
bash scripts/run_core_h2d_matrix.sh
```

固定双卡通信小矩阵：

```bash
OUT_DIR=results/core-pair-d2d-matrix-smoke-20260706 \
DEVICE_PAIR=0,1 \
RANK0_CPUS=0-1 \
RANK1_CPUS=2-3 \
SIZES_MB=1 \
ITERS=1 \
WARMUP=0 \
REPEATS=1 \
BACKEND=auto \
DEVICE_KIND=auto \
bash scripts/run_core_pair_d2d_matrix.sh
```

## 结果

- H2D smoke：`8/8` 点成功，生成 `reports/A3-core-h2d-matrix-smoke.xlsx`。
- 固定双卡通信 smoke：`8/8` rank 侧指标成功，生成 `reports/A3-core-pair-d2d-matrix-smoke.xlsx`。
- 两个 Excel 都通过 zip 完整性检查，并用工作簿引擎导入和渲染验证。

## 限制

这轮使用 `SIZES_MB=1`、`ITERS=1`、`WARMUP=0`，只证明功能链路和热力图维度正确，不代表稳定带宽上限。正式定位 pod4/pod8 时建议：

```bash
CPUS=auto DEVICES=auto SIZES_MB=16,64,256 ITERS=20 WARMUP=5 REPEATS=3 bash scripts/run_core_h2d_matrix.sh
DEVICE_PAIR=0,1 RANK0_CPUS=auto RANK1_CPUS=auto SIZES_MB=16,64,256 ITERS=20 WARMUP=5 REPEATS=3 bash scripts/run_core_pair_d2d_matrix.sh
```

