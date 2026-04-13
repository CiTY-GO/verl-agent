# verl-agent ALFWorld 提交脚本说明

## 脚本文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `run_alfworld_hope.sh` | 训练脚本 | ALFWorld 训练脚本 |
| `run_webshop_hope.sh` | 训练脚本 | WebShop 训练脚本 |
| `setup.py` | Python 安装脚本 | verl-agent 包安装 |
| `submit_alfworld_hotel.sh` | 提交脚本 | 使用 Hotel 镜像 |
| `submit_alfworld_hope.sh` | 提交脚本 | 使用基础镜像 + 本地 conda |

## 两个提交脚本对比

### 1. submit_alfworld_hotel.sh（推荐）

**使用 Hotel 完整镜像**

```bash
bash submit_alfworld_hotel.sh [gpu_num] [worker_num]
bash submit_alfworld_hotel.sh 2 1  # 2 GPU, 1 worker (默认)
```

| 配置项 | 值 |
|--------|-----|
| Docker 镜像 | `training_cuda12.6_conda_python3.12_torch2.8_vllm0.11_verl0.6_fa3_mtai` |
| CUDA | 12.6 |
| Python | 3.12 |
| PyTorch | 2.8 |
| vllm | 0.11 |
| verl | 0.6 |
| flash-attn | fa3 |
| 环境 | 镜像自带，无需本地 conda |

**优点：**
- ✅ 版本更新（torch2.8, vllm0.11, verl0.6）
- ✅ 无需本地 conda 环境
- ✅ 镜像已验证可用

---

### 2. submit_alfworld_hope.sh（备选）

**使用基础镜像 + 本地 conda 环境**

```bash
bash submit_alfworld_hope.sh [gpu_num] [worker_num]
bash submit_alfworld_hope.sh 2 1  # 2 GPU, 1 worker (默认)
```

| 配置项 | 值 |
|--------|-----|
| Docker 镜像 | `training_cuda12.4_cudnn9.8_glibc2.31` |
| CUDA | 12.4 |
| Python | 3.12 |
| PyTorch | 2.6.0 |
| vllm | 0.8.5 |
| verl | 0.3.1.dev |
| flash-attn | 2.7.4.post1 |
| 环境 | 本地 conda: `/mnt/.../miniconda3/envs/verl-agent` |

**优点：**
- ✅ 使用你精心配置的环境
- ✅ 版本控制在你手中

---

## 修正内容

1. **run_alfworld_hope.sh**
   - ✅ 移除 `pip3 install vllm==0.8.5`（避免与 hotel 镜像冲突）
   - ✅ GPU 数量改为动态获取（从参数传入）

2. **submit_alfworld_hotel.sh**
   - ✅ 传递 GPU_NUM 参数给训练脚本

---

## 使用建议

1. **优先尝试** `submit_alfworld_hotel.sh`（Hotel 镜像版本）
2. 如果遇到问题，再尝试 `submit_alfworld_hope.sh`（本地环境版本）

## 查看任务状态

```bash
# 使用任务 ID 查看状态
hope ps <task_id>

# 示例
hope ps psx5r8ixrjzmgsfz
```

或者在 MLP 控制台查看日志。
