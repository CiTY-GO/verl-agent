# verl-agent Hope 提交指南

本文档介绍如何使用 Hope 提交 verl-agent 实验到美团远程集群。

## 📋 目录

- [环境要求](#环境要求)
- [Docker 镜像](#docker-镜像)
- [快速开始](#快速开始)
- [环境自动安装](#环境自动安装)
- [脚本说明](#脚本说明)
- [配置参数](#配置参数)
- [常见问题](#常见问题)

## 🚀 环境要求

### 本地环境

- Python 3.8+
- Hope 客户端工具
- 访问美团集群权限

### 远程集群

- Docker 镜像包含所需依赖
- GPU 资源（A100/H100/H20）

### 可用资源

| 资源类型 | 状态 | 位置 |
|-----------|--------|------|
| **Qwen2.5-VL-3B-Instruct** | ✅ 已下载 | `/VLLM/Qwen/Qwen2.5-VL-3B-Instruct` |
| **Qwen2.5-VL-7B-Instruct** | ✅ 已下载 | `/VLLM/Qwen/Qwen2.5-VL-7B-Instruct` |
| **Qwen3-VL-2B-Instruct** | ✅ 已下载 | `/VLLM/Qwen/Qwen3-VL-2B-Instruct` |
| **ALFWorld 环境** | ❌ 未安装 | 将在容器中自动安装 |
| **WebShop 环境** | ❌ 未安装 | 将在容器中自动安装 |

## 🐳 Docker 镜像

使用的 Docker 镜像：
```
registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/ai-search/training_codelab_verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.13.0-preview_3f97798e:v1.0.0
```

### 镜像包含的组件

- **verl**: 0.4 (高于 verl-agent 要求的 0.3.1.dev)
- **sglang**: 0.4.6.post5 (符合要求)
- **vllm**: 0.8.5 (符合要求)
- **mcore**: 0.13.0 (Megatron Core)
- **CUDA**: 12.4
- **PyTorch**: 2.x

### 兼容性分析

| 组件 | verl-agent 要求 | Docker 镜像 | 状态 |
|------|----------------|-------------|------|
| verl | 0.3.1.dev | 0.4 | ✅ 兼容（镜像版本更新） |
| sglang | 0.4.6.post5 | 0.4.6.post5 | ✅ 完全匹配 |
| vllm | 0.8.5 | 0.8.5 | ✅ 完全匹配 |
| torch | 2.6.0 | 2.x | ✅ 兼容 |

## 🎯 快速开始

### 1. 提交 ALFWorld 训练（2 GPU，vllm 引擎）

```bash
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/verl-agent
bash submit_verl_agent.sh alfworld
```

### 2. 提交 WebShop 训练（2 GPU，vllm 引擎）

```bash
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/verl-agent
bash submit_verl_agent.sh webshop
```

### 3. 使用 sglang 引擎

```bash
bash submit_verl_agent.sh alfworld sglang
```

### 4. 使用 8 GPU

```bash
bash submit_verl_agent.sh alfworld vllm 8
```

### 5. 使用特定提交脚本

```bash
# ALFWorld 专用提交脚本
bash submit_alfworld.sh

# 修改 submit_alfworld.sh 中的 GPU_NUM 参数来调整 GPU 数量
```

## 📦 环境自动安装

训练脚本会在容器中自动检测并安装所需的环境：

### ALFWorld 环境
- 自动安装：`gymnasium==0.29.1`, `stable-baselines3==2.6.0`, `alfworld`, `vllm==0.8.5`
- 自动下载：PDDL & Game 文件和预训练检测器
- 数据缓存位置：`~/.cache/alfworld/`

### WebShop 环境
- 自动安装：`gymnasium==0.29.1`, `stable-baselines3==2.6.0`, `vllm==0.8.5`
- 自动下载：WebShop 产品和指令数据（使用 `-d small` 下载子集）
- 数据位置：`agent_system/environments/env_package/webshop/webshop/data/`

**注意**：环境安装会在首次运行时自动进行，后续运行会跳过已安装的环境。

## 📝 脚本说明

### 1. 训练脚本

#### `run_alfworld_hope.sh`
ALFWorld 环境训练脚本，包含：
- 数据准备（自动检测）
- GiGPO 算法训练
- 默认配置：2 GPU，vllm 引擎

#### `run_webshop_hope.sh`
WebShop 环境训练脚本，包含：
- 数据准备（自动检测）
- GiGPO 算法训练
- 默认配置：2 GPU，vllm 引擎

### 2. 提交脚本

#### `submit_alfworld.sh`
ALFWorld 专用提交脚本：
- 生成 Hope 配置文件
- 自动替换 Docker 镜像
- 提交到集群

#### `submit_verl_agent.sh`（推荐）
通用提交脚本，支持：
- 多任务类型（alfworld, webshop）
- 多引擎选择（vllm, sglang）
- 灵活的 GPU 数量配置

## ⚙️ 配置参数

### 训练参数

在 `run_alfworld_hope.sh` 和 `run_webshop_hope.sh` 中可以修改以下参数：

```bash
# 数据大小
train_data_size=16
val_data_size=128

# GiGPO 参数
group_size=8
mode="mean_std_norm"  # 或 "mean_norm"

# 模型路径
MODEL_PATH=$BASE/VLLM/Qwen/Qwen2.5-VL-3B-Instruct

# 训练参数
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.kl_loss_coef=0.01
algorithm.gamma=0.95
algorithm.gigpo.step_advantage_w=1.0

# 环境参数
env.max_steps=50
env.rollout.n=$group_size

# 训练轮数
trainer.total_epochs=150
trainer.test_freq=5
```

### 提交参数

在 `submit_verl_agent.sh` 中可以修改以下参数：

```bash
# 任务参数
TASK=${1:-alfworld}        # 任务类型
ENGINE=${2:-vllm}         # 引擎类型
GPU_NUM=${3:-2}           # GPU 数量

# 集群参数
WORKER=1                   # Worker 数量
MIS_ID=chentianyu18        # MIS ID
PRIORITY=1                 # 优先级
MAX_RETRY=1                # 最大重试次数
```

### Hope 配置参数

生成的 Hope 配置文件包含：

```ini
[resource]
usergroup = hadoop-mtai
queue = root.zw05_training_cluster.hadoop-llm.pool

[roles]
workers = 1
worker.memory = 237000
worker.vcore = 176
worker.gcores80g = 2  # 根据 GPU_NUM 调整

[docker]
afo.docker.image.name = registry-offlinebiz.sankuai.com/custom_prod/com.sankuai.data.hadoop.gpu/ai-search/training_codelab_verl0.4-sglang0.4.6.post5-vllm0.8.5-mcore0.13.0-preview_3f97798e:v1.0.0
```

## 🔧 高级配置

### 1. 使用 H100 GPU

修改 `create_hope_verl.py` 的调用，添加 `--use_h100 true`：

```bash
python3 /path/to/create_hope_verl.py \
    --worker ${WORKER} \
    --gpu_num ${GPU_NUM} \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}" \
    --use_h100 true
```

### 2. 使用 H20 GPU

添加 `--use_h20 true` 参数。

### 3. 安装额外依赖

使用 `--requirement` 参数指定 requirements 文件：

```bash
python3 /path/to/create_hope_verl.py \
    --worker ${WORKER} \
    --gpu_num ${GPU_NUM} \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}" \
    --requirement "/path/to/requirements.txt"
```

### 4. 调整内存和 CPU

在 `create_hope_verl.py` 调用中添加：

```bash
--memory 300000 \
--vcore 200
```

## ❓ 常见问题

### Q1: 提交失败，提示找不到 hope 命令

**A**: 确保 PATH 包含 hope 命令路径：
```bash
export PATH=/home/sankuai/conda/bin:${PATH}
```

### Q2: 训练失败，提示缺少依赖

**A**: 使用 `--requirement` 参数安装额外依赖，或在训练脚本开头添加：
```bash
pip3 install package_name
```

### Q3: 如何查看训练进度？

**A**: 训练日志会输出到控制台，可以通过 Hope 的日志系统查看。

### Q4: 如何保存训练检查点？

**A**: 修改训练脚本中的 `trainer.save_freq` 参数：
```bash
trainer.save_freq=10  # 每 10 个 epoch 保存一次
```

### Q5: 如何使用不同的模型？

**A**: 修改训练脚本中的 `MODEL_PATH` 变量：
```bash
# 可用模型：
MODEL_PATH=$BASE/VLLM/Qwen/Qwen2.5-VL-3B-Instruct  # 默认使用
MODEL_PATH=$BASE/VLLM/Qwen/Qwen2.5-VL-7B-Instruct  # 更大的模型
MODEL_PATH=$BASE/VLLM/Qwen/Qwen3-VL-2B-Instruct   # Qwen3 系列
```

### Q6: 如何调整 tensor_model_parallel_size？

**A**: 修改训练脚本中的 `actor_rollout_ref.rollout.tensor_model_parallel_size` 参数：
```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=4  # 4 GPU 并行
```

### Q7: 环境安装失败怎么办？

**A**: 训练脚本会在容器中自动安装 ALFWorld 和 WebShop 环境。如果遇到安装失败：
1. 检查 Docker 镜像是否包含必要的依赖（pip、bash 等）
2. 查看日志中的具体错误信息
3. 如需手动调试，可以在本地环境先安装测试

### Q8: 首次运行和后续运行有什么区别？

**A**:
- **首次运行**：会自动下载和安装环境数据（ALFWorld 的 PDDL 文件、WebShop 的产品数据）
- **后续运行**：会跳过已安装的环境，直接开始训练

## 📊 监控和管理

### 查看任务状态

```bash
hope ps  # 查看所有任务
hope ps <job_id>  # 查看特定任务
```

### 停止任务

```bash
hope kill <job_id>
```

### 查看日志

```bash
hope logs <job_id>
hope logs <job_id> -f  # 实时查看
```

## 📚 参考资料

- [verl-agent 官方文档](https://github.com/langfengQ/verl-agent)
- [veRL 官方文档](https://github.com/volcengine/verl)
- [Hope 使用文档](内部文档链接)

## 🤝 贡献

如有问题或建议，请联系：
- MIS ID: chentianyu18
- 项目路径: `/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/verl-agent`
