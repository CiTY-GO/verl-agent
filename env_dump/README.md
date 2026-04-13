# env_dump — 线上真实运行环境依赖快照

本目录包含内网 Hope 集群**实际运行实验**时各 conda 环境的完整 pip 包列表，
直接从 `miniconda3/envs/` 下各环境的 pip binary 中 dump，**100% 反映真实运行环境**。

---

## 内网使用镜像

```
training_cuda12.6_conda_python3.12_torch2.8_vllm0.11_verl0.6_fa3_mtai_fd0a5390:v1.0.0
```

---

## 各任务实际使用的环境

通过分析 `run_alfworld_hope.sh` / `run_search_hope.sh` / `run_webshop_hope.sh` 得到：

### ALFWorld 训练（`submit_alfworld_hope.sh`）

| 组件 | 使用环境 | 对应文件 |
|------|---------|---------|
| 主训练进程（verl PPO） | 镜像内置 conda env（`python3`） | `pip_freeze_verl-agent.txt` |
| ALFWorld 游戏环境 | 同上，运行时 `pip3 install alfworld` | 同上 |

### Search 训练（`submit_search_hope.sh`）

| 组件 | 使用环境 | 对应文件 |
|------|---------|---------|
| 主训练进程（verl PPO） | 镜像内置 conda env（`python3`） | `pip_freeze_verl-agent.txt` |
| Retrieval Server（faiss + e5） | `retriever-cty`（`${ENV_RETRIEVER}/bin/python`） | `pip_freeze_retriever-cty.txt` |
| skyrl_gym（search gym） | 镜像内置，运行时 `pip3 install` | `pip_freeze_verl-agent.txt` |

### WebShop 训练（`submit_webshop_hope.sh`）

| 组件 | 使用环境 | 对应文件 |
|------|---------|---------|
| **全部**（主训练 + WebShop 环境） | `webshop-cty`（`${ENV_WEBSHOP}/bin/python3`） | `pip_freeze_webshop-cty.txt` |

> WebShop 是唯一一个**完全用独立 conda 环境**运行的任务，包括 verl 主训练进程也走 `webshop-cty`。

### dump_mirror_env（`submit_dump_env.sh`）

| 组件 | 使用环境 |
|------|---------|
| pip freeze / conda list | 镜像默认 `python`（基础系统层，非训练环境） |

> ⚠️ 注意：`dump_mirror_env.sh` dump 出的是镜像系统层的包，**不是**训练用的 conda 环境。
> 本目录的文件是直接从 `miniconda3/envs/*/bin/pip freeze` 得到的，才是真实训练环境。

---

## 文件说明

| 文件 | Python 版本 | 用途 | 包数量 |
|------|------------|------|--------|
| `pip_freeze_verl-agent.txt` | **3.12.0** | **ALFWorld / Search 主训练环境**（镜像内置） | 230 |
| `pip_freeze_webshop-cty.txt` | 3.10.12 | **WebShop 全环境**（含 verl 训练 + WebShop 仿真） | 292 |
| `pip_freeze_retriever-cty.txt` | 3.10.20 | Search Retrieval Server（faiss-gpu + e5） | 153 |
| `pip_freeze_search-cty.txt` | 3.12.13 | Search gym（skyrl_gym，轻量） | 58 |
| `pip_freeze_verl_vllm.txt` | 3.10.0 | verl + vllm 完整备用环境 | 441 |
| `pip_freeze_vlm-r1.txt` | 3.11.8 | VLM-R1 视觉语言模型环境 | 280 |

---

## 关键版本（主训练环境 `verl-agent`，Python 3.12）

```
# 框架核心
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
vllm==0.8.5
xformers==0.0.29.post2
flash-attn  # FA3，镜像内置，未出现在 pip freeze 中

# verl 生态
transformers==4.51.1
accelerate==1.13.0
peft==0.18.1
liger_kernel==0.7.0
tensordict==0.6.2
ray==2.54.1

# 任务环境
alfworld==0.4.2
gymnasium==0.29.1
stable_baselines3==2.6.0

# 数据 & 训练工具
datasets==4.8.4
wandb==0.25.1
hydra-core==1.3.2
numpy==2.2.6
pandas==3.0.2
pyarrow==23.0.1
```

---

## 外网复现方法

### ALFWorld / Search 主训练环境

```bash
conda create -n verl-agent python=3.12 -y
conda activate verl-agent

# PyTorch（CUDA 12.4 wheel，兼容 12.6）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# vLLM
pip install vllm==0.8.5

# Flash Attention（镜像内置，外网需手动编译）
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 其余依赖（按 pip_freeze 精确安装）
pip install -r env_dump/pip_freeze_verl-agent.txt --ignore-installed torch torchvision torchaudio vllm

# 安装项目
pip install -e .
```

### WebShop 环境

```bash
conda create -n webshop-cty python=3.10 -y
conda activate webshop-cty

# conda 依赖（Lucene 索引需要 Java）
conda install mkl -y
conda install -c conda-forge faiss-cpu openjdk=11 -y

# pip 依赖
pip install -r env_dump/pip_freeze_webshop-cty.txt
```

### Search Retrieval Server 环境

```bash
conda create -n retriever-cty python=3.10 -y
conda activate retriever-cty

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install -r env_dump/pip_freeze_retriever-cty.txt --ignore-installed torch torchvision
```

---

## 模型

```bash
# 主模型：Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# Search 检索模型：e5-base-v2
huggingface-cli download intfloat/e5-base-v2 --local-dir ./models/e5-base-v2
```
