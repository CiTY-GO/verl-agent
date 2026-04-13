# env_dump — 线上真实环境依赖快照

本目录包含内网 Hope 集群实际运行实验时，各 conda 环境的完整 pip 包列表。
由 `dump_mirror_env.sh` 在容器内直接 dump，**100% 反映真实运行环境**。

## 内网镜像

```
training_cuda12.6_conda_python3.12_torch2.8_vllm0.11_verl0.6_fa3_mtai_fd0a5390:v1.0.0
```

## 各环境说明

| 文件 | Python | 用途 | 包数量 |
|------|--------|------|--------|
| `pip_freeze_verl-agent.txt` | 3.12.0 | **主训练环境**（ALFWorld / verl PPO 训练） | 230 |
| `pip_freeze_verl_vllm.txt` | 3.10.0 | verl + vllm 完整环境（含 megatron 等） | 441 |
| `pip_freeze_webshop-cty.txt` | 3.10.12 | WebShop 任务环境 | 292 |
| `pip_freeze_vlm-r1.txt` | 3.11.8 | VLM-R1 视觉语言模型环境 | 280 |
| `pip_freeze_retriever-cty.txt` | 3.10.20 | Search 检索服务（faiss + pyserini） | 153 |
| `pip_freeze_search-cty.txt` | 3.12.13 | Search gym 环境（skyrl_gym） | 58 |

## 关键版本（主训练环境 verl-agent）

```
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
vllm==0.8.5
transformers==4.51.1
accelerate==1.13.0
ray==2.54.1
tensordict==0.6.2
peft==0.18.1
liger_kernel==0.7.0
xformers==0.0.29.post2
flash-attn (FA3, 镜像内置)
alfworld==0.4.2
gymnasium==0.29.1
stable_baselines3==2.6.0
wandb==0.25.1
hydra-core==1.3.2
datasets==4.8.4
numpy==2.2.6
pandas==3.0.2
pyarrow==23.0.1
```

## 外网复现方法

```bash
# 1. 创建 Python 3.12 conda 环境
conda create -n verl-agent python=3.12 -y
conda activate verl-agent

# 2. 安装 PyTorch（CUDA 12.4 wheel，兼容 12.6）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. 安装 vLLM
pip install vllm==0.8.5

# 4. 按 pip_freeze 安装剩余依赖
pip install -r env_dump/pip_freeze_verl-agent.txt

# 5. 安装项目本身
pip install -e .
```

> **注意**：`flash-attn` 在内网镜像中预装，外网需单独安装：
> ```bash
> MAX_JOBS=4 pip install flash-attn --no-build-isolation
> ```
