#!/bin/bash
set -e
source ~/.bashrc 2>/dev/null || true
export PATH=/home/sankuai/conda/bin:$PATH
export http_proxy="http://10.229.18.27:8412"
export https_proxy="http://10.229.18.27:8412"
export PYTHONNOUSERSITE=1

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
WORKDIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/verl-agent
ENV_SEARCH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/../miniconda3/envs/search-cty
ENV_RETRIEVER=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/../miniconda3/envs/retriever-cty
PIP_OPTS="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo "====== Search 环境初始化 ======"

# ========================
# search-cty: Python 3.12，只装 skyrl_gym + gym
# ========================
[ -d "$ENV_SEARCH" ] || conda create --prefix "$ENV_SEARCH" python=3.12 -y

if [ ! -f "$ENV_SEARCH/.step_search" ]; then
    echo "--- install skyrl_gym ---"
    conda run --no-capture-output --prefix "$ENV_SEARCH" pip install \
        -e "$WORKDIR/agent_system/environments/env_package/search/third_party" \
        gym==0.26.2 \
        $PIP_OPTS
    touch "$ENV_SEARCH/.step_search"
fi

if [ ! -f "$ENV_SEARCH/.setup_done" ]; then
    touch "$ENV_SEARCH/.setup_done"
    echo "✅ search-cty 完成"
fi

# ========================
# retriever-cty: Python 3.10，faiss-gpu + retrieval server
# ========================
[ -d "$ENV_RETRIEVER" ] || conda create --prefix "$ENV_RETRIEVER" python=3.10 -y

if [ ! -f "$ENV_RETRIEVER/.step_numpy" ]; then
    conda run --no-capture-output --prefix "$ENV_RETRIEVER" conda install numpy==1.26.4 -y
    touch "$ENV_RETRIEVER/.step_numpy"
fi

if [ ! -f "$ENV_RETRIEVER/.step_torch" ]; then
    echo "--- install torch for retriever ---"
    conda run --no-capture-output --prefix "$ENV_RETRIEVER" pip install \
        torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    touch "$ENV_RETRIEVER/.step_torch"
fi

if [ ! -f "$ENV_RETRIEVER/.step_deps" ]; then
    echo "--- install retriever deps ---"
    conda run --no-capture-output --prefix "$ENV_RETRIEVER" pip install \
        transformers datasets pyserini huggingface_hub uvicorn fastapi \
        $PIP_OPTS
    touch "$ENV_RETRIEVER/.step_deps"
fi

if [ ! -f "$ENV_RETRIEVER/.step_faiss" ]; then
    echo "--- install faiss-gpu (slow) ---"
    conda run --no-capture-output --prefix "$ENV_RETRIEVER" conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
    touch "$ENV_RETRIEVER/.step_faiss"
fi

if [ ! -f "$ENV_RETRIEVER/.setup_done" ]; then
    touch "$ENV_RETRIEVER/.setup_done"
    echo "✅ retriever-cty 完成"
fi

# ========================
# 生成训练数据
# ========================
SEARCH_DATA=$BASE/data/search/searchR1_processed_direct
if [ ! -f "$SEARCH_DATA/train.parquet" ]; then
    echo "--- generate searchR1 data ---"
    mkdir -p "$SEARCH_DATA"
    conda run --no-capture-output --prefix "$ENV_SEARCH" python \
        "$WORKDIR/examples/data_preprocess/preprocess_search_r1_dataset.py" \
        --local_dir "$SEARCH_DATA"
    echo "✅ 训练数据生成完成"
else
    echo "--- search data already exists ---"
fi

unset http_proxy https_proxy PYTHONNOUSERSITE
echo ""
echo "✅ Search 环境初始化完成！"
