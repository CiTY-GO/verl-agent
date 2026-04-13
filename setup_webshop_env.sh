#!/bin/bash
set -e
source ~/.bashrc 2>/dev/null || true
export PATH=/home/sankuai/conda/bin:$PATH
export http_proxy="http://10.229.18.27:8412"
export https_proxy="http://10.229.18.27:8412"
export PYTHONNOUSERSITE=1   # 屏蔽 ~/.local，避免旧包污染

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
WORKDIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/verl-agent
WEBSHOP_PKG=$WORKDIR/agent_system/environments/env_package/webshop/webshop
ENV=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/../miniconda3/envs/webshop-cty
PIP_OPTS="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"

echo "====== WebShop 环境初始化 ======"

# Step 1: 创建 Python 3.10 环境
[ -d "$ENV" ] || conda create --prefix "$ENV" python=3.10 -y

# Step 2: conda 依赖（索引构建需要）
if [ ! -f "$ENV/.step_conda_deps" ]; then
    echo "--- conda deps ---"
    conda run --no-capture-output --prefix "$ENV" conda install mkl -y
    conda run --no-capture-output --prefix "$ENV" conda install -c conda-forge faiss-cpu openjdk=11 -y
    touch "$ENV/.step_conda_deps"
fi

# Step 3: pip 依赖（WebShop requirements，固定 pyarrow 避免冲突）
if [ ! -f "$ENV/.step_pip_deps" ]; then
    echo "--- pip deps ---"
    conda run --no-capture-output --prefix "$ENV" pip install \
        regex itsdangerous gymnasium==0.29.1 stable-baselines3==2.6.0 \
        "pyarrow>=14.0" \
        -r "$WEBSHOP_PKG/requirements.txt" \
        $PIP_OPTS
    touch "$ENV/.step_pip_deps"
fi

# Step 4: spacy 模型（代理下载 whl）
if [ ! -f "$ENV/.step_spacy_models" ]; then
    echo "--- spacy models ---"
    conda run --no-capture-output --prefix "$ENV" pip install \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
    touch "$ENV/.step_spacy_models"
fi

# Step 5: 拷贝数据
if [ ! -d "$WEBSHOP_PKG/data" ] || [ -z "$(ls -A $WEBSHOP_PKG/data 2>/dev/null)" ]; then
    echo "--- copy data ---"
    mkdir -p "$WEBSHOP_PKG/data"
    cp -r "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/fangyangyi/data/agent/webshop/"* "$WEBSHOP_PKG/data/"
    echo "✅ 数据已拷贝: $(ls $WEBSHOP_PKG/data/)"
else
    echo "--- data already exists ---"
fi

# Step 6: 建 Lucene 索引
SEARCH_ENGINE=$WEBSHOP_PKG/search_engine
if [ ! -d "$SEARCH_ENGINE/indexes" ] || [ -z "$(ls -A $SEARCH_ENGINE/indexes 2>/dev/null)" ]; then
    echo "--- build lucene index (10-20 min) ---"
    mkdir -p "$SEARCH_ENGINE/resources" "$SEARCH_ENGINE/resources_100" \
             "$SEARCH_ENGINE/resources_1k" "$SEARCH_ENGINE/resources_100k" \
             "$SEARCH_ENGINE/indexes"
    cd "$SEARCH_ENGINE"
    conda run --no-capture-output --prefix "$ENV" python convert_product_file_format.py
    conda run --no-capture-output --prefix "$ENV" bash run_indexing.sh
    cd "$WORKDIR"
    echo "✅ 索引建立完成"
else
    echo "--- indexes already exist ---"
fi

unset http_proxy https_proxy PYTHONNOUSERSITE
echo ""
echo "✅ WebShop 环境初始化完成！"
