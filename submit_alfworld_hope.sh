#!/bin/bash
set -x

# ============================================================
# verl-agent ALFWorld 训练 Hope 提交脚本 (镜像版本)
# ============================================================

# 使用方法:
# bash submit_alfworld.sh [gpu_num] [worker_num]
# bash submit_alfworld.sh 2 1  # 2 GPU, 1 worker (默认)

# ============================================================
# 参数解析
# ============================================================

# 使用 conda shell.bash hook（与参考脚本一致）
eval "$(conda shell.bash hook)"

GPU_NUM=${1:-2}           # GPU 数量
WORKER=${2:-1}            # Worker 数量
MIS_ID=chentianyu18        # MIS ID
PRIORITY=1                 # 优先级
MAX_RETRY=1                # 最大重试次数

# 目录配置
BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
HOPE_DIR=$VERL_AGENT/hope_files
CREATE_HOPE_SCRIPT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/hotel/verl0.6.1/utils/create_hope_verl.py

# 创建 hope 目录
mkdir -p $HOPE_DIR

# ============================================================
# 生成 Hope 配置文件
# ============================================================
echo "=== 生成 Hope 配置文件 ==="

HOPE_FILE=$HOPE_DIR/chentianyu18_alfworld_mirror${GPU_NUM}gpus_$(date +%Y%m%d_%H%M%S).hope

# 训练命令：直接调用 run_alfworld_hope.sh（使用镜像自带的 verl-agent conda 环境）
# env.resources_per_worker.num_cpus=0.1：每个 AlfworldWorker 占 0.1 CPU
# train_batch_size=16 * group_size=8 = 128个并行环境，需要 128*0.1=12.8 CPU
TRAIN_CMD="export RAY_memory_monitor_refresh_ms=0 && \
export RAY_memory_usage_threshold=1.0 && \
export RAY_DISABLE_MEMORY_MONITOR=1 && \
export RAY_DISABLE_DASHBOARD=1 && \
export OPENBLAS_NUM_THREADS=1 && \
export OMP_NUM_THREADS=1 && \
cd ${VERL_AGENT} && bash run_alfworld_hope.sh vllm ${GPU_NUM}"

# 使用 create_hope_verl.py 生成配置文件
# vcore=88：给容器分配足够的 CPU（Ray 需要 80 CPU，额外留余量）
# memory=150000：相应增加内存
python3 ${CREATE_HOPE_SCRIPT} \
    --worker ${WORKER} \
    --gpu_num ${GPU_NUM} \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}" \
    --board_dir "${BASE}/experiments/tensorboard" \
    --max_retry ${MAX_RETRY} \
    --vcore 88 \
    --memory 150000

echo "=== Hope 配置文件已生成: ${HOPE_FILE} ==="

# ============================================================
# 显示生成的 Hope 配置
# ============================================================
echo "=== Hope 配置文件内容 ==="
cat ${HOPE_FILE}

# ============================================================
# 提交到 Hope 集群
# ============================================================
echo "=== 提交到 Hope 集群 ==="
echo "GPU 数量: ${GPU_NUM}"
echo "Worker 数量: ${WORKER}"
echo "优先级: P${PRIORITY}"
echo "使用镜像: training_cuda12.6_conda_python3.12_torch2.8_vllm0.11_verl0.6_fa3_mtai"

# 设置 Hope 登录模式
export HOPE_LOGIN_MODE=dx_confirm

# 登录并提交
hope login ${MIS_ID}
hope run ${HOPE_FILE} -Dmlp.sche.priority=P${PRIORITY} --files=${VERL_AGENT}

echo "=== 提交完成 ==="
echo "Hope 文件: ${HOPE_FILE}"
