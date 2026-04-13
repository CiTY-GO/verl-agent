#!/bin/bash
set -x

eval "$(conda shell.bash hook)"

GPU_NUM=${1:-2}
WORKER=${2:-1}
MIS_ID=chentianyu18
PRIORITY=1
MAX_RETRY=1

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
HOPE_DIR=$VERL_AGENT/hope_files
CREATE_HOPE_SCRIPT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/hotel/verl0.6.1/utils/create_hope_verl.py

mkdir -p $HOPE_DIR

HOPE_FILE=$HOPE_DIR/chentianyu18_webshop_${GPU_NUM}gpus_$(date +%Y%m%d_%H%M%S).hope

TRAIN_CMD="export RAY_memory_monitor_refresh_ms=0 && \
export RAY_memory_usage_threshold=1.0 && \
export RAY_DISABLE_MEMORY_MONITOR=1 && \
export RAY_DISABLE_DASHBOARD=1 && \
export OPENBLAS_NUM_THREADS=1 && \
export OMP_NUM_THREADS=1 && \
cd ${VERL_AGENT} && bash run_webshop_hope.sh vllm ${GPU_NUM}"

python3 ${CREATE_HOPE_SCRIPT} \
    --worker ${WORKER} \
    --gpu_num ${GPU_NUM} \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}" \
    --board_dir "${BASE}/experiments/tensorboard" \
    --max_retry ${MAX_RETRY} \
    --vcore 88 \
    --memory 400000

echo "=== Hope 配置文件内容 ==="
cat ${HOPE_FILE}

echo "=== 提交到 Hope 集群 ==="
export HOPE_LOGIN_MODE=dx_confirm
hope login ${MIS_ID}
hope run ${HOPE_FILE} -Dmlp.sche.priority=P${PRIORITY} --files=${VERL_AGENT}

echo "=== 提交完成 ==="
