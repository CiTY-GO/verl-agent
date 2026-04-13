#!/bin/bash
set -x

eval "$(conda shell.bash hook)"

MIS_ID=chentianyu18
BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
HOPE_DIR=$VERL_AGENT/hope_files
CREATE_HOPE_SCRIPT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/hotel/verl0.6.1/utils/create_hope_verl.py

mkdir -p $HOPE_DIR

HOPE_FILE=$HOPE_DIR/dump_env_$(date +%Y%m%d_%H%M%S).hope

TRAIN_CMD="cd ${VERL_AGENT} && bash dump_mirror_env.sh"

python3 ${CREATE_HOPE_SCRIPT} \
    --worker 1 \
    --gpu_num 1 \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}" \
    --board_dir "${BASE}/data/verl-agent/tensorboard" \
    --max_retry 1 \
    --vcore 22 \
    --memory 40000

echo "=== Hope 配置文件内容 ==="
cat ${HOPE_FILE}

export HOPE_LOGIN_MODE=dx_confirm
hope login ${MIS_ID}
hope run ${HOPE_FILE} -Dmlp.sche.priority=P1 --files=${VERL_AGENT}

echo "=== 提交完成，结果将保存到: ${BASE}/mirror_env_dump/ ==="
