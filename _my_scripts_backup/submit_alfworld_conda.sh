#!/bin/bash
set -x

# ============================================================
# verl-agent ALFWorld 训练 Hope 提交脚本
# ============================================================

# 使用方法:
# bash submit_alfworld_hope.sh [gpu_num] [worker_num]
# bash submit_alfworld_hope.sh 2 1  # 2 GPU, 1 worker (默认)

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
DATA_DIR=$BASE/data/verl-agent/text
MODEL_PATH=$BASE/VLLM/Qwen/Qwen2.5-1.5B-Instruct

# Conda 环境路径
CONDA_ENV="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/miniconda3/envs/verl-agent"

# 创建 hope 目录
mkdir -p $HOPE_DIR

# ============================================================
# 生成 Hope 配置文件
# ============================================================
echo "=== 生成 Hope 配置文件 ==="

HOPE_FILE=$HOPE_DIR/chentianyu18_alfworld_gigpo_${GPU_NUM}gpus_$(date +%Y%m%d_%H%M%S).hope

# 训练命令：通过 PATH 指定 conda 环境（不使用 conda activate）
TRAIN_CMD="export PATH=${CONDA_ENV}/bin:\$PATH && \
export LD_LIBRARY_PATH=${CONDA_ENV}/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-12.4/targets/x86_64-linux/lib:\$LD_LIBRARY_PATH && \
export PYTHONPATH=${VERL_AGENT}:\$PYTHONPATH && \
cd ${VERL_AGENT} && \
export HYDRA_FULL_ERROR=1 && \
ulimit -n 65535 && \
export VLLM_ATTENTION_BACKEND=XFORMERS && \
export RAY_memory_monitor_refresh_ms=0 && \
export RAY_memory_usage_threshold=1.0 && \
export RAY_DISABLE_MEMORY_MONITOR=1 && \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_std_norm \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    trainer.critic_warmup=0 \
    trainer.logger='[\"console\"]' \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    ray_init.num_cpus=32"

# 使用 create_hope.py 生成配置文件（使用基础镜像）
python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/create_hope.py \
    --worker ${WORKER} \
    --gpu_num ${GPU_NUM} \
    --mis_id ${MIS_ID} \
    --script_path "${TRAIN_CMD}" \
    --save_path "${HOPE_FILE}"

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

# 设置 Hope 登录模式
export HOPE_LOGIN_MODE=dx_confirm

# 登录并提交
hope login ${MIS_ID}
hope run ${HOPE_FILE} -Dmlp.sche.priority=P${PRIORITY} --files=${VERL_AGENT}

echo "=== 提交完成 ==="
echo "Hope 文件: ${HOPE_FILE}"
