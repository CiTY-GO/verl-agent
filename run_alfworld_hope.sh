#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535
# 增加进程/线程限制，解决 OpenBLAS 线程创建失败问题
ulimit -u unlimited
ulimit -s unlimited
# 限制 OpenBLAS 线程数，避免创建过多线程
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
DATA_DIR=$BASE/data/verl-agent/text
MODEL_PATH=$BASE/models/Qwen2.5-1.5B-Instruct

# ALFWorld 游戏数据路径（存放在 dolphinfs 共享存储，容器可直接访问）
export ALFWORLD_DATA=$BASE/data/alfworld/game_data

# ============================================================
# Step 1: 准备数据（如果不存在）
# ============================================================
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "=== 准备 verl-agent parquet 数据 ==="
    mkdir -p $DATA_DIR
    cd $VERL_AGENT
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size 16 \
        --val_data_size 128
    echo "=== 数据准备完成 ==="
else
    echo "=== verl-agent 数据已存在，跳过 ==="
fi

ls -la $DATA_DIR/

# ============================================================
# Step 2: 安装 ALFWorld 环境（如果未安装）
# ============================================================
echo "=== 检查 ALFWorld 环境 ==="
if ! python3 -c "import alfworld" 2>/dev/null; then
    echo "=== 安装 ALFWorld Python 包 ==="
    pip3 install gymnasium==0.29.1
    pip3 install stable-baselines3==2.6.0
    pip3 install alfworld
    echo "=== ALFWorld 环境安装完成 ==="
else
    echo "=== ALFWorld 环境已安装，跳过 ==="
fi

# 验证 ALFWORLD_DATA 数据目录可访问
echo "=== 验证 ALFWORLD_DATA: $ALFWORLD_DATA ==="
if [ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]; then
    echo "错误: ALFWorld 数据目录不存在: $ALFWORLD_DATA/json_2.1.1/train"
    exit 1
fi
echo "=== ALFWorld 数据验证通过，train games: $(ls $ALFWORLD_DATA/json_2.1.1/train | wc -l) ==="

# ============================================================
# Step 3: 启动训练
# ============================================================
echo "=== 开始 ALFWorld GiGPO 训练 ==="
cd $VERL_AGENT
export PYTHONPATH=$VERL_AGENT:$PYTHONPATH
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/experiments/checkpoints/alfworld/gigpo_alfworld_1.5b_${TIMESTAMP}
mkdir -p $CKPT_DIR
echo "=== Checkpoint 目录: $CKPT_DIR ==="
TB_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/experiments/tensorboard/${TIMESTAMP}
mkdir -p $TB_DIR
echo "=== TensorBoard 目录: $TB_DIR ==="

ENGINE=${1:-vllm}
GPU_NUM=${2:-2}
# export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================
# Step 4: 启动 GPU 持续计算（防止因显存利用率低被 kill）
# ============================================================
echo "=== 启动 GPU 持续计算任务 ==="
GPU_KEEPER_SCRIPT=$BASE/gpu_continuous_compute.py
python3 ${GPU_KEEPER_SCRIPT} --gpu_number ${GPU_NUM} --duration 0 > /tmp/gpu_keeper_${TIMESTAMP}.log 2>&1 &
GPU_KEEPER_PID=$!
echo "GPU 持续计算任务 PID: $GPU_KEEPER_PID"
sleep 2  # 等待 GPU 占用任务启动

# Ray 内存配置，防止 OOM
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
export RAY_DISABLE_MEMORY_MONITOR=1
# 禁用 Ray Dashboard（避免 OpenTelemetry 兼容性问题）
export RAY_DISABLE_DASHBOARD=1

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"

# num_cpus_per_env_worker:
# 每个 AlfworldWorker 占 0.1 个 CPU（参考 fangyangyi 的配置）
# train_batch_size * group_size = 16 * 8 = 128 个并发环境 → 共占 12.8 CPU，资源足够
num_cpus_per_env_worker=0.1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
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
    actor_rollout_ref.rollout.name=$ENGINE \
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
    actor_rollout_ref.actor.checkpoint.contents="['model','hf_model','optimizer','extra']" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name='gigpo_alfworld_1.5b' \
    trainer.default_local_dir=${TB_DIR} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True

# 训练结束后停止 GPU 持续计算任务
echo "=== 停止 GPU 持续计算任务 ==="
kill $GPU_KEEPER_PID 2>/dev/null || true
echo "=== 训练完成 ==="
