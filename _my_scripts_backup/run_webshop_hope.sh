#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
DATA_DIR=$BASE/data/verl-agent/text
MODEL_PATH=$BASE/VLLM/Qwen/Qwen2.5-VL-3B-Instruct

# ============================================================
# Step 1: 准备数据（如果不存在）
# ============================================================
if [ ! -d "$DATA_DIR" ]; then
    echo "=== 准备 WebShop 数据 ==="
    mkdir -p $DATA_DIR

    cd $VERL_AGENT
    python3 prepare_agent_data.py \
        --output_dir $DATA_DIR \
        --train_size 16 \
        --val_size 128
    echo "=== 数据准备完成 ==="
else
    echo "=== WebShop 数据已存在，跳过 ==="
fi

ls -la $DATA_DIR/

# ============================================================
# Step 2: 安装 WebShop 环境（如果未安装）
# ============================================================
echo "=== 检查 WebShop 环境 ==="
WEBSHOP_DIR=$VERL_AGENT/agent_system/environments/env_package/webshop/webshop

if ! python3 -c "import web_agent_site" 2>/dev/null; then
    echo "=== 安装 WebShop 环境 ==="
    pip3 install gymnasium==0.29.1
    pip3 install stable-baselines3==2.6.0
    pip3 install vllm==0.8.5

    # 安装 WebShop
    cd $WEBSHOP_DIR
    # 尝试下载小数据集（更快）
    bash setup.sh -d small 2>/dev/null || echo "WebShop setup 失败或已安装，跳过"

    # 返回 verl-agent 目录
    cd $VERL_AGENT

    echo "=== WebShop 环境安装完成 ==="
else
    echo "=== WebShop 环境已安装，跳过 ==="
fi

# ============================================================
# Step 3: 启动训练
# ============================================================
echo "=== 开始 WebShop GiGPO 训练 ==="
cd $VERL_AGENT

ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=16
val_data_size=128
group_size=8
mode="mean_std_norm"

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
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=webshop/WebShopEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gigpo_qwen2.5_vl_3b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True
