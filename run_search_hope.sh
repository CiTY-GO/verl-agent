#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535
ulimit -u unlimited
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

BASE=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research
VERL_AGENT=$BASE/verl-agent
TRAIN_DATA=$BASE/data/search/train_data/searchR1_processed_direct/train.parquet
VAL_DATA=$BASE/data/search/train_data/searchR1_processed_direct/test.parquet
MODEL_PATH=$BASE/models/Qwen2.5-1.5B-Instruct
ENV_RETRIEVER=$BASE/../miniconda3/envs/retriever-cty
E5_MODEL=$BASE/models/e5-base-v2
SEARCH_INDEX=$BASE/data/search/faiss_index/e5_Flat.index
SEARCH_CORPUS=$BASE/data/search/faiss_index/wiki-18.jsonl

cd $VERL_AGENT
export PYTHONPATH=$VERL_AGENT:$PYTHONPATH

# 安装 skyrl_gym（Search 环境，镜像里没有）
if ! python3 -c "import skyrl_gym" 2>/dev/null; then
    pip3 install -e $VERL_AGENT/agent_system/environments/env_package/search/third_party \
        -i http://pip.sankuai.com/simple --trusted-host pip.sankuai.com --quiet
    pip3 install gym==0.26.2 \
        -i http://pip.sankuai.com/simple --trusted-host pip.sankuai.com --quiet
fi

# Ray 内存配置
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=1.0
export RAY_DISABLE_MEMORY_MONITOR=1
export RAY_DISABLE_DASHBOARD=1

# ============================================================
# Step 1: 后台启动 retrieval server（用 retriever-cty 环境）
# ============================================================
echo "=== 启动 retrieval server ==="
${ENV_RETRIEVER}/bin/python examples/search/retriever/retrieval_server.py \
    --index_path $SEARCH_INDEX \
    --corpus_path $SEARCH_CORPUS \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model $E5_MODEL \
    --faiss_gpu \
    --port 8000 > /tmp/retrieval_server.log 2>&1 &
RETRIEVER_PID=$!
echo "retrieval server PID: $RETRIEVER_PID"

# 等待 server 启动（最多 120 秒）
echo "=== 等待 retrieval server 就绪 ==="
for i in $(seq 1 24); do
    if curl -s http://127.0.0.1:8000/retrieve -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "topk": 1}' > /dev/null 2>&1; then
        echo "✅ retrieval server 已就绪（${i}次检查）"
        break
    fi
    echo "等待中... (${i}/24)"
    sleep 5
done

# ============================================================
# Step 2: 生成时间戳 checkpoint 目录
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR=$BASE/experiments/checkpoints/search/gigpo_search_1.5b_${TIMESTAMP}
mkdir -p $CKPT_DIR
echo "=== Checkpoint 目录: $CKPT_DIR ==="
TB_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/chentianyu18/research/experiments/tensorboard/${TIMESTAMP}
mkdir -p $TB_DIR
echo "=== TensorBoard 目录: $TB_DIR ==="

ENGINE=${1:-vllm}
GPU_NUM=${2:-2}

# ============================================================
# Step 3: 启动 GPU 持续计算（防止因显存利用率低被 kill）
# ============================================================
echo "=== 启动 GPU 持续计算任务 ==="
GPU_KEEPER_SCRIPT=$BASE/gpu_continuous_compute.py
python3 ${GPU_KEEPER_SCRIPT} --gpu_number ${GPU_NUM} --duration 0 > /tmp/gpu_keeper_${TIMESTAMP}.log 2>&1 &
GPU_KEEPER_PID=$!
echo "GPU 持续计算任务 PID: $GPU_KEEPER_PID"
sleep 2  # 等待 GPU 占用任务启动

train_data_size=64
val_data_size=128
group_size=5
mode="mean_std_norm"
enable_similarity=True
similarity_thresh=0.9

# ============================================================
# Step 3: 启动训练
# ============================================================
echo "=== 开始 Search GiGPO 训练 ==="
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    actor_rollout_ref.actor.checkpoint.contents="['model','hf_model','optimizer','extra']" \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.gigpo.enable_similarity=$enable_similarity \
    algorithm.gigpo.similarity_thresh=$similarity_thresh \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name='gigpo_search_1.5b' \
    trainer.default_local_dir=${TB_DIR} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 \
    trainer.val_before_train=False

# 训练结束后停止 GPU 持续计算任务和 retrieval server
echo "=== 停止 GPU 持续计算任务 ==="
kill $GPU_KEEPER_PID 2>/dev/null || true
echo "=== 停止 retrieval server ==="
kill $RETRIEVER_PID 2>/dev/null || true
echo "=== 训练完成 ==="
