#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "usage: $0 grpo|gspo [extra Hydra overrides...]" >&2
  exit 2
fi
ALGO=$1; shift
[[ $ALGO == grpo || $ALGO == gspo ]] || { echo "algorithm must be grpo or gspo" >&2; exit 2; }

PY=${PY:-/home/zhangzongyuan/anaconda3/envs/radargym-rl/bin/python}
ROOT=/home/zhangzongyuan/Myproject/drivelm/DriveLM-main
HERE=$ROOT/reproduction/qwen_vl_v040_trajectory_rl
DATA=/mnt/data/zzy/drivelm/reproduction/qwen_vl_v040_trajectory_rl/dataset
BASE=/mnt/data/zzy/drivelm/models/Qwen2.5-VL-7B-Instruct
ADAPTER=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v039b-mol-sweep/expert_planning/checkpoint-700
OUT=/mnt/data/zzy/drivelm/models/qwen2.5-vl-7b-drivelm-v040-trajectory-${ALGO}
POLICY_LOSS=vanilla
[[ $ALGO == gspo ]] && POLICY_LOSS=gspo

# Physical indices pin the three RTX 5090s and deliberately exclude the A6000.
# vLLM 0.15 resolves visible devices with integer parsing and rejects UUID strings.
export CUDA_VISIBLE_DEVICES=0,2,3
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# Full-dataset preflight: max text prompt is 789 tokens. Six images are each
# capped at 100352 px (=128 merged Qwen-VL tokens), so the conservative
# multimodal upper bound is about 1557 tokens, safely below 3072. Disabling
# VERL's duplicate filter avoids its multiprocessing image-tensor RAM leak.
"$PY" "$HERE/run_verl_main.py" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/train.parquet" \
  data.val_files="$DATA/dev.parquet" \
  data.image_key=images \
  data.train_batch_size=12 \
  data.val_batch_size=12 \
  data.max_prompt_length=3072 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  +data.mm_processor_kwargs.min_pixels=25088 \
  +data.mm_processor_kwargs.max_pixels=100352 \
  actor_rollout_ref.model.path="$BASE" \
  actor_rollout_ref.model.lora_adapter_path="$ADAPTER" \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.policy_loss.loss_mode="$POLICY_LOSS" \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  +actor_rollout_ref.rollout.limit_images=6 \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_pixels=25088 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels=100352 \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=0.8 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$HERE/trajectory_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name=radarmind_drivelm_v040 \
  trainer.experiment_name="trajectory_${ALGO}" \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=3 \
  trainer.nnodes=1 \
  trainer.total_training_steps=100 \
  trainer.total_epochs=20 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.val_before_train=True \
  trainer.default_local_dir="$OUT" \
  "$@"
