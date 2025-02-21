export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export HF_DATASETS_CACHE="/workspace/model_cache/datasets"
export HF_HOME="/workspace/model_cache"
# Use token from environment variable - DO NOT hardcode tokens
if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

torchrun --nproc_per_node=1 \
    --master_port="12345" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir "./checkpoints/grpo_clevr_2b" \
    --model_name_or_path "bytedance-research/UI-TARS-7B-DPO" \
    --dataset_name "leonardPKU/clevr_cogen_a_train" \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --deepspeed scripts/zero3.json