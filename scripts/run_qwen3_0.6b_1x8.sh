#!/usr/bin/env bash
set -euo pipefail

# Run Qwen-3-0.6B on 1 node with 8 GPUs using `accelerate launch`.
# Requirements: python, torch with CUDA, accelerate, transformers, safetensors.
# Example install:
#   pip install --upgrade pip
#   pip install torch --index-url https://download.pytorch.org/whl/cu126  # pick your CUDA
#   pip install accelerate transformers safetensors
#
# Usage:
#   ./scripts/run_qwen3_0.6b_1x8.sh --model "Qwen/Qwen-3-0.6B" --prompt "Hello" --max_new_tokens 128

MODEL="Qwen/Qwen-3-0.6B"
PROMPT="Hello, Qwen!"
CTX_LEN=8192
MAX_NEW_TOKENS=128
TEMPERATURE=0.0

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) MODEL="$2"; shift 2 ;;
    --prompt) PROMPT="$2"; shift 2 ;;
    --ctx_len) CTX_LEN="$2"; shift 2 ;;
    --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --save-tokenizer) SAVE_TOKENIZER=1; shift 1 ;;
    --profile) PROFILE=1; shift 1 ;;
    -h|--help) sed -n '1,240p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12355}

# NCCL tuning for single-node multi-GPU
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=OFF

echo "Launching Qwen inference with model=${MODEL} ctx_len=${CTX_LEN}"

if [ "${PROFILE:-0}" == "1" ]; then
  echo "Profile mode requested — running profiler for Qwen (small)."
  # run profiler wrapper which will produce profiled-time-miniset entries
  bash /home/cyn/projects/SuperScalerFor5090/profiler/scripts/profile_small_qwen.sh
  exit 0
fi

if [ "${SAVE_TOKENIZER:-0}" == "1" ]; then
  echo "Saving tokenizer locally to runtime/vocabs/qwen ..."
  python3 /home/cyn/projects/SuperScalerFor5090/scripts/fetch_qwen_tokenizer.py --model "$MODEL" --out "/home/cyn/projects/SuperScalerFor5090/runtime/vocabs/qwen"
  echo "Tokenizer saved. You can now run the execute wrapper which will pick up vocabs/qwen automatically."
  exit 0
fi

# Use accelerate launch to let transformers/accelerate handle sharding/device_map='auto'.
# If you prefer `torchrun`, you can adapt this to `torchrun --nproc_per_node=8`.
accelerate launch --num_processes 8 --num_machines 1 \
  /home/cyn/projects/SuperScalerFor5090/scripts/run_qwen_infer.py \
  --model_name_or_path "$MODEL" \
  --ctx_len "$CTX_LEN" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE"
