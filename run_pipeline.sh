#!/bin/bash
# Pipeline: download + extract models sequentially on USTC 107 platform
# Usage: bash run_pipeline.sh
set -euo pipefail

ENV=~/miniconda3/envs/ttt/bin/python3
DIR=~/tokens-to-thoughts-reproduce
LOG=/tmp/pipeline.log

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="__HF_TOKEN_HERE__"

cd "$DIR"
mkdir -p models_embeddings

run_model() {
  local model="$1"
  local name=$(echo "$model" | sed 's|.*/||')
  echo -e "\n====== $model ======" | tee -a "$LOG"
  date | tee -a "$LOG"
  $ENV extract.py --model "$model" 2>&1 | tee -a "$LOG"
  echo "DONE: $model" | tee -a "$LOG"
  ls -lh "models_embeddings/${name}_static.npy" 2>/dev/null || echo "WARNING: No output for $model"
}

# Smallest first
run_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
run_model "google/gemma-2-9b"
run_model "meta-llama/Meta-Llama-3.1-8B"
run_model "Qwen/Qwen2.5-32B"

echo -e "\n====== ALL DONE ======" | tee -a "$LOG"
date | tee -a "$LOG"