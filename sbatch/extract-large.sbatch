#!/bin/bash
#SBATCH -J extract-A_school_large
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH -t 03:50:00
#SBATCH -o logs/extract-large-%j.out
#SBATCH -e logs/extract-large-%j.err

set -euo pipefail

# ===== 环境 =====
cd $HOME/tokens-to-thoughts-reproduce

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttt
set -u

# ===== 设置 HF 镜像（加速下载）=====
export HF_ENDPOINT=https://hf-mirror.com
# 如需访问 Llama/Gemma 等需要登录的模型，取消注释下一行并填入你的 HF token
# export HF_TOKEN=hf_xxxxxxxxxxxx

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ===== 提取 70B/72B 大模型（需要大量内存）=====
# 逐个提取，大模型可能需要更长时间
python extract.py --model meta-llama/Meta-Llama-3.1-70B
python extract.py --model Qwen/Qwen2.5-72B

echo "=== A_school large models extract done ==="

echo ""
echo "=== Job finished: $(date) ==="