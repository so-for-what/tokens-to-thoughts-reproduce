#!/bin/bash
#SBATCH -J extract-A_school
#SBATCH -p Students
#SBATCH --qos=qos_stu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -t 23:00:00
#SBATCH -o logs/extract-%j.out
#SBATCH -e logs/extract-%j.err

set -euo pipefail

# ===== 环境 =====
cd $HOME/tokens-to-thoughts-reproduce

set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ttt
set -u

# ===== 设置 HF 镜像（加速下载）=====
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=__FILL_YOUR_HF_TOKEN_HERE__

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo ""

# ===== 提取 A_school 模型（从大到小，逐一下载+提取）=====
# ✅ 已完成（已提交 GitHub）:
#   - Qwen2.5-14B
#   - Qwen2.5-32B
#   - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
#   - google/gemma-2-9b

# 1. Llama-3.1-8B
python extract.py --model meta-llama/Meta-Llama-3.1-8B

# 2. DeepSeek-R1-Distill-Qwen-32B（新的 32B 模型）
python extract.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

echo "=== A_school extract done ==="

# 70B 和 72B 模型需要更多资源，单独提交
echo ""
echo "注意：70B/72B 模型因内存限制未包含在此作业中"
echo "如需提取请使用 extract-large.sbatch"

echo ""
echo "=== Job finished: $(date) ==="

