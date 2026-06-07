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
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ===== 提取 A_school 模型（从小到大，逐一下载+提取）=====
# Qwen2.5-14B 已完成，跳过

# 1. 8B 模型（最快）
python extract.py --model meta-llama/Meta-Llama-3.1-8B

# 2. 9B 模型
python extract.py --model google/gemma-2-9b

# 3. 14B 模型
python extract.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# 4. 32B 模型（最慢，放最后）
python extract.py --model Qwen/Qwen2.5-32B

echo "=== A_school extract done ==="

# 70B 和 72B 模型需要更多资源，单独提交
echo ""
echo "注意：70B/72B 模型因内存限制未包含在此作业中"
echo "如需提取请使用 extract-large.sbatch"

echo ""
echo "=== Job finished: $(date) ==="