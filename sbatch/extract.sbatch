#!/bin/bash
#SBATCH -J extract-A_school
#SBATCH -p Students
#SBATCH --qos=qos_stu_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH -t 03:50:00
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
# 如需访问 Llama/Gemma 等需要登录的模型，取消注释下一行并填入你的 HF token
# export HF_TOKEN=hf_xxxxxxxxxxxx

echo "=== Job started: $(date) ==="
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ===== 提取 A_school 模型 =====
# 逐一提取，避免 OOM
python extract.py --model meta-llama/Meta-Llama-3.1-8B
python extract.py --model Qwen/Qwen2.5-14B
python extract.py --model Qwen/Qwen2.5-32B
python extract.py --model google/gemma-2-9b
python extract.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

echo "=== A_school extract done ==="

# 70B 和 72B 模型需要更多资源，单独提交
echo ""
echo "注意：70B/72B 模型因内存限制未包含在此作业中"
echo "如需提取请使用 extract-large.sbatch"

echo ""
echo "=== Job finished: $(date) ==="