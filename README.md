# From Tokens to Thoughts — 复现管线

> Shani et al. (2025) — [arXiv:2505.17117](https://arxiv.org/abs/2505.17117)
>
> 三人分工 · 双环境并行 · 三步分析

---

## 📊 数据集

**`human_concepts.csv`** — 论文官方发布，1105 items，24 categories

| 子数据集 | 来源 | 条目数 |
|---------|------|:----:|
| Rosch1973 | Rosch (1973) "On the internal structure of perceptual and semantic categories" | 48 |
| Rosch1975 | Rosch (1975) "Cognitive representations of semantic categories" | 565 |
| McCloskey1978 | McCloskey & Glucksberg (1978) "Natural categories: Well defined or fuzzy sets?" | 492 |

**字段：** `item`（条目名） · `category`（类别） · `typicality`（人的典型性评分） · `subdataset`（来源）

---

## 🧬 整体管线

```
┌── 阶段①：写代码（🧑A, 此时没跑任何模型）───────┐
│  human_concepts.csv ──► extract.py + analysis.py │
│  🧑A 写完后 git push → 🧑B🧑C git pull          │
└─────────────────────────┬───────────────────────┘
                          │
                          ▼
┌── 阶段②：唯一一次 Embedding 提取 ─────────────────┐
│                                                    │
│  🧑A (GPU+学校) ─┬─ 本地 4060: 14个模型  ~30min  │
│                   ├─ 学校 A100: 8个大模型 ~60min   │
│                   └─ OLMo 57ckpt       ~60min     │
│                                                    │
│  🧑B (有GPU) ──── 本地 4060: 6-8个模型  ~30min    │
│                                                    │
│  🧑C (无GPU) ──── CPU: Word2Vec·GloVe  &lt;1min    │
│                                                    │
│  产出统一: models_embeddings/{model}.npy           │
│                                                    │
└─────────────────────────┬─────────────────────────┘
                          │  🧑A 收齐所有 .npy
                          ▼
┌── 阶段③：分析 + 出图（全员CPU可参与）─────────────┐
│  ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │  RQ1     │ │  RQ2     │ │  RQ3     │         │
│  │ AMI聚类  │ │Spearman ρ│ │  ℒ 曲线  │         │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘         │
│       └────────────┼────────────┘                │
│                    ▼                              │
│             ┌──────────┐                          │
│             │ 汇总出图  │                          │
│             │ Figure1-3│                          │
│             └──────────┘                          │
└───────────────────────────────────────────────────┘
```

---

## 👥 实际分工（更新版）

| | 🧑A（你） | 🧑B | 🧑C |
|:--|:---------:|:---:|:---:|
| **GPU** | ✅ RTX 4060 | ✅ RTX 4060 | ❌ 无 GPU |
| **学校平台** | ✅ USTC 107 | ❌ | ❌ |
| **角色** | 主力（代码+学校+少量本地） | 副手（多数本地模型） | 辅助（分析+做图+验证） |

### 🧑A — 主力（~80min + 排队）

| 任务 | 模型 | 时间 |
|:----|:----|:---:|
| 📝 写代码 | extract/analysis/utils + push GitHub | 一次性 |
| 🖥 本地 4060（仅核心 Encoder） | BERT-base/large · RoBERTa · DeBERTa · GPT-2 | ~5min |
| 🏫 学校 A100（大模型） | Llama-8B/70B · Qwen-14B/32B/72B · DeepSeek-14B/32B · Gemma-9B | ~60min+排队 |
| 🏫 OLMo 57 checkpoint | 训练动态分析 | ~60min |
| 📊 汇总分析 | 收齐全部 .npy → RQ1+RQ2+RQ3 → 出图 | ~10min |

### 🧑B — 副手（有 GPU, ~30min）

| 模型 | 环境 |
|:----|:----:|
| GPT-2-medium · Qwen2.5-0.5B/1.5B | RTX 4060 |
| Llama-3.2-1B · Phi-2 · Gemma-2-2B · Qwen2.5-4B | RTX 4060 |
| Phi-4-mini · DeepSeek-Distill-7B · OLMo-6ckpt | 学校（🧑A 提交） |

产出 .npy 交 🧑A 汇总。

### 🧑C — 辅助（无 GPU, 全程参与）

| 能做 | 命令 | 说明 |
|:----|:----|:-----|
| 🟢 Word2Vec + GloVe | `python extract_static.py --method all` | CPU 秒出 |
| 🟢 写分析代码 + 调试 | 协助 🧑A 验证 analysis.py | 等第一批 .npy 出来 |
| 🟢 跑分析出图 | `python analysis.py` | 全部 CPU，拿到 .npy 就能跑 |
| 🟢 数值验证 | 对比论文结果 vs 复现结果 | — |
| 🟢 图表润色 | 配色/排版/标注 | — |

---

## 📐 三个研究问题

### RQ1 — 类别边界对齐（AMI/NMI/ARI）

```
embs → k-means(K=人类类别数) × 100次 → AMI vs 人类类别标签
每层 hidden state 都算 → 取 Peak AMI
```

**核心发现：** 所有模型 AMI > 随机基线。BERT-large(340M) 达 AMI=0.60，超过 100× 更大的解码器。

### RQ2 — 典型性结构（Spearman ρ）

```
cosine(E(item), E(category_name)) → Spearman ρ vs human_typicality
静态层 + Peak AMI 层分别算
```

**核心发现：** ρ 普遍 < 0.2。LLM 能分门别类，但不懂门类内部的"核心成员"和"边缘成员"的区别。

### RQ3 — 压缩-意义权衡（ℒ 目标函数）

```
遍历 K=[2…50] → k-means(K) → ℒ = I(X;C) + β·Distortion
画 LLM 曲线 vs 人类 ℒ（固定 K）
```

**核心发现：** LLM 曲线在人类点下方 → LLM 更"优"但牺牲语义细腻度。

**公式细节：**
- `I(X;C) = log₂|X| - (1/|X|) Σ |C_c| log₂|C_c|` （压缩项）
- `Distortion = (1/|X|) Σ |C_c| · σ²_c` （失真项，σ²_c = 簇内方差）
- `β = 1.0`（默认）

---

## 📁 文件说明

| 文件 | 谁用 | 一句话 |
|:-----|:----|:-------|
| `human_concepts.csv` | 🧑🧑🧑 | 论文官方数据集（1105 items），全员共用 |
| `extract.py` | 🧑A🧑B | `python extract.py --model bert-base` — 从 HF 模型提 embedding |
| `extract_static.py` | 🧑C | `python extract_static.py --method all` — Word2Vec/GloVe，CPU 秒出 |
| `analysis.py` | 🧑A🧑C | `python analysis.py` — RQ1+RQ2+RQ3 分析出图 |
| `utils.py` | 🧑A🧑C | ℒ计算 / k-means / AMI / Spearman — analysis.py 调用的工具箱 |
| `requirements.txt` | 🧑🧑🧑 | `pip install -r requirements.txt` — 一次性装依赖 |
| `pipeline.html` | 🧑🧑🧑 | 可视化管线图，浏览器打开 |
| `sbatch/extract.sbatch` | 🧑A | 学校平台提交脚本（5 个中大型模型） |
| `sbatch/extract-large.sbatch` | 🧑A | 学校平台提交脚本（70B/72B 模型） |

|---

## 🏫 USTC 107 平台提交指南

如果你有 USTC 107 平台权限，sbatch 脚本已准备好：

| 脚本 | 用途 |
|:-----|:-----|
| `sbatch/extract.sbatch` | 提取 5 个中大型模型（Llama-8B · Qwen-14B/32B · Gemma-9B · DeepSeek-14B），~4h |
| `sbatch/extract-large.sbatch` | 单独提取 70B/72B 模型，需更多内存 |

**登陆后步骤：**

```bash
# 1. 克隆仓库
git clone https://github.com/so-for-what/tokens-to-thoughts-reproduce.git
cd tokens-to-thoughts-reproduce

# 2. 装 conda 环境（一次性）
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n ttt python=3.11 -y
conda activate ttt
conda tos accept
CONDA_OVERRIDE_CUDA="12.4" conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install transformers datasets scikit-learn numpy scipy sentencepiece

# 3. 提交作业
sbatch sbatch/extract.sbatch

# 4. 查看状态
squeue -u $USER
cat logs/extract-*.out   # 看输出
cat logs/extract-*.err   # 看错误
```

> ⚠️ 学校平台**不支持 SSH/SFTP**，只能用网页文件管理器传文件。
> 默认 QOS 限制：4 CPU / 1 GPU / 4GB 内存 / 4 小时。70B/72B 需单独提高内存限制。

---

## ⚠️ 注意事项

| 项目 | 要求 |
|:----|:-----|
| Prompt 模板 | 统一 `"This is a {word}. "`（带尾随空格） |
| 池化策略 | 统一 average pooling over subtokens |
| 聚类初始化 | k-means 重复 100 次随机初始化取平均 |
| β 参数 | 先跑 `β=1.0`，敏感性分析可选 |
| 数据集 | 所有人共用同一份 `human_concepts.csv` |
| 环境差异 | CUDA/Python/GPU 型号不影响 embedding 数值 |
| 网络 | 学校平台注意 HF 代理配置 / `HF_TOKEN` |

---

## 📄 可视化管线

仓库里的 `pipeline.html` 是完整的可视化流程图，下载后用浏览器打开即可查看。

> 直接下载：https://raw.githubusercontent.com/so-for-what/tokens-to-thoughts-reproduce/main/pipeline.html
> 或者本地 `C:\projects\hermes\reproduce_tokens_to_thoughts\pipeline.html`