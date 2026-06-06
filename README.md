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
human_concepts.csv (1105 items)
         │
         ▼
   ┌──────────────────────────────┐
   │  extract.py                  │  ← 统一代码，所有人通用
   │  prompt → 取 embedding       │
   │  avg pooling over subtokens  │
   └────────────┬─────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 ┌──────┐  ┌──────┐  ┌──────────┐
 │ 🧑A  │  │ 🧑B  │  │  🧑C    │
 │ 本地  │  │ 本地  │  │ 学校平台 │
 │12模型 │  │ 8模型 │  │ 8个大模型│
 └──┬───┘  └──┬───┘  └────┬─────┘
    │         │           │
    └─────────┼───────────┘
              ▼
    ┌─────────────────┐
    │  models_embeddings/ │  ← 每人产出 .npy 文件
    │  {model_name}.npy   │
    │  shape: (1105, dim) │
    └─────────┬───────────┘
              │
    ┌─────────┼───────────┐
    ▼         ▼           ▼
 ┌──────┐ ┌──────┐ ┌──────────┐
 │ RQ1  │ │ RQ2  │ │  RQ3     │
 │ AMI  │ │Spear.│ │  ℒ 曲线  │
 │聚类对齐│ │典型性 │ │ 压缩权衡 │
 └──┬───┘ └──┬───┘ └────┬─────┘
    │         │           │
    └─────────┼───────────┘
              ▼
       ┌──────────┐
       │  汇总出图  │
       │  Figure1-3│
       └──────────┘
```

---

## 👥 三人分工

### 🧑A — 写代码 + 本地模型 (~40分钟)

| 负责 | 模型 | 环境 | 时间 |
|:----|:----|:----:|:---:|
| 📝 编写 extract.py / analysis.py / utils.py | — | — | — |
| 💻 跑经典 + Encoder | Word2Vec · GloVe · BERT-base/large · RoBERTa · DeBERTa | CPU / 4060 | ~2min |
| 💻 跑小 Decoder | GPT-2 · GPT-2-medium · Qwen2.5-0.5B/1.5B · Llama-3.2-1B · Phi-2 | 4060 | ~15min |
| 📊 汇总分析 | 收齐所有 .npy → 跑 RQ1+RQ2+RQ3 → 出图 | CPU | ~5min |

### 🧑B — 本地模型并行 (~30分钟)

| 模型 | 环境 | 时间 |
|:----|:----:|:---:|
| Gemma-2-2B | 4060 | ~4min |
| Qwen2.5-4B | 4060/学校 | ~6min |
| Phi-4-mini (3.8B) | 学校 | ~5min |
| DeepSeek-Distill-Qwen-7B | 学校 | ~8min |
| OLMo-7B (子集6 ckpt) | 学校 | ~15min |

### 🧑C — 学校平台大模型 (~60分钟)

| 模型 | 参数量 | 时间 |
|:----|:-----:|:---:|
| Llama-3.1-8B | 8B | ~10min |
| Gemma-2-9B | 9B | ~10min |
| Qwen2.5-14B | 14B | ~15min |
| DeepSeek-Distill-Qwen-14B | 14B | ~15min |
| Qwen2.5-32B | 32B | ~20min |
| DeepSeek-Distill-32B | 32B | ~20min |
| Llama-3.1-70B | 70B | ~30min |
| Qwen2.5-72B | 72B | ~30min |
| OLMo-7B (完整 57 ckpt) ⚠️ | 7B | ~60min |

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

## 📁 代码结构

```
tokens-to-thoughts-reproduce/
├── human_concepts.csv          # 数据集（已就绪）
├── extract.py                  # [🧑A写] 模型名参数化，通用
├── analysis.py                 # [🧑A写] RQ1+RQ2+RQ3 统一入口
├── utils.py                    # [🧑A写] ℒ计算、聚类、指标
├── models_embeddings/          # 全员产出（.npy + metadata.json）
│   ├── bert-large-uncased.npy
│   └── ...
├── results/                    # 出图、出表
│   ├── rq1_ami_scatter.png
│   ├── rq2_spearman_bar.png
│   ├── rq3_L_curve.png
│   └── summary_table.csv
├── pipeline.html               # 可视化管线（本地浏览器打开）
└── .gitignore
```

### Embedding 格式规范

```python
# models_embeddings/{model_name}.npy
# shape: (1105, hidden_dim)
# 行序 = human_concepts.csv 行序

# {model_name}_metadata.json
{
  "model": "bert-large-uncased",
  "type": "encoder",
  "num_layers": 24,
  "dim": 1024,
  "prompt": "This is a {word}. ",
  "pooling": "avg",
  "subdatasets": ["Rosch1973", "Rosch1975", "McCloskey1978"]
}
```

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