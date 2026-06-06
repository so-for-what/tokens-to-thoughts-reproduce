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
| **角色** | 主力 | 副手 | 辅助 |

### 🧑A — 主力（GPU + 学校平台, ~2h）

| 承担 | 内容 | 时间 |
|:----|:----|:---:|
| 📝 写代码 | extract.py + analysis.py + utils.py，push GitHub | 一次性 |
| 🖥 本地 4060 | BERT/RoBERTa/DeBERTa/GPT-2/Qwen0.5B/1.5B/Llama1B/Phi-2 等 14 个 | ~30min |
| 🏫 学校 A100 | Llama-8B/70B · Qwen-14B/32B/72B · DeepSeek-14B/32B · Gemma-9B · OLMo 57ckpt | ~60min+排队 |
| 📊 汇总分析 | 收齐 🧑B🧑C 的 .npy → RQ1+RQ2+RQ3 → 出图 | ~10min |

### 🧑B — 副手（有 GPU, ~30min）

| 模型 | 环境 |
|:----|:----:|
| Gemma-2-2B · Qwen2.5-4B | 4060 |
| Phi-4-mini · DeepSeek-Distill-7B | 学校（🧑A 提交） |
| OLMo-7B（6 ckpt 子集） | 学校（🧑A 提交） |

产出 .npy → 交给 🧑A 汇总。如果 A 的 4060 来不及，可以分担部分小模型。

### 🧑C — 辅助（无 GPU, 全程参与）

| 能做 | 说明 |
|:----|:-----|
| 🟢 Word2Vec · GloVe | CPU 秒出，作为基线 |
| 🟢 分析代码调试 | 协助 🧑A 写 / 验证 analysis.py |
| 🟢 跑分析出图 | 阶段③ 全部 CPU 运算，拿到 .npy 就能跑 |
| 🟢 OLMo 训练动态 | 非 GPU 密集型 |
| 🟢 数值验证 | 复现结果 vs 论文结果对比 |
| 🟢 图表设计 | 配色 / 排版 / 标注 |

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