"""
extract.py — 从 HuggingFace 模型中提取 embedding

用法:
  python extract.py --model bert-base-uncased
  python extract.py --model all                  # 跑责任人名下的所有模型
  python extract.py --model bert-base --cpu       # 强制 CPU

产出:
  models_embeddings/{model}.npy        shape: (1105, hidden_dim)
  models_embeddings/{model}_metadata.json
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM

# ── 路径 ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "human_concepts.csv")
OUT_DIR  = os.path.join(BASE_DIR, "models_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

PROMPT_TEMPLATE = "This is a {word}. "   # 论文指定的中性模板
POOLING = "avg"                          # 平均池化 over subtokens

# ── 模型清单 ────────────────────────────────────────
# 按责任人分组
MODELS = {
    "A_local": [    # 🧑A 本地 4060（只抽核心 Encoder）
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-large",
        "microsoft/deberta-large",
        "gpt2",
    ],
    "A_school": [   # 🧑A 学校 A100
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-70B",
        "Qwen/Qwen2.5-14B",
        "Qwen/Qwen2.5-32B",
        "Qwen/Qwen2.5-72B",
        "google/gemma-2-9b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        # OLMo 57 checkpoints 单独脚本处理
    ],
    "B": [          # 🧑B 本地 4060（其余的 Decoder）
        "gpt2-medium",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        "meta-llama/Llama-3.2-1B",
        "microsoft/phi-2",
        "google/gemma-2-2b",
        "Qwen/Qwen2-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ],
    "C": [          # 🧑C CPU（秒出）
        # Word2Vec 和 GloVe 用 gensim，单独脚本
    ],
}

ALL_MODELS = []

# model_path = os.path.join(BASE_DIR, "Qwen2-7B-Instruct")
for group in MODELS.values():
    ALL_MODELS.extend(group)

def get_model_info(model_name):
    """返回 (short_name, is_encoder)"""
    short = model_name.split("/")[-1]
    is_encoder = any(k in model_name.lower() for k in ["bert", "roberta", "deberta", "electra"])
    return short, is_encoder


def extract_embeddings(model_name, device="cuda"):
    """对数据集中的每个 item 提取静态和上下文 embedding"""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} ...")
    short_name, is_encoder = get_model_info(model_name)

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    
    # Use float32 for compatibility (some models like DeBERTa
    # fail at forward pass with float16 even if loading succeeds)
    model = AutoModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    ).to(device)
    #model = AutoModelForCausalLM.from_pretrained(
    # model_path,
    # local_files_only=True,
    # trust_remote_code=True,
    # low_cpu_mem_usage=True,
    # torch_dtype=torch.float16,
    # )
    # print(f"  Model loaded. Layers: {config.num_hidden_layers}, Dim: {config.hidden_size}")
    # model = model.to(device)  # 强制 float32，兼容性更好（某些模型如 DeBERTa 即使加载成功，float16 也会前向失败）
    model.eval()

    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    print(f"  Layers: {num_layers}, Dim: {hidden_dim}, Device: {device}")

    # 读取数据集
    df = pd.read_csv(CSV_PATH)
    items = df["item"].tolist()
    n = len(items)
    print(f"  Items: {n}")

    # 准备 prompt
    texts = [PROMPT_TEMPLATE.format(word=item) for item in items]

    # 批量提取（避免 OOM）
    batch_size = 32
    # 静态 embedding 矩阵 (n, dim)
    static_embs = np.zeros((n, hidden_dim), dtype=np.float32)
    # 上下文 embedding 每层 (n, dim) — 只取最后一层做分析演示，完整逐层分析可选
    # 为节省时间，我们先存最后一层；逐层分析可在 analysis.py 中按需重抽或存储全部层
    last_layer_embs = np.zeros((n, hidden_dim), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            batch_items = items[start:end]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)

            # 静态 embedding: input embedding layer (E 矩阵)
            # transformers 不直接暴露 E 矩阵, 我们用第一层 hidden state 近似
            # 或者取 word_embeddings 权重
            if hasattr(model, "get_input_embeddings"):
                wte = model.get_input_embeddings().weight
                # input_ids -> lookup
                static = wte[inputs["input_ids"]]  # (batch, seq_len, dim)
            else:
                static = outputs.hidden_states[0]

            # 上下文: 最后 hidden layer
            contextual = outputs.hidden_states[-1]  # (batch, seq_len, dim)

            # average pooling over subtokens（忽略特殊 token）
            attention_mask = inputs["attention_mask"]  # (batch, seq_len)
            for i in range(len(batch_texts)):
                mask = attention_mask[i].bool()
                s = static[i][mask].mean(dim=0).cpu().numpy()
                c = contextual[i][mask].mean(dim=0).cpu().numpy()
                static_embs[start + i] = s
                last_layer_embs[start + i] = c

            if (start // batch_size) % 5 == 0:
                print(f"  [{start}/{n}]")

    # 保存
    base_name = short_name.replace("-", "_").replace(".", "_")
    np.save(os.path.join(OUT_DIR, f"{base_name}_static.npy"), static_embs)
    np.save(os.path.join(OUT_DIR, f"{base_name}.npy"), last_layer_embs)

    # metadata
    meta = {
        "model": model_name,
        "short_name": short_name,
        "type": "encoder" if is_encoder else "decoder",
        "num_layers": num_layers,
        "dim": hidden_dim,
        "num_items": n,
        "prompt": PROMPT_TEMPLATE,
        "pooling": POOLING,
        "device": device,
    }
    with open(os.path.join(OUT_DIR, f"{base_name}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [OK] Saved {base_name}.npy (last layer) + {base_name}_static.npy")
    return base_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model name (HF) or 'all' / 'A_local' / 'A_school' / 'B' / 'C'")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CPU mode")

    # 解析模型清单
    if args.model == "all":
        models = ALL_MODELS
    elif args.model in MODELS:
        models = MODELS[args.model]
    else:
        models = [args.model]

    print(f"Models to run: {len(models)}")
    for m in models:
        t0 = time.time()
        try:
            extract_embeddings(m, device=device)
            elapsed = time.time() - t0
            print(f"  [TIME] {elapsed:.1f}s")
        except Exception as e:
            print(f"  [FAIL] {e}")

    print("\n[Done]")


if __name__ == "__main__":
    main()