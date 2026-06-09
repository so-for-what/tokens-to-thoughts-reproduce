"""
extract_static.py — 🧑C 专用：提取 Word2Vec / GloVe 静态 embedding

用法:
  python extract_static.py --method word2vec
  python extract_static.py --method glove
  python extract_static.py --method all          # 跑两种

CPU 即刻完成。
"""
import argparse, json, os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "human_concepts.csv")
OUT_DIR  = os.path.join(BASE_DIR, "models_embeddings")
os.makedirs(OUT_DIR, exist_ok=True)

# 下载地址（自动下载到 ~/.gensim）
# Word2Vec: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
# GloVe: https://nlp.stanford.edu/data/glove.840B.300d.zip

METHODS = {
    "word2vec": "word2vec-google-news-300",
    "glove": "glove-wiki-gigaword-300",
}


def extract_gensim(method):
    """用 gensim 的内置数据集"""
    import gensim.downloader as api

    model_name = METHODS[method]
    print(f"Loading {method} ({model_name})...")
    model = api.load(model_name)  # 自动下载缓存
    dim = model.vector_size
    print(f"  Dim: {dim}")

    df = pd.read_csv(CSV_PATH)
    items = df["item"].tolist()
    n = len(items)
    embs = np.zeros((n, dim), dtype=np.float32)

    missing = 0
    for i, item in enumerate(items):
        item_clean = item.lower().replace(" ", "_")
        if item_clean in model:
            embs[i] = model[item_clean]
        elif item.lower() in model:
            embs[i] = model[item.lower()]
        else:
            # 取平均 fallback
            parts = item.lower().split()
            vecs = [model[p] for p in parts if p in model]
            if vecs:
                embs[i] = np.mean(vecs, axis=0)
            else:
                missing += 1

    if missing:
        print(f"  ⚠ {missing}/{n} items not found, zero-initialized")

    # 保存
    np.save(os.path.join(OUT_DIR, f"{method}.npy"), embs)
    meta = {
        "model": method,
        "short_name": method,
        "type": "classic",
        "num_layers": 0,
        "dim": dim,
        "num_items": n,
        "prompt": "n/a (static)",
        "pooling": "n/a",
    }
    with open(os.path.join(OUT_DIR, f"{method}_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✅ Saved {method}.npy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="all",
                        choices=["word2vec", "glove", "all"])
    args = parser.parse_args()

    methods = ["word2vec", "glove"] if args.method == "all" else [args.method]
    for m in methods:
        extract_gensim(m)
    print("✨ Done!")


if __name__ == "__main__":
    main()