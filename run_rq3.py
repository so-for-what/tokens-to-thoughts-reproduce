#!/usr/bin/env python3
"""
RQ3: Compression-Meaning Trade-off (ℒ curves).
Subset: 8 representative models, McCloskey1978, static embeddings, k=2..50, n_init=50.
"""
import json, os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans

BASE_DIR = "/tmp/ttt"
CSV = os.path.join(BASE_DIR, "human_concepts.csv")
EMB_DIR = os.path.join(BASE_DIR, "models_embeddings")
OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV)

# Enumerate models
files = glob(os.path.join(EMB_DIR, "*_static.npy"))
models = []
for f in files:
    name = os.path.basename(f).replace("_static.npy", "")
    meta_path = os.path.join(EMB_DIR, f"{name}_metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    models.append({"key": name, "name": meta.get("model", name), "type": meta.get("type", "unknown")})

# Select representative subset
rq3_keys = {"word2vec", "glove", "bert_base_uncased", "roberta_large",
            "gpt2", "Qwen2_7B", "Qwen2_5_0_5B", "phi_1"}
rq3_models = [m for m in models if m["key"] in rq3_keys]
print(f"RQ3: {len(rq3_models)} models (subset)\n", flush=True)

def compute_complexity(labels):
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    h_x = np.log2(n)
    h_x_given_c = np.sum(counts * np.log2(counts)) / n
    return h_x - h_x_given_c

def compute_distortion(embeddings, labels):
    n = len(embeddings)
    total = 0.0
    for c in np.unique(labels):
        mask = labels == c
        pts = embeddings[mask]
        centroid = pts.mean(axis=0)
        var = np.sum((pts - centroid) ** 2) / len(pts)
        total += len(pts) * var
    return total / n

def compute_l(embeddings, labels, beta=1.0):
    return compute_complexity(labels) + beta * compute_distortion(embeddings, labels)

# Use McCloskey1978 (largest dataset)
ds_name = "McCloskey1978"
mask = df["subdataset"] == ds_name
idx = np.where(mask)[0]
cat_labels = df.loc[mask, "category"].tolist()
cat_to_id = {c: i for i, c in enumerate(sorted(set(cat_labels)))}
y_true = np.array([cat_to_id[c] for c in cat_labels])

results = []
for m in rq3_models:
    static_path = os.path.join(EMB_DIR, f"{m['key']}_static.npy")
    static = np.load(static_path) if os.path.exists(static_path) else None
    if static is None:
        print(f"  ⚠ {m['name']}: no static embeddings, skipping", flush=True)
        continue

    static_sub = static[idx]
    k_range = list(range(2, min(51, len(static_sub) // 2)))
    l_values = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=50, random_state=42)
        pred = km.fit_predict(static_sub)
        l = compute_l(static_sub, pred, beta=1.0)
        l_values.append(l)

    human_l = compute_l(static_sub, y_true, beta=1.0)
    human_k = len(np.unique(y_true))

    results.append({
        "model": m["name"], "model_key": m["key"], "type": m["type"],
        "dataset": ds_name, "human_l": human_l, "human_k": human_k,
        "k_values": k_range, "l_values": l_values,
    })
    print(f"  {m['name'][:40]:40s} human_k={human_k}  human_ℒ={human_l:.2f}", flush=True)

# Save summaries
rq3_summary = pd.DataFrame([{
    "model": r["model"], "model_key": r["model_key"], "type": r["type"],
    "dataset": r["dataset"], "human_l": r["human_l"], "human_k": r["human_k"],
} for r in results])
rq3_summary.to_csv(os.path.join(OUT_DIR, "rq3_summary.csv"), index=False)

json.dump(results, open(os.path.join(OUT_DIR, "rq3_curves.json"), "w"))

# Plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each model gets its own panel
n_models = len(results)
cols = min(4, n_models)
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
fig.patch.set_facecolor("#020618")
axes = axes.flatten() if n_models > 1 else [axes]

colors = {"encoder": "#22d3ee", "decoder": "#34d399", "classic": "#fbbf24", "unknown": "#94a3b8"}

for i, r in enumerate(results):
    ax = axes[i]
    ax.set_facecolor("#0f172a")
    ax.plot(r["k_values"], r["l_values"], color=colors.get(r["type"], "#94a3b8"),
            lw=1.5, alpha=0.9)
    # Mark human point
    ax.scatter([r["human_k"]], [r["human_l"]], color="white", s=40, zorder=5,
               edgecolors="#f43f5e", linewidths=1.5, label=f"Human (k={r['human_k']})")
    ax.set_xlabel("k (clusters)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("ℒ", color="#94a3b8", fontsize=8)
    model_short = r["model"].split("/")[-1][:18]
    ax.set_title(f"{model_short} [{r['type']}]", color="white", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.legend(fontsize=6, labelcolor="#94a3b8")

# Hide unused axes
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rq3_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# Also a combined plot (all models on one axis)
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#020618")
ax.set_facecolor("#0f172a")
for r in results:
    ax.plot(r["k_values"], r["l_values"],
            color=colors.get(r["type"], "#94a3b8"),
            lw=1.5, alpha=0.8, label=r["model"].split("/")[-1][:20])
    ax.scatter([r["human_k"]], [r["human_l"]],
               color=colors.get(r["type"], "#94a3b8"), s=30, zorder=5,
               edgecolors="white", linewidths=0.8)
ax.scatter([], [], color="white", s=40, edgecolors="#f43f5e", linewidths=1.5,
           label="Human ℒ", marker="o")
ax.set_xlabel("k (clusters)", color="#94a3b8")
ax.set_ylabel("ℒ = I(X;C) + β·Distortion", color="#94a3b8")
ax.set_title("RQ3: Compression-Meaning Trade-off", color="white")
ax.tick_params(colors="#94a3b8")
for spine in ax.spines.values():
    spine.set_color("#1e293b")
ax.legend(fontsize=7, labelcolor="#94a3b8")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rq3_L_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ RQ3 complete: {len(results)} models", flush=True)
print(f"   results/rq3_summary.csv", flush=True)
print(f"   results/rq3_curves.json", flush=True)
print(f"   results/rq3_curves.png", flush=True)
print(f"   results/rq3_L_curve.png", flush=True)