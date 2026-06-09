#!/usr/bin/env python3
"""
RQ1: Categorical Alignment — n_init=100 (paper standard).
24 models × 3 datasets × static+ctx. Output: results/rq1_n100_results.csv
"""
import json, os, sys
import numpy as np
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score

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

print(f"RQ1 n_init=100: {len(models)} models\n", flush=True)

results = []
for mi, m in enumerate(models):
    static = np.load(os.path.join(EMB_DIR, f"{m['key']}_static.npy"))
    ctx_path = os.path.join(EMB_DIR, f"{m['key']}.npy")
    ctx = np.load(ctx_path) if os.path.exists(ctx_path) else None

    for ds_name in df["subdataset"].unique():
        mask = df["subdataset"] == ds_name
        idx = np.where(mask)[0]
        cat_labels = df.loc[mask, "category"].tolist()
        cat_to_id = {c: i for i, c in enumerate(sorted(set(cat_labels)))}
        y_true = np.array([cat_to_id[c] for c in cat_labels])
        k = len(cat_to_id)

        for emb, etype in [(static[idx], "static"), (ctx[idx] if ctx is not None else static[idx], "ctx")]:
            km = KMeans(n_clusters=k, n_init=100, random_state=42)
            pred = km.fit_predict(emb)
            ami = adjusted_mutual_info_score(y_true, pred)
            nmi = normalized_mutual_info_score(y_true, pred)
            ari = adjusted_rand_score(y_true, pred)
            results.append({
                "model": m["name"], "model_key": m["key"], "type": m["type"],
                "dataset": ds_name, "emb_type": etype,
                "ami": ami, "nmi": nmi, "ari": ari,
            })

    ds_short = ds_name.replace("McCloskey1978", "MC").replace("Rosch", "R")
    if etype == "static":
        print(f"  [{mi+1}/{len(models)}] {m['name'][:40]} [{ds_short}] AMI={ami:.4f}  NMI={nmi:.4f}", flush=True)

rq1 = pd.DataFrame(results)
rq1.to_csv(os.path.join(OUT_DIR, "rq1_n100_results.csv"), index=False)

# Generate plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {"encoder": "#22d3ee", "decoder": "#34d399", "classic": "#fbbf24", "unknown": "#94a3b8"}

for etype, etype_label in [("static", "Static"), ("ctx", "Contextual")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#020618")
    for ei, ds_name in enumerate(df["subdataset"].unique()):
        ax = axes[ei]
        ax.set_facecolor("#0f172a")
        sub = rq1[(rq1["emb_type"] == etype) & (rq1["dataset"] == ds_name)]
        sorted_sub = sub.sort_values("ami", ascending=False)
        ax.bar(range(len(sorted_sub)), sorted_sub["ami"].values,
               color=[colors.get(t, "#94a3b8") for t in sorted_sub["type"].values],
               edgecolor="white", linewidth=0.3)
        ax.set_xticks(range(len(sorted_sub)))
        labels = [n.split("/")[-1][:14] for n in sorted_sub["model"].values]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6, color="#94a3b8")
        ax.set_ylabel("AMI", color="#94a3b8")
        ds_title = ds_name.replace("McCloskey1978", "McCloskey1978").replace("Rosch", "Rosch")
        ax.set_title(f"RQ1 ({etype_label}): {ds_title}", color="white", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"rq1_n100_{etype}.png"), dpi=150, bbox_inches="tight")
    plt.close()

# Also generate the combined figure used in the report
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#020618")
for ei, (etype, title) in enumerate([("static", "Static"), ("ctx", "Contextual")]):
    ax = axes[ei]
    ax.set_facecolor("#0f172a")
    agg = rq1[rq1["emb_type"] == etype].groupby(["model", "type", "model_key"]).agg(ami_mean=("ami", "mean")).reset_index().sort_values("ami_mean", ascending=False)
    ax.bar(range(len(agg)), agg["ami_mean"].values,
           color=[colors.get(t, "#94a3b8") for t in agg["type"].values],
           edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(agg)))
    labels = [n.split("/")[-1][:14] for n in agg["model"].values]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6, color="#94a3b8")
    ax.set_ylabel("AMI", color="#94a3b8")
    ax.set_title(f"RQ1: {title} (avg across datasets)", color="white", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rq1_n100_combined.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n✅ RQ1 n_init=100 complete: {len(rq1)} rows", flush=True)
print(f"   results/rq1_n100_results.csv", flush=True)
print(f"   results/rq1_n100_static.png", flush=True)
print(f"   results/rq1_n100_ctx.png", flush=True)
print(f"   results/rq1_n100_combined.png", flush=True)