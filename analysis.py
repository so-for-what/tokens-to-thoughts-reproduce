"""
analysis.py — 三路分析统一入口

用法:
  python analysis.py                         # 跑所有已提取的模型
  python analysis.py --model bert_base        # 只跑某个模型
  python analysis.py --quick                  # 快速模式(少K值)

产出:
  results/rq1_ami_scatter.png
  results/rq2_spearman_bar.png
  results/rq3_L_curve.png
  results/summary.csv
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "human_concepts.csv")
EMB_DIR  = os.path.join(BASE_DIR, "models_embeddings")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from utils import run_rq1, run_rq2, run_rq3


def load_embeddings(model_key):
    """
    读取一个模型的 embedding
    返回 (static_embs, contextual_embs, metadata)
    
    文件命名约定:
      {short_name}_static.npy   — 静态 embedding
      {short_name}.npy          — 上下文 embedding（最后一层）
      {short_name}_metadata.json
    """
    static_path = os.path.join(EMB_DIR, f"{model_key}_static.npy")
    ctx_path    = os.path.join(EMB_DIR, f"{model_key}.npy")
    meta_path   = os.path.join(EMB_DIR, f"{model_key}_metadata.json")

    static = np.load(static_path) if os.path.exists(static_path) else None
    ctx    = np.load(ctx_path) if os.path.exists(ctx_path) else None
    meta   = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    return static, ctx, meta


def get_available_models():
    """扫描 models_embeddings/ 下的可用模型"""
    files = glob(os.path.join(EMB_DIR, "*_static.npy"))
    models = []
    for f in files:
        name = os.path.basename(f).replace("_static.npy", "")
        # 尝试读 metadata
        meta_path = os.path.join(EMB_DIR, f"{name}_metadata.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            models.append({
                "key": name,
                "name": meta.get("model", name),
                "type": meta.get("type", "unknown"),
                "layers": meta.get("num_layers", 0),
                "dim": meta.get("dim", 0),
            })
        else:
            models.append({"key": name, "name": name, "type": "unknown", "layers": 0, "dim": 0})
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="模型 key (不传则跑所有)")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式: 少 K 值, 少初始化次数")
    args = parser.parse_args()

    # 加载数据集
    df = pd.read_csv(CSV_PATH)

    # 标准化典型性评分 (按子数据集归一化)
    df["typicality_norm"] = np.nan
    for ds in df["subdataset"].unique():
        mask = df["subdataset"] == ds
        vals = df.loc[mask, "typicality"]
        mn, mx = vals.min(), vals.max()
        if mx > mn:
            df.loc[mask, "typicality_norm"] = (vals - mn) / (mx - mn)
        else:
            df.loc[mask, "typicality_norm"] = 1.0

    # 获取可用模型
    if args.model:
        models = [m for m in get_available_models() if m["key"] == args.model]
        if not models:
            print(f"❌ 模型 {args.model} 未找到 embedding")
            return
    else:
        models = get_available_models()

    if not models:
        print("❌ models_embeddings/ 中没有找到 embedding 文件")
        print("请先运行 extract.py 提取 embedding")
        return

    print(f"\n📊 找到 {len(models)} 个模型，开始分析...\n")

    n_init = 20 if args.quick else 100
    k_range = [2, 3, 5, 8, 10, 15, 20, 30, 50] if args.quick else None

    results = []

    for m in models:
        print(f"{'='*50}")
        print(f"📌 {m['name']} ({m['type']})")
        static, ctx, meta = load_embeddings(m["key"])

        if static is None:
            print(f"  ⏭ 无静态 embedding")
            continue

        # 对每个子数据集分别做分析
        for ds_name in df["subdataset"].unique():
            mask = df["subdataset"] == ds_name
            idx = np.where(mask)[0]
            item_names = df.loc[mask, "item"].tolist()
            cat_labels = df.loc[mask, "category"].tolist()

            # 转为数值标签
            cat_to_id = {c: i for i, c in enumerate(sorted(set(cat_labels)))}
            y_true = np.array([cat_to_id[c] for c in cat_labels])
            k = len(cat_to_id)

            # 取对应行的 embedding
            static_sub = static[idx]
            ctx_sub = ctx[idx] if ctx is not None else static_sub

            # RQ1
            try:
                r1_static = run_rq1(static_sub, y_true, k, n_init=n_init)
                r1_ctx    = run_rq1(ctx_sub, y_true, k, n_init=n_init)
            except Exception as e:
                print(f"  ⚠ RQ1 failed: {e}")
                r1_static = r1_ctx = {"ami": np.nan, "nmi": np.nan, "ari": np.nan}

            # RQ2 — 需要类别名 embedding（简化: 不做逐层 Peak AMI）
            # 这里我们只对静态 embedding 做典型性分析
            typicality = df.loc[mask, "typicality_norm"].values
            try:
                # 类别名 embedding：取每个类别第一个 item 的 embedding 作为代理
                cat_embs = []
                for item_name in item_names:
                    cat_name = df.loc[mask & (df["item"] == item_name), "category"].values[0]
                    # 在全部数据中找到类别名的 index（简化：用类别名作为 prompt）
                    cat_mask = df["item"] == cat_name.lower()
                    if cat_mask.any():
                        cat_idx = np.where(cat_mask)[0][0]
                        cat_emb = static[cat_idx]
                    else:
                        # fallback: 用该类别的 centroid
                        cat_mask2 = df["category"] == cat_name
                        cat_idx2 = np.where(cat_mask2)[0]
                        cat_emb = static[cat_idx2].mean(axis=0)
                    cat_embs.append(cat_emb)
                cat_embs = np.array(cat_embs)
                rho, p = run_rq2(static_sub, cat_embs, typicality)
            except Exception as e:
                print(f"  ⚠ RQ2 failed: {e}")
                rho, p = np.nan, np.nan

            # RQ3
            try:
                r3_static = run_rq3(static_sub, y_true, beta=1.0, k_range=k_range)
            except Exception as e:
                print(f"  ⚠ RQ3 failed: {e}")
                r3_static = {"k_values": [], "l_values": [], "human_l": np.nan, "human_k": k}

            results.append({
                "model": m["name"],
                "model_key": m["key"],
                "type": m["type"],
                "dataset": ds_name,
                "n_items": len(idx),
                "k_categories": k,
                "rq1_ami_static": r1_static["ami"],
                "rq1_nmi_static": r1_static["nmi"],
                "rq1_ari_static": r1_static["ari"],
                "rq1_ami_ctx": r1_ctx["ami"],
                "rq2_spearman_rho": rho,
                "rq2_spearman_p": p,
                "rq3_human_l": r3_static["human_l"],
                "rq3_human_k": r3_static["human_k"],
            })

            ds_short = ds_name.replace("McCloskey1978", "MC").replace("Rosch", "R")
            ami_str = f"{r1_static['ami']:.3f}" if not np.isnan(r1_static['ami']) else "N/A"
            rho_str = f"{rho:.3f}" if not np.isnan(rho) else "N/A"
            print(f"  [{ds_short}] AMI={ami_str}  ρ={rho_str}")

    # 保存汇总表
    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ 汇总表已保存: {summary_path}")

    # 生成图表
    plot_results(summary_df)

    print("\n✨ 分析完成！")


def plot_results(df):
    """生成 RQ1/RQ2/RQ3 图表"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib 未安装，跳过出图")
        return

    # 配色
    colors = {"encoder": "#22d3ee", "decoder": "#34d399", "unknown": "#94a3b8"}
    markers = {"encoder": "s", "decoder": "o", "unknown": "x"}

    # ── RQ1: AMI 散点图 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#020618")
    ax.set_facecolor("#0f172a")

    for typ in ["encoder", "decoder", "unknown"]:
        subset = df[df["type"] == typ]
        if subset.empty:
            continue
        # 用 dim 作为参数量的代理（简化）
        x = np.arange(len(subset))
        ax.scatter(x, subset["rq1_ami_static"],
                   c=colors.get(typ, "#94a3b8"),
                   marker=markers.get(typ, "x"),
                   label=typ, s=60, edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Model Index", color="#94a3b8")
    ax.set_ylabel("AMI", color="#94a3b8")
    ax.set_title("RQ1: Categorical Alignment (Static Embeddings)", color="white")
    ax.tick_params(colors="#94a3b8")
    ax.legend()
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rq1_ami_scatter.png"), dpi=150)
    plt.close()
    print(f"  📈 RQ1 图已保存")

    # ── RQ2: Spearman 柱状图 ──
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#020618")
    ax.set_facecolor("#0f172a")

    # 每模型取跨数据集的平均 ρ
    agg = df.groupby(["model", "type", "model_key"]).agg(
        rho_mean=("rq2_spearman_rho", "mean"),
        rho_std=("rq2_spearman_rho", "std"),
    ).reset_index().dropna()
    agg = agg.sort_values("rho_mean", ascending=False)

    bars = ax.bar(range(len(agg)), agg["rho_mean"],
                  color=[colors.get(t, "#94a3b8") for t in agg["type"]],
                  edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([n.split("/")[-1][:15] for n in agg["model"]], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Spearman ρ", color="#94a3b8")
    ax.set_title("RQ2: Typicality Alignment", color="white")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    ax.axhline(y=0, color="#475569", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rq2_spearman_bar.png"), dpi=150)
    plt.close()
    print(f"  📊 RQ2 图已保存")

    # ── RQ3: ℒ 曲线 ──
    # 简化：只对第一个数据集的第一个 Encoder 和第一个 Decoder 画示意
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#020618")
    ax.set_facecolor("#0f172a")

    for model_key in df["model_key"].unique()[:3]:
        sub = df[df["model_key"] == model_key]
        if sub.empty:
            continue
        # 这里简化处理，完整版需要读取每个模型存储的 ℒ 曲线数据
        # 标记人类 ℒ
        human_l = sub["rq3_human_l"].values[0]
        human_k = sub["rq3_human_k"].values[0]
        ax.axhline(y=human_l, linestyle="--", color="#fb923c", alpha=0.5)
        ax.scatter(human_k, human_l, c="#fb923c", s=100, marker="*", zorder=5)

    ax.set_xlabel("Number of Clusters (K)", color="#94a3b8")
    ax.set_ylabel("ℒ (lower is more optimal)", color="#94a3b8")
    ax.set_title("RQ3: Compression-Meaning Trade-off", color="white")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rq3_L_curve.png"), dpi=150)
    plt.close()
    print(f"  📈 RQ3 图已保存")


if __name__ == "__main__":
    main()