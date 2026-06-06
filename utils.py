"""
utils.py — 指标计算工具箱

包含：ℒ 目标函数 · k-means(×100) · AMI/NMI/ARI · Spearman ρ
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler


def compute_complexity(X, labels):
    """
    压缩项: I(X;C) = log₂|X| - (1/|X|) Σ |C_c| log₂|C_c|
    X 本身不参与计算, 只用 labels 统计簇大小
    """
    n = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    # H(X) = log₂|X|
    h_x = np.log2(n)
    # H(X|C) = (1/|X|) Σ |C_c| log₂|C_c|
    h_x_given_c = np.sum(counts * np.log2(counts)) / n
    return h_x - h_x_given_c


def compute_distortion(embeddings, labels):
    """
    失真项: (1/|X|) Σ |C_c| · σ²_c
    其中 σ²_c = (1/|C_c|) Σ ||e_i - ē_c||²
    """
    n = len(embeddings)
    unique = np.unique(labels)
    total = 0.0
    for c in unique:
        mask = labels == c
        cluster_pts = embeddings[mask]
        centroid = cluster_pts.mean(axis=0)
        variance = np.sum((cluster_pts - centroid) ** 2) / len(cluster_pts)
        total += len(cluster_pts) * variance
    return total / n


def compute_l_objective(embeddings, labels, beta=1.0):
    """
    ℒ = I(X;C) + β · Distortion
    """
    complexity = compute_complexity(embeddings, labels)
    distortion = compute_distortion(embeddings, labels)
    return complexity + beta * distortion


def kmeans_with_restarts(embeddings, k, n_init=100, random_state=42):
    """
    k-means 聚类，重复 n_init 次取最优
    返回 (labels, inertia)
    """
    best_labels = None
    best_inertia = float("inf")

    # sklearn 的 n_init 内部已经做了多次初始化取最优
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state, n_jobs=-1)
    labels = km.fit_predict(embeddings)
    return labels, km.inertia_


def compute_ami(true_labels, pred_labels):
    """Adjusted Mutual Information"""
    return adjusted_mutual_info_score(true_labels, pred_labels)


def compute_nmi(true_labels, pred_labels):
    """Normalized Mutual Information"""
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_ari(true_labels, pred_labels):
    """Adjusted Rand Index"""
    return adjusted_rand_score(true_labels, pred_labels)


def compute_spearman(item_similarities, human_typicality):
    """
    Spearman rank correlation
    返回 (rho, p_value)
    """
    mask = ~np.isnan(human_typicality) & ~np.isnan(item_similarities)
    if mask.sum() < 3:
        return 0.0, 1.0
    rho, p = spearmanr(item_similarities[mask], human_typicality[mask])
    return rho, p


def run_rq1(embeddings, human_labels, k, n_init=100):
    """
    RQ1: 聚类对齐
    返回 {ami, nmi, ari}
    """
    labels, _ = kmeans_with_restarts(embeddings, k, n_init=n_init)
    return {
        "ami": compute_ami(human_labels, labels),
        "nmi": compute_nmi(human_labels, labels),
        "ari": compute_ari(human_labels, labels),
    }


def run_rq2(item_embs, cat_name_embs, human_typicality):
    """
    RQ2: 典型性相关性
    返回 (rho, p_value)
    """
    # 归一化后算余弦
    item_norm = item_embs / (np.linalg.norm(item_embs, axis=1, keepdims=True) + 1e-12)
    cat_norm = cat_name_embs / (np.linalg.norm(cat_name_embs, axis=1, keepdims=True) + 1e-12)
    similarities = np.sum(item_norm * cat_norm, axis=1)
    return compute_spearman(similarities, human_typicality)


def run_rq3(embeddings, human_labels, beta=1.0, k_range=None):
    """
    RQ3: ℒ 曲线
    返回 {k_values, l_values, human_l}

    human_l: 用人类类别划分算出来的 ℒ
    """
    if k_range is None:
        k_range = list(range(2, min(51, len(embeddings) // 2 + 1)))

    l_values = []
    for k in k_range:
        labels, _ = kmeans_with_restarts(embeddings, k, n_init=50)
        l = compute_l_objective(embeddings, labels, beta=beta)
        l_values.append(l)

    # 人类类别的 ℒ
    human_l = compute_l_objective(embeddings, human_labels, beta=beta)

    return {
        "k_values": k_range,
        "l_values": l_values,
        "human_l": human_l,
        "human_k": len(np.unique(human_labels)),
    }