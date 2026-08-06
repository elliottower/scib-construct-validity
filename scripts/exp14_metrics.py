"""Exp14: Transfer-aware embedding metrics — pure Python logic.

Computes 9 metrics per (tissue, model) condition:
  Ablations (Part A): RCS-baseline, RCS-PCA10, RCS-trimmed, RCS-normalized, RCS-combined
  New metrics (Part B): PAD, SCC, MMD, CCA

All metrics take source/target embeddings + labels and return a single float.
Modal wrapper in modal_exp14_transfer_metrics.py calls these.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder


SEED = 20260727


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroids(X, labels, types, trim=0.0):
    """Compute centroids for each type. If trim > 0, use trimmed mean."""
    centroids = []
    for t in types:
        mask = labels == t
        pts = X[mask]
        if trim > 0 and len(pts) > 4:
            from scipy.stats import trim_mean
            centroids.append(np.array([trim_mean(pts[:, d], trim) for d in range(pts.shape[1])]))
        else:
            centroids.append(pts.mean(axis=0))
    return np.stack(centroids)


def _pairwise_dist_matrix(centroids):
    """Euclidean pairwise distance matrix."""
    k = len(centroids)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = np.linalg.norm(centroids[i] - centroids[j])
            D[i, j] = D[j, i] = d
    return D


def _upper_tri(D):
    """Upper triangular values."""
    return D[np.triu_indices(len(D), k=1)]


# ---------------------------------------------------------------------------
# Part A: RCS ablations
# ---------------------------------------------------------------------------

def rcs_baseline(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """RCS: simple mean centroids, native dim, raw distances, upper-tri Spearman."""
    c_src = _centroids(X_src, labels_src, shared_types, trim=0.0)
    c_tgt = _centroids(X_tgt, labels_tgt, shared_types, trim=0.0)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


def rcs_pca10(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Ablation A1: RCS on PCA-10 reduced embeddings."""
    n_src = len(X_src)
    X_all = np.vstack([X_src, X_tgt])
    n_comp = min(10, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_red = pca.fit_transform(X_all)
    return rcs_baseline(X_red[:n_src], X_red[n_src:], labels_src, labels_tgt, shared_types)


def rcs_trimmed(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Ablation A2: RCS with 5% trimmed-mean centroids."""
    c_src = _centroids(X_src, labels_src, shared_types, trim=0.05)
    c_tgt = _centroids(X_tgt, labels_tgt, shared_types, trim=0.05)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


def rcs_normalized(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Ablation A3: RCS with column-max normalized distance matrix."""
    c_src = _centroids(X_src, labels_src, shared_types, trim=0.0)
    c_tgt = _centroids(X_tgt, labels_tgt, shared_types, trim=0.0)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    col_max_src = D_src.max(axis=0, keepdims=True)
    col_max_tgt = D_tgt.max(axis=0, keepdims=True)
    col_max_src[col_max_src == 0] = 1.0
    col_max_tgt[col_max_tgt == 0] = 1.0
    D_src = D_src / col_max_src
    D_tgt = D_tgt / col_max_tgt
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


def rcs_combined(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Ablation A4: PCA-10 + trimmed mean + column-max normalization."""
    n_src = len(X_src)
    X_all = np.vstack([X_src, X_tgt])
    n_comp = min(10, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_red = pca.fit_transform(X_all)
    X_s, X_t = X_red[:n_src], X_red[n_src:]

    c_src = _centroids(X_s, labels_src, shared_types, trim=0.05)
    c_tgt = _centroids(X_t, labels_tgt, shared_types, trim=0.05)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    col_max_src = D_src.max(axis=0, keepdims=True)
    col_max_tgt = D_tgt.max(axis=0, keepdims=True)
    col_max_src[col_max_src == 0] = 1.0
    col_max_tgt[col_max_tgt == 0] = 1.0
    D_src = D_src / col_max_src
    D_tgt = D_tgt / col_max_tgt
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


# ---------------------------------------------------------------------------
# Part B: New transfer-aware metrics
# ---------------------------------------------------------------------------

def proxy_a_distance(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """B1: Proxy A-distance — domain classifier error rate.

    Low PAD = similar distributions = better transfer.
    Returns NEGATED value so higher = better (for correlation with F1).
    """
    n_s, n_t = len(X_src), len(X_tgt)
    X = np.vstack([X_src, X_tgt])
    y = np.concatenate([np.zeros(n_s), np.ones(n_t)])

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, y_train)
    error = 1.0 - clf.score(X_test, y_test)
    pad = 2.0 * (1.0 - 2.0 * error)
    return -pad


def source_classifier_confidence(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """B2: Source classifier confidence on target data.

    Train cell-type classifier on source, measure mean max-probability on target.
    """
    mask_src = np.isin(labels_src, shared_types)
    mask_tgt = np.isin(labels_tgt, shared_types)

    le = LabelEncoder()
    le.fit(shared_types)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_src[mask_src], le.transform(labels_src[mask_src]))

    probs = clf.predict_proba(X_tgt[mask_tgt])
    return float(np.mean(np.max(probs, axis=1)))


def mmd_score(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """B3: Maximum Mean Discrepancy with Gaussian kernel (median heuristic).

    Computed on PCA-50 to keep kernel computation tractable.
    Returns NEGATED value so higher = better (for correlation with F1).
    """
    n_src = len(X_src)
    X_all = np.vstack([X_src, X_tgt])
    n_components = min(50, X_all.shape[1], X_all.shape[0] - 1)
    if n_components < X_all.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_all = pca.fit_transform(X_all)
    X_s, X_t = X_all[:n_src], X_all[n_src:]

    # Subsample if too many cells (kernel matrix is O(n^2))
    rng = np.random.default_rng(SEED)
    max_n = 500
    if len(X_s) > max_n:
        idx = rng.choice(len(X_s), max_n, replace=False)
        X_s = X_s[idx]
    if len(X_t) > max_n:
        idx = rng.choice(len(X_t), max_n, replace=False)
        X_t = X_t[idx]

    # Median heuristic for bandwidth
    all_pts = np.vstack([X_s[:200], X_t[:200]])
    dists = cdist(all_pts, all_pts, 'sqeuclidean')
    sigma2 = np.median(dists[np.triu_indices(len(all_pts), k=1)])
    if sigma2 == 0:
        sigma2 = 1.0

    def rbf_kernel(A, B):
        sq = cdist(A, B, 'sqeuclidean')
        return np.exp(-sq / (2 * sigma2))

    K_ss = rbf_kernel(X_s, X_s)
    K_tt = rbf_kernel(X_t, X_t)
    K_st = rbf_kernel(X_s, X_t)

    m, n = len(X_s), len(X_t)
    mmd2 = (K_ss.sum() / (m * m) - 2 * K_st.sum() / (m * n) + K_tt.sum() / (n * n))
    return -float(mmd2)


def class_conditional_alignment(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """B4: Class-conditional alignment.

    For each shared type: 1 - ||c_src - c_tgt|| / (radius_src + eps),
    where radius_src = mean L2 distance of source points to their centroid.
    Using L2 radius (not per-dim std) so the normalizer scales with dimension.
    Higher = better alignment.
    """
    eps = 1e-8
    alignments = []
    for t in shared_types:
        pts_s = X_src[labels_src == t]
        pts_t = X_tgt[labels_tgt == t]
        if len(pts_s) < 2 or len(pts_t) < 2:
            continue
        c_s = pts_s.mean(axis=0)
        c_t = pts_t.mean(axis=0)
        radius_s = np.mean(np.linalg.norm(pts_s - c_s, axis=1))
        dist = np.linalg.norm(c_s - c_t)
        alignments.append(1.0 - dist / (radius_s + eps))
    return float(np.mean(alignments)) if alignments else 0.0


def transfer_f1(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Ground truth: macro-averaged transfer F1."""
    mask_src = np.isin(labels_src, shared_types)
    mask_tgt = np.isin(labels_tgt, shared_types)
    le = LabelEncoder()
    le.fit(shared_types)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_src[mask_src], le.transform(labels_src[mask_src]))
    y_pred = clf.predict(X_tgt[mask_tgt])
    return float(f1_score(le.transform(labels_tgt[mask_tgt]), y_pred, average="macro"))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_METRICS = {
    # Part A: ablations
    "rcs_baseline": rcs_baseline,
    "rcs_pca10": rcs_pca10,
    "rcs_trimmed": rcs_trimmed,
    "rcs_normalized": rcs_normalized,
    "rcs_combined": rcs_combined,
    # Part B: new metrics
    "pad": proxy_a_distance,
    "scc": source_classifier_confidence,
    "mmd": mmd_score,
    "cca": class_conditional_alignment,
}


def compute_all_metrics(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    """Compute all 9 metrics + F1 for one (tissue, model) condition."""
    result = {}
    for name, fn in ALL_METRICS.items():
        try:
            result[name] = fn(X_src, X_tgt, labels_src, labels_tgt, shared_types)
        except Exception as e:
            result[name] = None
            result[f"{name}_error"] = str(e)
    result["f1"] = transfer_f1(X_src, X_tgt, labels_src, labels_tgt, shared_types)
    return result
