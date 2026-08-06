"""Exp15: SCC stress test — cross-classifier robustness + bootstrap CIs.

Tests whether SCC's strong F1 prediction survives when the SCC classifier
differs from the F1 classifier (breaking the shared-machinery confound).

Classifiers tested for SCC:
  - Logistic regression (original, same as F1 ground truth)
  - kNN (k=15)
  - Random forest (100 trees)
  - Linear SVM (probability via Platt scaling)

F1 ground truth always uses logistic regression (matching the main panel).

Also computes all 9 original exp14 metrics for completeness.

Modal wrapper in modal_exp15_scc_stress_test.py calls these.
"""
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import trim_mean
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


SEED = 20260801


def _centroids(X, labels, types, trim=0.0):
    centroids = []
    for t in types:
        mask = labels == t
        pts = X[mask]
        if trim > 0 and len(pts) > 4:
            centroids.append(np.array([trim_mean(pts[:, d], trim) for d in range(pts.shape[1])]))
        else:
            centroids.append(pts.mean(axis=0))
    return np.stack(centroids)


def _pairwise_dist_matrix(centroids):
    k = len(centroids)
    D = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            d = np.linalg.norm(centroids[i] - centroids[j])
            D[i, j] = D[j, i] = d
    return D


def _upper_tri(D):
    return D[np.triu_indices(len(D), k=1)]


def rcs_baseline(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    c_src = _centroids(X_src, labels_src, shared_types, trim=0.0)
    c_tgt = _centroids(X_tgt, labels_tgt, shared_types, trim=0.0)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


def rcs_pca10(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    n_src = len(X_src)
    X_all = np.vstack([X_src, X_tgt])
    n_comp = min(10, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    X_red = pca.fit_transform(X_all)
    return rcs_baseline(X_red[:n_src], X_red[n_src:], labels_src, labels_tgt, shared_types)


def rcs_trimmed(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    c_src = _centroids(X_src, labels_src, shared_types, trim=0.05)
    c_tgt = _centroids(X_tgt, labels_tgt, shared_types, trim=0.05)
    D_src = _pairwise_dist_matrix(c_src)
    D_tgt = _pairwise_dist_matrix(c_tgt)
    rho, _ = stats.spearmanr(_upper_tri(D_src), _upper_tri(D_tgt))
    return float(rho) if np.isfinite(rho) else 0.0


def rcs_normalized(X_src, X_tgt, labels_src, labels_tgt, shared_types):
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


def proxy_a_distance(X_src, X_tgt, labels_src, labels_tgt, shared_types):
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


def _scc_with_classifier(clf, X_src, X_tgt, labels_src, labels_tgt, shared_types):
    mask_src = np.isin(labels_src, shared_types)
    mask_tgt = np.isin(labels_tgt, shared_types)
    le = LabelEncoder()
    le.fit(shared_types)
    clf.fit(X_src[mask_src], le.transform(labels_src[mask_src]))
    probs = clf.predict_proba(X_tgt[mask_tgt])
    return float(np.mean(np.max(probs, axis=1)))


def scc_logreg(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    return _scc_with_classifier(clf, X_src, X_tgt, labels_src, labels_tgt, shared_types)


def scc_knn(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    clf = KNeighborsClassifier(n_neighbors=15)
    return _scc_with_classifier(clf, X_src, X_tgt, labels_src, labels_tgt, shared_types)


def scc_rf(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    return _scc_with_classifier(clf, X_src, X_tgt, labels_src, labels_tgt, shared_types)


def scc_svm(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    clf = SVC(kernel='linear', probability=True, max_iter=2000, random_state=SEED)
    return _scc_with_classifier(clf, X_src, X_tgt, labels_src, labels_tgt, shared_types)


def mmd_score(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    n_src = len(X_src)
    X_all = np.vstack([X_src, X_tgt])
    n_components = min(50, X_all.shape[1], X_all.shape[0] - 1)
    if n_components < X_all.shape[1]:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_all = pca.fit_transform(X_all)
    X_s, X_t = X_all[:n_src], X_all[n_src:]
    rng = np.random.default_rng(SEED)
    max_n = 500
    if len(X_s) > max_n:
        idx = rng.choice(len(X_s), max_n, replace=False)
        X_s = X_s[idx]
    if len(X_t) > max_n:
        idx = rng.choice(len(X_t), max_n, replace=False)
        X_t = X_t[idx]
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
    mask_src = np.isin(labels_src, shared_types)
    mask_tgt = np.isin(labels_tgt, shared_types)
    le = LabelEncoder()
    le.fit(shared_types)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_src[mask_src], le.transform(labels_src[mask_src]))
    y_pred = clf.predict(X_tgt[mask_tgt])
    return float(f1_score(le.transform(labels_tgt[mask_tgt]), y_pred, average="macro"))


ALL_METRICS = {
    "rcs_baseline": rcs_baseline,
    "rcs_pca10": rcs_pca10,
    "rcs_trimmed": rcs_trimmed,
    "rcs_normalized": rcs_normalized,
    "rcs_combined": rcs_combined,
    "pad": proxy_a_distance,
    "scc_logreg": scc_logreg,
    "scc_knn": scc_knn,
    "scc_rf": scc_rf,
    "scc_svm": scc_svm,
    "mmd": mmd_score,
    "ccal": class_conditional_alignment,
}


def compute_all_metrics(X_src, X_tgt, labels_src, labels_tgt, shared_types):
    result = {}
    for name, fn in ALL_METRICS.items():
        try:
            result[name] = fn(X_src, X_tgt, labels_src, labels_tgt, shared_types)
        except Exception as e:
            result[name] = None
            result[f"{name}_error"] = str(e)
    result["f1"] = transfer_f1(X_src, X_tgt, labels_src, labels_tgt, shared_types)
    return result
