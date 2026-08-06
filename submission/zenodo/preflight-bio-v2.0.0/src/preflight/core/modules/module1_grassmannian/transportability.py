"""Geometric transportability measures.

Ported from paper_clinical experiments (sheaf H^1, Grassmannian principal
angles, Fisher-Rao off-geodesic score). Operates on embedding matrices
(samples x features) per cohort.
"""
import numpy as np
from scipy import linalg, stats
from scipy.stats import chi2


# ======================================================================
# Grassmannian geometry primitives
# ======================================================================

def principal_angles(U1, U2):
    """Principal angles between two subspaces (orthonormal bases)."""
    M = U1.T @ U2
    svals = linalg.svdvals(M)
    svals = np.clip(svals, -1, 1)
    return np.arccos(svals)


def geodesic_distance(U1, U2):
    """Grassmannian geodesic distance = L2 norm of principal angles."""
    return float(np.linalg.norm(principal_angles(U1, U2)))


def transport_matrix(U_from, U_to):
    """Parallel transport on Gr(k,d) via the canonical connection."""
    M = U_from.T @ U_to
    u, s, vt = linalg.svd(M)
    return u @ vt


def cocycle_holonomy(subspaces):
    """Holonomy around a cycle of subspaces.

    Returns (holonomy_matrix, deviation_from_identity).
    """
    k = subspaces[0].shape[1]
    composed = np.eye(k)
    for idx in range(len(subspaces)):
        U_from = subspaces[idx]
        U_to = subspaces[(idx + 1) % len(subspaces)]
        T = transport_matrix(U_from, U_to)
        composed = T @ composed
    deviation = np.linalg.norm(composed - np.eye(k), "fro")
    return composed, float(deviation)


# ======================================================================
# Subspace extraction
# ======================================================================

def top_k_subspace(X, k):
    """Extract top-k PCA subspace from data matrix X (samples x features).

    Returns orthonormal basis (features x k) and explained variance ratios.
    """
    X_centered = X - X.mean(axis=0)
    U, s, Vt = linalg.svd(X_centered, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    return Vt[:k].T, explained[:k]


# ======================================================================
# Sheaf H^1 cross-cohort consistency
# ======================================================================

def sheaf_h1_two_cohort(X1, X2, k):
    """Sheaf H^1 test for two cohorts.

    Builds a 2-node graph with one edge. Stalks are top-k subspaces.
    Reports holonomy deviation, principal-angle distance, and whether
    H^1 != 0 (subspaces are not related by a single rotation).

    Returns dict with:
        h1_nonzero: bool
        holonomy_norm: float (Frobenius deviation from identity)
        geodesic_dist: float
        principal_angles: array of angles
        explained_var_c1: array
        explained_var_c2: array
    """
    U1, ev1 = top_k_subspace(X1, k)
    U2, ev2 = top_k_subspace(X2, k)

    angles = principal_angles(U1, U2)
    gdist = float(np.linalg.norm(angles))

    _, holonomy_norm = cocycle_holonomy([U1, U2])

    return {
        "h1_nonzero": gdist > 0.1,
        "holonomy_norm": holonomy_norm,
        "geodesic_dist": gdist,
        "principal_angles": angles,
        "explained_var_c1": ev1,
        "explained_var_c2": ev2,
    }


def sheaf_h1_multi_cohort(cohort_matrices, k):
    """Sheaf H^1 for multiple cohorts arranged in a cycle.

    cohort_matrices: list of (samples x features) arrays.
    Returns holonomy around the full cycle.
    """
    subspaces = []
    explained = []
    for X in cohort_matrices:
        U, ev = top_k_subspace(X, k)
        subspaces.append(U)
        explained.append(ev)

    hol_mat, hol_norm = cocycle_holonomy(subspaces)

    pairwise = []
    for i in range(len(subspaces)):
        j = (i + 1) % len(subspaces)
        pairwise.append({
            "pair": (i, j),
            "geodesic_dist": geodesic_distance(subspaces[i], subspaces[j]),
            "principal_angles": principal_angles(subspaces[i], subspaces[j]),
        })

    return {
        "holonomy_norm": hol_norm,
        "holonomy_matrix": hol_mat,
        "pairwise": pairwise,
        "explained_variances": explained,
    }


# ======================================================================
# Fisher-Rao off-geodesic / confound-leakage score
# ======================================================================

def fisher_rao_confound_score(X1, X2, batch_labels_1, batch_labels_2, y1, y2):
    """Fisher-Rao style confound-leakage score.

    Quantifies how much of the cross-cohort embedding shift is
    batch/acquisition confound vs biological signal.

    X1, X2: embedding matrices (samples x features)
    batch_labels_1, batch_labels_2: batch/site labels per sample
    y1, y2: clinical labels per sample

    Returns dict with:
        raw_shift: L2 distance between cohort centroids
        label_shift: L2 distance between label-conditioned centroids
        batch_shift: L2 distance between batch-conditioned centroids
        confound_fraction: batch_shift / raw_shift
        signal_fraction: 1 - confound_fraction
    """
    centroid_1 = X1.mean(axis=0)
    centroid_2 = X2.mean(axis=0)
    raw_shift = float(np.linalg.norm(centroid_1 - centroid_2))

    label_centroids_1 = {
        lbl: X1[y1 == lbl].mean(axis=0) for lbl in np.unique(y1)
    }
    label_centroids_2 = {
        lbl: X2[y2 == lbl].mean(axis=0) for lbl in np.unique(y2)
    }
    common_labels = set(label_centroids_1.keys()) & set(label_centroids_2.keys())
    label_shift = float(np.mean([
        np.linalg.norm(label_centroids_1[lbl] - label_centroids_2[lbl])
        for lbl in common_labels
    ]))

    confound_fraction = 1.0 - (label_shift / raw_shift) if raw_shift > 0 else 0.0
    confound_fraction = np.clip(confound_fraction, 0, 1)

    return {
        "raw_shift": raw_shift,
        "label_shift": label_shift,
        "confound_fraction": float(confound_fraction),
        "signal_fraction": float(1.0 - confound_fraction),
    }


# ======================================================================
# Directional transport predictors (v2 — fixes symmetric mismatch)
# ======================================================================

def discriminant_direction(X, y):
    """Fisher's linear discriminant direction (unit vector).

    Returns the direction that maximizes class separation:
    w = Sw^{-1} (mu_1 - mu_0), normalized to unit length.
    Falls back to centroid difference if Sw is singular.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Need exactly 2 classes, got {len(classes)}")
    X0 = X[y == classes[0]]
    X1 = X[y == classes[1]]
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    diff = mu1 - mu0

    Sw = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)
    try:
        w = linalg.solve(Sw, diff, assume_a='pos')
    except linalg.LinAlgError:
        w = linalg.lstsq(Sw, diff)[0]

    norm = np.linalg.norm(w)
    if norm < 1e-12:
        return diff / (np.linalg.norm(diff) + 1e-12)
    return w / norm


def directional_transport_score(X_from, y_from, X_to, y_to, k=10):
    """Directional transport predictor: projection of between-cohort shift
    onto the source's classification-relevant subspace.

    For ordered pair (from -> to):
    1. Compute PCA subspaces for both cohorts
    2. Find Fisher discriminant direction in source cohort
    3. Transport it to target's subspace via parallel transport on Gr(k,d)
    4. Measure how much of the centroid shift lands on this direction

    Returns dict with:
        score: float — signed projection (positive = shift along discriminant)
        abs_score: float — absolute projection magnitude
        shift_norm: float — total centroid shift magnitude
        discriminant_alignment: float — cos(angle) between shift and discriminant
    """
    U_from, _ = top_k_subspace(X_from, k)
    U_to, _ = top_k_subspace(X_to, k)

    w_full = discriminant_direction(X_from, y_from)
    w_in_sub = U_from @ (U_from.T @ w_full)
    w_in_sub_norm = np.linalg.norm(w_in_sub)
    if w_in_sub_norm < 1e-12:
        w_in_sub = w_full
    else:
        w_in_sub = w_in_sub / w_in_sub_norm

    T = transport_matrix(U_from, U_to)
    w_sub_coords = U_from.T @ w_in_sub
    w_transported_coords = T @ w_sub_coords
    w_transported = U_to @ w_transported_coords
    w_transported = w_transported / (np.linalg.norm(w_transported) + 1e-12)

    shift = X_to.mean(axis=0) - X_from.mean(axis=0)
    shift_norm = float(np.linalg.norm(shift))

    projection = float(shift @ w_transported)
    alignment = projection / (shift_norm + 1e-12)

    return {
        "score": projection,
        "abs_score": abs(projection),
        "shift_norm": shift_norm,
        "discriminant_alignment": alignment,
    }


def bracket_norm_score(X_from, y_from, X_to, y_to):
    """Bracket-norm batch/label decomposition.

    Decomposes cross-cohort shift into:
    - batch component: shift in the label-averaged centroid
    - label component: shift in the between-class direction
    - interaction: the remainder (batch x label entanglement)

    Unlike Fisher-Rao confound score, this handles the degenerate case
    where label-conditioned centroids shift more than overall centroids.
    """
    classes = np.unique(np.concatenate([y_from, y_to]))

    mu_from = X_from.mean(axis=0)
    mu_to = X_to.mean(axis=0)
    total_shift = mu_to - mu_from
    total_norm = float(np.linalg.norm(total_shift))

    label_shifts = []
    for c in classes:
        mask_from = y_from == c
        mask_to = y_to == c
        if mask_from.sum() == 0 or mask_to.sum() == 0:
            continue
        mu_c_from = X_from[mask_from].mean(axis=0)
        mu_c_to = X_to[mask_to].mean(axis=0)
        label_shifts.append(mu_c_to - mu_c_from)

    if len(label_shifts) == 0:
        return {"batch_norm": total_norm, "label_norm": 0.0,
                "interaction_norm": 0.0, "total_norm": total_norm, "score": 0.0}

    mean_label_shift = np.mean(label_shifts, axis=0)
    batch_component = mean_label_shift
    batch_norm = float(np.linalg.norm(batch_component))

    within_from = np.zeros_like(mu_from)
    within_to = np.zeros_like(mu_to)
    for c in classes:
        mask_from = y_from == c
        mask_to = y_to == c
        if mask_from.sum() == 0 or mask_to.sum() == 0:
            continue
        within_from += X_from[mask_from].mean(axis=0) * (mask_from.sum() / len(y_from))
        within_to += X_to[mask_to].mean(axis=0) * (mask_to.sum() / len(y_to))

    label_direction_from = np.zeros_like(mu_from)
    label_direction_to = np.zeros_like(mu_to)
    for i, c in enumerate(classes):
        mask_from = y_from == c
        mask_to = y_to == c
        if mask_from.sum() > 0:
            label_direction_from += ((-1) ** i) * X_from[mask_from].mean(axis=0)
        if mask_to.sum() > 0:
            label_direction_to += ((-1) ** i) * X_to[mask_to].mean(axis=0)

    label_change = label_direction_to - label_direction_from
    label_norm = float(np.linalg.norm(label_change))

    interaction = total_shift - batch_component
    interaction_norm = float(np.linalg.norm(interaction))

    score = interaction_norm / (batch_norm + 1e-12)

    return {
        "batch_norm": batch_norm,
        "label_norm": label_norm,
        "interaction_norm": interaction_norm,
        "total_norm": total_norm,
        "score": score,
    }


# ======================================================================
# Dumb baselines (spec 4.4 — geometry must beat these)
# ======================================================================

def centroid_distance(X1, X2):
    """Euclidean distance between cohort mean vectors."""
    return float(np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0)))


def domain_classifier_auc(X1, X2):
    """AUC of a classifier distinguishing cohort 1 from cohort 2.

    Uses logistic regression with L2 penalty. Higher AUC = more
    distinguishable cohorts = more distribution shift.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def variance_ratio(X1, X2):
    """Ratio of total variance between cohorts (larger / smaller)."""
    var1 = float(np.var(X1))
    var2 = float(np.var(X2))
    if min(var1, var2) < 1e-12:
        return float("inf")
    return max(var1, var2) / min(var1, var2)


def all_baselines(X1, X2):
    """Compute all three dumb baselines."""
    return {
        "centroid_dist": centroid_distance(X1, X2),
        "domain_auc": domain_classifier_auc(X1, X2),
        "variance_ratio": variance_ratio(X1, X2),
    }


# ======================================================================
# Sheaf Q test (scalar stalks — for completeness / comparison)
# ======================================================================

def sheaf_q_test(estimates):
    """Sheaf consistency test on scalar estimates.

    estimates: dict of {node_name: {"beta": float, "se": float}}
    Returns (p_value, Q, df, node_scores).
    """
    nodes = sorted(estimates.keys())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    d0 = np.zeros((len(edges), n))
    for e_idx, (u, v) in enumerate(edges):
        d0[e_idx, node_idx[u]] = -1.0
        d0[e_idx, node_idx[v]] = 1.0

    stalks = np.array([estimates[nd]["beta"] for nd in nodes])
    ses = np.array([estimates[nd]["se"] for nd in nodes])

    obs = d0 @ stalks
    Sigma = d0 @ np.diag(ses ** 2) @ d0.T

    _, s, _ = np.linalg.svd(d0, full_matrices=False)
    rank = int(np.sum(s > 1e-10))

    Sigma_pinv = np.linalg.pinv(Sigma, rcond=1e-10)
    Q = float(obs @ Sigma_pinv @ obs)
    df = rank
    p = 1.0 - chi2.cdf(Q, df) if df > 0 else 1.0

    return p, Q, df
