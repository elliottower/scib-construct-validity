"""Transportability measures: geometric, distributional, and classifier-based.

Core primitives for measuring cross-site embedding shift.
"""
import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance_nd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC


def principal_angles(U1, U2):
    """Principal angles between two subspaces (orthonormal bases)."""
    M = U1.T @ U2
    svals = linalg.svdvals(M)
    svals = np.clip(svals, -1, 1)
    return np.arccos(svals)


def geodesic_distance(U1, U2):
    """Grassmannian geodesic distance = L2 norm of principal angles."""
    return float(np.linalg.norm(principal_angles(U1, U2)))


def top_k_subspace(X, k):
    """Top-k PCA subspace from data matrix X (samples x features).

    Returns orthonormal basis (features x k) and explained variance ratios.
    """
    X_centered = X - X.mean(axis=0)
    U, s, Vt = linalg.svd(X_centered, full_matrices=False)
    explained = (s ** 2) / np.sum(s ** 2)
    return Vt[:k].T, explained[:k]


def sheaf_h1_two_cohort(X1, X2, k):
    """Sheaf H^1 test: geodesic distance + explained variance for two cohorts."""
    U1, ev1 = top_k_subspace(X1, k)
    U2, ev2 = top_k_subspace(X2, k)
    angles = principal_angles(U1, U2)
    gdist = float(np.linalg.norm(angles))
    return {
        "geodesic_dist": gdist,
        "principal_angles": angles,
        "explained_var_c1": ev1,
        "explained_var_c2": ev2,
    }


def centroid_distance(X1, X2):
    """Euclidean distance between cohort mean vectors."""
    return float(np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0)))


def domain_classifier_auc(X1, X2):
    """AUC of a logistic regression distinguishing cohort 1 from cohort 2."""
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def spectral_matching_degradation(X1, X2, labels1, labels2):
    """Measure cross-instrument compound identification degradation.

    Restricted to compounds appearing in both instruments. For shared
    compounds: within-instrument hit@1 (nearest neighbor has same label)
    vs cross-instrument hit@1. Degradation = within - cross.

    Falls back to unrestricted matching if fewer than 5 compounds overlap.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    shared = set(labels1) & set(labels2)

    if len(shared) >= 5:
        mask1 = np.array([l in shared for l in labels1])
        mask2 = np.array([l in shared for l in labels2])
        X1s, L1s = X1[mask1], labels1[mask1]
        X2s, L2s = X2[mask2], labels2[mask2]
    else:
        X1s, L1s = X1, labels1
        X2s, L2s = X2, labels2

    sim_within = cosine_similarity(X1s)
    np.fill_diagonal(sim_within, -np.inf)
    nn_within = sim_within.argmax(axis=1)
    within_hit1 = float(np.mean(L1s[nn_within] == L1s))

    sim_cross = cosine_similarity(X2s, X1s)
    nn_cross = sim_cross.argmax(axis=1)
    cross_hit1 = float(np.mean(L1s[nn_cross] == L2s))

    return within_hit1, cross_hit1, within_hit1 - cross_hit1


def sheaf_h1_multi_cohort(cohorts, k):
    """Sheaf H¹ obstruction across multiple cohorts.

    Computes the Frobenius norm of the H¹ coboundary matrix.
    Higher value = less consistent subspace alignment across cohorts.
    """
    bases = []
    for X in cohorts:
        U, _ = top_k_subspace(X, k)
        bases.append(U)

    n = len(bases)
    total_obstruction = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            angles = principal_angles(bases[i], bases[j])
            total_obstruction += float(np.sum(angles ** 2))
            pair_count += 1

    mean_obstruction = total_obstruction / max(pair_count, 1)
    return {
        "h1_norm": float(np.sqrt(total_obstruction)),
        "mean_pair_obstruction": float(np.sqrt(mean_obstruction)),
        "n_cohorts": n,
        "n_pairs": pair_count,
    }


def permutation_null(X1, X2, k, n_perms=500):
    """Permutation null for geodesic distance.

    Pools X1 and X2, randomly splits into two groups of the same sizes,
    computes geodesic distance. Returns null distribution.
    """
    X_pool = np.vstack([X1, X2])
    n1 = len(X1)
    rng = np.random.default_rng(42)
    null_dists = np.empty(n_perms)
    for i in range(n_perms):
        idx = rng.permutation(len(X_pool))
        X1_perm = X_pool[idx[:n1]]
        X2_perm = X_pool[idx[n1:]]
        U1, _ = top_k_subspace(X1_perm, k)
        U2, _ = top_k_subspace(X2_perm, k)
        null_dists[i] = geodesic_distance(U1, U2)
    return null_dists


def bootstrap_correlation(xs, ys, n_boot=2000, seed=42):
    """Bootstrap Spearman correlation with 95% CI."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    n = len(xs)
    xs, ys = np.asarray(xs), np.asarray(ys)
    observed_r, observed_p = stats.spearmanr(xs, ys)
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        r, _ = stats.spearmanr(xs[idx], ys[idx])
        boot_rs[i] = r
    ci_lo = float(np.percentile(boot_rs, 2.5))
    ci_hi = float(np.percentile(boot_rs, 97.5))
    return {
        "r": float(observed_r),
        "p": float(observed_p),
        "ci_95": (ci_lo, ci_hi),
        "boot_std": float(np.std(boot_rs)),
    }


def mmd_rbf(X1, X2, gamma=None):
    """Maximum Mean Discrepancy with RBF kernel.

    Standard distribution shift metric from Gretton et al. (2012).
    gamma defaults to 1/d (median heuristic approximation).
    """
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    K11 = np.exp(-gamma * cdist(X1, X1, 'sqeuclidean'))
    K22 = np.exp(-gamma * cdist(X2, X2, 'sqeuclidean'))
    K12 = np.exp(-gamma * cdist(X1, X2, 'sqeuclidean'))
    n1, n2 = len(X1), len(X2)
    return float(K11.sum() / (n1 * n1) + K22.sum() / (n2 * n2) - 2 * K12.sum() / (n1 * n2))


def sliced_wasserstein(X1, X2, n_projections=100, seed=42):
    """Sliced Wasserstein distance (1D projections averaged).

    Approximation of Wasserstein distance tractable in high dimensions.
    """
    rng = np.random.default_rng(seed)
    d = X1.shape[1]
    projections = rng.standard_normal((n_projections, d))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    dists = []
    for proj in projections:
        p1 = X1 @ proj
        p2 = X2 @ proj
        p1_sorted = np.sort(p1)
        p2_sorted = np.sort(p2)
        n1, n2 = len(p1), len(p2)
        if n1 == n2:
            dists.append(float(np.mean(np.abs(p1_sorted - p2_sorted))))
        else:
            grid = np.linspace(0, 1, max(n1, n2), endpoint=False)
            interp1 = np.interp(grid, np.linspace(0, 1, n1, endpoint=False), p1_sorted)
            interp2 = np.interp(grid, np.linspace(0, 1, n2, endpoint=False), p2_sorted)
            dists.append(float(np.mean(np.abs(interp1 - interp2))))
    return float(np.mean(dists))


def proxy_a_distance(X1, X2):
    """Proxy A-distance (Ben-David et al., 2007).

    d_A = 2(1 - 2*error) where error is the classification error
    of a linear SVM trained to distinguish the two domains.
    Higher = more different distributions. Range [0, 2].
    """
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LinearSVC(C=1.0, max_iter=20000, dual=False)
    scores = cross_val_score(clf, X_s, y, cv=5, scoring='accuracy')
    error = 1.0 - float(scores.mean())
    return float(2.0 * (1.0 - 2.0 * error))


def all_distances(X1, X2, k=10, compute_expensive=True):
    """Compute all distance metrics between two cohort embeddings.

    When compute_expensive=False, skips MMD, Sliced Wasserstein, and proxy
    A-distance (these don't depend on k and only need to be computed once).
    """
    sheaf = sheaf_h1_two_cohort(X1, X2, k)
    result = {
        "geodesic_dist": sheaf["geodesic_dist"],
        "principal_angles": sheaf["principal_angles"],
        "centroid_dist": centroid_distance(X1, X2),
        "domain_auc": domain_classifier_auc(X1, X2),
        "explained_var_c1": sheaf["explained_var_c1"],
        "explained_var_c2": sheaf["explained_var_c2"],
    }
    if compute_expensive:
        result["mmd"] = mmd_rbf(X1, X2)
        result["sliced_wasserstein"] = sliced_wasserstein(X1, X2)
        result["proxy_a_distance"] = proxy_a_distance(X1, X2)
    return result
