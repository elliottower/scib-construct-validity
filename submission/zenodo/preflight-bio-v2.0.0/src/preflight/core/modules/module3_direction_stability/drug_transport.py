"""Transport metrics for drug perturbation signatures across cell lines.

Adapts the bracket norm concept from neural population geometry to
drug perturbation biology. In neural data, bracket norm measures how
the choice-coding direction rotates as evidence strength changes.
Here, the "choice coding" is the drug perturbation signature (already
drug - control), and the contextual axis is cell line identity.

Key insight: a drug whose perturbation signature points in the same
direction across cell lines has a mechanism that transports. One whose
direction rotates across cell lines has a context-dependent mechanism.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd


def direction_stability(signatures: dict[str, np.ndarray]) -> dict:
    """How stable is the drug's response direction across cell lines?

    This is the bracket norm analogue: high instability = the mechanism
    direction rotates across contexts, like high bracket norm means the
    choice-coding direction rotates across evidence levels.

    Args:
        signatures: {cell_line_name: (n_genes,) z-score vector}

    Returns:
        dict with direction_instability (bracket norm analogue),
        mean/min pairwise cosine, mean rotation angle.
    """
    names = sorted(signatures.keys())
    if len(names) < 2:
        return {"direction_instability": np.nan, "n_cell_lines": len(names)}

    vecs = np.array([signatures[n] for n in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    unit_vecs = vecs / norms

    cosines = unit_vecs @ unit_vecs.T
    np.fill_diagonal(cosines, np.nan)
    pairwise_cosines = cosines[~np.isnan(cosines)]

    mean_cos = float(np.mean(pairwise_cosines))
    min_cos = float(np.min(pairwise_cosines))

    angles = np.arccos(np.clip(pairwise_cosines, -1, 1))
    mean_angle_deg = float(np.degrees(np.mean(angles)))

    return {
        "direction_instability": 1.0 - mean_cos,
        "mean_pairwise_cosine": mean_cos,
        "min_pairwise_cosine": min_cos,
        "mean_rotation_angle_deg": mean_angle_deg,
        "n_cell_lines": len(names),
        "cell_lines": names,
    }


def magnitude_stability(signatures: dict[str, np.ndarray]) -> dict:
    """How consistent is the drug's effect magnitude across cell lines?

    A drug with stable direction but unstable magnitude may have a
    dose-sensitive mechanism that still transports qualitatively.

    Args:
        signatures: {cell_line_name: (n_genes,) z-score vector}
    """
    names = sorted(signatures.keys())
    norms = np.array([np.linalg.norm(signatures[n]) for n in names])

    if len(norms) < 2 or np.mean(norms) < 1e-10:
        return {"magnitude_cv": np.nan, "mean_norm": 0.0}

    return {
        "magnitude_cv": float(np.std(norms) / np.mean(norms)),
        "mean_norm": float(np.mean(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }


def frechet_variance(signatures: dict[str, np.ndarray]) -> float:
    """Frechet variance on the unit sphere.

    Measures dispersion of normalized signatures — the spherical analogue
    of variance. Zero when all signatures point in the same direction.
    """
    names = sorted(signatures.keys())
    if len(names) < 2:
        return np.nan

    vecs = np.array([signatures[n] for n in names])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    unit_vecs = vecs / norms

    frechet_mean = unit_vecs.mean(axis=0)
    frechet_mean = frechet_mean / (np.linalg.norm(frechet_mean) + 1e-10)

    cosines = unit_vecs @ frechet_mean
    angles = np.arccos(np.clip(cosines, -1, 1))
    return float(np.mean(angles**2))


def subspace_transport(
    signatures_multi: dict[str, np.ndarray],
    k: int = 3,
) -> dict:
    """Grassmannian transport when multiple signatures per cell line.

    When a drug has signatures at multiple doses or time points within
    a cell line, the response defines a subspace. This measures whether
    response subspaces are consistent across cell lines.

    Args:
        signatures_multi: {cell_line: (n_sigs, n_genes) matrix}
            Must have n_sigs >= k per cell line.
        k: subspace dimension
    """
    names = sorted(signatures_multi.keys())
    valid = {n: signatures_multi[n] for n in names if signatures_multi[n].shape[0] >= k}
    if len(valid) < 2:
        return {"mean_grassmannian_distance": np.nan, "n_valid_cell_lines": len(valid)}

    subspaces = {}
    for name, X in valid.items():
        X_centered = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = svd(X_centered, full_matrices=False)
        subspaces[name] = Vt[:k].T  # (n_genes, k)

    sub_names = sorted(subspaces.keys())
    n = len(sub_names)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            U, V = subspaces[sub_names[i]], subspaces[sub_names[j]]
            _, s, _ = svd(U.T @ V, full_matrices=False)
            s = np.clip(s, -1, 1)
            angles = np.arccos(s)
            d = float(np.sqrt(np.sum(angles**2)))
            distances[i, j] = d
            distances[j, i] = d

    triu = distances[np.triu_indices(n, k=1)]
    return {
        "mean_grassmannian_distance": float(np.mean(triu)),
        "max_grassmannian_distance": float(np.max(triu)),
        "n_valid_cell_lines": len(valid),
        "cell_lines": sub_names,
    }


def gene_level_consistency(signatures: dict[str, np.ndarray], top_k: int = 50) -> dict:
    """Do the same genes drive the signature across cell lines?

    Measures whether the top up/down-regulated genes are consistent,
    independent of the full-vector direction metric.

    Args:
        signatures: {cell_line: (n_genes,) z-score vector}
        top_k: number of top genes per direction to compare
    """
    names = sorted(signatures.keys())
    if len(names) < 2:
        return {"mean_top_gene_jaccard": np.nan}

    top_up_sets = {}
    top_down_sets = {}
    for name in names:
        s = signatures[name]
        up_idx = set(np.argsort(s)[-top_k:])
        down_idx = set(np.argsort(s)[:top_k])
        top_up_sets[name] = up_idx
        top_down_sets[name] = down_idx

    up_jaccards = []
    down_jaccards = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            up_j = len(top_up_sets[n1] & top_up_sets[n2]) / len(top_up_sets[n1] | top_up_sets[n2])
            down_j = len(top_down_sets[n1] & top_down_sets[n2]) / len(top_down_sets[n1] | top_down_sets[n2])
            up_jaccards.append(up_j)
            down_jaccards.append(down_j)

    return {
        "mean_top_gene_jaccard_up": float(np.mean(up_jaccards)),
        "mean_top_gene_jaccard_down": float(np.mean(down_jaccards)),
        "mean_top_gene_jaccard": float(np.mean(up_jaccards + down_jaccards)),
    }


def compute_all_transport_metrics(
    signatures: dict[str, np.ndarray],
    signatures_multi: dict[str, np.ndarray] | None = None,
    top_k_genes: int = 50,
    subspace_k: int = 3,
) -> dict:
    """Compute all transport metrics for a single drug.

    Args:
        signatures: {cell_line: (n_genes,) consensus z-score vector}
        signatures_multi: optional {cell_line: (n_sigs, n_genes)} for subspace analysis
        top_k_genes: for gene-level consistency
        subspace_k: subspace dimension for Grassmannian
    """
    result = {}
    result.update(direction_stability(signatures))
    result.update(magnitude_stability(signatures))
    result["frechet_variance"] = frechet_variance(signatures)
    result.update(gene_level_consistency(signatures, top_k=top_k_genes))

    if signatures_multi is not None:
        sub = subspace_transport(signatures_multi, k=subspace_k)
        result["mean_grassmannian_distance"] = sub["mean_grassmannian_distance"]
        result["max_grassmannian_distance"] = sub.get("max_grassmannian_distance", np.nan)

    return result
