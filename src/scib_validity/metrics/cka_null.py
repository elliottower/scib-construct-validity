"""CKA null saturation bound for embedding evaluation.

At high embedding dimension d, linear CKA between independent random
centroid matrices converges to a predictable floor that depends only on
the number of cell types k and the embedding dimension d.  When d >= 512
and k is small (typical in single-cell benchmarks), random embeddings
score higher than trained models.

The closed-form approximation is:

    E[CKA] ≈ 1 - 1.06 * (k - 1) / (d + k)

where k = number of shared cell types and d = embedding dimension.

Reference:
    Tower (2026). Construct Validity Failure in Single-Cell Embedding
    Evaluation: Null Saturation Bounds and Source Classifier Confidence.
"""
import numpy as np


def cka_null(k: int, d: int) -> float:
    """Expected CKA between independent random centroid matrices.

    Args:
        k: Number of cell types (centroid rows).
        d: Embedding dimension.

    Returns:
        Approximate expected CKA under the null (independent Gaussian centroids).
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 (got {k})")
    if d < 1:
        raise ValueError(f"d must be >= 1 (got {d})")
    return 1.0 - 1.06 * (k - 1) / (d + k)


def cka_certifiable(observed_cka: float, k: int, d: int, margin: float = 0.0) -> bool:
    """Check whether an observed CKA exceeds the null saturation floor.

    Args:
        observed_cka: The CKA value to test.
        k: Number of cell types.
        d: Embedding dimension.
        margin: Safety margin above the null floor (default 0).

    Returns:
        True if observed_cka > cka_null(k, d) + margin, meaning the CKA
        score carries information beyond what random embeddings would produce.
    """
    return observed_cka > cka_null(k, d) + margin


def cka_null_empirical(k: int, d: int, n_trials: int = 5000, seed: int = 42) -> dict:
    """Compute empirical CKA null distribution via Monte Carlo.

    Generates pairs of independent Gaussian centroid matrices and
    computes linear CKA for each pair.

    Args:
        k: Number of centroids (rows).
        d: Embedding dimension (columns).
        n_trials: Number of random pairs to generate.
        seed: Random seed.

    Returns:
        Dict with keys: mean, std, ci_lower, ci_upper (2.5th/97.5th percentiles).
    """
    rng = np.random.default_rng(seed)
    H = np.eye(k) - np.ones((k, k)) / k
    vals = []
    for _ in range(n_trials):
        X = rng.standard_normal((k, d))
        Y = rng.standard_normal((k, d))
        Kx = H @ X @ X.T @ H
        Ky = H @ Y @ Y.T @ H
        num = np.trace(Kx @ Ky)
        denom = np.sqrt(np.trace(Kx @ Kx) * np.trace(Ky @ Ky))
        vals.append(num / denom if denom > 1e-12 else 0.0)
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "ci_lower": float(np.percentile(vals, 2.5)),
        "ci_upper": float(np.percentile(vals, 97.5)),
    }
