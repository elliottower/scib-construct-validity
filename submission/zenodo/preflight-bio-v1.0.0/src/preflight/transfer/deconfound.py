"""Causal deconfounding — remove session/batch direction before correction.

When session S and outcome Y both cause features X, naive correction
removes S→X indiscriminately, potentially damaging Y→X signal wherever
the two share features. Double ML (Chernozhukov et al., 2018) projects
out the session direction first, making subsequent corrections less
likely to distort analyte/outcome signal.

From the Dig4Bio case study: +14% private R² for simple linear models,
neutral-to-negative for tree ensembles that already handle confounding
implicitly.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class DeconfoundResult:
    """Result of session deconfounding."""
    source_deconfounded: np.ndarray
    target_deconfounded: np.ndarray
    n_directions_removed: int
    session_direction: np.ndarray
    explained_variance_removed: float
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Deconfounded: removed {self.n_directions_removed} session direction(s)\n"
            f"  Variance removed: {self.explained_variance_removed:.1%}\n"
            f"  Session direction norm: {np.linalg.norm(self.session_direction):.4f}"
        )


def deconfound(
    source: np.ndarray,
    target: np.ndarray,
    n_directions: int = 1,
    method: str = "logistic",
) -> DeconfoundResult:
    """Remove session-predictive directions from both datasets.

    Fits a classifier to distinguish source from target, extracts the
    decision direction in original feature space, and projects it out.
    The resulting features retain analyte/outcome signal that is
    orthogonal to the session confound.

    Best combined with a correction method (ratio, location-scale) applied
    AFTER deconfounding — the correction then operates on features from
    which the session direction has been removed.

    Args:
        source: (n_source, d) feature matrix (labeled, e.g. calibration).
        target: (n_target, d) feature matrix (unlabeled, e.g. test).
        n_directions: number of confounding directions to remove. 1 is
            usually sufficient; >3 tends to remove too much signal.
        method: "logistic" (default) or "pca_residual".
            logistic: remove the direction most predictive of session.
            pca_residual: remove top PCA directions of the session
                mean difference (faster, less targeted).
    """
    if method == "logistic":
        return _deconfound_logistic(source, target, n_directions)
    elif method == "pca_residual":
        return _deconfound_pca(source, target, n_directions)
    raise ValueError(f"Unknown method: {method}")


def _deconfound_logistic(
    source: np.ndarray, target: np.ndarray, n_directions: int,
) -> DeconfoundResult:
    X_all = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    all_directions = []
    X_residual = X_all.copy()

    for i in range(n_directions):
        scaler_i = StandardScaler()
        X_s = scaler_i.fit_transform(X_residual)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X_s, y)

        # Session direction in original (unscaled) space
        w = clf.coef_[0] / scaler_i.scale_
        w = w / (np.linalg.norm(w) + 1e-10)
        all_directions.append(w)

        # Project out this direction from the data
        X_centered = X_residual - X_residual.mean(axis=0)
        X_residual = X_centered - np.outer(X_centered @ w, w) + X_residual.mean(axis=0)

    # Compute how much variance was removed
    total_var = np.var(X_all, axis=0).sum()
    residual_var = np.var(X_residual, axis=0).sum()
    var_removed = max(0, (total_var - residual_var) / (total_var + 1e-10))

    # Split back and re-center to source mean
    source_deconf = X_residual[:len(source)]
    target_deconf_raw = X_residual[len(source):]

    # Re-center target to source mean (the deconfounding changes means)
    target_deconf = target_deconf_raw - target_deconf_raw.mean(axis=0) + source_deconf.mean(axis=0)

    return DeconfoundResult(
        source_deconfounded=source_deconf,
        target_deconfounded=target_deconf,
        n_directions_removed=n_directions,
        session_direction=all_directions[0] if all_directions else np.zeros(source.shape[1]),
        explained_variance_removed=float(var_removed),
        details={
            "method": "logistic",
            "n_directions": n_directions,
        },
    )


def _deconfound_pca(
    source: np.ndarray, target: np.ndarray, n_directions: int,
) -> DeconfoundResult:
    diff = source.mean(axis=0) - target.mean(axis=0)
    X_all = np.vstack([source, target])

    from scipy.linalg import svd

    # Use the mean difference + PCA of concatenated data to find
    # session-predictive directions
    X_centered = X_all - X_all.mean(axis=0)
    _, _, Vt = svd(X_centered, full_matrices=False)

    # Project diff onto PCA space, find which PCA directions carry session info
    diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
    projections = np.abs(Vt @ diff_norm)
    top_k = np.argsort(projections)[-n_directions:][::-1]

    # Remove those PCA directions
    V_remove = Vt[top_k].T  # (d, n_directions)
    X_residual = X_centered - X_centered @ V_remove @ V_remove.T + X_all.mean(axis=0)

    total_var = np.var(X_all, axis=0).sum()
    residual_var = np.var(X_residual, axis=0).sum()
    var_removed = max(0, (total_var - residual_var) / (total_var + 1e-10))

    source_deconf = X_residual[:len(source)]
    target_deconf_raw = X_residual[len(source):]
    target_deconf = target_deconf_raw - target_deconf_raw.mean(axis=0) + source_deconf.mean(axis=0)

    return DeconfoundResult(
        source_deconfounded=source_deconf,
        target_deconfounded=target_deconf,
        n_directions_removed=n_directions,
        session_direction=diff_norm,
        explained_variance_removed=float(var_removed),
        details={
            "method": "pca_residual",
            "n_directions": n_directions,
            "pca_directions_used": top_k.tolist(),
        },
    )
