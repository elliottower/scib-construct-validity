"""#1: Predict target performance from source CV score + divergence.

Uses a simplified Ben-David bound:
  target_error ≤ source_error + divergence_term + irreducible_term

The divergence_term is estimated from the domain classifier AUC (M2).
The irreducible_term is estimated from the Grassmannian geodesic (M1).
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


@dataclass
class PerformancePrediction:
    estimated_target_score: float
    lower_bound: float
    upper_bound: float
    source_cv_score: float
    divergence_penalty: float
    irreducible_gap: float
    domain_auc: float
    details: dict

    def summary(self) -> str:
        return (
            f"Source CV: {self.source_cv_score:.3f}\n"
            f"Estimated target: {self.estimated_target_score:.3f} "
            f"[{self.lower_bound:.3f}, {self.upper_bound:.3f}]\n"
            f"Divergence penalty: -{self.divergence_penalty:.3f} "
            f"(domain AUC={self.domain_auc:.3f})\n"
            f"Irreducible gap: -{self.irreducible_gap:.3f}"
        )


def predict_target_performance(
    source: np.ndarray,
    target: np.ndarray,
    source_cv_score: float,
    score_type: str = "r2",
) -> PerformancePrediction:
    """Estimate expected target performance from source CV + distribution divergence.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        source_cv_score: cross-validated score on source (R² or accuracy).
        score_type: "r2" or "accuracy" — affects how divergence maps to penalty.

    Returns:
        PerformancePrediction with estimated range.
    """
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    auc_scores = cross_val_score(clf, X_s, y, cv=5, scoring="roc_auc")
    domain_auc = float(np.mean(auc_scores))

    # Divergence penalty: maps AUC ∈ [0.5, 1.0] to penalty ∈ [0, 1]
    # AUC = 0.5 means no shift (penalty = 0), AUC = 1.0 means total shift
    d_hat = 2.0 * (domain_auc - 0.5)
    d_hat = max(0.0, min(1.0, d_hat))

    # Subspace divergence as irreducible gap estimate
    from scipy.linalg import svd
    k = min(10, min(source.shape) - 1, min(target.shape) - 1)
    def _basis(X, k):
        Xc = X - X.mean(axis=0)
        _, s, Vt = svd(Xc, full_matrices=False)
        return Vt[:k].T
    U_s, U_t = _basis(source, k), _basis(target, k)
    M = U_s.T @ U_t
    svals = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    angles = np.arccos(svals)
    geo = float(np.linalg.norm(angles))
    max_geo = (np.pi / 2) * np.sqrt(k)
    subspace_gap = geo / max_geo * 0.2  # scaled to [0, 0.2]

    if score_type == "r2":
        # For R², divergence can make scores negative
        penalty = d_hat ** 1.5 * (1.0 + source_cv_score)
        estimated = source_cv_score - penalty - subspace_gap
        margin = 0.15 + 0.1 * d_hat
    else:
        # For accuracy, penalty is bounded by source score
        penalty = d_hat * source_cv_score * 0.5
        estimated = source_cv_score - penalty - subspace_gap
        margin = 0.10 + 0.05 * d_hat

    return PerformancePrediction(
        estimated_target_score=estimated,
        lower_bound=estimated - margin,
        upper_bound=min(source_cv_score, estimated + margin),
        source_cv_score=source_cv_score,
        divergence_penalty=penalty,
        irreducible_gap=subspace_gap,
        domain_auc=domain_auc,
        details={
            "d_hat": d_hat,
            "geodesic_normalized": geo / max_geo,
            "subspace_gap_scaled": subspace_gap,
        },
    )
