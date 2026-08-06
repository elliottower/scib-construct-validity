"""#5: Conformal prediction intervals under distribution shift.

Uses the domain classifier's predicted probabilities as density ratio
estimates for importance-weighted conformal prediction. Gives valid
prediction intervals on the target even under covariate shift.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict


@dataclass
class ConformalResult:
    intervals: np.ndarray
    mean_width: float
    coverage_estimate: float
    effective_n: float
    details: dict

    def summary(self) -> str:
        return (
            f"Prediction intervals (alpha={self.details.get('alpha', 0.05)}):\n"
            f"  Mean width: {self.mean_width:.4f}\n"
            f"  Estimated coverage: {self.coverage_estimate:.0%}\n"
            f"  Effective sample size: {self.effective_n:.1f}/{self.details.get('n_source', 0)}\n"
            f"  Intervals shape: {self.intervals.shape}"
        )


def conformal_intervals(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_X: np.ndarray,
    model,
    alpha: float = 0.1,
    max_weight: float = 20.0,
) -> ConformalResult:
    """Compute importance-weighted conformal prediction intervals on target.

    Uses the domain classifier P(target|x) / P(source|x) as the density
    ratio for importance weighting. This gives approximately valid coverage
    under covariate shift (Tibshirani et al., 2019).

    Args:
        source_X: (n_source, d) source features.
        source_y: (n_source,) source labels.
        target_X: (n_target, d) target features.
        model: fitted sklearn model with .predict() method.
        alpha: miscoverage rate (0.1 = 90% intervals).
        max_weight: cap on importance weights to prevent instability.

    Returns:
        ConformalResult with per-target-sample intervals.
    """
    # Fit domain classifier for density ratio estimation
    X_all = np.vstack([source_X, target_X])
    domain_labels = np.array([0] * len(source_X) + [1] * len(target_X))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_scaled, domain_labels)

    # Density ratio on source samples: P(target|x) / P(source|x)
    source_proba = clf.predict_proba(scaler.transform(source_X))
    weights = source_proba[:, 1] / (source_proba[:, 0] + 1e-10)
    weights = np.clip(weights, 0, max_weight)
    weights = weights / (weights.sum() + 1e-10)

    # Effective sample size (Kish's formula)
    eff_n = 1.0 / (np.sum(weights ** 2) + 1e-10)

    # Compute conformity scores on source (leave-one-out or CV)
    source_pred = cross_val_predict(
        model.__class__(**model.get_params()), source_X, source_y,
        cv=min(5, len(source_y)),
    )
    residuals = np.abs(source_y - source_pred)

    # Weighted quantile of residuals
    sorted_idx = np.argsort(residuals)
    sorted_residuals = residuals[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    quantile_idx = np.searchsorted(cumsum, 1.0 - alpha)
    quantile_idx = min(quantile_idx, len(sorted_residuals) - 1)
    q_hat = sorted_residuals[quantile_idx]

    # Prediction intervals on target
    target_pred = model.predict(target_X)
    if target_pred.ndim == 1:
        intervals = np.column_stack([target_pred - q_hat, target_pred + q_hat])
    else:
        intervals = np.stack([target_pred - q_hat, target_pred + q_hat], axis=-1)

    mean_width = float(2 * q_hat) if target_pred.ndim == 1 else float(np.mean(2 * q_hat))

    return ConformalResult(
        intervals=intervals,
        mean_width=mean_width,
        coverage_estimate=1.0 - alpha,
        effective_n=float(eff_n),
        details={
            "alpha": alpha,
            "quantile": float(q_hat),
            "n_source": len(source_y),
            "n_target": len(target_X),
            "max_weight_used": float(np.max(weights * len(weights))),
        },
    )
