"""Functional alignment verification.

The critical gap exposed by the Dig4Bio case study: distributional
alignment (M2 AUC → 0) does not guarantee functional alignment
(preserving P(Y|X)). SNV + ratio correction matched distributions
perfectly but destroyed predictive signal (CV R² 0.94 → private R² -0.83).

This module checks whether a correction preserves prediction quality
by comparing CV performance before and after correction on labeled data.
"""

import numpy as np
from dataclasses import dataclass, field
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold


@dataclass
class FunctionalCheckResult:
    """Result of checking whether a correction preserves prediction quality."""
    r2_before: float
    r2_after: float
    r2_change: float
    is_destructive: bool
    per_target_r2_before: list[float] = field(default_factory=list)
    per_target_r2_after: list[float] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "DESTRUCTIVE" if self.is_destructive else "SAFE"
        lines = [
            f"Functional alignment: {status}",
            f"  CV R² before correction: {self.r2_before:.3f}",
            f"  CV R² after correction:  {self.r2_after:.3f}",
            f"  Change: {self.r2_change:+.3f}",
        ]
        if self.per_target_r2_before:
            lines.append("  Per-target breakdown:")
            for i, (b, a) in enumerate(zip(self.per_target_r2_before, self.per_target_r2_after)):
                delta = a - b
                flag = " ← DROP" if delta < -0.05 else ""
                lines.append(f"    target {i}: {b:.3f} → {a:.3f} ({delta:+.3f}){flag}")
        return "\n".join(lines)


def check_functional_alignment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_source_corrected: np.ndarray,
    destructive_threshold: float = -0.05,
    cv: int = 5,
    model_type: str = "ridge",
) -> FunctionalCheckResult:
    """Check if a correction preserves prediction quality on labeled data.

    Compares CV R² of a model trained on the original data vs. the
    corrected data. If R² drops significantly, the correction is
    destroying predictive signal.

    This catches the failure mode where distributional alignment (M2 AUC → 0)
    coexists with functional misalignment (worse predictions). A correction
    that scores well on M2 but fails this check is removing signal, not drift.

    Args:
        X_source: (n, d) original labeled feature matrix.
        y_source: (n,) or (n, k) target values.
        X_source_corrected: (n, d) the same samples after correction has been
            applied. To test a source→target correction, apply the correction
            to source data (using source as both reference and target) and
            check if the model still works on the corrected features.
        destructive_threshold: R² drop below this flags the correction as
            destructive. Default -0.05 (5% drop).
        cv: number of cross-validation folds.
        model_type: "ridge" or "pls".
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    y = np.atleast_2d(y_source)
    if y.shape[0] == len(X_source) and y.ndim == 2:
        pass
    else:
        y = y_source.reshape(-1, 1) if y_source.ndim == 1 else y_source

    def _make_model():
        if model_type == "pls":
            return PLSRegression(n_components=min(5, X_source.shape[1], len(X_source) - 1))
        return Ridge(alpha=1.0)

    r2_before_list = []
    r2_after_list = []

    for j in range(y.shape[1]):
        yj = y[:, j]
        scores_before = cross_val_score(_make_model(), X_source, yj, cv=kf, scoring="r2")
        scores_after = cross_val_score(_make_model(), X_source_corrected, yj, cv=kf, scoring="r2")
        r2_before_list.append(float(np.mean(scores_before)))
        r2_after_list.append(float(np.mean(scores_after)))

    mean_before = float(np.mean(r2_before_list))
    mean_after = float(np.mean(r2_after_list))
    change = mean_after - mean_before

    return FunctionalCheckResult(
        r2_before=mean_before,
        r2_after=mean_after,
        r2_change=change,
        is_destructive=change < destructive_threshold,
        per_target_r2_before=r2_before_list,
        per_target_r2_after=r2_after_list,
        details={
            "n_samples": len(X_source),
            "n_features": X_source.shape[1],
            "n_targets": y.shape[1],
            "cv_folds": cv,
            "model_type": model_type,
            "threshold": destructive_threshold,
        },
    )
