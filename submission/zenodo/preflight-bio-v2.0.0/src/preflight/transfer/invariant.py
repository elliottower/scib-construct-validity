"""#8: Shift-invariant feature selection.

Identify features that are stable across source and target distributions.
Training on invariant features gives a model that's robust by construction,
even without explicit domain correction.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InvariantFeatureResult:
    mask: np.ndarray
    n_invariant: int
    n_total: int
    pct_invariant: float
    feature_shift_scores: np.ndarray

    def summary(self) -> str:
        return (
            f"Invariant features: {self.n_invariant}/{self.n_total} "
            f"({self.pct_invariant:.0%})\n"
            f"Mean shift score: {np.mean(self.feature_shift_scores):.4f}\n"
            f"Most shifted indices: {np.argsort(self.feature_shift_scores)[-5:][::-1].tolist()}"
        )


def invariant_features(
    source: np.ndarray,
    target: np.ndarray,
    ratio_threshold: float = 0.05,
    ks_threshold: float = 0.1,
) -> InvariantFeatureResult:
    """Identify features that are stable across source and target.

    A feature is invariant if:
      (a) |ratio_of_means - 1| < ratio_threshold, AND
      (b) the two-sample KS statistic < ks_threshold

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        ratio_threshold: max allowable deviation of mean ratio from 1.
        ks_threshold: max allowable KS statistic.

    Returns:
        InvariantFeatureResult with boolean mask and per-feature shift scores.
    """
    from scipy.stats import ks_2samp

    d = source.shape[1]
    s_mean = source.mean(axis=0)
    t_mean = target.mean(axis=0)
    ratios = s_mean / (t_mean + 1e-10)
    ratio_dev = np.abs(ratios - 1.0)

    ks_stats = np.zeros(d)
    for j in range(d):
        ks_stats[j] = ks_2samp(source[:, j], target[:, j]).statistic

    # Combined shift score: geometric mean of ratio deviation and KS stat
    shift_scores = np.sqrt(ratio_dev * ks_stats)

    mask = (ratio_dev < ratio_threshold) & (ks_stats < ks_threshold)

    return InvariantFeatureResult(
        mask=mask,
        n_invariant=int(mask.sum()),
        n_total=d,
        pct_invariant=float(mask.mean()),
        feature_shift_scores=shift_scores,
    )
