"""#9: Drift monitor — track distribution shift over time.

Maintains a running estimate of shift magnitude so production systems
can detect when a model's input distribution has drifted enough to
require re-calibration or re-training.
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class DriftSnapshot:
    timestamp: str
    mmd: float
    domain_auc: float
    tier: int
    n_samples: int


@dataclass
class DriftMonitor:
    """Stateful monitor that accumulates batches and tracks drift over time."""

    reference: np.ndarray
    history: list[DriftSnapshot] = field(default_factory=list)
    _ref_mean: np.ndarray = field(init=False, repr=False)
    _ref_cov_diag: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._ref_mean = self.reference.mean(axis=0)
        self._ref_cov_diag = np.var(self.reference, axis=0) + 1e-10

    def check(self, batch: np.ndarray, timestamp: str = "") -> DriftSnapshot:
        """Score a new batch against the reference distribution.

        Args:
            batch: (n_batch, d) feature matrix from the incoming stream.
            timestamp: optional ISO-format timestamp string.

        Returns:
            DriftSnapshot with MMD estimate, domain AUC, and tier.
        """
        b_mean = batch.mean(axis=0)

        # Squared MMD estimate (mean embedding distance, standardized)
        diff = self._ref_mean - b_mean
        mmd = float(np.sqrt(np.sum(diff ** 2 / self._ref_cov_diag)))

        # Domain classifier AUC (quick logistic regression)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        X = np.vstack([self.reference, batch])
        y = np.array([0] * len(self.reference) + [1] * len(batch))
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=500, C=1.0)
        try:
            cv = min(5, min(len(self.reference), len(batch)))
            auc = float(np.mean(cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")))
        except Exception:
            auc = 0.5

        # Tier from AUC: <0.55 = no shift (Tier 7+), 0.55-0.7 = mild (Tier 5-6),
        # 0.7-0.85 = moderate (Tier 3-4), >0.85 = severe (Tier 1-2)
        if auc < 0.55:
            tier = 7
        elif auc < 0.65:
            tier = 6
        elif auc < 0.75:
            tier = 5
        elif auc < 0.85:
            tier = 4
        elif auc < 0.95:
            tier = 3
        else:
            tier = 2

        snap = DriftSnapshot(
            timestamp=timestamp,
            mmd=mmd,
            domain_auc=auc,
            tier=tier,
            n_samples=len(batch),
        )
        self.history.append(snap)
        return snap

    def summary(self) -> str:
        if not self.history:
            return "No drift checks recorded."
        lines = [f"Drift monitor: {len(self.history)} checks"]
        lines.append(f"{'Time':<20} | {'MMD':>6} | {'AUC':>5} | {'Tier':>4} | {'N':>5}")
        lines.append("-" * 50)
        for s in self.history:
            lines.append(f"{s.timestamp:<20} | {s.mmd:>6.3f} | {s.domain_auc:>5.3f} | {s.tier:>4} | {s.n_samples:>5}")

        latest = self.history[-1]
        if latest.tier <= 4:
            lines.append(f"\nWARNING: significant drift detected (Tier {latest.tier})")
        return "\n".join(lines)

    def is_drifted(self, tier_threshold: int = 5) -> bool:
        """True if the most recent check shows drift below the tier threshold."""
        if not self.history:
            return False
        return self.history[-1].tier < tier_threshold
