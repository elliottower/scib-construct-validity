"""#6: Multi-source optimal weighting.

When multiple source domains are available, weight each by similarity to
the target. Closer sources get higher weight in the ensemble.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


@dataclass
class MultiSourceWeights:
    weights: np.ndarray
    source_names: list[str]
    distances: np.ndarray
    details: dict

    def summary(self) -> str:
        lines = [f"{'Source':<20} | {'Weight':>7} | {'AUC':>5}"]
        lines.append("-" * 40)
        for name, w, d in zip(self.source_names, self.weights, self.distances):
            lines.append(f"{name:<20} | {w:>7.3f} | {d:>5.3f}")
        return "\n".join(lines)


def multi_source_weights(
    sources: list[np.ndarray],
    target: np.ndarray,
    source_names: list[str] | None = None,
    method: str = "inverse_auc",
) -> MultiSourceWeights:
    """Compute per-source weights proportional to similarity with target.

    Args:
        sources: list of (n_i, d) feature matrices, one per source domain.
        target: (n_target, d) target feature matrix.
        source_names: optional names for each source.
        method: "inverse_auc" (default) or "inverse_mmd".

    Returns:
        MultiSourceWeights with normalized weights.
    """
    if source_names is None:
        source_names = [f"source_{i}" for i in range(len(sources))]

    distances = np.zeros(len(sources))

    for i, src in enumerate(sources):
        if method == "inverse_auc":
            X = np.vstack([src, target])
            y = np.array([0] * len(src) + [1] * len(target))
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            try:
                scores = cross_val_score(clf, X_s, y, cv=min(5, min(len(src), len(target))),
                                          scoring="roc_auc")
                distances[i] = float(np.mean(scores))
            except Exception:
                distances[i] = 1.0

        elif method == "inverse_mmd":
            s_mean = src.mean(axis=0)
            t_mean = target.mean(axis=0)
            distances[i] = float(np.linalg.norm(s_mean - t_mean))

    # Convert distances to weights (closer = higher weight)
    if method == "inverse_auc":
        # AUC 0.5 = identical, 1.0 = maximally different
        similarities = np.maximum(0, 1.0 - 2.0 * (distances - 0.5))
    else:
        # MMD: smaller = more similar
        similarities = 1.0 / (distances + 1e-6)

    weights = similarities / (similarities.sum() + 1e-10)

    return MultiSourceWeights(
        weights=weights,
        source_names=source_names,
        distances=distances,
        details={
            "method": method,
            "n_sources": len(sources),
            "most_similar": source_names[int(np.argmax(weights))],
            "least_similar": source_names[int(np.argmin(weights))],
        },
    )
