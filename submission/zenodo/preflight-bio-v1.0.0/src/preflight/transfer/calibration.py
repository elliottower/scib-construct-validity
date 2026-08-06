"""#4: Targeted calibration design — which samples to re-measure?

Given a shift diagnosis, recommends which source samples would be most
informative to re-measure on the target instrument. Prioritizes samples
that span the shifted feature dimensions.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationDesign:
    selected_indices: np.ndarray
    n_selected: int
    coverage_score: float
    details: dict

    def summary(self) -> str:
        return (
            f"Selected {self.n_selected} calibration samples\n"
            f"Coverage of shifted dimensions: {self.coverage_score:.3f}\n"
            f"Indices: {self.selected_indices.tolist()}"
        )


def suggest_calibration_samples(
    source: np.ndarray,
    target: np.ndarray,
    n_budget: int = 10,
    feature_importances: np.ndarray | None = None,
) -> CalibrationDesign:
    """Select source samples that maximally span the shifted feature dimensions.

    Strategy: project source samples onto the top principal components of the
    shift (weighted by feature importances from the domain classifier), then
    greedily select samples that maximize coverage of this projection.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        n_budget: number of samples to select.
        feature_importances: per-feature shift importances from audit().
            If None, computed from a domain classifier.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if feature_importances is None:
        X = np.vstack([source, target])
        y = np.array([0] * len(source) + [1] * len(target))
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(scaler.fit_transform(X), y)
        feature_importances = np.abs(clf.coef_[0])

    # Weight features by importance, project source onto top components
    weights = feature_importances / (feature_importances.sum() + 1e-10)
    source_weighted = source * np.sqrt(weights)[np.newaxis, :]

    # SVD on weighted source to find directions of maximum shifted variance
    from scipy.linalg import svd
    Xc = source_weighted - source_weighted.mean(axis=0)
    _, _, Vt = svd(Xc, full_matrices=False)
    k = min(n_budget, min(source.shape) - 1)
    projection = source_weighted @ Vt[:k].T  # (n_source, k)

    # Greedy maximin selection: pick sample farthest from already selected
    n = len(source)
    n_budget = min(n_budget, n)
    selected = [np.argmax(np.linalg.norm(projection, axis=1))]
    min_dists = np.full(n, np.inf)

    for _ in range(n_budget - 1):
        last = projection[selected[-1]]
        dists = np.linalg.norm(projection - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -np.inf
        selected.append(int(np.argmax(min_dists)))

    selected = np.array(selected)

    # Coverage: fraction of shifted variance explained by selected samples
    selected_proj = projection[selected]
    total_var = np.var(projection, axis=0).sum()
    selected_var = np.var(selected_proj, axis=0).sum()
    coverage = float(selected_var / (total_var + 1e-10))

    return CalibrationDesign(
        selected_indices=selected,
        n_selected=len(selected),
        coverage_score=coverage,
        details={
            "top_shifted_features": np.argsort(feature_importances)[-10:][::-1].tolist(),
            "projection_dims": k,
        },
    )
