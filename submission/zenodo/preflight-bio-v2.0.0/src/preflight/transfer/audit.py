"""Transfer Audit: classify the type of distribution shift between source and target.

Given source (n_source, d) and target (n_target, d) feature matrices, decomposes
the shift into actionable categories:

  MULTIPLICATIVE  — per-feature scaling (non-uniform ratio of means)
  ADDITIVE        — per-feature location shift with uniform scaling
  ROTATIONAL      — subspace misalignment (high geodesic, moderate domain AUC)
  SUPPORT_MISMATCH — non-overlapping feature regions
  MIXED           — no single category dominates
  NONE            — no detectable shift

Each type maps to a specific correction in preflight.transfer.steer.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.stats import iqr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.linalg import svd

from preflight.core.runner import run as preflight_run, PreflightReport


@dataclass
class ShiftDiagnosis:
    """Result of a transfer audit."""
    shift_type: str
    confidence: float
    domain_auc: float
    details: dict = field(default_factory=dict)
    feature_importances: np.ndarray | None = None
    recommendations: list[str] = field(default_factory=list)
    preflight: PreflightReport | None = None

    def summary(self) -> str:
        lines = [
            f"Shift type: {self.shift_type} (confidence={self.confidence:.2f})",
            f"Domain AUC: {self.domain_auc:.3f}",
        ]
        if self.preflight is not None:
            lines.append(f"Preflight: Tier {self.preflight.overall_tier}, score={self.preflight.overall_score:.3f}")
            for name, r in sorted(self.preflight.module_results.items()):
                lines.append(f"  {name}: {r.score:.3f} (T{r.tier})")
        for k, v in self.details.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.4f}")
            elif not isinstance(v, np.ndarray):
                lines.append(f"  {k}: {v}")
        if self.recommendations:
            lines.append("Recommendations:")
            for r in self.recommendations:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    def top_shifted_features(self, n: int = 10) -> np.ndarray:
        """Indices of the n features most responsible for the domain shift."""
        if self.feature_importances is None:
            return np.array([], dtype=int)
        return np.argsort(self.feature_importances)[-n:][::-1]


def _domain_classifier_auc(
    source: np.ndarray, target: np.ndarray,
) -> tuple[float, np.ndarray]:
    X = np.vstack([source, target])
    y = np.array([0] * len(source) + [1] * len(target))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    auc_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring="roc_auc")
    auc = float(np.mean(auc_scores))
    clf.fit(X_scaled, y)
    importances = np.abs(clf.coef_[0])
    return auc, importances


def _ratio_pattern(source_mean: np.ndarray, target_mean: np.ndarray) -> dict:
    ratios = source_mean / (target_mean + 1e-10)
    cv = float(np.std(ratios) / (np.mean(ratios) + 1e-10))
    pct_far = float(np.mean(np.abs(ratios - 1.0) > 0.05))
    is_uniform = cv < 0.05 and (np.max(ratios) - np.min(ratios)) < 0.15
    return {
        "ratio_cv": cv,
        "ratio_range": float(np.max(ratios) - np.min(ratios)),
        "ratio_iqr": float(iqr(ratios)),
        "median_ratio": float(np.median(ratios)),
        "pct_features_shifted_5pct": pct_far,
        "is_uniform_scaling": is_uniform,
    }


def _diff_pattern(
    source_mean: np.ndarray, target_mean: np.ndarray, source_std: np.ndarray,
) -> dict:
    diffs = source_mean - target_mean
    norm_diffs = diffs / (source_std + 1e-10)
    cv = float(np.std(norm_diffs) / (np.abs(np.mean(norm_diffs)) + 1e-10))
    return {
        "mean_normalized_diff": float(np.mean(norm_diffs)),
        "std_normalized_diff": float(np.std(norm_diffs)),
        "diff_cv": cv,
        "is_uniform_additive": cv < 0.5,
        "pct_features_shifted_1sd": float(np.mean(np.abs(norm_diffs) > 1.0)),
    }


def _support_overlap(source: np.ndarray, target: np.ndarray) -> dict:
    s_min, s_max = source.min(axis=0), source.max(axis=0)
    t_min, t_max = target.min(axis=0), target.max(axis=0)
    intersection = np.maximum(0, np.minimum(s_max, t_max) - np.maximum(s_min, t_min))
    union = np.maximum(s_max, t_max) - np.minimum(s_min, t_min) + 1e-10
    overlap = intersection / union
    return {
        "mean_feature_overlap": float(np.mean(overlap)),
        "min_feature_overlap": float(np.min(overlap)),
        "pct_features_no_overlap": float(np.mean(overlap < 0.1)),
    }


def _subspace_divergence(source: np.ndarray, target: np.ndarray, k: int) -> dict:
    def _top_k(X, k):
        X_c = X - X.mean(axis=0)
        _, s, Vt = svd(X_c, full_matrices=False)
        ev = (s ** 2) / np.sum(s ** 2)
        return Vt[:k].T, ev[:k]

    k = min(k, min(source.shape) - 1, min(target.shape) - 1)
    U_s, ev_s = _top_k(source, k)
    U_t, ev_t = _top_k(target, k)

    M = U_s.T @ U_t
    svals = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    angles = np.arccos(svals)
    geodesic = float(np.linalg.norm(angles))
    max_geo = (np.pi / 2) * np.sqrt(k)

    return {
        "geodesic_distance": geodesic,
        "max_geodesic": max_geo,
        "normalized_geodesic": geodesic / max_geo,
        "mean_principal_angle_deg": float(np.mean(np.degrees(angles))),
        "max_principal_angle_deg": float(np.max(np.degrees(angles))),
        "explained_var_source": float(np.sum(ev_s)),
        "explained_var_target": float(np.sum(ev_t)),
    }


def audit(
    source: np.ndarray,
    target: np.ndarray,
    auc_threshold: float = 0.6,
    k: int = 10,
) -> ShiftDiagnosis:
    """Classify the distribution shift between source and target.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        auc_threshold: domain AUC below this means no significant shift.
        k: subspace dimension for Grassmannian analysis.

    Returns:
        ShiftDiagnosis with shift type, confidence, feature importances,
        and recommended corrections.
    """
    auc, importances = _domain_classifier_auc(source, target)

    if auc < auc_threshold:
        return ShiftDiagnosis(
            shift_type="NONE",
            confidence=1.0 - auc,
            domain_auc=auc,
            details={"reason": f"Domain AUC={auc:.3f} below threshold {auc_threshold}"},
            feature_importances=importances,
            recommendations=["No significant shift detected. Standard modeling should work."],
        )

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_std = source.std(axis=0)

    ratio_info = _ratio_pattern(source_mean, target_mean)
    diff_info = _diff_pattern(source_mean, target_mean, source_std)
    support_info = _support_overlap(source, target)
    subspace_info = _subspace_divergence(source, target, k)

    details = {**ratio_info, **diff_info, **support_info, **subspace_info}
    recs = []

    # Classification hierarchy: support mismatch > multiplicative > additive > rotational > mixed
    if support_info["pct_features_no_overlap"] > 0.1:
        shift_type = "SUPPORT_MISMATCH"
        confidence = support_info["pct_features_no_overlap"]
        recs.append(f"{support_info['pct_features_no_overlap']:.0%} of features have <10% support overlap.")
        recs.append("Restrict to overlapping feature ranges or use domain-invariant features.")

    elif not ratio_info["is_uniform_scaling"] and ratio_info["pct_features_shifted_5pct"] > 0.3:
        shift_type = "MULTIPLICATIVE"
        confidence = min(1.0, ratio_info["ratio_cv"] * 2)
        ratios = source_mean / (target_mean + 1e-10)
        top_shifted = np.argsort(np.abs(ratios - 1.0))[-10:][::-1]
        recs.append(f"Per-feature ratio CV={ratio_info['ratio_cv']:.3f}, "
                     f"{ratio_info['pct_features_shifted_5pct']:.0%} of features shifted >5%.")
        recs.append("Apply per-feature ratio correction: target *= (source_mean / target_mean).")
        recs.append(f"Median ratio={ratio_info['median_ratio']:.3f}. Use trimmed mean for robustness.")
        recs.append(f"Most shifted feature indices: {top_shifted.tolist()}")

    elif ratio_info["is_uniform_scaling"] and diff_info["pct_features_shifted_1sd"] > 0.1:
        shift_type = "ADDITIVE"
        confidence = min(1.0, diff_info["pct_features_shifted_1sd"])
        recs.append(f"Uniform scaling (ratio CV={ratio_info['ratio_cv']:.3f}) with location shifts.")
        recs.append("Apply location-scale normalization: match moments of target to source.")

    elif subspace_info["normalized_geodesic"] > 0.4:
        shift_type = "ROTATIONAL"
        confidence = subspace_info["normalized_geodesic"]
        recs.append(f"Subspace misalignment: geodesic={subspace_info['geodesic_distance']:.3f} "
                     f"({subspace_info['normalized_geodesic']:.0%} of max).")
        recs.append("Apply CORAL or transfer component analysis.")

    else:
        shift_type = "MIXED"
        confidence = auc - 0.5
        recs.append("Shift does not fit a single category. Try ratio correction first.")

    return ShiftDiagnosis(
        shift_type=shift_type,
        confidence=confidence,
        domain_auc=auc,
        details=details,
        feature_importances=importances,
        recommendations=recs,
    )


def diagnose(
    source: np.ndarray,
    target: np.ndarray,
    source_labels: np.ndarray | None = None,
    target_labels: np.ndarray | None = None,
    metadata: dict | None = None,
    k: int = 5,
) -> ShiftDiagnosis:
    """Unified diagnosis: shift classification + preflight scoring in one call.

    Runs both the shift-type classifier (audit) and the preflight module
    pipeline (M1-M6), returning a ShiftDiagnosis with the PreflightReport
    attached. This is the recommended entry point for agents.

    Args:
        source: (n_source, d) embedding matrix.
        target: (n_target, d) embedding matrix.
        source_labels: Optional cell-type or condition labels for source.
        target_labels: Optional cell-type or condition labels for target.
        metadata: Optional dict with graph, records, estimates for M5-M7.
        k: Subspace dimension for Grassmannian / LDA.
    """
    shift = audit(source, target, k=k)

    pf_metadata = dict(metadata) if metadata else {}
    if source_labels is not None:
        pf_metadata["source_labels"] = source_labels
    if target_labels is not None:
        pf_metadata["target_labels"] = target_labels

    report = preflight_run(
        source, target,
        metadata=pf_metadata if pf_metadata else None,
        hyperparameters={"k": k},
    )
    shift.preflight = report

    if report.overall_tier >= 5 and shift.shift_type != "NONE":
        shift.recommendations.append(
            f"Preflight tier {report.overall_tier} suggests embeddings are already "
            f"well-aligned. Correction may be unnecessary."
        )
    elif report.overall_tier <= 2:
        shift.recommendations.append(
            f"Preflight tier {report.overall_tier}: severe structural mismatch. "
            f"Correction alone may not be sufficient."
        )

    m4 = report.module_results.get("m4_domain_validity")
    if m4 and m4.tier <= 3:
        shift.recommendations.append(
            f"M4 domain validity is low (T{m4.tier}): effective dimensionality "
            f"mismatch between source and target. Check for embedding collapse."
        )

    return shift
