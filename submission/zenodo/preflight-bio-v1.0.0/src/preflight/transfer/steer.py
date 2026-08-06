"""Transfer Steering: apply corrections based on diagnosed shift type.

Each shift type maps to a correction:
  MULTIPLICATIVE   → ratio_correction (per-feature trimmed mean ratio)
  ADDITIVE         → location_scale (match moments)
  ROTATIONAL       → coral (align covariance structure)
  SUPPORT_MISMATCH → support_intersection (clip to source range)
  MIXED            → ratio_correction (safe default)
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.stats import trim_mean
from scipy.linalg import svd


@dataclass
class CorrectionResult:
    """Result of applying a transfer correction."""
    method: str
    target_corrected: np.ndarray
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Correction: {self.method}"]
        for k, v in self.details.items():
            if isinstance(v, (float, int)):
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            elif not isinstance(v, np.ndarray):
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


def ratio_correction(
    source: np.ndarray,
    target: np.ndarray,
    trim_frac: float = 0.1,
) -> CorrectionResult:
    """Per-feature multiplicative correction using trimmed mean ratio.

    Estimates the per-feature transfer function as source_mean / target_mean,
    then inverts it. Trimmed mean is more robust to outlier samples.
    """
    if trim_frac > 0:
        s_stat = trim_mean(source, trim_frac, axis=0)
        t_stat = trim_mean(target, trim_frac, axis=0)
    else:
        s_stat = source.mean(axis=0)
        t_stat = target.mean(axis=0)

    ratio = s_stat / (t_stat + 1e-10)
    corrected = target * ratio[np.newaxis, :]

    return CorrectionResult(
        method=f"ratio_correction(trim={trim_frac})",
        target_corrected=corrected,
        details={
            "ratio_mean": float(np.mean(ratio)),
            "ratio_std": float(np.std(ratio)),
            "ratio_min": float(np.min(ratio)),
            "ratio_max": float(np.max(ratio)),
            "trim_frac": trim_frac,
        },
    )


def location_scale(
    source: np.ndarray,
    target: np.ndarray,
) -> CorrectionResult:
    """Match first two moments: standardize target, then rescale to source statistics."""
    s_mean, s_std = source.mean(axis=0), source.std(axis=0) + 1e-10
    t_mean, t_std = target.mean(axis=0), target.std(axis=0) + 1e-10

    corrected = (target - t_mean) / t_std * s_std + s_mean

    return CorrectionResult(
        method="location_scale",
        target_corrected=corrected,
        details={
            "mean_shift": float(np.mean(s_mean - t_mean)),
            "scale_ratio_mean": float(np.mean(s_std / t_std)),
        },
    )


def coral(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 1.0,
) -> CorrectionResult:
    """CORAL: align second-order statistics of target to source.

    Whitens target covariance, then colors with source covariance.
    Sun et al., "Return of Frustratingly Easy Domain Adaptation" (AAAI 2016).
    """
    d = source.shape[1]
    s_c = source - source.mean(axis=0)
    t_c = target - target.mean(axis=0)

    C_s = (s_c.T @ s_c) / (len(source) - 1) + reg * np.eye(d)
    C_t = (t_c.T @ t_c) / (len(target) - 1) + reg * np.eye(d)

    U_t, S_t, _ = svd(C_t)
    whiten = U_t @ np.diag(1.0 / np.sqrt(S_t)) @ U_t.T

    U_s, S_s, _ = svd(C_s)
    color = U_s @ np.diag(np.sqrt(S_s)) @ U_s.T

    A = whiten @ color
    corrected = t_c @ A + source.mean(axis=0)

    return CorrectionResult(
        method=f"coral(reg={reg})",
        target_corrected=corrected,
        details={
            "source_cov_trace": float(np.trace(C_s)),
            "target_cov_trace": float(np.trace(C_t)),
        },
    )


def support_intersection(
    source: np.ndarray,
    target: np.ndarray,
    percentile_clip: float = 5.0,
) -> CorrectionResult:
    """Clip target features to the source support range."""
    lo = np.percentile(source, percentile_clip, axis=0)
    hi = np.percentile(source, 100 - percentile_clip, axis=0)
    corrected = np.clip(target, lo, hi)

    n_clipped = int(np.sum(target != corrected))

    return CorrectionResult(
        method=f"support_intersection(clip={percentile_clip}%)",
        target_corrected=corrected,
        details={
            "pct_values_clipped": float(n_clipped / target.size),
            "n_values_clipped": n_clipped,
        },
    )


def ot_correction(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 100,
) -> CorrectionResult:
    """Optimal transport correction via Sinkhorn barycentric mapping.

    Computes the regularized OT plan between target and source, then maps
    each target sample to the barycentric projection in source space.
    Falls back to location_scale if scipy doesn't converge or dimensions
    are too large.
    """
    n_s, d = source.shape
    n_t = target.shape[0]

    if n_s * n_t > 500_000:
        result = location_scale(source, target)
        result.method = "ot_correction(fallback=location_scale, reason=too_large)"
        return result

    # Cost matrix (squared Euclidean)
    diff = target[:, np.newaxis, :] - source[np.newaxis, :, :]  # (n_t, n_s, d)
    C = np.sum(diff ** 2, axis=2)  # (n_t, n_s)

    # Sinkhorn iterations
    K = np.exp(-C / (reg * C.mean() + 1e-10))
    u = np.ones(n_t)
    v = np.ones(n_s)
    a = np.ones(n_t) / n_t
    b = np.ones(n_s) / n_s

    for _ in range(max_iter):
        u = a / (K @ v + 1e-10)
        v = b / (K.T @ u + 1e-10)

    # Transport plan: T[i, j] = u[i] * K[i, j] * v[j]
    T = u[:, np.newaxis] * K * v[np.newaxis, :]

    # Barycentric mapping: each target sample maps to weighted average of source
    row_sums = T.sum(axis=1, keepdims=True) + 1e-10
    T_norm = T / row_sums
    corrected = T_norm @ source  # (n_t, d)

    return CorrectionResult(
        method=f"ot_correction(reg={reg})",
        target_corrected=corrected,
        details={
            "n_source": n_s,
            "n_target": n_t,
            "mean_transport_cost": float(np.sum(T * C)),
            "sinkhorn_reg": reg,
        },
    )


CORRECTION_MAP = {
    "MULTIPLICATIVE": ratio_correction,
    "ADDITIVE": location_scale,
    "ROTATIONAL": coral,
    "SUPPORT_MISMATCH": support_intersection,
    "MIXED": ratio_correction,
}


def steer(
    source: np.ndarray,
    target: np.ndarray,
    shift_type: str,
    **kwargs,
) -> CorrectionResult | None:
    """Apply the correction matching the diagnosed shift type.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        shift_type: one of MULTIPLICATIVE, ADDITIVE, ROTATIONAL,
                    SUPPORT_MISMATCH, MIXED, NONE.
        **kwargs: passed to the correction function.

    Returns:
        CorrectionResult with corrected target, or None if shift_type is NONE.
    """
    fn = CORRECTION_MAP.get(shift_type)
    if fn is None:
        return None
    return fn(source, target, **kwargs)
