"""#10: Systematic benchmark utilities.

Generate synthetic shift datasets and evaluate correction pipelines end-to-end.
Useful for comparing methods before applying them to real data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SyntheticShift:
    source: np.ndarray
    target: np.ndarray
    target_clean: np.ndarray
    shift_type: str
    params: dict


@dataclass
class BenchmarkResult:
    method: str
    mse_before: float
    mse_after: float
    improvement_pct: float
    domain_auc_before: float
    domain_auc_after: float
    details: dict = field(default_factory=dict)


def generate_multiplicative_shift(
    n_source: int = 200,
    n_target: int = 100,
    d: int = 50,
    scale_range: tuple[float, float] = (0.5, 2.0),
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SyntheticShift:
    """Generate synthetic data with multiplicative per-feature shift."""
    rng = rng or np.random.default_rng()
    source = rng.normal(5.0, 1.0, (n_source, d))
    scales = rng.uniform(scale_range[0], scale_range[1], d)
    target_clean = rng.normal(5.0, 1.0, (n_target, d))
    target = target_clean * scales + rng.normal(0, noise, (n_target, d))
    return SyntheticShift(
        source=source, target=target, target_clean=target_clean,
        shift_type="multiplicative", params={"scales": scales, "noise": noise},
    )


def generate_additive_shift(
    n_source: int = 200,
    n_target: int = 100,
    d: int = 50,
    shift_range: tuple[float, float] = (-2.0, 2.0),
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SyntheticShift:
    """Generate synthetic data with additive per-feature shift."""
    rng = rng or np.random.default_rng()
    source = rng.normal(5.0, 1.0, (n_source, d))
    offsets = rng.uniform(shift_range[0], shift_range[1], d)
    target_clean = rng.normal(5.0, 1.0, (n_target, d))
    target = target_clean + offsets + rng.normal(0, noise, (n_target, d))
    return SyntheticShift(
        source=source, target=target, target_clean=target_clean,
        shift_type="additive", params={"offsets": offsets, "noise": noise},
    )


def generate_rotational_shift(
    n_source: int = 200,
    n_target: int = 100,
    d: int = 50,
    angle_deg: float = 30.0,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> SyntheticShift:
    """Generate synthetic data with rotational (covariance) shift."""
    rng = rng or np.random.default_rng()
    source = rng.normal(0, 1, (n_source, d))

    # Random rotation matrix via QR decomposition
    A = rng.normal(0, 1, (d, d))
    angle_rad = np.radians(angle_deg)
    A = np.eye(d) + np.sin(angle_rad) * (A - A.T) / 2
    Q, _ = np.linalg.qr(A)

    target_clean = rng.normal(0, 1, (n_target, d))
    target = target_clean @ Q.T + rng.normal(0, noise, (n_target, d))
    return SyntheticShift(
        source=source, target=target, target_clean=target_clean,
        shift_type="rotational", params={"angle_deg": angle_deg, "noise": noise},
    )


def generate_support_shift(
    n_source: int = 200,
    n_target: int = 100,
    d: int = 50,
    drop_frac: float = 0.2,
    rng: np.random.Generator | None = None,
) -> SyntheticShift:
    """Generate synthetic data where target has different support (some features zeroed)."""
    rng = rng or np.random.default_rng()
    source = rng.normal(5.0, 1.0, (n_source, d))
    target_clean = rng.normal(5.0, 1.0, (n_target, d))

    n_drop = max(1, int(d * drop_frac))
    drop_idx = rng.choice(d, n_drop, replace=False)
    target = target_clean.copy()
    target[:, drop_idx] = 0.0
    return SyntheticShift(
        source=source, target=target, target_clean=target_clean,
        shift_type="support_mismatch",
        params={"dropped_features": drop_idx.tolist(), "drop_frac": drop_frac},
    )


ALL_GENERATORS = {
    "multiplicative": generate_multiplicative_shift,
    "additive": generate_additive_shift,
    "rotational": generate_rotational_shift,
    "support": generate_support_shift,
}


def run_benchmark(
    correction_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    method_name: str = "custom",
    shift_types: list[str] | None = None,
    n_trials: int = 5,
    rng: np.random.Generator | None = None,
    **gen_kwargs,
) -> list[BenchmarkResult]:
    """Run a correction function against all synthetic shift types.

    Args:
        correction_fn: takes (source, target) and returns corrected_target.
        method_name: label for the method.
        shift_types: which shift types to test (default: all).
        n_trials: number of random trials per shift type.
        rng: random generator.
        **gen_kwargs: passed to each generator.

    Returns:
        List of BenchmarkResult, one per (shift_type, trial).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    rng = rng or np.random.default_rng()
    if shift_types is None:
        shift_types = list(ALL_GENERATORS.keys())

    results = []
    for stype in shift_types:
        gen = ALL_GENERATORS[stype]
        for trial in range(n_trials):
            syn = gen(rng=rng, **gen_kwargs)

            # MSE before
            mse_before = float(np.mean((syn.target - syn.target_clean) ** 2))

            # Apply correction
            corrected = correction_fn(syn.source, syn.target)
            mse_after = float(np.mean((corrected - syn.target_clean) ** 2))

            improvement = (mse_before - mse_after) / (mse_before + 1e-10) * 100

            # Domain AUC before and after
            def _auc(s, t):
                X = np.vstack([s, t])
                y = np.array([0] * len(s) + [1] * len(t))
                scaler = StandardScaler()
                clf = LogisticRegression(max_iter=500, C=1.0)
                try:
                    scores = cross_val_score(clf, scaler.fit_transform(X), y,
                                              cv=min(5, min(len(s), len(t))),
                                              scoring="roc_auc")
                    return float(np.mean(scores))
                except Exception:
                    return 0.5

            auc_before = _auc(syn.source, syn.target)
            auc_after = _auc(syn.source, corrected)

            results.append(BenchmarkResult(
                method=method_name,
                mse_before=mse_before,
                mse_after=mse_after,
                improvement_pct=improvement,
                domain_auc_before=auc_before,
                domain_auc_after=auc_after,
                details={"shift_type": stype, "trial": trial},
            ))

    return results


def format_benchmark_results(results: list[BenchmarkResult]) -> str:
    """Format benchmark results into a readable summary table."""
    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        st = r.details.get("shift_type", "unknown")
        by_type.setdefault(st, []).append(r)

    lines = [f"{'Shift Type':<18} | {'MSE Before':>10} | {'MSE After':>10} | {'Improv%':>7} | {'AUC Before':>10} | {'AUC After':>10}"]
    lines.append("-" * 75)

    for stype, rlist in by_type.items():
        mse_b = np.mean([r.mse_before for r in rlist])
        mse_a = np.mean([r.mse_after for r in rlist])
        imp = np.mean([r.improvement_pct for r in rlist])
        auc_b = np.mean([r.domain_auc_before for r in rlist])
        auc_a = np.mean([r.domain_auc_after for r in rlist])
        lines.append(f"{stype:<18} | {mse_b:>10.4f} | {mse_a:>10.4f} | {imp:>6.1f}% | {auc_b:>10.3f} | {auc_a:>10.3f}")

    return "\n".join(lines)
