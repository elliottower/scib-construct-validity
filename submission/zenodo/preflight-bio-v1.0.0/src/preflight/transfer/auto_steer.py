"""#7: Auto-steer — try all corrections, pick the best by preflight score.

Removes the classification step as a bottleneck. Instead of diagnosing the
shift type and applying one correction, tries every available correction in
parallel and returns the one that produces the highest preflight composite score.
"""

import numpy as np
from dataclasses import dataclass, field

from preflight.transfer.audit import diagnose, ShiftDiagnosis
from preflight.transfer.steer import (
    ratio_correction, location_scale, coral, support_intersection,
    CorrectionResult,
)
from preflight.core.runner import run as preflight_run


@dataclass
class AutoSteerResult:
    best_method: str
    best_correction: CorrectionResult
    best_tier: int
    best_score: float
    initial_diagnosis: ShiftDiagnosis | None = None
    all_results: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = []
        if self.initial_diagnosis:
            d = self.initial_diagnosis
            lines.append(f"Initial: {d.shift_type} (AUC={d.domain_auc:.3f})")
            if d.preflight:
                lines.append(f"  Preflight: Tier {d.preflight.overall_tier}, score={d.preflight.overall_score:.3f}")
            lines.append("")
        lines.append(f"Best: {self.best_method} (Tier {self.best_tier}, score={self.best_score:.3f})")
        lines.append("")
        lines.append(f"{'Method':<30} | {'Tier':>4} | {'Score':>6} | {'M2 AUC':>7}")
        lines.append("-" * 55)
        for r in self.all_results:
            lines.append(f"{r['method']:<30} | {r['tier']:>4} | {r['score']:>6.3f} | {r['m2_auc']:>7.3f}")
        return "\n".join(lines)


def auto_steer(
    source: np.ndarray,
    target: np.ndarray,
    include_coral: bool = True,
    max_features_for_coral: int = 200,
) -> AutoSteerResult:
    """Try all available corrections and pick the best by preflight score.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        include_coral: whether to try CORAL (expensive for high-d).
        max_features_for_coral: skip CORAL if d exceeds this.

    Returns:
        AutoSteerResult with the best correction and comparison table.
    """
    initial = diagnose(source, target)

    corrections = [
        ("no_correction", None),
        ("ratio_mean", lambda: ratio_correction(source, target, trim_frac=0.0)),
        ("ratio_trimmed_10", lambda: ratio_correction(source, target, trim_frac=0.1)),
        ("ratio_trimmed_20", lambda: ratio_correction(source, target, trim_frac=0.2)),
        ("location_scale", lambda: location_scale(source, target)),
        ("support_intersection", lambda: support_intersection(source, target)),
    ]

    d = source.shape[1]
    if include_coral and d <= max_features_for_coral:
        corrections.append(("coral", lambda: coral(source, target, reg=1.0)))

    modules = ["m1_grassmannian", "m2_domain_shift", "m3_direction_stability", "m4_domain_validity"]
    all_results = []
    best = None

    for name, correction_fn in corrections:
        if correction_fn is None:
            corrected = target
            cr = None
        else:
            cr = correction_fn()
            corrected = cr.target_corrected

        report = preflight_run(source, corrected, modules=modules)
        m2 = report.module_results.get("m2_domain_shift")
        m2_auc = m2.details.get("domain_classifier_auc", 0.5) if m2 else 0.5

        entry = {
            "method": name,
            "tier": report.overall_tier,
            "score": report.overall_score,
            "m2_auc": m2_auc,
            "correction": cr,
        }
        all_results.append(entry)

        if best is None or report.overall_score > best["score"]:
            best = entry

    return AutoSteerResult(
        best_method=best["method"],
        best_correction=best["correction"],
        best_tier=best["tier"],
        best_score=best["score"],
        initial_diagnosis=initial,
        all_results=[{k: v for k, v in r.items() if k != "correction"} for r in all_results],
    )
