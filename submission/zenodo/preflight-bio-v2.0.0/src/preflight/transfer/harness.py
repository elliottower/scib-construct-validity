"""Transfer Harness: iterative audit → steer → verify loop.

Runs the full pipeline:
  1. Audit: classify the shift type
  2. Steer: apply the recommended correction
  3. Verify: re-run preflight to check if the correction worked
  4. Repeat until convergence or max iterations

Usage:
    from preflight.transfer import run_harness
    report = run_harness(source, target)
    print(report.summary())
"""

import time
import numpy as np
from dataclasses import dataclass, field

from preflight.transfer.audit import audit, diagnose, ShiftDiagnosis
from preflight.transfer.steer import steer, CorrectionResult, CORRECTION_MAP
from preflight.core.runner import run as preflight_run


@dataclass
class IterationResult:
    iteration: int
    diagnosis: ShiftDiagnosis
    correction: CorrectionResult | None
    preflight_tier: int | None
    preflight_score: float | None
    preflight_m2_auc: float | None


@dataclass
class TransferReport:
    """Full report from the transfer harness."""
    iterations: list[IterationResult] = field(default_factory=list)
    final_target: np.ndarray | None = None
    converged: bool = False

    @property
    def initial_tier(self) -> int | None:
        return self.iterations[0].preflight_tier if self.iterations else None

    @property
    def final_tier(self) -> int | None:
        return self.iterations[-1].preflight_tier if self.iterations else None

    @property
    def corrections_applied(self) -> list[str]:
        return [it.correction.method for it in self.iterations if it.correction]

    def summary(self) -> str:
        lines = ["=" * 60, "TRANSFER HARNESS REPORT", "=" * 60, ""]
        for it in self.iterations:
            d = it.diagnosis
            tier = f"Tier {it.preflight_tier}" if it.preflight_tier else "N/A"
            auc = f"{it.preflight_m2_auc:.3f}" if it.preflight_m2_auc is not None else "N/A"
            lines.append(f"Iteration {it.iteration}:")
            lines.append(f"  Shift: {d.shift_type} (confidence={d.confidence:.2f}, AUC={d.domain_auc:.3f})")
            lines.append(f"  Preflight: {tier}, score={it.preflight_score:.3f}, M2 AUC={auc}")
            if it.correction:
                lines.append(f"  Applied: {it.correction.method}")
            lines.append("")

        lines.append(f"Converged: {self.converged}")
        if self.iterations:
            lines.append(f"Progression: Tier {self.initial_tier} -> Tier {self.final_tier}")
        return "\n".join(lines)


def _verify(source: np.ndarray, target: np.ndarray) -> tuple[int, float, float]:
    report = preflight_run(
        source, target,
        modules=["m1_grassmannian", "m2_domain_shift", "m3_direction_stability", "m4_domain_validity"],
    )
    m2 = report.module_results.get("m2_domain_shift")
    m2_auc = m2.details.get("domain_classifier_auc", 0.5) if m2 else 0.5
    return report.overall_tier, report.overall_score, m2_auc


def run_harness(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 3,
    target_tier: int = 5,
    verbose: bool = False,
) -> TransferReport:
    """Run the iterative audit -> steer -> verify loop.

    Args:
        source: (n_source, d) feature matrix.
        target: (n_target, d) feature matrix.
        max_iterations: maximum correction attempts.
        target_tier: stop when preflight tier reaches this level.
        verbose: print progress to stdout.

    Returns:
        TransferReport with per-iteration diagnostics and the final
        corrected target matrix.
    """
    report = TransferReport()
    current = target.copy()
    tried = set()

    for i in range(max_iterations + 1):
        if verbose:
            print(f"\nIteration {i}")

        diagnosis = diagnose(source, current)
        pf = diagnosis.preflight
        tier = pf.overall_tier if pf else _verify(source, current)[0]
        score = pf.overall_score if pf else _verify(source, current)[1]
        m2 = pf.module_results.get("m2_domain_shift") if pf else None
        m2_auc = m2.details.get("domain_classifier_auc", 0.5) if m2 else 0.5

        if verbose:
            print(f"  {diagnosis.shift_type} (AUC={diagnosis.domain_auc:.3f}) -> Tier {tier}, M2 AUC={m2_auc:.3f}")

        correction = None

        if tier >= target_tier or diagnosis.shift_type == "NONE":
            report.iterations.append(IterationResult(i, diagnosis, None, tier, score, m2_auc))
            report.converged = True
            report.final_target = current
            if verbose:
                print(f"  Converged at Tier {tier}")
            break

        # Try the recommended correction, or an alternative if already tried
        if diagnosis.shift_type not in tried:
            correction = steer(source, current, diagnosis.shift_type)
            tried.add(diagnosis.shift_type)
        else:
            for alt in ["MULTIPLICATIVE", "ADDITIVE", "ROTATIONAL", "SUPPORT_MISMATCH"]:
                if alt not in tried and CORRECTION_MAP.get(alt) is not None:
                    correction = steer(source, current, alt)
                    tried.add(alt)
                    break

        report.iterations.append(IterationResult(i, diagnosis, correction, tier, score, m2_auc))

        if correction is None:
            report.final_target = current
            break

        if verbose:
            print(f"  Applied: {correction.method}")

        current = correction.target_corrected
    else:
        report.final_target = current

    return report
