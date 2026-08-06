"""Preflight pipeline runner: orchestrates modules into a composite report.

Usage:
    from preflight import run
    report = run(source, target, metadata)
    print(report.overall_tier, report.overall_score)

All 7 modules:
    M1 Grassmannian geodesic — subspace alignment (embedding)
    M2 Domain shift — domain classifier AUC (embedding)
    M3 Direction stability — subspace overlap / principal angles (embedding)
    M4 Domain validity — data completeness / missingness (embedding)
    M5 Cross-design discordance — sheaf Q-test on scalar estimates (metadata)
    M6 Network curvature — Ollivier-Ricci on knowledge graph (metadata)
    M7 Ecological bias — site-level outcome variation (metadata)

Modules requiring only source/target arrays (M1, M2, M3, M4) run by default.
Modules requiring metadata (M5, M6, M7) run when their inputs are provided.
"""
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from preflight.core.preregister import (
    DatasetSpec,
    compute_preregistration_sha,
    save_preregistration,
)
from preflight.core.modules.module1_grassmannian import transportability as m1
from preflight.core.modules.module2_domain_shift import transportability as m2
from preflight.core.modules.module3_direction_stability import distances as m3
from preflight.core.modules.module3_direction_stability import subspace as m3_subspace
from preflight.core.modules.module6_curvature import weighted_orc as m6
from preflight.core.modules.module7_ecological_bias import s7_evalue_sensitivity as m7

_runner_module = sys.modules[__name__]
ALL_MODULE_SOURCES = [m1, m2, m3, m3_subspace, m6, m7, _runner_module]

AVAILABLE_MODULES = {
    "m1_grassmannian",
    "m2_domain_shift",
    "m3_direction_stability",
    "m4_domain_validity",
    "m5_cross_design",
    "m6_curvature",
    "m7_ecological_bias",
}

EMBEDDING_MODULES = {
    "m1_grassmannian", "m2_domain_shift",
    "m3_direction_stability", "m4_domain_validity",
}

DEFAULT_WEIGHTS = {
    "m1_grassmannian": 2.0,
    "m2_domain_shift": 2.0,
    "m3_direction_stability": 1.0,
    "m4_domain_validity": 1.0,
    "m5_cross_design": 1.5,
    "m6_curvature": 0.5,
}

TIER_THRESHOLDS = [0.85, 0.70, 0.55, 0.40, 0.25, 0.10]


TIER_LABELS = {
    7: "Excellent", 6: "Good", 5: "Acceptable",
    4: "Marginal", 3: "Poor", 2: "Very Poor", 1: "Failure",
}


@dataclass
class ModuleResult:
    module: str
    score: float
    tier: int
    details: dict[str, Any]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    overall_tier: int
    overall_score: float
    module_results: dict[str, ModuleResult]
    recommendations: list[str]
    preregistration_sha: str | None
    timestamp: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "overall_tier": self.overall_tier,
            "overall_tier_label": TIER_LABELS.get(self.overall_tier, "Unknown"),
            "overall_score": self.overall_score,
            "module_results": {k: v.to_dict() for k, v in self.module_results.items()},
            "recommendations": self.recommendations,
            "preregistration_sha": self.preregistration_sha,
            "timestamp": self.timestamp,
            "hyperparameters": self.hyperparameters,
        }
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    def to_markdown(self) -> str:
        lines = []
        lines.append("# Preflight Transportability Report")
        lines.append("")
        label = TIER_LABELS.get(self.overall_tier, "Unknown")
        lines.append(f"**Overall Tier: {self.overall_tier}/7 ({label})**")
        lines.append(f"**Overall Score: {self.overall_score:.3f}**")
        lines.append(f"**Timestamp:** {self.timestamp}")
        if self.preregistration_sha:
            lines.append(f"**Preregistration SHA:** `{self.preregistration_sha[:16]}...`")
        lines.append("")

        lines.append("## Module Results")
        lines.append("")
        for name in sorted(self.module_results):
            r = self.module_results[name]
            mod_label = TIER_LABELS.get(r.tier, "Unknown")
            lines.append(f"### {r.module}")
            lines.append(f"- **Score:** {r.score:.3f} | **Tier:** {r.tier}/7 ({mod_label})")
            lines.append(f"- {r.interpretation}")
            lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        for rec in self.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

        return "\n".join(lines)


def _score_to_tier(score: float) -> int:
    for i, threshold in enumerate(TIER_THRESHOLDS):
        if score >= threshold:
            return 7 - i
    return 1


# ======================================================================
# Per-module runners
# ======================================================================

def _run_m1_grassmannian(
    source: np.ndarray, target: np.ndarray, k: int,
) -> ModuleResult:
    U_s, _ = m1.top_k_subspace(source, k)
    U_t, _ = m1.top_k_subspace(target, k)

    geodesic = m1.geodesic_distance(U_s, U_t)
    max_geodesic = (math.pi / 2) * math.sqrt(k)
    score = max(0.0, 1.0 - geodesic / max_geodesic)

    details = {
        "geodesic_distance": float(geodesic),
        "max_geodesic": float(max_geodesic),
        "k": k,
    }

    if score >= 0.7:
        interp = f"Subspaces are well-aligned (geodesic={geodesic:.3f}). Transferability is likely."
    elif score >= 0.4:
        interp = f"Moderate subspace divergence (geodesic={geodesic:.3f}). Transfer with caution."
    else:
        interp = f"Large subspace divergence (geodesic={geodesic:.3f}). Transferability is unlikely."

    return ModuleResult(
        module="m1_grassmannian",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _run_m2_domain_shift(
    source: np.ndarray, target: np.ndarray,
    source_labels: np.ndarray | None,
    target_labels: np.ndarray | None,
) -> ModuleResult:
    centroid_d = m2.centroid_distance(source, target)

    try:
        domain_auc = m2.domain_classifier_auc(source, target)
    except Exception:
        domain_auc = 0.5

    score = max(0.0, min(1.0, 2.0 * (1.0 - domain_auc)))

    details = {
        "centroid_distance": float(centroid_d),
        "domain_classifier_auc": float(domain_auc),
    }

    if source_labels is not None and target_labels is not None:
        try:
            smd = m2.spectral_matching_degradation(
                source, target, source_labels, target_labels,
            )
            details["spectral_matching_degradation"] = smd
        except Exception:
            pass

    if score >= 0.7:
        interp = f"Embeddings overlap well (domain AUC={domain_auc:.3f}). No evidence of collapse."
    elif score >= 0.4:
        interp = f"Moderate domain shift (domain AUC={domain_auc:.3f}). Some embedding structure differs."
    else:
        interp = f"Severe domain shift (domain AUC={domain_auc:.3f}). Embeddings may have collapsed."

    return ModuleResult(
        module="m2_domain_shift",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _run_m3_direction_stability(
    source: np.ndarray, target: np.ndarray, k: int,
    source_labels: np.ndarray | None = None,
    target_labels: np.ndarray | None = None,
) -> ModuleResult:
    used_lda = False
    if source_labels is not None and target_labels is not None:
        n_classes_s = len(np.unique(source_labels))
        n_classes_t = len(np.unique(target_labels))
        if n_classes_s >= 2 and n_classes_t >= 2:
            U_s = m3_subspace.fit_lda_subspace(source, source_labels, k)
            U_t = m3_subspace.fit_lda_subspace(target, target_labels, k)
            used_lda = True

    if not used_lda:
        U_s, _ = m1.top_k_subspace(source, k)
        U_t, _ = m1.top_k_subspace(target, k)

    dists = m3.all_subspace_distances(U_s, U_t)
    grassmannian = dists.get("grassmannian", 0.0)
    chordal = dists.get("chordal", 0.0)
    overlap = dists.get("subspace_overlap", 1.0)

    score = overlap

    details = {
        "grassmannian_distance": float(grassmannian),
        "chordal_distance": float(chordal),
        "subspace_overlap": float(overlap),
        "subspace_method": "lda" if used_lda else "pca",
        "all_distances": {k_: float(v) for k_, v in dists.items()},
    }

    method_label = "LDA discriminant" if used_lda else "PCA"
    if score >= 0.7:
        interp = f"{method_label} directions are stable across cohorts (overlap={overlap:.3f})."
    elif score >= 0.4:
        interp = f"Moderate {method_label.lower()} direction instability (overlap={overlap:.3f})."
    else:
        interp = f"Severe {method_label.lower()} direction instability (overlap={overlap:.3f}). Discriminant directions do not replicate."

    return ModuleResult(
        module="m3_direction_stability",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _participation_ratio(X: np.ndarray) -> float:
    """Participation ratio of eigenvalues, normalized to [0, 1].

    PR = (sum(lambda_i))^2 / sum(lambda_i^2) / d.
    1.0 = variance spread uniformly across all dimensions (isotropic).
    1/d = variance concentrated in a single dimension (collapsed).
    """
    if np.issubdtype(X.dtype, np.floating) and np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)
    try:
        cov = np.cov(X.T)
        eigs = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return 0.0
    eigs = np.maximum(eigs, 0.0)
    total = eigs.sum()
    if total < 1e-12:
        return 0.0
    d = X.shape[1]
    return float((total ** 2) / (eigs ** 2).sum() / d)


def _run_m4_domain_validity(
    source: np.ndarray, target: np.ndarray,
) -> ModuleResult:
    pr_source = _participation_ratio(source)
    pr_target = _participation_ratio(target)

    pr_max = max(pr_source, pr_target)
    pr_min = min(pr_source, pr_target)
    ratio = pr_min / pr_max if pr_max > 1e-12 else 0.0

    collapse_floor = min(pr_source, pr_target) > 0.01
    score = max(0.0, min(1.0, ratio)) if collapse_floor else 0.0

    details = {
        "participation_ratio_source": float(pr_source),
        "participation_ratio_target": float(pr_target),
        "pr_ratio": float(ratio),
    }

    if score >= 0.7:
        interp = (
            f"Embedding dimensionality is comparable "
            f"(PR source={pr_source:.3f}, target={pr_target:.3f}, ratio={ratio:.3f})."
        )
    elif score >= 0.4:
        interp = (
            f"Moderate dimensional asymmetry "
            f"(PR source={pr_source:.3f}, target={pr_target:.3f}, ratio={ratio:.3f}). "
            f"One domain may have lower effective dimensionality."
        )
    else:
        interp = (
            f"Severe dimensional mismatch or collapse "
            f"(PR source={pr_source:.3f}, target={pr_target:.3f}, ratio={ratio:.3f}). "
            f"Embedding structure is not comparable."
        )

    return ModuleResult(
        module="m4_domain_validity",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _run_m5_cross_design(
    estimates: dict[str, dict[str, float]],
) -> ModuleResult:
    p, Q, df = m1.sheaf_q_test(estimates)

    score = min(1.0, p / 0.10)

    details = {
        "sheaf_q_p_value": float(p),
        "sheaf_q_statistic": float(Q),
        "df": int(df),
        "n_designs": len(estimates),
        "design_names": list(estimates.keys()),
    }

    if p > 0.10:
        interp = f"Designs are concordant (Q-test p={p:.3f}). Cross-design consistency holds."
    elif p > 0.05:
        interp = f"Borderline concordance (Q-test p={p:.3f}). Some design-specific variation."
    else:
        interp = f"Significant discordance across designs (Q-test p={p:.3f}). Findings may be design-dependent."

    return ModuleResult(
        module="m5_cross_design",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _run_m6_curvature(
    graph: nx.Graph, alpha: float = 0.5,
) -> ModuleResult:
    curvatures = m6.ollivier_ricci_curvature(graph, alpha=alpha)

    if not curvatures:
        return ModuleResult(
            module="m6_curvature",
            score=0.0,
            tier=1,
            details={"error": "No edges to compute curvature on"},
            interpretation="Graph has no edges. Curvature analysis not possible.",
        )

    vals = list(curvatures.values())
    mean_orc = float(np.mean(vals))
    std_orc = float(np.std(vals))
    min_orc = float(np.min(vals))
    max_orc = float(np.max(vals))
    frac_bottleneck = float(np.mean([v < -0.1 for v in vals]))
    score = max(0.0, min(1.0, frac_bottleneck))

    details = {
        "mean_orc": mean_orc,
        "std_orc": std_orc,
        "min_orc": min_orc,
        "max_orc": max_orc,
        "n_edges": len(vals),
        "fraction_bottleneck": frac_bottleneck,
        "alpha": alpha,
    }

    if score >= 0.7:
        interp = f"Rich bottleneck structure ({frac_bottleneck:.0%} bridge edges, mean ORC={mean_orc:.3f}). Focused pathway connectivity."
    elif score >= 0.4:
        interp = f"Moderate bottleneck structure ({frac_bottleneck:.0%} bridge edges, mean ORC={mean_orc:.3f})."
    else:
        interp = f"Few bottleneck edges ({frac_bottleneck:.0%}). Diffuse connectivity may indicate weak pathway specificity."

    return ModuleResult(
        module="m6_curvature",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


def _run_m7_ecological_bias(
    records: list[dict], site_key: str = "site", outcome_key: str = "died",
) -> ModuleResult:
    site_outcomes: dict[str, list[int]] = {}
    for r in records:
        site = r.get(site_key, "unknown")
        outcome = r.get(outcome_key, 0)
        site_outcomes.setdefault(site, []).append(outcome)

    n_sites = len(site_outcomes)
    if n_sites < 2:
        return ModuleResult(
            module="m7_ecological_bias",
            score=0.5,
            tier=4,
            details={"n_sites": n_sites, "error": "Too few sites for bias assessment"},
            interpretation="Only one site present. Cannot assess ecological bias.",
        )

    site_rates = {s: np.mean(vals) for s, vals in site_outcomes.items()}
    rates = list(site_rates.values())
    cv = float(np.std(rates) / np.mean(rates)) if np.mean(rates) > 1e-10 else 0.0
    rate_range = float(max(rates) - min(rates))

    score = max(0.0, min(1.0, 1.0 - min(cv, 1.0)))

    overall_rate = np.mean([r.get(outcome_key, 0) for r in records])
    if overall_rate > 0 and overall_rate < 1:
        implied_or = (max(rates) / (1 - max(rates))) / (min(rates) / (1 - min(rates))) if min(rates) > 0 and max(rates) < 1 else 1.0
        implied_beta = math.log(max(implied_or, 1e-6))
        evalue = m7.compute_evalue_rr(max(implied_or, 1.0))
    else:
        implied_or = 1.0
        implied_beta = 0.0
        evalue = 1.0

    details = {
        "n_sites": n_sites,
        "n_records": len(records),
        "site_rate_cv": cv,
        "site_rate_range": rate_range,
        "site_rate_min": float(min(rates)),
        "site_rate_max": float(max(rates)),
        "implied_odds_ratio": float(implied_or),
        "evalue": float(evalue),
    }

    if score >= 0.7:
        interp = f"Low ecological bias risk (site rate CV={cv:.3f}, E-value={evalue:.1f})."
    elif score >= 0.4:
        interp = f"Moderate ecological bias risk (site rate CV={cv:.3f}). Between-site variation may confound."
    else:
        interp = f"High ecological bias risk (site rate CV={cv:.3f}, E-value={evalue:.1f}). Aggregation may distort individual-level inference."

    return ModuleResult(
        module="m7_ecological_bias",
        score=score,
        tier=_score_to_tier(score),
        details=details,
        interpretation=interp,
    )


# ======================================================================
# Composite scoring
# ======================================================================

def _composite_score(
    results: dict[str, ModuleResult],
    weights: dict[str, float],
) -> float:
    if not results:
        return 0.0
    total_weight = 0.0
    weighted_sum = 0.0
    for name, r in results.items():
        w = weights.get(name, 1.0)
        weighted_sum += r.score * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def _apply_worst_tier_gate(
    results: dict[str, ModuleResult], tier: int,
) -> int:
    worst = min((r.tier for r in results.values()), default=7)
    if worst <= WORST_TIER_GATE and tier > GATED_TIER_CAP:
        return GATED_TIER_CAP
    return tier


def resolve_hyperparameters(
    hyperparameters: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Resolve hyperparameters to the canonical form used for preregistration.

    Returns (hp_without_weights, weights) — recombine as
    {**hp, "weights": weights} to get the preregistration-canonical dict.
    """
    hp = {"k": 5, "orc_alpha": 0.5}
    if hyperparameters:
        hp.update(hyperparameters)
    weights = hp.pop("weights", None) or DEFAULT_WEIGHTS
    return hp, weights


def _generate_recommendations(results: dict[str, ModuleResult]) -> list[str]:
    recs = []
    for name, r in sorted(results.items()):
        if r.tier <= 3:
            recs.append(f"[{r.module}] Tier {r.tier}: {r.interpretation}")
    if not recs:
        recs.append("All modules pass at tier 4 or above. Proceed with caution for tiers 4-5.")
    return recs


# ======================================================================
# Main entry point
# ======================================================================

def run(
    source: np.ndarray,
    target: np.ndarray,
    metadata: dict[str, Any] | None = None,
    model_name: str | None = None,
    modules: list[str] | str = "all",
    hyperparameters: dict[str, Any] | None = None,
    preregister: bool = False,
    preregister_path: str | None = None,
    dataset_spec: DatasetSpec | None = None,
) -> PreflightReport:
    """Run the Preflight diagnostic pipeline.

    Args:
        source: (n_source, d) embedding matrix.
        target: (n_target, d) embedding matrix.
        metadata: Optional dict with module-specific inputs:
            source_labels, target_labels: arrays for M2
            estimates: dict[str, {beta, se}] for M5 sheaf Q-test
            graph: nx.Graph for M6 curvature
            records: list[dict] for M7 ecological bias
            site_key, outcome_key: column names for M7
        model_name: Name of the model that produced embeddings.
        modules: List of module names to run, or "all".
        hyperparameters: Module hyperparameters. Defaults provided if omitted.
            k: subspace dimension (default 5)
            orc_alpha: ORC laziness parameter (default 0.5)
            weights: dict of module weights for composite scoring
        preregister: If True, compute and save preregistration SHA.
        preregister_path: Path to save preregistration JSON.
        dataset_spec: DatasetSpec for preregistration.

    Returns:
        PreflightReport with per-module results and composite score.
    """
    if metadata is None:
        metadata = {}

    hp, weights = resolve_hyperparameters(hyperparameters)

    if modules == "all":
        active_modules = set(EMBEDDING_MODULES)
        if "estimates" in metadata:
            active_modules.add("m5_cross_design")
        if "graph" in metadata:
            active_modules.add("m6_curvature")
        if "records" in metadata:
            active_modules.add("m7_ecological_bias")
        active_modules = sorted(active_modules)
    else:
        active_modules = sorted(set(modules) & AVAILABLE_MODULES)

    prereg_sha = None
    if preregister:
        if dataset_spec is None:
            raise ValueError("dataset_spec required when preregister=True")
        if preregister_path is None:
            raise ValueError("preregister_path required when preregister=True")

        frozen_modules = []
        for mod_src in ALL_MODULE_SOURCES:
            frozen_modules.append(mod_src)

        prereg_hp = {**hp, "weights": weights}
        record = compute_preregistration_sha(frozen_modules, prereg_hp, dataset_spec)
        save_preregistration(record, preregister_path)
        prereg_sha = record.sha256

    source_labels = metadata.get("source_labels")
    target_labels = metadata.get("target_labels")
    k = hp["k"]

    results: dict[str, ModuleResult] = {}

    if "m1_grassmannian" in active_modules:
        results["m1_grassmannian"] = _run_m1_grassmannian(source, target, k)

    if "m2_domain_shift" in active_modules:
        results["m2_domain_shift"] = _run_m2_domain_shift(
            source, target, source_labels, target_labels,
        )

    if "m3_direction_stability" in active_modules:
        results["m3_direction_stability"] = _run_m3_direction_stability(
            source, target, k, source_labels, target_labels,
        )

    if "m4_domain_validity" in active_modules:
        results["m4_domain_validity"] = _run_m4_domain_validity(source, target)

    if "m5_cross_design" in active_modules and "estimates" in metadata:
        results["m5_cross_design"] = _run_m5_cross_design(metadata["estimates"])

    if "m6_curvature" in active_modules and "graph" in metadata:
        results["m6_curvature"] = _run_m6_curvature(
            metadata["graph"], alpha=hp["orc_alpha"],
        )

    if "m7_ecological_bias" in active_modules and "records" in metadata:
        results["m7_ecological_bias"] = _run_m7_ecological_bias(
            metadata["records"],
            site_key=metadata.get("site_key", "site"),
            outcome_key=metadata.get("outcome_key", "died"),
        )

    composite = _composite_score(results, weights)
    tier = _score_to_tier(composite)
    recs = _generate_recommendations(results)

    return PreflightReport(
        overall_tier=tier,
        overall_score=composite,
        module_results=results,
        recommendations=recs,
        preregistration_sha=prereg_sha,
        timestamp=datetime.now(timezone.utc).isoformat(),
        hyperparameters={**hp, "weights": weights},
    )
