"""Tool registry — self-describing catalog of all transfer tools.

Each tool is registered with metadata describing what it does, when to
use it, what assumptions it makes, and what to check after using it.
An agent (Claude via MCP, or any LLM with tool access) can query the
registry to decide which tools to apply for a given problem.

The registry replaces the rigid audit→steer→verify pipeline with a
toolkit that an intelligent agent can compose freely.
"""

from dataclasses import dataclass, field


@dataclass
class ToolSpec:
    name: str
    module: str
    function: str
    category: str
    description: str
    when_to_use: str
    assumptions: str
    check_after: str
    combines_with: list[str] = field(default_factory=list)
    caution: str = ""


REGISTRY: list[ToolSpec] = [
    # ── Diagnosis ──
    ToolSpec(
        name="diagnose",
        module="preflight.transfer.audit",
        function="diagnose(source, target, source_labels=None, target_labels=None, metadata=None)",
        category="diagnosis",
        description="Unified diagnosis: runs shift classification AND preflight module scoring (M1-M6) in one call. Returns ShiftDiagnosis with PreflightReport attached.",
        when_to_use="Always run first. The recommended entry point — gives you both the shift type (for choosing corrections) and the preflight tier (for gauging severity).",
        assumptions="Source and target have the same feature space. Pass labels when available for better M3 direction stability (uses LDA instead of PCA).",
        check_after="shift_type guides correction choice. preflight.overall_tier gauges severity. Check per-module scores: M4 low = dimensional collapse, M6 low = diffuse connectivity.",
        combines_with=["steer", "auto_steer", "deconfound", "check_functional_alignment"],
    ),
    ToolSpec(
        name="audit",
        module="preflight.transfer.audit",
        function="audit(source, target)",
        category="diagnosis",
        description="Classify distribution shift type only (without preflight scoring). Use diagnose() instead for full analysis.",
        when_to_use="When you only need shift type classification without the full preflight pipeline. Faster than diagnose().",
        assumptions="Source and target have the same feature space (same number of columns, same meaning per column).",
        check_after="Look at domain_auc (>0.6 = significant shift), shift_type (guides correction choice), and feature_importances (which features drive the shift).",
        combines_with=["steer", "auto_steer", "deconfound"],
    ),
    ToolSpec(
        name="preflight_run",
        module="preflight.core.runner",
        function="run(source, target, metadata=None)",
        category="diagnosis",
        description="Run all preflight modules (M1 subspace alignment, M2 domain shift, M3 direction stability, M4 domain validity, M6 curvature) and return composite score + tier. Pass metadata with graph/records/estimates for M5-M7.",
        when_to_use="When you need the preflight tier and per-module scores without shift-type classification. Use diagnose() to get both.",
        assumptions="Source and target are embedding matrices of the same dimensionality. Labels improve M3 (LDA vs PCA).",
        check_after="overall_tier (1-7, higher=better). Per-module tiers identify specific problems: M1 low=subspace misaligned, M2 low=domain shift, M3 low=discriminant directions unstable, M4 low=dimensional collapse.",
        combines_with=["audit", "diagnose"],
    ),
    ToolSpec(
        name="detect_concept_drift",
        module="preflight.transfer.concept_drift",
        function="detect_concept_drift(source_X, source_y, target_X, target_y)",
        category="diagnosis",
        description="Test whether P(Y|X) differs between domains, not just P(X). If concept drift exists, no feature correction will help.",
        when_to_use="When you have labels for both domains. Run before investing in correction — if P(Y|X) has changed, corrections are futile.",
        assumptions="Both domains have labels. Regression or classification task.",
        check_after="has_concept_drift=True means the relationship between features and outcome has changed. Consider domain-specific modeling instead of correction.",
        combines_with=["audit"],
        caution="Requires labels for both domains, which is often unavailable for the target.",
    ),
    ToolSpec(
        name="invariant_features",
        module="preflight.transfer.invariant",
        function="invariant_features(source, target)",
        category="diagnosis",
        description="Identify features stable across domains using KS statistic + ratio deviation. Training on invariant features gives robustness without explicit correction.",
        when_to_use="When you want to select features that are inherently transferable rather than correcting all features.",
        assumptions="Features that are statistically similar across domains carry the same signal.",
        check_after="pct_invariant tells you what fraction of features survived. If very low (<10%), the shift is pervasive and feature selection alone won't help.",
        combines_with=["audit", "steer"],
    ),
    ToolSpec(
        name="predict_target_performance",
        module="preflight.transfer.predict",
        function="predict_target_performance(source, target, source_cv_score)",
        category="diagnosis",
        description="Estimate expected target performance from source CV + domain divergence using a simplified Ben-David bound.",
        when_to_use="To set expectations before deploying. Tells you 'your model will probably score X on target' given the measured shift.",
        assumptions="Source CV score is an honest estimate of in-distribution performance. Shift is covariate shift (P(X) changes, P(Y|X) constant).",
        check_after="If estimated_target_score is much lower than source_cv_score, the shift is too large for reliable transfer without correction.",
        combines_with=["audit"],
    ),

    # ── Correction ──
    ToolSpec(
        name="ratio_correction",
        module="preflight.transfer.steer",
        function="ratio_correction(source, target, trim_frac=0.1)",
        category="correction",
        description="Per-feature multiplicative correction using trimmed mean ratio. Inverts the per-feature transfer function source_mean/target_mean.",
        when_to_use="When audit says MULTIPLICATIVE. The workhorse for instrument calibration, batch effects, intensity drift. trim_frac=0.1 is robust to outliers.",
        assumptions="Shift is per-feature multiplicative (each feature scaled by a different constant). No cross-feature interactions in the drift.",
        check_after="Run audit() again — domain_auc should drop. Run check_functional_alignment() to verify predictions are preserved.",
        combines_with=["deconfound", "check_functional_alignment"],
        caution="Do NOT stack with SNV or other per-sample normalizations that also remove multiplicative scatter — double correction destroys signal.",
    ),
    ToolSpec(
        name="location_scale",
        module="preflight.transfer.steer",
        function="location_scale(source, target)",
        category="correction",
        description="Match first two moments: standardize target, rescale to source statistics.",
        when_to_use="When audit says ADDITIVE. Good for location shifts where each feature has a different offset.",
        assumptions="Shift is location + scale (mean and variance differ per feature, but the shape of each feature's distribution is preserved).",
        check_after="Run audit() again. Check that prediction means are centered correctly.",
        combines_with=["deconfound", "check_functional_alignment"],
    ),
    ToolSpec(
        name="coral",
        module="preflight.transfer.steer",
        function="coral(source, target, reg=1.0)",
        category="correction",
        description="CORAL: align second-order statistics by whitening target covariance and coloring with source covariance.",
        when_to_use="When audit says ROTATIONAL. The covariance structure has changed (e.g., feature correlations differ across domains).",
        assumptions="Shift is primarily in the covariance structure. Requires enough samples to estimate covariance reliably (n > d).",
        check_after="Run audit() again — geodesic distance should drop.",
        combines_with=["deconfound"],
        caution="Expensive for high-dimensional data (d > 200). Use PCA reduction first if needed.",
    ),
    ToolSpec(
        name="ot_correction",
        module="preflight.transfer.steer",
        function="ot_correction(source, target, reg=0.1)",
        category="correction",
        description="Optimal transport via Sinkhorn barycentric mapping. Maps each target sample to a weighted combination of source samples.",
        when_to_use="When the shift is complex and doesn't fit multiplicative/additive/rotational categories. Nonparametric.",
        assumptions="Source and target have similar support. Falls back to location_scale for large inputs.",
        check_after="Check that corrected samples look reasonable — OT can produce averages that wash out structure.",
        combines_with=["deconfound"],
        caution="O(n_source * n_target) memory. Falls back automatically if too large.",
    ),
    ToolSpec(
        name="support_intersection",
        module="preflight.transfer.steer",
        function="support_intersection(source, target, percentile_clip=5.0)",
        category="correction",
        description="Clip target features to the source support range. Prevents extrapolation.",
        when_to_use="When audit says SUPPORT_MISMATCH. Target values fall outside the range the model was trained on.",
        assumptions="Values outside source range are artifacts, not real signal.",
        check_after="Check pct_values_clipped — if very high, the domains may be fundamentally incompatible.",
        combines_with=["ratio_correction", "location_scale"],
    ),
    ToolSpec(
        name="deconfound",
        module="preflight.transfer.deconfound",
        function="deconfound(source, target, n_directions=1)",
        category="correction",
        description="Remove session-predictive directions from feature space using Double ML. Projects out the logistic regression decision boundary between source and target.",
        when_to_use="BEFORE other corrections, when session/batch is a confound that correlates with outcome signal. Most helpful for linear downstream models (PLS, Ridge). Less helpful when using tree ensembles that handle confounding implicitly.",
        assumptions="The session direction is separable from outcome-relevant directions. With n_directions=1, assumes a single dominant confound axis.",
        check_after="Run audit() — AUC should drop. Run check_functional_alignment() — R² should be preserved or improve. If R² drops, the session direction carried outcome signal.",
        combines_with=["ratio_correction", "location_scale", "check_functional_alignment"],
        caution="n_directions > 3 tends to remove too much signal. Always check functional alignment after deconfounding.",
    ),

    # ── Verification ──
    ToolSpec(
        name="check_functional_alignment",
        module="preflight.transfer.functional",
        function="check_functional_alignment(X_source, y_source, X_source_corrected)",
        category="verification",
        description="Check whether a correction preserves prediction quality by comparing CV R² before and after. Catches the failure mode where distributional alignment (M2 AUC→0) coexists with functional misalignment (worse predictions).",
        when_to_use="After ANY correction, before trusting it. The SNV disaster: M2 AUC dropped to 0 (perfect distributional match) but predictions went from 0.80 to -0.83 R². This check would have caught it.",
        assumptions="Labeled data available for the source/calibration domain. Correction applied to source data as a self-test.",
        check_after="is_destructive=True means STOP — the correction is removing signal. Try a different correction or reduce its strength.",
        combines_with=["ratio_correction", "location_scale", "coral", "deconfound"],
    ),

    # ── Orchestration ──
    ToolSpec(
        name="auto_steer",
        module="preflight.transfer.auto_steer",
        function="auto_steer(source, target)",
        category="orchestration",
        description="Try all available corrections and pick the best by preflight composite score. Removes the need to classify shift type first.",
        when_to_use="When you're not sure which correction to use, or want to compare all options systematically.",
        assumptions="Preflight composite score is a good proxy for correction quality. Note: this measures distributional alignment, not functional alignment.",
        check_after="The best method's preflight tier. But ALSO run check_functional_alignment() — auto_steer optimizes distributional alignment which can diverge from functional alignment.",
        combines_with=["check_functional_alignment"],
        caution="Does not check functional alignment. A correction that scores Tier 7 on preflight but destroys predictions will still be selected.",
    ),
    ToolSpec(
        name="run_harness",
        module="preflight.transfer.harness",
        function="run_harness(source, target, max_iterations=3)",
        category="orchestration",
        description="Iterative audit→steer→verify loop. Applies corrections until preflight tier exceeds threshold.",
        when_to_use="For hands-off correction: let the pipeline find and fix the shift iteratively.",
        assumptions="Each correction step makes independent progress. Convergence = preflight tier above threshold.",
        check_after="converged flag, final_tier, corrections_applied list.",
        combines_with=["check_functional_alignment"],
    ),

    # ── Uncertainty ──
    ToolSpec(
        name="conformal_intervals",
        module="preflight.transfer.conformal",
        function="conformal_intervals(source_X, source_y, target_X, model, alpha=0.1)",
        category="uncertainty",
        description="Importance-weighted conformal prediction intervals. Uses domain classifier density ratios for valid coverage under covariate shift.",
        when_to_use="When you need prediction intervals (not just point predictions) that account for distribution shift.",
        assumptions="Covariate shift (P(Y|X) constant). Requires a fitted model.",
        check_after="mean_width (narrower = more confident), effective_n (low = severe shift, intervals less reliable).",
        combines_with=["ratio_correction", "audit"],
    ),
    ToolSpec(
        name="suggest_calibration_samples",
        module="preflight.transfer.calibration",
        function="suggest_calibration_samples(source, target, n_budget=10)",
        category="uncertainty",
        description="Select source samples to re-measure on target instrument for maximum calibration coverage.",
        when_to_use="When you can measure a few source samples on the target device. Tells you WHICH samples to re-measure for maximum information.",
        assumptions="Re-measuring is possible but expensive. Samples selected to span the shifted dimensions.",
        check_after="coverage_score (closer to 1.0 = selected samples represent the shifted variation well).",
        combines_with=["audit"],
    ),
    ToolSpec(
        name="multi_source_weights",
        module="preflight.transfer.multi_source",
        function="multi_source_weights(sources, target)",
        category="uncertainty",
        description="When multiple source domains exist, weight each by similarity to target.",
        when_to_use="Multi-site or multi-instrument studies where you want to weight sources by relevance.",
        assumptions="Closer sources in distribution space are more informative for the target.",
        check_after="weights array — sources with near-zero weight should be excluded.",
        combines_with=["audit"],
    ),

    # ── Monitoring ──
    ToolSpec(
        name="DriftMonitor",
        module="preflight.transfer.monitor",
        function="DriftMonitor(reference=X_ref).check(batch)",
        category="monitoring",
        description="Stateful drift detector. Feed batches of new data, get alerts when distribution drifts past a threshold.",
        when_to_use="Production deployment: monitor incoming data for drift that would degrade model performance.",
        assumptions="Reference distribution is representative of good performance.",
        check_after="is_drifted() flag. If True, re-run the correction pipeline on new data.",
        combines_with=["audit", "ratio_correction"],
    ),

    # ── Evaluation ──
    ToolSpec(
        name="run_benchmark",
        module="preflight.transfer.benchmark",
        function="run_benchmark(correction_fn, shift_types=[...], n_trials=5)",
        category="evaluation",
        description="Evaluate a correction function against synthetic shift datasets. Generates multiplicative, additive, rotational, and support shifts.",
        when_to_use="Before applying a correction to real data — check how it performs on known ground truth.",
        assumptions="Synthetic shifts approximate real-world shifts. Useful for method selection, not final validation.",
        check_after="improvement_pct per shift type. A correction that helps on multiplicative but hurts on additive tells you about its assumptions.",
        combines_with=[],
    ),
]


def get_tools(category: str | None = None) -> list[ToolSpec]:
    """Get tools, optionally filtered by category."""
    if category is None:
        return REGISTRY
    return [t for t in REGISTRY if t.category == category]


def get_tool(name: str) -> ToolSpec | None:
    """Get a specific tool by name."""
    for t in REGISTRY:
        if t.name == name:
            return t
    return None


def describe_toolkit() -> str:
    """Human-readable description of all available tools."""
    categories = {}
    for t in REGISTRY:
        categories.setdefault(t.category, []).append(t)

    lines = ["# Transfer Toolkit", ""]
    for cat in ["diagnosis", "correction", "verification", "orchestration", "uncertainty", "monitoring", "evaluation"]:
        tools = categories.get(cat, [])
        if not tools:
            continue
        lines.append(f"## {cat.title()}")
        lines.append("")
        for t in tools:
            lines.append(f"### {t.name}")
            lines.append(f"`{t.function}`")
            lines.append(f"{t.description}")
            lines.append(f"**When**: {t.when_to_use}")
            if t.caution:
                lines.append(f"**Caution**: {t.caution}")
            lines.append(f"**Check after**: {t.check_after}")
            if t.combines_with:
                lines.append(f"**Combines with**: {', '.join(t.combines_with)}")
            lines.append("")
    return "\n".join(lines)


def suggest_workflow(shift_type: str, has_labels: bool = False, preflight_tier: int | None = None) -> list[str]:
    """Suggest a tool workflow given a diagnosed shift type.

    Returns tool names in suggested execution order.
    """
    steps = ["diagnose"]

    if preflight_tier is not None and preflight_tier >= 5 and shift_type == "NONE":
        return steps

    if has_labels:
        steps.append("detect_concept_drift")

    if shift_type == "MULTIPLICATIVE":
        steps.extend(["deconfound", "ratio_correction"])
    elif shift_type == "ADDITIVE":
        steps.extend(["deconfound", "location_scale"])
    elif shift_type == "ROTATIONAL":
        steps.extend(["deconfound", "coral"])
    elif shift_type == "SUPPORT_MISMATCH":
        steps.append("support_intersection")
    elif shift_type == "MIXED":
        steps.append("auto_steer")
    elif shift_type == "NONE":
        return steps

    if has_labels:
        steps.append("check_functional_alignment")

    steps.append("diagnose")

    return steps
