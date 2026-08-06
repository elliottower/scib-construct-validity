"""Transfer toolkit — self-describing tools for distribution shift.

Tools by category:

  Diagnosis:
    diagnose()                  — unified: shift type + preflight scoring (M1-M6)
    audit()                     — classify shift type only
    preflight_run()             — preflight module scoring only
    detect_concept_drift()      — test whether P(Y|X) changed
    invariant_features()        — find shift-stable features
    predict_target_performance() — Ben-David bound estimate

  Correction:
    ratio_correction()          — per-feature multiplicative fix
    location_scale()            — match first two moments
    coral()                     — align covariance structure
    ot_correction()             — optimal transport mapping
    support_intersection()      — clip to source support range
    deconfound()                — remove session-predictive direction (Double ML)

  Verification:
    check_functional_alignment() — CV R² before vs after correction

  Orchestration:
    auto_steer()                — try all corrections, pick best
    run_harness()               — iterative audit -> steer -> verify

  Uncertainty:
    conformal_intervals()       — importance-weighted prediction intervals
    suggest_calibration_samples() — targeted re-measurement design
    multi_source_weights()      — multi-source domain weighting

  Monitoring:
    DriftMonitor                — streaming drift detection

  Evaluation:
    run_benchmark()             — synthetic shift evaluation

  Registry:
    get_tools() / get_tool()    — query tool metadata
    describe_toolkit()          — human-readable catalog
    suggest_workflow()          — suggest tool sequence for a shift type

Usage:
    from preflight.transfer import diagnose, audit, check_functional_alignment
    from preflight.transfer import describe_toolkit, suggest_workflow
"""

from preflight.transfer.audit import audit, diagnose, ShiftDiagnosis
from preflight.transfer.steer import (
    steer, CorrectionResult,
    ratio_correction, location_scale, coral, support_intersection, ot_correction,
)
from preflight.transfer.harness import run_harness, TransferReport
from preflight.transfer.predict import predict_target_performance, PerformancePrediction
from preflight.transfer.concept_drift import detect_concept_drift, ConceptDriftResult
from preflight.transfer.calibration import suggest_calibration_samples, CalibrationDesign
from preflight.transfer.conformal import conformal_intervals, ConformalResult
from preflight.transfer.multi_source import multi_source_weights, MultiSourceWeights
from preflight.transfer.auto_steer import auto_steer, AutoSteerResult
from preflight.transfer.invariant import invariant_features, InvariantFeatureResult
from preflight.transfer.monitor import DriftMonitor, DriftSnapshot
from preflight.transfer.benchmark import (
    run_benchmark, format_benchmark_results, BenchmarkResult,
    generate_multiplicative_shift, generate_additive_shift,
    generate_rotational_shift, generate_support_shift,
)
from preflight.transfer.functional import check_functional_alignment, FunctionalCheckResult
from preflight.transfer.deconfound import deconfound, DeconfoundResult
from preflight.transfer.registry import (
    ToolSpec, REGISTRY, get_tools, get_tool, describe_toolkit, suggest_workflow,
)
from preflight.transfer.mcp_server import serve_stdio, _call_tool, _tool_schema

__all__ = [
    # Core pipeline
    "diagnose", "audit", "ShiftDiagnosis",
    "steer", "CorrectionResult",
    "run_harness", "TransferReport",
    # Corrections
    "ratio_correction", "location_scale", "coral", "support_intersection", "ot_correction",
    "deconfound", "DeconfoundResult",
    # Verification
    "check_functional_alignment", "FunctionalCheckResult",
    # Extensions
    "predict_target_performance", "PerformancePrediction",
    "detect_concept_drift", "ConceptDriftResult",
    "suggest_calibration_samples", "CalibrationDesign",
    "conformal_intervals", "ConformalResult",
    "multi_source_weights", "MultiSourceWeights",
    "auto_steer", "AutoSteerResult",
    "invariant_features", "InvariantFeatureResult",
    "DriftMonitor", "DriftSnapshot",
    "run_benchmark", "format_benchmark_results", "BenchmarkResult",
    "generate_multiplicative_shift", "generate_additive_shift",
    "generate_rotational_shift", "generate_support_shift",
    # Registry
    "ToolSpec", "REGISTRY", "get_tools", "get_tool", "describe_toolkit", "suggest_workflow",
    # MCP server
    "serve_stdio", "_call_tool", "_tool_schema",
]
