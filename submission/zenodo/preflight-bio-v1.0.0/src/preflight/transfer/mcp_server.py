"""MCP server exposing the transfer toolkit as tools for autonomous agents.

Wraps each tool in the registry as an MCP tool with structured input/output.
The agent gets tool descriptions from ToolSpec metadata and can compose
tools freely based on when_to_use / check_after / combines_with guidance.

Usage:
    python -m preflight.transfer.mcp_server
"""

import json
import sys

import numpy as np

from preflight.transfer.registry import REGISTRY, ToolSpec


def _tool_schema(spec: ToolSpec) -> dict:
    """Build an MCP tool definition from a ToolSpec."""
    description_parts = [
        spec.description,
        f"\nWhen to use: {spec.when_to_use}",
        f"\nAssumptions: {spec.assumptions}",
        f"\nCheck after: {spec.check_after}",
    ]
    if spec.caution:
        description_parts.append(f"\nCaution: {spec.caution}")
    if spec.combines_with:
        description_parts.append(f"\nCombines with: {', '.join(spec.combines_with)}")

    return {
        "name": f"preflight_{spec.name}",
        "description": "".join(description_parts),
        "inputSchema": _input_schema(spec),
    }


def _input_schema(spec: ToolSpec) -> dict:
    """Infer JSON Schema for tool inputs from the function signature."""
    base_matrix = {
        "type": "object",
        "properties": {
            "source": {
                "type": "array",
                "description": "(n_source, d) feature matrix as nested list",
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "target": {
                "type": "array",
                "description": "(n_target, d) feature matrix as nested list",
                "items": {"type": "array", "items": {"type": "number"}},
            },
        },
        "required": ["source", "target"],
    }

    name = spec.name

    if name == "diagnose":
        schema = dict(base_matrix)
        schema["properties"] = dict(schema["properties"])
        schema["properties"]["source_labels"] = {
            "type": "array", "items": {"type": "integer"},
            "description": "Optional cell-type labels for source",
        }
        schema["properties"]["target_labels"] = {
            "type": "array", "items": {"type": "integer"},
            "description": "Optional cell-type labels for target",
        }
        schema["properties"]["k"] = {
            "type": "integer", "default": 5,
            "description": "Subspace dimension",
        }
        return schema

    if name in ("audit", "preflight_run", "ratio_correction", "location_scale",
                "coral", "support_intersection", "ot_correction",
                "invariant_features", "auto_steer"):
        return base_matrix

    if name == "deconfound":
        schema = dict(base_matrix)
        schema["properties"] = dict(schema["properties"])
        schema["properties"]["n_directions"] = {
            "type": "integer", "default": 1,
            "description": "Number of confound directions to remove",
        }
        return schema

    if name == "check_functional_alignment":
        return {
            "type": "object",
            "properties": {
                "X_source": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "(n, d) original source features",
                },
                "y_source": {
                    "type": "array", "items": {"type": "number"},
                    "description": "(n,) source labels",
                },
                "X_source_corrected": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "(n, d) corrected source features",
                },
            },
            "required": ["X_source", "y_source", "X_source_corrected"],
        }

    if name == "detect_concept_drift":
        return {
            "type": "object",
            "properties": {
                "source_X": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "source_y": {"type": "array", "items": {"type": "number"}},
                "target_X": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "target_y": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["source_X", "source_y", "target_X", "target_y"],
        }

    if name == "predict_target_performance":
        schema = dict(base_matrix)
        schema["properties"] = dict(schema["properties"])
        schema["properties"]["source_cv_score"] = {
            "type": "number",
            "description": "CV score from source domain",
        }
        schema["required"] = ["source", "target", "source_cv_score"]
        return schema

    if name == "suggest_calibration_samples":
        schema = dict(base_matrix)
        schema["properties"] = dict(schema["properties"])
        schema["properties"]["n_budget"] = {
            "type": "integer", "default": 10,
            "description": "Number of calibration samples to select",
        }
        return schema

    if name == "multi_source_weights":
        return {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "description": "List of source matrices",
                    "items": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                },
                "target": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
            },
            "required": ["sources", "target"],
        }

    if name == "DriftMonitor":
        return {
            "type": "object",
            "properties": {
                "reference": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "Reference distribution matrix",
                },
                "batch": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                    "description": "New batch to check for drift",
                },
            },
            "required": ["reference", "batch"],
        }

    return base_matrix


def _call_tool(name: str, arguments: dict) -> dict:
    """Execute a tool and return JSON-serializable result."""

    def _to_array(v):
        return np.array(v, dtype=np.float64)

    if name == "preflight_diagnose":
        from preflight.transfer.audit import diagnose
        result = diagnose(
            _to_array(arguments["source"]),
            _to_array(arguments["target"]),
            source_labels=np.array(arguments["source_labels"]) if "source_labels" in arguments else None,
            target_labels=np.array(arguments["target_labels"]) if "target_labels" in arguments else None,
            k=arguments.get("k", 5),
        )
        return {
            "shift_type": result.shift_type,
            "confidence": result.confidence,
            "domain_auc": result.domain_auc,
            "recommendations": result.recommendations,
            "preflight_tier": result.preflight.overall_tier if result.preflight else None,
            "preflight_score": result.preflight.overall_score if result.preflight else None,
            "module_scores": {
                k: {"score": v.score, "tier": v.tier}
                for k, v in result.preflight.module_results.items()
            } if result.preflight else {},
        }

    if name == "preflight_audit":
        from preflight.transfer.audit import audit
        result = audit(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {
            "shift_type": result.shift_type,
            "confidence": result.confidence,
            "domain_auc": result.domain_auc,
            "recommendations": result.recommendations,
        }

    if name == "preflight_preflight_run":
        from preflight.core.runner import run as preflight_run
        result = preflight_run(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {
            "overall_tier": result.overall_tier,
            "overall_score": result.overall_score,
            "module_scores": {
                k: {"score": v.score, "tier": v.tier, "interpretation": v.interpretation}
                for k, v in result.module_results.items()
            },
        }

    if name == "preflight_ratio_correction":
        from preflight.transfer.steer import ratio_correction
        result = ratio_correction(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {"method": result.method, "target_corrected_shape": list(result.target_corrected.shape)}

    if name == "preflight_location_scale":
        from preflight.transfer.steer import location_scale
        result = location_scale(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {"method": result.method, "target_corrected_shape": list(result.target_corrected.shape)}

    if name == "preflight_coral":
        from preflight.transfer.steer import coral
        result = coral(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {"method": result.method, "target_corrected_shape": list(result.target_corrected.shape)}

    if name == "preflight_deconfound":
        from preflight.transfer.deconfound import deconfound
        result = deconfound(
            _to_array(arguments["source"]), _to_array(arguments["target"]),
            n_directions=arguments.get("n_directions", 1),
        )
        return {
            "n_directions_removed": result.n_directions_removed,
            "explained_variance_removed": result.explained_variance_removed,
        }

    if name == "preflight_check_functional_alignment":
        from preflight.transfer.functional import check_functional_alignment
        result = check_functional_alignment(
            _to_array(arguments["X_source"]),
            np.array(arguments["y_source"]),
            _to_array(arguments["X_source_corrected"]),
        )
        return {
            "is_destructive": result.is_destructive,
            "r2_before": result.r2_before,
            "r2_after": result.r2_after,
            "r2_change": result.r2_change,
        }

    if name == "preflight_detect_concept_drift":
        from preflight.transfer.concept_drift import detect_concept_drift
        result = detect_concept_drift(
            _to_array(arguments["source_X"]), np.array(arguments["source_y"]),
            _to_array(arguments["target_X"]), np.array(arguments["target_y"]),
        )
        return {
            "has_concept_drift": result.has_concept_drift,
            "drift_severity": result.drift_severity,
        }

    if name == "preflight_predict_target_performance":
        from preflight.transfer.predict import predict_target_performance
        result = predict_target_performance(
            _to_array(arguments["source"]), _to_array(arguments["target"]),
            source_cv_score=arguments["source_cv_score"],
        )
        return {
            "estimated_target_score": result.estimated_target_score,
            "divergence_penalty": result.divergence_penalty,
            "upper_bound": result.upper_bound,
        }

    if name == "preflight_auto_steer":
        from preflight.transfer.auto_steer import auto_steer
        result = auto_steer(_to_array(arguments["source"]), _to_array(arguments["target"]))
        return {
            "best_method": result.best_method,
            "best_tier": result.best_tier,
            "best_score": result.best_score,
            "all_results": result.all_results,
        }

    if name == "preflight_DriftMonitor":
        from preflight.transfer.monitor import DriftMonitor
        mon = DriftMonitor(reference=_to_array(arguments["reference"]))
        snap = mon.check(_to_array(arguments["batch"]))
        return {
            "tier": snap.tier,
            "mmd": snap.mmd,
            "domain_auc": snap.domain_auc,
            "is_drifted": mon.is_drifted(),
        }

    raise ValueError(f"Unknown tool: {name}")


def serve_stdio():
    """Run the MCP server on stdin/stdout using JSON-RPC."""
    tools = [_tool_schema(spec) for spec in REGISTRY
             if spec.name not in ("run_benchmark", "run_harness",
                                  "conformal_intervals", "suggest_calibration_samples",
                                  "multi_source_weights", "support_intersection",
                                  "ot_correction", "invariant_features")]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method")

        if method == "initialize":
            _respond(msg_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "preflight-transfer", "version": "0.1.0"},
            })

        elif method == "tools/list":
            _respond(msg_id, {"tools": tools})

        elif method == "tools/call":
            params = msg.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            try:
                result = _call_tool(tool_name, arguments)
                _respond(msg_id, {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                })
            except Exception as e:
                _respond(msg_id, {
                    "content": [{"type": "text", "text": f"Error: {e}"}],
                    "isError": True,
                })

        elif method == "notifications/initialized":
            pass

        else:
            _respond(msg_id, None, error={"code": -32601, "message": f"Unknown method: {method}"})


def _respond(msg_id, result, error=None):
    response = {"jsonrpc": "2.0", "id": msg_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
