"""SHA-256 preregistration for Preflight scoring pipelines.

Freezes the scorer (module code + hyperparameters + dataset spec) into a
reproducible hash BEFORE running on real data. The hash covers:
  1. Source code of every module used in the run
  2. Hyperparameters (k, thresholds, metric choices)
  3. Dataset specification (NOT the data itself — just the identifier)
"""
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DatasetSpec:
    """Identifies what data will be evaluated (without containing it)."""
    name: str
    source_description: str
    target_description: str
    n_source: int | None = None
    n_target: int | None = None
    d: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreregistrationRecord:
    """Immutable record of a preregistration event."""
    sha256: str
    timestamp: str
    modules_used: list[str]
    hyperparameters: dict[str, Any]
    dataset_spec: dict[str, Any]
    module_code_lengths: dict[str, int]


def _hash_module_source(module) -> str:
    """Get the source code of a module for hashing."""
    try:
        return inspect.getsource(module)
    except (TypeError, OSError):
        return ""


def _collect_module_sources(modules: list) -> dict[str, str]:
    """Collect source code from a list of modules or functions."""
    sources = {}
    for mod in modules:
        name = getattr(mod, "__name__", str(mod))
        src = _hash_module_source(mod)
        if src:
            sources[name] = src
    return sources


def compute_preregistration_sha(
    modules: list,
    hyperparameters: dict[str, Any],
    dataset_spec: DatasetSpec,
) -> PreregistrationRecord:
    """Compute SHA-256 over scorer code + hyperparameters + dataset spec.

    This hash uniquely identifies the analysis configuration. Any change to
    module code, hyperparameters, or dataset specification produces a
    different hash.
    """
    hasher = hashlib.sha256()

    module_sources = _collect_module_sources(modules)
    for name in sorted(module_sources):
        hasher.update(f"MODULE:{name}\n".encode())
        hasher.update(module_sources[name].encode())

    hp_json = json.dumps(hyperparameters, sort_keys=True, default=str)
    hasher.update(f"HYPERPARAMETERS:{hp_json}\n".encode())

    ds_json = json.dumps(asdict(dataset_spec), sort_keys=True, default=str)
    hasher.update(f"DATASET:{ds_json}\n".encode())

    sha = hasher.hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    return PreregistrationRecord(
        sha256=sha,
        timestamp=timestamp,
        modules_used=[getattr(m, "__name__", str(m)) for m in modules],
        hyperparameters=hyperparameters,
        dataset_spec=asdict(dataset_spec),
        module_code_lengths={name: len(src) for name, src in module_sources.items()},
    )


def save_preregistration(record: PreregistrationRecord, path: Path) -> Path:
    """Save preregistration record to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(record), f, indent=2)
    return path


def load_preregistration(path: Path) -> PreregistrationRecord:
    """Load a saved preregistration record."""
    with open(path) as f:
        data = json.load(f)
    return PreregistrationRecord(**data)


def verify_preregistration(
    record: PreregistrationRecord,
    modules: list,
    hyperparameters: dict[str, Any],
    dataset_spec: DatasetSpec,
) -> tuple[bool, str]:
    """Verify that current config matches a saved preregistration.

    Returns (matches: bool, message: str).
    """
    current = compute_preregistration_sha(modules, hyperparameters, dataset_spec)
    if current.sha256 == record.sha256:
        return True, f"SHA match: {current.sha256[:16]}..."

    diffs = []
    if current.modules_used != record.modules_used:
        diffs.append(f"modules: {record.modules_used} -> {current.modules_used}")
    if current.hyperparameters != record.hyperparameters:
        diffs.append("hyperparameters changed")
    if current.dataset_spec != record.dataset_spec:
        diffs.append("dataset spec changed")
    if current.module_code_lengths != record.module_code_lengths:
        changed = []
        for name in set(current.module_code_lengths) | set(record.module_code_lengths):
            old = record.module_code_lengths.get(name, 0)
            new = current.module_code_lengths.get(name, 0)
            if old != new:
                changed.append(f"{name}: {old}->{new} chars")
        diffs.append(f"module code changed: {', '.join(changed)}")

    return False, f"SHA mismatch. Differences: {'; '.join(diffs)}"
