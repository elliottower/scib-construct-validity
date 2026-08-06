"""Preflight CLI — transportability diagnostics from the command line.

Usage:
    preflight run --source embeddings_a.npy --target embeddings_b.npy
    preflight run --source a.h5ad --target b.h5ad --obsm-key X_geneformer
    preflight run --source a.npy --target b.npy --preregister --output report.json
    preflight run --source a.npy --target b.npy --modules m1_grassmannian m2_domain_shift
    preflight run --source a.npy --target b.npy --format markdown
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from preflight.core.preregister import DatasetSpec
from preflight.core.runner import run
from preflight.extractors.anndata_loader import load_h5ad


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    elif p.suffix == ".npz":
        data = np.load(p)
        keys = list(data.keys())
        return data[keys[0]]
    elif p.suffix == ".csv":
        return np.loadtxt(p, delimiter=",", skiprows=1)
    elif p.suffix == ".tsv":
        return np.loadtxt(p, delimiter="\t", skiprows=1)
    else:
        return np.load(p)


def _load_h5ad(
    path: str,
    obsm_key: str | None = None,
    obs_label_key: str | None = None,
    obs_batch_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    return load_h5ad(
        path,
        obsm_key=obsm_key,
        obs_label_key=obs_label_key,
        obs_batch_key=obs_batch_key,
    )


def _load_metadata(path: str) -> dict:
    p = Path(path)
    with open(p) as f:
        return json.load(f)


def cmd_run(args: argparse.Namespace) -> int:
    source_path = Path(args.source)
    target_path = Path(args.target)
    is_h5ad_source = source_path.suffix == ".h5ad"
    is_h5ad_target = target_path.suffix == ".h5ad"

    metadata = {}

    if is_h5ad_source:
        source, source_labels, source_meta = _load_h5ad(
            args.source,
            obsm_key=getattr(args, "obsm_key", None),
            obs_label_key=getattr(args, "obs_label", None),
            obs_batch_key=getattr(args, "obs_batch", None),
        )
        if source_labels is not None:
            metadata["source_labels"] = source_labels
        metadata.update({f"source_{k}": v for k, v in source_meta.items()})
    else:
        source = _load_array(args.source)

    if is_h5ad_target:
        target, target_labels, target_meta = _load_h5ad(
            args.target,
            obsm_key=getattr(args, "obsm_key", None),
            obs_label_key=getattr(args, "obs_label", None),
            obs_batch_key=getattr(args, "obs_batch", None),
        )
        if target_labels is not None:
            metadata["target_labels"] = target_labels
        metadata.update({f"target_{k}": v for k, v in target_meta.items()})
    else:
        target = _load_array(args.target)

    if not is_h5ad_source and args.labels_source:
        metadata["source_labels"] = np.load(args.labels_source)
    if not is_h5ad_target and args.labels_target:
        metadata["target_labels"] = np.load(args.labels_target)
    if args.metadata:
        metadata.update(_load_metadata(args.metadata))

    hyperparameters = {"k": args.k}

    modules = args.modules if args.modules else "all"

    dataset_spec = None
    preregister_path = None
    if args.preregister:
        source_path = Path(args.source)
        target_path = Path(args.target)
        dataset_spec = DatasetSpec(
            name=f"{source_path.stem}_vs_{target_path.stem}",
            source_description=f"{source_path.name} ({source.shape[0]}x{source.shape[1]})",
            target_description=f"{target_path.name} ({target.shape[0]}x{target.shape[1]})",
            n_source=source.shape[0],
            n_target=target.shape[0],
            d=source.shape[1],
        )
        preregister_path = args.preregister_path or str(
            Path(args.output).with_suffix(".prereg.json") if args.output
            else "preflight_prereg.json"
        )

    report = run(
        source=source,
        target=target,
        metadata=metadata if metadata else None,
        modules=modules,
        hyperparameters=hyperparameters,
        preregister=args.preregister,
        preregister_path=preregister_path,
        dataset_spec=dataset_spec,
    )

    if args.format == "json":
        output_text = report.to_json()
    elif args.format == "markdown":
        output_text = report.to_markdown()
    else:
        output_text = report.to_json()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text)
        print(f"Report saved to {out_path}")
    else:
        print(output_text)

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from preflight.core.preregister import load_preregistration

    record = load_preregistration(args.prereg_file)
    print(f"Preregistration SHA: {record.sha256}")
    print(f"Timestamp: {record.timestamp}")
    print(f"Modules: {', '.join(record.modules_used)}")
    print(f"Code lengths: {record.module_code_lengths}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight",
        description="Transportability diagnostics for foundation models in biology",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run diagnostic pipeline")
    run_parser.add_argument("--source", required=True, help="Path to source data (.npy, .npz, .csv, .h5ad)")
    run_parser.add_argument("--target", required=True, help="Path to target data (.npy, .npz, .csv, .h5ad)")
    run_parser.add_argument("--labels-source", help="Path to source labels (.npy)")
    run_parser.add_argument("--labels-target", help="Path to target labels (.npy)")
    run_parser.add_argument("--obsm-key", dest="obsm_key", help="Key in .obsm for embeddings (h5ad only, e.g. X_pca)")
    run_parser.add_argument("--obs-label", dest="obs_label", help="Column in .obs for cell labels (h5ad only)")
    run_parser.add_argument("--obs-batch", dest="obs_batch", help="Column in .obs for batch/site (h5ad only)")
    run_parser.add_argument("--metadata", help="Path to metadata JSON")
    run_parser.add_argument("--modules", nargs="+", help="Modules to run (default: all embedding modules)")
    run_parser.add_argument("-k", type=int, default=5, help="Subspace dimension (default: 5)")
    run_parser.add_argument("--preregister", action="store_true", help="Freeze scorer SHA before running")
    run_parser.add_argument("--preregister-path", help="Path for preregistration JSON")
    run_parser.add_argument("--output", "-o", help="Output file path (default: stdout)")
    run_parser.add_argument("--format", "-f", choices=["json", "markdown"], default="json", help="Output format")

    verify_parser = subparsers.add_parser("verify", help="Inspect a preregistration record")
    verify_parser.add_argument("prereg_file", help="Path to preregistration JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "verify":
        return cmd_verify(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
