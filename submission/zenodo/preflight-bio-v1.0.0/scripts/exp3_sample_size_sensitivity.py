"""Exp 3: Sample size sensitivity analysis.

Preregistered experiment: for a single tissue (lung) and embedding
(Geneformer), vary cell count from 200 to 10000, run 3 replicates
per count, and measure score stability.

Key optimization: pull the full 10000-cell pool ONCE, then subsample
from it for each replicate.

Usage:
    .venv-census/bin/python scripts/exp3_sample_size_sensitivity.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
EMBEDDING = "geneformer"

SOURCE_FILTER = (
    "tissue_ontology_term_id == 'UBERON:0002048' "
    "and is_primary_data == True "
    "and assay_ontology_term_id == 'EFO:0009922'"
)
TARGET_FILTER = (
    "tissue_ontology_term_id == 'UBERON:0002048' "
    "and is_primary_data == True "
    "and assay_ontology_term_id == 'EFO:0011025'"
)

CELL_COUNTS = [200, 500, 1000, 2000, 5000, 10000]
REPLICATES = 3
REPLICATE_SEEDS = [0, 1, 2]
MAX_POOL = 10000

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

OUTPUT_DIR = Path("results/sensitivity")
PREREG_DIR = Path("docs/frozen_prereg_v6")


def _build_m7_records(source_adata, target_adata):
    records = []
    for adata in [source_adata, target_adata]:
        for _, row in adata.obs.iterrows():
            records.append({
                "site": str(row["dataset_id"]),
                "outcome": 0 if str(row.get("disease", "normal")) == "normal" else 1,
            })
    return records


def _build_m6_knn_graph(X_source, X_target, k=10, max_nodes=500):
    rng = np.random.default_rng(0)
    X = np.vstack([X_source, X_target])
    n = X.shape[0]
    if n > max_nodes:
        idx = rng.choice(n, size=max_nodes, replace=False)
        X = X[idx]
        n = max_nodes
    D = cdist(X, X, metric="cosine")
    np.fill_diagonal(D, np.inf)
    k_actual = min(k, n - 1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        neighbors = np.argsort(D[i])[:k_actual]
        for j in neighbors:
            G.add_edge(i, int(j))
    return G


def _query_pool(census, cellxgene_census, filter_str, max_cells, seed):
    """Pull a large pool of cells for subsequent subsampling."""
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Querying IDs: {filter_str[:70]}...")
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    n_total = len(obs_df)
    print(f"  Total matching cells: {n_total}")

    joinids = obs_df["soma_joinid"].values
    if n_total > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_cells, replace=False)
        idx.sort()
        joinids = joinids[idx]
        print(f"  Subsampled pool to: {len(joinids)} cells")

    print(f"  [{datetime.now(timezone.utc).isoformat()}] Pulling embeddings for {len(joinids)} cells...")
    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=[EMBEDDING],
    )
    print(f"  AnnData: {adata.n_obs} cells, obsm keys: {list(adata.obsm.keys())}")
    return adata


def main():
    try:
        import cellxgene_census
        import cellxgene_census.experimental
    except ImportError:
        print("ERROR: cellxgene-census not installed.")
        print("Run: uv pip install --python .venv-census/bin/python --only-binary :all: cellxgene-census")
        return 1

    from preflight.core.preregister import (
        DatasetSpec,
        load_preregistration,
        verify_preregistration,
    )
    from preflight.core.runner import (
        ALL_MODULE_SOURCES,
        resolve_hyperparameters,
        run,
    )
    from preflight.extractors.anndata_loader import extract_arrays
    from preflight.probes import run_probes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Exp 3: Sample size sensitivity")
    print(f"Census: {CENSUS_VERSION}, Embedding: {EMBEDDING}")
    print(f"Cell counts: {CELL_COUNTS}, Replicates: {REPLICATES}")

    # ==================================================================
    # STEP 1: Load and verify frozen preregistration
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [1/6] Loading frozen preregistration...")

    prereg_path = PREREG_DIR / "exp3_sample_size_sensitivity.json"
    if not prereg_path.exists():
        print(f"ERROR: Frozen prereg not found at {prereg_path}")
        print("Run: python scripts/freeze_preregistration_v4.py")
        return 1

    prereg_record = load_preregistration(prereg_path)
    frozen_sha = prereg_record.sha256
    print(f"  Frozen SHA: {frozen_sha[:32]}...")

    hp, weights = resolve_hyperparameters({"k": 5})
    canonical_hp = {**hp, "weights": weights}

    dataset_spec = DatasetSpec(
        name="exp3_sample_size_sensitivity",
        source_description="Lung 10x 3' v3 (geneformer), varying cell counts",
        target_description="Lung 10x 5' v2 (geneformer), varying cell counts",
        n_source=None,
        n_target=None,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": "geneformer",
            "cell_counts": [200, 500, 1000, 2000, 5000, 10000],
            "replicates_per_count": 3,
            "tissue": "lung",
        },
    )

    match, msg = verify_preregistration(prereg_record, ALL_MODULE_SOURCES, canonical_hp, dataset_spec)
    if not match:
        print(f"  VERIFICATION FAILED: {msg}")
        return 1
    print(f"  VERIFIED: {msg}")

    # ==================================================================
    # STEP 2: Pull the full 10000-cell pools ONCE
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [2/6] Pulling full cell pools...")

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("\n  Source pool:")
        src_pool_adata = _query_pool(census, cellxgene_census, SOURCE_FILTER, MAX_POOL, seed=42)

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        print("\n  Target pool:")
        tgt_pool_adata = _query_pool(census, cellxgene_census, TARGET_FILTER, MAX_POOL, seed=43)

    X_src_pool, _, _ = extract_arrays(src_pool_adata, obsm_key=EMBEDDING, obs_label_key="cell_type")
    X_tgt_pool, _, _ = extract_arrays(tgt_pool_adata, obsm_key=EMBEDDING, obs_label_key="cell_type")
    print(f"  Source pool embeddings: {X_src_pool.shape}")
    print(f"  Target pool embeddings: {X_tgt_pool.shape}")

    n_src_pool = X_src_pool.shape[0]
    n_tgt_pool = X_tgt_pool.shape[0]

    # ==================================================================
    # STEP 3: Run pipeline for each cell_count x replicate
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [3/6] Running {len(CELL_COUNTS)} x {REPLICATES} conditions...")

    # results[cell_count][rep_idx] = {...}
    all_results = {}
    total_runs = len(CELL_COUNTS) * REPLICATES

    with tqdm(total=total_runs, desc="Sensitivity runs") as pbar:
        for cell_count in CELL_COUNTS:
            all_results[cell_count] = []

            for rep_idx, seed in enumerate(REPLICATE_SEEDS):
                print(f"\n  [{datetime.now(timezone.utc).isoformat()}] "
                      f"n={cell_count}, rep={rep_idx} (seed={seed})")

                rng = np.random.default_rng(seed)

                # Subsample from pools
                actual_src_n = min(cell_count, n_src_pool)
                actual_tgt_n = min(cell_count, n_tgt_pool)
                src_idx = np.sort(rng.choice(n_src_pool, size=actual_src_n, replace=False))
                tgt_idx = np.sort(rng.choice(n_tgt_pool, size=actual_tgt_n, replace=False))

                X_src = X_src_pool[src_idx]
                X_tgt = X_tgt_pool[tgt_idx]
                src_sub = src_pool_adata[src_idx].copy()
                tgt_sub = tgt_pool_adata[tgt_idx].copy()

                print(f"    Subsampled: src={X_src.shape}, tgt={X_tgt.shape}")

                # Build metadata
                metadata = {}
                metadata["source_labels"] = src_sub.obs["cell_type"].values
                metadata["target_labels"] = tgt_sub.obs["cell_type"].values

                m7_records = _build_m7_records(src_sub, tgt_sub)
                n_sites = len(set(r["site"] for r in m7_records))
                if n_sites >= 2:
                    metadata["records"] = m7_records
                    metadata["site_key"] = "site"
                    metadata["outcome_key"] = "outcome"

                metadata["graph"] = _build_m6_knn_graph(X_src, X_tgt)

                # Run pipeline
                report = run(
                    source=X_src,
                    target=X_tgt,
                    metadata=metadata,
                    hyperparameters={"k": 5},
                    preregister=True,
                    preregister_path=str(prereg_path),
                    dataset_spec=dataset_spec,
                )
                print(f"    Tier: {report.overall_tier}/7, Score: {report.overall_score:.3f}")

                # Run probes
                combined_X = np.vstack([X_src, X_tgt])
                combined_labels = np.concatenate([
                    src_sub.obs["cell_type"].values,
                    tgt_sub.obs["cell_type"].values,
                ])
                combined_donors = np.concatenate([
                    src_sub.obs["donor_id"].values,
                    tgt_sub.obs["donor_id"].values,
                ])
                n_donors = len(np.unique(combined_donors))
                probe_k = min(5, n_donors)

                probe_results = None
                if probe_k >= 2:
                    probe_results = run_probes(
                        embeddings={EMBEDDING: combined_X},
                        labels=combined_labels,
                        batch_ids=combined_donors,
                        k=probe_k,
                    )

                _res_entry = {
                    "report": report,
                    "probe_results": probe_results,
                    "cell_count": cell_count,
                    "replicate": rep_idx,
                    "seed": seed,
                }
                all_results[cell_count].append(_res_entry)
                _inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"
                _inc_rec = {
                    "cell_count": cell_count, "replicate": rep_idx, "seed": seed,
                    **report.to_dict(),
                }
                with open(_inc_path, "a") as _jf:
                    _jf.write(json.dumps(_inc_rec, default=str) + "\n")
                    _jf.flush()

                pbar.update(1)

    # ==================================================================
    # STEP 4: Compute mean and SD per module per cell count
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [4/6] Aggregating results...")

    # Collect module names from first valid report
    module_names = None
    for cc in CELL_COUNTS:
        for res in all_results[cc]:
            if res["report"] is not None:
                module_names = sorted(res["report"].module_results.keys())
                break
        if module_names:
            break

    if module_names is None:
        print("ERROR: No valid reports produced")
        return 1

    # module_stats[module][cell_count] = {"mean": ..., "sd": ..., "values": [...]}
    module_stats = {m: {} for m in module_names}
    module_stats["overall"] = {}
    probe_stats = {}

    for cell_count in CELL_COUNTS:
        reps = all_results[cell_count]
        # Module scores
        for mod in module_names:
            scores = [r["report"].module_results[mod].score
                      for r in reps if mod in r["report"].module_results]
            if scores:
                module_stats[mod][cell_count] = {
                    "mean": float(np.mean(scores)),
                    "sd": float(np.std(scores)),
                    "values": [float(s) for s in scores],
                }

        # Overall score
        overall_scores = [r["report"].overall_score for r in reps]
        module_stats["overall"][cell_count] = {
            "mean": float(np.mean(overall_scores)),
            "sd": float(np.std(overall_scores)),
            "values": [float(s) for s in overall_scores],
        }

        # Probe F1
        f1_scores = []
        for r in reps:
            pr = r.get("probe_results")
            if pr and "cell_type" in pr.get("results_by_probe", {}):
                ct = pr["results_by_probe"]["cell_type"].get(EMBEDDING)
                if ct:
                    f1_scores.append(ct.mean)
        if f1_scores:
            probe_stats[cell_count] = {
                "mean": float(np.mean(f1_scores)),
                "sd": float(np.std(f1_scores)),
                "values": [float(s) for s in f1_scores],
            }

    print("\n  --- Module score mean +/- SD by cell count ---")
    header = f"  {'Module':<28}" + "".join(f"  n={cc:>5}" for cc in CELL_COUNTS)
    print(header)
    print("  " + "-" * (28 + 9 * len(CELL_COUNTS)))
    for mod in list(module_names) + ["overall"]:
        row = f"  {mod:<28}"
        for cc in CELL_COUNTS:
            stats_entry = module_stats[mod].get(cc)
            if stats_entry:
                row += f"  {stats_entry['mean']:.3f}+/-{stats_entry['sd']:.3f}"
            else:
                row += "      n/a"
        print(row)

    if probe_stats:
        print("\n  Probe F1:")
        for cc in CELL_COUNTS:
            ps = probe_stats.get(cc)
            if ps:
                print(f"    n={cc}: {ps['mean']:.3f} +/- {ps['sd']:.3f}")

    # ==================================================================
    # STEP 5: Test hypotheses
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [5/6] Testing hypotheses...")

    # H3.1: Module scores stabilize (SD < 0.05 for all modules) at <= 2000 cells
    print("\n  H3.1: Score stabilization (SD < 0.05 for all modules at <= 2000 cells)")
    stabilization_n = None
    for cc in CELL_COUNTS:
        if cc > 2000:
            break
        all_stable = True
        for mod in module_names:
            stats_entry = module_stats[mod].get(cc)
            if stats_entry and stats_entry["sd"] >= 0.05:
                all_stable = False
                break
        if all_stable:
            stabilization_n = cc
            break

    if stabilization_n is not None:
        h3_1_pass = True
        print(f"    Stabilizes at n={stabilization_n}")
    else:
        h3_1_pass = False
        # Check what the max SD is at 2000
        max_sd_at_2000 = 0.0
        worst_mod = ""
        for mod in module_names:
            stats_entry = module_stats[mod].get(2000)
            if stats_entry and stats_entry["sd"] > max_sd_at_2000:
                max_sd_at_2000 = stats_entry["sd"]
                worst_mod = mod
        print(f"    Not stabilized at n<=2000 (worst: {worst_mod} SD={max_sd_at_2000:.4f} at n=2000)")
    print(f"    H3.1 {'PASS' if h3_1_pass else 'FAIL'}")

    # H3.2: Cell-type probe F1 increases monotonically with cell count up to 2000
    print("\n  H3.2: Probe F1 monotonically increasing up to n=2000")
    counts_up_to_2000 = [cc for cc in CELL_COUNTS if cc <= 2000]
    f1_means = [probe_stats[cc]["mean"] for cc in counts_up_to_2000 if cc in probe_stats]
    if len(f1_means) >= 2:
        monotonic = all(f1_means[i] <= f1_means[i + 1] for i in range(len(f1_means) - 1))
        h3_2_pass = monotonic
        for i, cc in enumerate(counts_up_to_2000):
            if cc in probe_stats:
                print(f"    n={cc}: F1 = {probe_stats[cc]['mean']:.4f}")
        print(f"    Monotonic: {monotonic}")
    else:
        h3_2_pass = False
        print(f"    SKIP: insufficient probe data ({len(f1_means)} points)")
    print(f"    H3.2 {'PASS' if h3_2_pass else 'FAIL'}")

    # H3.3: M6 curvature score stable (SD < 0.05) at <= 1000 cells
    print("\n  H3.3: M6 curvature SD < 0.05 at <= 1000 cells")
    m6_stable_n = None
    for cc in CELL_COUNTS:
        if cc > 1000:
            break
        m6_stats = module_stats.get("m6_curvature", {}).get(cc)
        if m6_stats and m6_stats["sd"] < 0.05:
            m6_stable_n = cc
            break

    if m6_stable_n is not None:
        h3_3_pass = True
        print(f"    M6 stable at n={m6_stable_n} (SD={module_stats['m6_curvature'][m6_stable_n]['sd']:.4f})")
    else:
        h3_3_pass = False
        m6_at_1000 = module_stats.get("m6_curvature", {}).get(1000)
        if m6_at_1000:
            print(f"    M6 at n=1000: SD={m6_at_1000['sd']:.4f}")
        else:
            print(f"    M6 curvature not available at n<=1000")
    print(f"    H3.3 {'PASS' if h3_3_pass else 'FAIL'}")

    # ==================================================================
    # STEP 6: Save results
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [6/6] Saving results...")

    summary = {
        "timestamp": timestamp,
        "frozen_prereg_sha": frozen_sha,
        "census_version": CENSUS_VERSION,
        "embedding": EMBEDDING,
        "cell_counts": CELL_COUNTS,
        "replicates": REPLICATES,
        "source_pool_size": n_src_pool,
        "target_pool_size": n_tgt_pool,
        "module_stats": {},
        "probe_stats": {str(k): v for k, v in probe_stats.items()},
        "hypotheses": {
            "H3.1_stabilization": {
                "pass": h3_1_pass,
                "stabilization_n": stabilization_n,
                "threshold_sd": 0.05,
                "threshold_n": 2000,
            },
            "H3.2_monotonic_f1": {
                "pass": h3_2_pass,
                "f1_means": {str(cc): probe_stats[cc]["mean"]
                             for cc in counts_up_to_2000 if cc in probe_stats},
            },
            "H3.3_m6_stability": {
                "pass": h3_3_pass,
                "stable_at_n": m6_stable_n,
                "threshold_sd": 0.05,
                "threshold_n": 1000,
            },
        },
    }

    for mod in list(module_names) + ["overall"]:
        summary["module_stats"][mod] = {
            str(cc): module_stats[mod].get(cc) for cc in CELL_COUNTS
        }

    # Save per-run reports
    for cell_count in CELL_COUNTS:
        for res in all_results[cell_count]:
            rep = res["replicate"]
            report_path = OUTPUT_DIR / f"report_n{cell_count}_rep{rep}_{timestamp}.json"
            res["report"].save_json(report_path)

    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # ==================================================================
    # Final report
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("EXP 3 RESULTS: Sample size sensitivity")
    print(f"{'=' * 60}")
    print(f"  Prereg SHA: {frozen_sha[:32]}...")
    print(f"  H3.1 stabilization at <=2000: {'PASS' if h3_1_pass else 'FAIL'}")
    print(f"  H3.2 monotonic F1:            {'PASS' if h3_2_pass else 'FAIL'}")
    print(f"  H3.3 M6 stable at <=1000:     {'PASS' if h3_3_pass else 'FAIL'}")
    n_pass = sum([h3_1_pass, h3_2_pass, h3_3_pass])
    print(f"  Total: {n_pass}/3 hypotheses confirmed")
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Done. Artifacts in {OUTPUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
