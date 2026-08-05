"""Modal: Exp17 — Marker gene recovery as non-classification ground truth.

Same 4 Census embeddings as exp16. Computes all 12 metrics + F1 (as before)
PLUS marker gene recovery (MGR) score per (tissue, model) and a ceiling
control per tissue. The summary then reports correlations of each metric
with both F1 and MGR (raw and ceiling-normalized), using block bootstrap
(resampling tissues, not conditions) for CIs.

Usage:
    modal run --detach scripts/modal_exp17_marker_gene_recovery.py
    modal volume ls preflight-results exp17_marker_gene_recovery
    modal volume get preflight-results exp17_marker_gene_recovery results/exp17_marker_gene_recovery
"""
import modal

app = modal.App("preflight-exp17-marker-gene-recovery")

vol = modal.Volume.from_name("preflight-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.14.1",
        "scikit-learn==1.6.0",
        "anndata==0.11.4",
        "cellxgene-census==1.16.2",
        "tqdm==4.67.1",
        "pandas==2.2.3",
        "matplotlib==3.9.3",
        "scanpy==1.11.5",
        "leidenalg==0.10.2",
        "igraph==0.11.8",
    )
    .add_local_file("scripts/exp15_scc_stress_test.py", "/app/exp15_scc_stress_test.py", copy=True)
    .add_local_file("scripts/exp17_marker_gene_recovery.py", "/app/exp17_marker_gene_recovery.py", copy=True)
    .workdir("/app")
)

VOL_BASE = "/vol/exp17_marker_gene_recovery"

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"
EMBEDDINGS = ["geneformer", "scvi", "scgpt", "uce"]
MAX_CELLS = 2000
MIN_SHARED_TYPES = 8
SEED = 20260801
TOP_N_GENES = 50

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768)
def run_tissue(tissue_index, tissue_name, tissue_id):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import cellxgene_census
    import numpy as np
    from scipy.sparse import issparse

    from exp15_scc_stress_test import compute_all_metrics
    from exp17_marker_gene_recovery import (
        marker_gene_recovery,
        marker_gene_recovery_ceiling,
        wilcoxon_markers,
    )

    def _ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    vol_dir = Path(f"{VOL_BASE}/tissue_{tissue_index:03d}")
    result_path = vol_dir / "results.json"

    if vol_dir.exists() and result_path.exists():
        existing = json.loads(result_path.read_text())
        if "ceiling" in existing and all(f"mgr_{m}" in existing for m in EMBEDDINGS):
            print(f"[{tissue_index}] {tissue_name}: already done, skipping", flush=True)
            return existing

    print(f"[{tissue_index}] {_ts()} {tissue_name}: downloading from Census...", flush=True)

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        sides = {}
        for side, assay_id, seed_offset in [
            ("source", SOURCE_ASSAY, 0),
            ("target", TARGET_ASSAY, 1),
        ]:
            obs_filter = (
                f"tissue_ontology_term_id == '{tissue_id}' "
                f"and is_primary_data == True "
                f"and disease == 'normal' "
                f"and assay_ontology_term_id == '{assay_id}'"
            )
            obs_df = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=obs_filter,
                column_names=["soma_joinid", "donor_id"],
            )
            all_ids = obs_df["soma_joinid"].values
            print(f"  {_ts()} {side}: {len(all_ids)} cells available", flush=True)
            if len(all_ids) == 0:
                print(f"  {_ts()} No cells for {side}, skipping tissue", flush=True)
                return None

            rng = np.random.default_rng(SEED + seed_offset)
            if len(all_ids) > MAX_CELLS:
                idx = rng.choice(len(all_ids), size=MAX_CELLS, replace=False)
                idx.sort()
                all_ids = all_ids[idx]

            adata = cellxgene_census.get_anndata(
                census, organism=ORGANISM,
                obs_value_filter=obs_filter,
                obs_coords=all_ids,
                obs_column_names=OBS_COLUMNS,
                obs_embeddings=EMBEDDINGS,
            )
            print(f"  {_ts()} {side}: {adata.n_obs} cells, "
                  f"{len(adata.obs['cell_type'].unique())} types", flush=True)
            sides[side] = adata

    src = sides["source"]
    tgt = sides["target"]
    shared = sorted(set(src.obs["cell_type"].values) & set(tgt.obs["cell_type"].values))
    k = len(shared)
    if k < MIN_SHARED_TYPES:
        print(f"  {_ts()} Only {k} shared types, skipping", flush=True)
        return None

    print(f"  {_ts()} {k} shared cell types", flush=True)

    src_labels = src.obs["cell_type"].values.astype(str)
    tgt_labels = tgt.obs["cell_type"].values.astype(str)

    X_raw_src = src.X.toarray() if issparse(src.X) else np.asarray(src.X)
    X_raw_tgt = tgt.X.toarray() if issparse(tgt.X) else np.asarray(tgt.X)

    print(f"  {_ts()} Computing reference markers (once for all embeddings)...", flush=True)
    ref_markers = wilcoxon_markers(X_raw_src, src_labels, shared, TOP_N_GENES)
    print(f"  {_ts()} Reference markers done", flush=True)

    print(f"  {_ts()} Computing ceiling control (true target labels)...", flush=True)
    ceiling = marker_gene_recovery_ceiling(
        X_raw_tgt, tgt_labels, shared, ref_markers, TOP_N_GENES,
    )
    print(f"  {_ts()} Ceiling MGR={ceiling['ceiling_score']:.3f}", flush=True)

    result = {
        "tissue": tissue_name,
        "tissue_index": tissue_index,
        "k": k,
        "n_src": src.n_obs,
        "n_tgt": tgt.n_obs,
        "ceiling": ceiling,
    }

    for emb_name in EMBEDDINGS:
        if emb_name not in src.obsm or emb_name not in tgt.obsm:
            print(f"  {_ts()} {emb_name}: missing from obsm, skipping", flush=True)
            continue

        X_src = np.asarray(src.obsm[emb_name])
        X_tgt = np.asarray(tgt.obsm[emb_name])

        if not (np.all(np.isfinite(X_src)) and np.all(np.isfinite(X_tgt))):
            print(f"  {_ts()} {emb_name}: non-finite values, skipping", flush=True)
            continue

        print(f"  {_ts()} {emb_name} (d={X_src.shape[1]}): computing 12 metrics + F1...",
              flush=True)
        metrics = compute_all_metrics(X_src, X_tgt, src_labels, tgt_labels, shared)
        result[f"metrics_{emb_name}"] = metrics
        print(f"  {_ts()} {emb_name}: F1={metrics['f1']:.3f}, "
              f"SCC_lr={metrics['scc_logreg']:.3f}", flush=True)

        print(f"  {_ts()} {emb_name}: computing marker gene recovery...", flush=True)
        mgr = marker_gene_recovery(
            X_raw_tgt, X_tgt, tgt_labels, shared, ref_markers,
            TOP_N_GENES,
        )
        result[f"mgr_{emb_name}"] = mgr
        norm = mgr["mgr_score"] / ceiling["ceiling_score"] if ceiling["ceiling_score"] > 0.01 else float("nan")
        print(f"  {_ts()} {emb_name}: MGR={mgr['mgr_score']:.3f} "
              f"(norm={norm:.3f}, {mgr['n_matched_types']} types, "
              f"{mgr['n_clusters']} clusters)", flush=True)

    vol_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()
    print(f"  {_ts()} Done: {tissue_name}", flush=True)
    return result


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=16384)
def compute_summary_stats(all_results):
    import json
    from pathlib import Path

    import numpy as np
    from scipy import stats

    rows = []
    for r in all_results:
        if r is None:
            continue
        tissue = r["tissue"]
        ceiling_score = r.get("ceiling", {}).get("ceiling_score", 0.0)
        for emb in EMBEDDINGS:
            metrics_key = f"metrics_{emb}"
            mgr_key = f"mgr_{emb}"
            if metrics_key not in r or mgr_key not in r:
                continue
            m = r[metrics_key]
            mgr = r[mgr_key]
            if m.get("f1") is None:
                continue
            mgr_raw = mgr["mgr_score"]
            mgr_norm = mgr_raw / ceiling_score if ceiling_score > 0.01 else float("nan")
            row = {
                "tissue": tissue,
                "model": emb,
                "f1": m["f1"],
                "mgr_score": mgr_raw,
                "mgr_normalized": mgr_norm,
                "ceiling_score": ceiling_score,
            }
            for metric_name in [
                "rcs_baseline", "rcs_pca10", "rcs_trimmed", "rcs_normalized",
                "rcs_combined", "pad", "scc_logreg", "scc_knn", "scc_rf",
                "scc_svm", "mmd", "ccal",
            ]:
                row[metric_name] = m.get(metric_name)
            rows.append(row)

    n_tissues = len(set(r["tissue"] for r in rows))
    print(f"\n=== SUMMARY: {len(rows)} conditions across {n_tissues} tissues ===\n",
          flush=True)

    metric_names = [
        "scc_logreg", "scc_knn", "scc_rf", "scc_svm",
        "mmd", "ccal",
        "rcs_baseline", "rcs_pca10", "rcs_trimmed", "rcs_normalized",
        "rcs_combined", "pad",
    ]

    f1_vals = np.array([r["f1"] for r in rows])
    mgr_vals = np.array([r["mgr_score"] for r in rows])
    mgr_norm_vals = np.array([r["mgr_normalized"] for r in rows])
    tissues = np.array([r["tissue"] for r in rows])
    unique_tissues = np.unique(tissues)

    summary = {
        "n_tissues": n_tissues,
        "n_conditions": len(rows),
        "ground_truths": {},
        "metrics": {},
        "within_tissue_tau": {},
        "ceiling_stats": {},
    }

    ceiling_scores = [r["ceiling_score"] for r in rows]
    summary["ceiling_stats"] = {
        "mean": float(np.nanmean(ceiling_scores)),
        "median": float(np.nanmedian(ceiling_scores)),
        "min": float(np.nanmin(ceiling_scores)),
        "max": float(np.nanmax(ceiling_scores)),
    }
    print(f"  Ceiling MGR: mean={summary['ceiling_stats']['mean']:.3f}, "
          f"range=[{summary['ceiling_stats']['min']:.3f}, "
          f"{summary['ceiling_stats']['max']:.3f}]", flush=True)

    rho_f1_mgr, p_f1_mgr = stats.spearmanr(f1_vals, mgr_vals)
    valid_norm = np.isfinite(mgr_norm_vals)
    rho_f1_mgr_norm = float("nan")
    p_f1_mgr_norm = float("nan")
    if valid_norm.sum() >= 5:
        rho_f1_mgr_norm, p_f1_mgr_norm = stats.spearmanr(
            f1_vals[valid_norm], mgr_norm_vals[valid_norm])
    summary["ground_truths"] = {
        "f1_mgr_raw": {"spearman_rho": float(rho_f1_mgr), "p": float(p_f1_mgr)},
        "f1_mgr_normalized": {"spearman_rho": float(rho_f1_mgr_norm), "p": float(p_f1_mgr_norm)},
    }
    print(f"\n  PRECONDITION — F1 vs MGR: rho={rho_f1_mgr:+.3f} (raw), "
          f"rho={rho_f1_mgr_norm:+.3f} (normalized)", flush=True)
    print(f"  Window [0.25, 0.80]: ", end="", flush=True)
    if 0.25 <= rho_f1_mgr <= 0.80:
        print("PASS", flush=True)
    else:
        print(f"FAIL (rho={rho_f1_mgr:.3f})", flush=True)

    print(f"\n{'Metric':<20} {'rho(F1)':<10} {'rho(MGR)':<10} "
          f"{'rho(nMGR)':<10} {'CI_MGR (block)':<24}", flush=True)
    print("-" * 80, flush=True)

    rng = np.random.default_rng(SEED)

    for mname in metric_names:
        vals = np.array([r[mname] for r in rows])
        valid = np.isfinite(vals) & np.isfinite(f1_vals) & np.isfinite(mgr_vals)

        if valid.sum() < 5:
            continue

        v = vals[valid]
        f = f1_vals[valid]
        g = mgr_vals[valid]
        t = tissues[valid]
        ut = np.unique(t)

        rho_f1, _ = stats.spearmanr(v, f)
        rho_mgr, p_mgr = stats.spearmanr(v, g)

        valid_n = valid & np.isfinite(mgr_norm_vals)
        rho_mgr_norm = float("nan")
        if valid_n.sum() >= 5:
            rho_mgr_norm, _ = stats.spearmanr(vals[valid_n], mgr_norm_vals[valid_n])

        n_boot = 10000
        boot_mgr = np.zeros(n_boot)
        for b in range(n_boot):
            sampled = rng.choice(ut, len(ut), replace=True)
            idx = np.concatenate([np.where(t == ti)[0] for ti in sampled])
            if len(idx) < 5:
                boot_mgr[b] = float("nan")
                continue
            r_val, _ = stats.spearmanr(v[idx], g[idx])
            boot_mgr[b] = r_val

        boot_valid = boot_mgr[np.isfinite(boot_mgr)]
        if len(boot_valid) > 100:
            ci_mgr = (float(np.percentile(boot_valid, 2.5)),
                      float(np.percentile(boot_valid, 97.5)))
        else:
            ci_mgr = (float("nan"), float("nan"))

        summary["metrics"][mname] = {
            "rho_f1": float(rho_f1),
            "rho_mgr": float(rho_mgr),
            "p_mgr": float(p_mgr),
            "rho_mgr_normalized": float(rho_mgr_norm),
            "ci_mgr_block": ci_mgr,
        }

        ci_str = f"[{ci_mgr[0]:+.2f}, {ci_mgr[1]:+.2f}]"
        print(f"  {mname:<18} {rho_f1:+.3f}     {rho_mgr:+.3f}     "
              f"{rho_mgr_norm:+.3f}     {ci_str}", flush=True)

    print(f"\n--- Within-tissue Kendall tau (SCC-LR vs MGR) ---", flush=True)
    for mname in metric_names:
        positive_count = 0
        total_count = 0
        taus = []
        for tissue in unique_tissues:
            tmask = tissues == tissue
            if tmask.sum() < 3:
                continue
            m_vals = np.array([r[mname] for r in rows])[tmask]
            g_vals = mgr_vals[tmask]
            if not np.all(np.isfinite(m_vals)) or not np.all(np.isfinite(g_vals)):
                continue
            if np.all(m_vals == m_vals[0]) or np.all(g_vals == g_vals[0]):
                continue
            tau, _ = stats.kendalltau(m_vals, g_vals)
            if np.isfinite(tau):
                total_count += 1
                taus.append(tau)
                if tau > 0:
                    positive_count += 1

        frac = positive_count / total_count if total_count > 0 else 0.0
        mean_tau = float(np.mean(taus)) if taus else float("nan")
        summary["within_tissue_tau"][mname] = {
            "positive": positive_count,
            "total": total_count,
            "fraction_positive": frac,
            "mean_tau": mean_tau,
        }
        print(f"  {mname:<18} {positive_count}/{total_count} positive "
              f"({frac:.0%}), mean tau={mean_tau:+.3f}", flush=True)

    summary_dir = Path(f"{VOL_BASE}/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "exp17_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(summary_dir / "exp17_all_rows.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    vol.commit()
    print(f"\n=== Summary saved to {summary_dir} ===", flush=True)
    return summary


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def orchestrate():
    import json
    from pathlib import Path

    import cellxgene_census

    print("[orchestrate] Discovering tissues...", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs_df = cellxgene_census.get_obs(
            census, ORGANISM,
            value_filter=(
                "is_primary_data == True "
                "and disease == 'normal'"
            ),
            column_names=["tissue", "tissue_ontology_term_id", "assay_ontology_term_id"],
        )

    tissue_map = {}
    for _, row in obs_df.drop_duplicates(subset=["tissue", "tissue_ontology_term_id"]).iterrows():
        tid = row["tissue_ontology_term_id"]
        tname = row["tissue"]
        tissue_map[tid] = tname

    src_tissues = set(obs_df[obs_df["assay_ontology_term_id"] == SOURCE_ASSAY]["tissue_ontology_term_id"])
    tgt_tissues = set(obs_df[obs_df["assay_ontology_term_id"] == TARGET_ASSAY]["tissue_ontology_term_id"])
    both = sorted(src_tissues & tgt_tissues)

    print(f"[orchestrate] Found {len(both)} tissues:", flush=True)
    for i, tid in enumerate(both):
        print(f"  [{i:2d}] {tissue_map[tid]}", flush=True)

    print(f"\n[orchestrate] Launching {len(both)} parallel containers...", flush=True)
    handles = []
    for i, tid in enumerate(both):
        h = run_tissue.spawn(i, tissue_map[tid], tid)
        handles.append(h)

    all_results = [h.get() for h in handles]

    n_ok = sum(1 for r in all_results if r is not None)
    n_skip = sum(1 for r in all_results if r is None)

    valid_results = [r for r in all_results if r is not None]

    if valid_results:
        print("[orchestrate] Computing summary statistics...", flush=True)
        compute_summary_stats.remote(valid_results)

    merged_dir = Path(f"{VOL_BASE}/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_dir / "exp17_all_tissues.json", "w") as f:
        json.dump(valid_results, f, indent=2, default=str)
    vol.commit()

    print(f"\n=== DONE === {n_ok} tissues completed, {n_skip} skipped", flush=True)


@app.local_entrypoint()
def main():
    orchestrate.remote()
