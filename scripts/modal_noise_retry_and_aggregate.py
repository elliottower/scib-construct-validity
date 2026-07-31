"""Retry brain_bog_pca_512 and aggregate all 24 noise results.

Usage:
    modal run --detach scripts/modal_noise_retry_and_aggregate.py
"""
import modal

app = modal.App("preflight-noise-retry")

vol = modal.Volume.from_name("preflight-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "anndata>=0.10.0",
        "cellxgene-census==1.16.2",
        "tqdm>=4.60.0",
        "pandas>=2.1.0",
        "scanpy>=1.10.0",
        "scib-metrics>=0.4.0",
        "pynndescent>=0.5.0",
        "jax[cpu]>=0.4.0,<0.5.0",
        "matplotlib>=3.7.0",
        "requests>=2.28.0",
    )
)

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713

SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}

SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"

REAL_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]

EMBEDDINGS_ORDER = [
    "geneformer", "scvi", "scgpt",
    "random_projection", "untrained_encoder", "bog_pca_512",
]

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

BIO_METRICS = [
    "nmi_leiden", "ari_leiden", "silhouette_label",
    "clisi", "isolated_label_asw",
]
BATCH_METRICS = [
    "silhouette_batch", "ilisi", "kbet",
    "graph_connectivity", "pcr_comparison",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=16384)
def retry_brain_bog():
    """Retry brain_bog_pca_512 — the one that S3-timed-out."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import anndata as ad
    import cellxgene_census
    import numpy as np
    import pandas as pd
    from scipy import sparse as sp
    from sklearn.decomposition import PCA

    def _ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    tissue_name = "brain"
    tissue_id = "UBERON:0000955"
    emb_name = "bog_pca_512"
    cond_key = f"{tissue_name}_{emb_name}"

    print(f"{_ts()} Retrying: {cond_key}")

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        src_filter = (
            f"tissue_ontology_term_id == '{tissue_id}' "
            f"and is_primary_data == True "
            f"and assay_ontology_term_id == '{SOURCE_ASSAY}'"
        )
        tgt_filter = (
            f"tissue_ontology_term_id == '{tissue_id}' "
            f"and is_primary_data == True "
            f"and assay_ontology_term_id == '{TARGET_ASSAY}'"
        )

        print(f"{_ts()} Pulling source cells...")
        obs_src = cellxgene_census.get_obs(
            census, ORGANISM, value_filter=src_filter,
            column_names=["soma_joinid", "donor_id"],
        )
        src_ids = obs_src["soma_joinid"].values
        if len(src_ids) > MAX_CELLS:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(src_ids), size=MAX_CELLS, replace=False)
            idx.sort()
            src_ids = src_ids[idx]

        src_adata = cellxgene_census.get_anndata(
            census, organism=ORGANISM,
            obs_value_filter=src_filter,
            obs_coords=src_ids,
            obs_column_names=OBS_COLUMNS,
        )
        print(f"  Source: {src_adata.n_obs} cells")

        print(f"{_ts()} Pulling target cells...")
        obs_tgt = cellxgene_census.get_obs(
            census, ORGANISM, value_filter=tgt_filter,
            column_names=["soma_joinid", "donor_id"],
        )
        tgt_ids = obs_tgt["soma_joinid"].values
        if len(tgt_ids) > MAX_CELLS:
            rng = np.random.default_rng(SEED + 1)
            idx = rng.choice(len(tgt_ids), size=MAX_CELLS, replace=False)
            idx.sort()
            tgt_ids = tgt_ids[idx]

        tgt_adata = cellxgene_census.get_anndata(
            census, organism=ORGANISM,
            obs_value_filter=tgt_filter,
            obs_coords=tgt_ids,
            obs_column_names=OBS_COLUMNS,
        )
        print(f"  Target: {tgt_adata.n_obs} cells")

    X_src_raw = src_adata.X.toarray() if sp.issparse(src_adata.X) else np.array(src_adata.X)
    X_tgt_raw = tgt_adata.X.toarray() if sp.issparse(tgt_adata.X) else np.array(tgt_adata.X)

    X_combined = np.log1p(np.vstack([X_src_raw, X_tgt_raw]).astype(np.float64))
    n_src = X_src_raw.shape[0]
    n_components = min(512, X_combined.shape[0], X_combined.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_combined).astype(np.float32)
    X_clean = X_pca

    n_src = src_adata.n_obs
    n_tgt = tgt_adata.n_obs
    labels_src = src_adata.obs["cell_type"].values
    labels_tgt = tgt_adata.obs["cell_type"].values
    donors_src = src_adata.obs["donor_id"].values
    donors_tgt = tgt_adata.obs["donor_id"].values

    X_raw_combined = sp.csr_matrix(np.vstack([
        X_src_raw[:n_src], X_tgt_raw[:n_tgt],
    ])) if X_src_raw.shape[1] == X_tgt_raw.shape[1] else sp.csr_matrix(
        np.zeros((n_src + n_tgt, 1))
    )
    obs_df = pd.DataFrame({
        "cell_type": np.concatenate([labels_src[:n_src], labels_tgt[:n_tgt]]),
        "batch": np.concatenate([np.full(n_src, "source"), np.full(n_tgt, "target")]),
        "donor_id": np.concatenate([donors_src[:n_src], donors_tgt[:n_tgt]]),
    })

    emb_std = X_clean.std(axis=0)
    emb_std = np.where(emb_std < 1e-8, 1e-8, emb_std)

    tissue_idx = 3  # brain
    emb_idx = 5     # bog_pca_512

    from scib_metrics import (
        nmi_ari_cluster_labels_leiden,
        silhouette_label,
        silhouette_batch,
        isolated_labels,
        pcr_comparison,
        graph_connectivity,
        clisi_knn,
        ilisi_knn,
        kbet_per_label,
    )
    from scib_metrics.nearest_neighbors import pynndescent

    def compute_scib_metrics(adata_in):
        results = {}
        X_emb = adata_in.obsm["X_emb"]
        labels = adata_in.obs["cell_type"].values
        batch = adata_in.obs["batch"].values

        nn_15 = pynndescent(X_emb, n_neighbors=15)
        nn_50 = pynndescent(X_emb, n_neighbors=50)
        nn_90 = pynndescent(X_emb, n_neighbors=90)

        try:
            nmi_ari = nmi_ari_cluster_labels_leiden(nn_15, labels, optimize_resolution=True)
            results["nmi_leiden"] = float(nmi_ari["nmi"])
            results["ari_leiden"] = float(nmi_ari["ari"])
        except Exception as e:
            print(f"    WARNING: NMI/ARI failed: {e}")
            results["nmi_leiden"] = None
            results["ari_leiden"] = None

        try:
            results["silhouette_label"] = float(silhouette_label(X_emb, labels))
        except Exception:
            results["silhouette_label"] = None
        try:
            results["silhouette_batch"] = float(silhouette_batch(X_emb, labels, batch))
        except Exception:
            results["silhouette_batch"] = None
        try:
            results["isolated_label_asw"] = float(isolated_labels(X_emb, labels, batch))
        except Exception:
            results["isolated_label_asw"] = None
        try:
            results["graph_connectivity"] = float(graph_connectivity(nn_15, labels))
        except Exception:
            results["graph_connectivity"] = None
        try:
            X_pre = adata_in.X if not sp.issparse(adata_in.X) else adata_in.X.toarray()
            results["pcr_comparison"] = float(pcr_comparison(X_pre, X_emb, batch, categorical=True))
        except Exception:
            results["pcr_comparison"] = None
        try:
            results["clisi"] = float(np.nanmean(clisi_knn(nn_90, labels)))
        except Exception:
            results["clisi"] = None
        try:
            results["ilisi"] = float(np.nanmean(ilisi_knn(nn_90, batch)))
        except Exception:
            results["ilisi"] = None
        try:
            kbet_scores = kbet_per_label(nn_50, batch, labels)
            results["kbet"] = float(np.nanmean([v for v in kbet_scores.values()]))
        except Exception:
            results["kbet"] = None
        return results

    scores_by_sigma = {}
    for sigma_idx, sigma in enumerate(SIGMAS):
        noise_seed = SEED + tissue_idx * 10000 + emb_idx * 1000 + sigma_idx
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0, 1, size=X_clean.shape).astype(np.float32)
        noise *= (sigma * emb_std)[np.newaxis, :]
        X_noisy = X_clean + noise

        adata = ad.AnnData(X=X_raw_combined, obs=obs_df.copy())
        adata.obsm["X_emb"] = X_noisy

        print(f"  {_ts()} sigma={sigma} computing metrics...")
        scores = compute_scib_metrics(adata)
        scores_by_sigma[sigma] = scores
        print(f"  {_ts()} sigma={sigma} done")

    result = {
        "condition": cond_key,
        "tissue": tissue_name,
        "embedding": emb_name,
        "emb_std_mean": float(emb_std.mean()),
        "scores_by_sigma": {str(s): sc for s, sc in scores_by_sigma.items()},
    }

    out_dir = Path("/vol/exp10_scib_audit/noise_parallel")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cond_key}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()
    print(f"{_ts()} {cond_key} saved")
    return result


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def aggregate_noise_results():
    """Collect all 24 per-condition JSONs and produce the final analysis."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    def _ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    vol.reload()
    noise_dir = Path("/vol/exp10_scib_audit/noise_parallel")

    all_results = {}
    for p in sorted(noise_dir.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        all_results[data["condition"]] = data

    print(f"{_ts()} Loaded {len(all_results)} conditions")

    SIGMAS_STR = ["0.01", "0.05", "0.1", "0.25", "0.5", "1.0", "2.0"]

    def check_monotonicity(scores_by_sigma):
        vals = [scores_by_sigma.get(s) for s in SIGMAS_STR]
        if any(v is None for v in vals):
            return None, "missing_values"
        non_decreasing = all(v1 <= v2 + 1e-10 for v1, v2 in zip(vals, vals[1:]))
        non_increasing = all(v1 >= v2 - 1e-10 for v1, v2 in zip(vals, vals[1:]))
        if non_decreasing or non_increasing:
            direction = "non_decreasing" if non_decreasing else "non_increasing"
            return True, direction
        return False, "non_monotonic"

    monotonicity = {}
    for metric in BIO_METRICS + BATCH_METRICS:
        non_mono_conditions = []
        mono_conditions = []
        missing = []

        for cond_key, cond_data in all_results.items():
            metric_by_sigma = {}
            for s in SIGMAS_STR:
                metric_by_sigma[s] = cond_data["scores_by_sigma"].get(s, {}).get(metric)

            is_mono, direction = check_monotonicity(metric_by_sigma)
            if is_mono is None:
                missing.append(cond_key)
            elif is_mono:
                mono_conditions.append({"condition": cond_key, "direction": direction})
            else:
                vals = [metric_by_sigma[s] for s in SIGMAS_STR]
                non_mono_conditions.append({
                    "condition": cond_key,
                    "values": [round(v, 6) if v is not None else None for v in vals],
                })

        n_non_mono = len(non_mono_conditions)
        verdict = "FAIL" if n_non_mono > 1 else "PASS"

        monotonicity[metric] = {
            "verdict": verdict,
            "n_non_monotonic": n_non_mono,
            "n_monotonic": len(mono_conditions),
            "n_missing": len(missing),
            "non_monotonic_conditions": non_mono_conditions,
            "monotonic_directions": {c["condition"]: c["direction"] for c in mono_conditions},
        }
        print(f"  {metric}: {verdict} ({n_non_mono}/{n_non_mono + len(mono_conditions)} non-monotonic)")

    prereg_predictions = {
        "nmi_leiden": "FAIL",
        "ari_leiden": "FAIL",
        "silhouette_label": "PASS",
        "clisi": "PASS",
        "isolated_label_asw": "PASS",
        "silhouette_batch": "PASS",
        "ilisi": "PASS",
        "kbet": "FAIL",
        "graph_connectivity": "PASS",
        "pcr_comparison": "PASS",
    }

    print(f"\n{'Metric':<25} {'Predicted':<10} {'Actual':<10} {'Match'}")
    print("-" * 55)
    for metric in BIO_METRICS + BATCH_METRICS:
        pred = prereg_predictions[metric]
        actual = monotonicity[metric]["verdict"]
        match = "YES" if pred == actual else "NO"
        print(f"{metric:<25} {pred:<10} {actual:<10} {match}")

    n_correct = sum(
        1 for m in BIO_METRICS + BATCH_METRICS
        if prereg_predictions[m] == monotonicity[m]["verdict"]
    )

    output = {
        "test": "test4_noise_dose_response",
        "status": "CONFIRMATORY",
        "prereg": "PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md",
        "freeze": "docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "census_version": CENSUS_VERSION,
        "seed": SEED,
        "sigmas": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
        "n_conditions": len(all_results),
        "monotonicity_verdicts": monotonicity,
        "prereg_predictions": prereg_predictions,
        "prediction_accuracy": f"{n_correct}/{len(BIO_METRICS + BATCH_METRICS)}",
        "raw_results": all_results,
    }

    out_path = Path("/vol/exp10_scib_audit/noise_dose_response.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    vol.commit()
    print(f"\n{_ts()} Final results saved to {out_path}")
    print(f"Prediction accuracy: {n_correct}/{len(BIO_METRICS + BATCH_METRICS)}")

    return output


@app.local_entrypoint()
def main():
    from datetime import datetime, timezone

    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} Retrying brain_bog_pca_512...")
    result = retry_brain_bog.remote()
    if "error" in result:
        print(f"  FAILED: {result['error']}")
        return
    print(f"  DONE: brain_bog_pca_512")

    print("Aggregating all 24 conditions...")
    final = aggregate_noise_results.remote()
    print(f"Prediction accuracy: {final.get('prediction_accuracy', '?')}")
    print("Done.")
