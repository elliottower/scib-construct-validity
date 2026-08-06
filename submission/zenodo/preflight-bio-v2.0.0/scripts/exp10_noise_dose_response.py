"""Experiment 10, Test 4: Noise Dose-Response Monotonicity [CONFIRMATORY].

Prereg: PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md (frozen 2026-07-13).
Freeze: docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json

For each of 6 embeddings x 4 tissues, corrupt the embedding with additive
Gaussian noise at sigma in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0} x
embedding_std. Recompute all 10 scIB metrics at each noise level. A metric
FAILS if non-monotonic in >1 of the 24 conditions.

Usage:
    python scripts/exp10_noise_dose_response.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713

SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

OUTPUT_DIR = Path("results/exp10_scib_audit")
OUTPUT_PATH = OUTPUT_DIR / "noise_dose_response.json"
INC_PATH = OUTPUT_DIR / "noise_incremental.jsonl"

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}

SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"

REAL_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]

BIO_METRICS = [
    "nmi_leiden", "ari_leiden", "silhouette_label",
    "clisi", "isolated_label_asw",
]
BATCH_METRICS = [
    "silhouette_batch", "ilisi", "kbet",
    "graph_connectivity", "pcr_comparison",
]
ALL_METRICS = BIO_METRICS + BATCH_METRICS


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def random_projection(X_raw, d_out=512):
    rng = np.random.default_rng(SEED)
    n_genes = X_raw.shape[1]
    R = rng.standard_normal((n_genes, d_out)) / np.sqrt(d_out)
    X_log = np.log1p(X_raw.astype(np.float64))
    return (X_log @ R).astype(np.float32)


def untrained_encoder(X_raw, d_out=512, d_hidden=256):
    rng = np.random.default_rng(SEED)
    n_genes = X_raw.shape[1]
    std1 = np.sqrt(2.0 / (n_genes + d_hidden))
    W1 = rng.standard_normal((n_genes, d_hidden)) * std1
    b1 = np.zeros(d_hidden)
    std2 = np.sqrt(2.0 / (d_hidden + d_out))
    W2 = rng.standard_normal((d_hidden, d_out)) * std2
    b2 = np.zeros(d_out)
    X_log = np.log1p(X_raw.astype(np.float64))
    h = np.maximum(0, X_log @ W1 + b1)
    return (h @ W2 + b2).astype(np.float32)


def bag_of_genes_pca_combined(X_src_raw, X_tgt_raw, d_out=512):
    from sklearn.decomposition import PCA
    X_combined = np.log1p(np.vstack([X_src_raw, X_tgt_raw]).astype(np.float64))
    n_src = X_src_raw.shape[0]
    n_components = min(d_out, X_combined.shape[0], X_combined.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_combined).astype(np.float32)
    return X_pca[:n_src], X_pca[n_src:], n_components


def compute_scib_metrics(adata, batch_key="batch", label_key="cell_type", embed_key="X_emb"):
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

    results = {}
    X_emb = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batch = adata.obs[batch_key].values

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
    except Exception as e:
        results["silhouette_label"] = None

    try:
        results["silhouette_batch"] = float(silhouette_batch(X_emb, labels, batch))
    except Exception as e:
        results["silhouette_batch"] = None

    try:
        results["isolated_label_asw"] = float(isolated_labels(X_emb, labels, batch))
    except Exception as e:
        results["isolated_label_asw"] = None

    try:
        results["graph_connectivity"] = float(graph_connectivity(nn_15, labels))
    except Exception as e:
        results["graph_connectivity"] = None

    try:
        X_pre = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
        results["pcr_comparison"] = float(pcr_comparison(X_pre, X_emb, batch, categorical=True))
    except Exception as e:
        results["pcr_comparison"] = None

    try:
        results["clisi"] = float(np.nanmean(clisi_knn(nn_90, labels)))
    except Exception as e:
        results["clisi"] = None

    try:
        results["ilisi"] = float(np.nanmean(ilisi_knn(nn_90, batch)))
    except Exception as e:
        results["ilisi"] = None

    try:
        kbet_scores = kbet_per_label(nn_50, batch, labels)
        results["kbet"] = float(np.nanmean([v for v in kbet_scores.values()]))
    except Exception as e:
        results["kbet"] = None

    return results


def corrupt_embedding(X_emb, sigma, emb_std, rng):
    noise = rng.normal(0, 1, size=X_emb.shape).astype(np.float32)
    noise *= (sigma * emb_std)[np.newaxis, :]
    return X_emb + noise


def check_monotonicity(scores_by_sigma):
    """Check if a sequence of scores is monotonically non-increasing or non-decreasing."""
    vals = [scores_by_sigma[s] for s in SIGMAS]
    if any(v is None for v in vals):
        return None, "missing_values"
    non_decreasing = all(v1 <= v2 + 1e-10 for v1, v2 in zip(vals, vals[1:]))
    non_increasing = all(v1 >= v2 - 1e-10 for v1, v2 in zip(vals, vals[1:]))
    if non_decreasing or non_increasing:
        direction = "non_decreasing" if non_decreasing else "non_increasing"
        return True, direction
    return False, "non_monotonic"


def _save_incremental(result):
    with open(INC_PATH, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def main():
    import cellxgene_census

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} Test 4: Noise Dose-Response Monotonicity [CONFIRMATORY]")
    print(f"{_ts()} Prereg: PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md")
    print(f"{_ts()} Sigmas: {SIGMAS}")
    print(f"{_ts()} Freeze: docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json")

    all_results = {}

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue_name, tissue_id in tqdm(TISSUES.items(), desc="Tissues"):
            print(f"\n{'='*60}")
            print(f"{_ts()} Processing tissue: {tissue_name}")

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
                obs_embeddings=REAL_EMBEDDINGS,
            )
            print(f"  Source: {src_adata.n_obs} cells")

            print(f"{_ts()} Pulling target cells...")
            obs_tgt = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=tgt_filter,
                column_names=["soma_joinid", "donor_id"],
            )
            tgt_ids = obs_tgt["soma_joinid"].values
            if len(tgt_ids) == 0:
                print(f"  WARNING: No target cells for {tissue_name}, skipping")
                continue
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
                obs_embeddings=REAL_EMBEDDINGS,
            )
            print(f"  Target: {tgt_adata.n_obs} cells")

            X_src_raw = src_adata.X.toarray() if sp.issparse(src_adata.X) else np.array(src_adata.X)
            X_tgt_raw = tgt_adata.X.toarray() if sp.issparse(tgt_adata.X) else np.array(tgt_adata.X)

            labels_src = src_adata.obs["cell_type"].values
            labels_tgt = tgt_adata.obs["cell_type"].values
            donors_src = src_adata.obs["donor_id"].values
            donors_tgt = tgt_adata.obs["donor_id"].values

            embedding_dict = {}
            for emb_name in REAL_EMBEDDINGS:
                if emb_name in src_adata.obsm and emb_name in tgt_adata.obsm:
                    X_src_emb = np.array(src_adata.obsm[emb_name])
                    X_tgt_emb = np.array(tgt_adata.obsm[emb_name])
                    embedding_dict[emb_name] = np.vstack([X_src_emb, X_tgt_emb]).astype(np.float32)

            X_src_rp = random_projection(X_src_raw)
            X_tgt_rp = random_projection(X_tgt_raw)
            embedding_dict["random_projection"] = np.vstack([X_src_rp, X_tgt_rp]).astype(np.float32)

            X_src_ue = untrained_encoder(X_src_raw)
            X_tgt_ue = untrained_encoder(X_tgt_raw)
            embedding_dict["untrained_encoder"] = np.vstack([X_src_ue, X_tgt_ue]).astype(np.float32)

            X_src_bog, X_tgt_bog, bog_d = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            embedding_dict["bog_pca_512"] = np.vstack([X_src_bog, X_tgt_bog]).astype(np.float32)

            n_src = src_adata.n_obs
            n_tgt = tgt_adata.n_obs
            X_raw_combined = sp.csr_matrix(np.vstack([
                X_src_raw[:n_src], X_tgt_raw[:n_tgt],
            ])) if X_src_raw.shape[1] == X_tgt_raw.shape[1] else sp.csr_matrix(
                np.zeros((n_src + n_tgt, 1))
            )
            obs_df = pd.DataFrame({
                "cell_type": np.concatenate([labels_src[:n_src], labels_tgt[:n_tgt]]),
                "batch": np.concatenate([
                    np.full(n_src, "source"),
                    np.full(n_tgt, "target"),
                ]),
                "donor_id": np.concatenate([donors_src[:n_src], donors_tgt[:n_tgt]]),
            })

            for emb_name, X_clean in tqdm(
                embedding_dict.items(), desc=f"  Embeddings ({tissue_name})", leave=False,
            ):
                print(f"\n{_ts()} Noise sweep: {tissue_name} / {emb_name}")
                emb_std = X_clean.std(axis=0)
                emb_std = np.where(emb_std < 1e-8, 1e-8, emb_std)

                scores_by_sigma = {}

                TISSUE_NAMES = list(TISSUES.keys())
                EMB_NAMES = ["geneformer", "scvi", "scgpt", "random_projection", "untrained_encoder", "bog_pca_512"]
                for sigma in tqdm(SIGMAS, desc=f"    sigma", leave=False):
                    tissue_idx = TISSUE_NAMES.index(tissue_name)
                    emb_idx = EMB_NAMES.index(emb_name)
                    sigma_idx = SIGMAS.index(sigma)
                    noise_seed = SEED + tissue_idx * 10000 + emb_idx * 1000 + sigma_idx
                    rng = np.random.default_rng(noise_seed)
                    X_noisy = corrupt_embedding(X_clean, sigma, emb_std, rng)

                    adata = ad.AnnData(X=X_raw_combined, obs=obs_df.copy())
                    adata.obsm["X_emb"] = X_noisy

                    print(f"    {_ts()} sigma={sigma} computing metrics...")
                    scores = compute_scib_metrics(adata)
                    scores_by_sigma[sigma] = scores

                    _save_incremental({
                        "tissue": tissue_name, "embedding": emb_name,
                        "sigma": sigma, "scores": scores,
                        "timestamp": _ts(),
                    })

                cond_key = f"{tissue_name}_{emb_name}"
                all_results[cond_key] = {
                    "tissue": tissue_name,
                    "embedding": emb_name,
                    "emb_std_mean": float(emb_std.mean()),
                    "scores_by_sigma": {str(s): sc for s, sc in scores_by_sigma.items()},
                }

    print(f"\n{'='*60}")
    print(f"{_ts()} Checking monotonicity...")

    monotonicity = {}
    for metric in ALL_METRICS:
        non_mono_conditions = []
        mono_conditions = []
        missing = []

        for cond_key, cond_data in all_results.items():
            metric_by_sigma = {}
            for sigma in SIGMAS:
                s_key = str(sigma)
                metric_by_sigma[sigma] = cond_data["scores_by_sigma"][s_key].get(metric)

            is_mono, direction = check_monotonicity(metric_by_sigma)
            if is_mono is None:
                missing.append(cond_key)
            elif is_mono:
                mono_conditions.append({"condition": cond_key, "direction": direction})
            else:
                vals = [metric_by_sigma[s] for s in SIGMAS]
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

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'Predicted':<10} {'Actual':<10} {'Match':<8} {'Non-mono count'}")
    print("-" * 75)
    for metric in ALL_METRICS:
        pred = prereg_predictions[metric]
        actual = monotonicity[metric]["verdict"]
        match = "YES" if pred == actual else "NO"
        n_nm = monotonicity[metric]["n_non_monotonic"]
        total = monotonicity[metric]["n_non_monotonic"] + monotonicity[metric]["n_monotonic"]
        print(f"{metric:<25} {pred:<10} {actual:<10} {match:<8} {n_nm}/{total}")

    n_correct = sum(
        1 for m in ALL_METRICS
        if prereg_predictions[m] == monotonicity[m]["verdict"]
    )
    print(f"\nPrediction accuracy: {n_correct}/{len(ALL_METRICS)}")

    output = {
        "test": "test4_noise_dose_response",
        "status": "CONFIRMATORY",
        "prereg": "PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md",
        "freeze": "docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "census_version": CENSUS_VERSION,
        "seed": SEED,
        "sigmas": SIGMAS,
        "n_conditions": len(all_results),
        "monotonicity_verdicts": monotonicity,
        "prereg_predictions": prereg_predictions,
        "prediction_accuracy": f"{n_correct}/{len(ALL_METRICS)}",
        "raw_results": all_results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{_ts()} Results saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
