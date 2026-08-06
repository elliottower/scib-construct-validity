"""Experiment 10, Test 5: Hyperparameter Sensitivity of Rankings [CONFIRMATORY].

Prereg: PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md (frozen 2026-07-13).
Freeze: docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json

Re-run clustering-based metrics (NMI, ARI) at Leiden resolutions {0.5, 1.0, 2.0}.
Re-run kNN-based metrics (cLISI, iLISI, kBET, graph_connectivity) at k in {15, 30, 90}.
For each metric, compute Kendall's tau between rankings of 6 embeddings at
different hyperparameter values within each tissue. Report per-tissue and
aggregate tau.

Usage:
    python scripts/exp10_hyperparameter_sensitivity.py
"""
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy import stats
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713

LEIDEN_RESOLUTIONS = [0.5, 1.0, 2.0]
K_VALUES = [15, 30, 90]

OUTPUT_DIR = Path("results/exp10_scib_audit")
OUTPUT_PATH = OUTPUT_DIR / "hyperparameter_sensitivity.json"
INC_PATH = OUTPUT_DIR / "hyperparam_incremental.jsonl"

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

EMBEDDINGS_ORDER = [
    "geneformer", "scvi", "scgpt",
    "random_projection", "untrained_encoder", "bog_pca_512",
]

LEIDEN_METRICS = ["nmi_leiden", "ari_leiden"]
KNN_METRICS = ["clisi", "ilisi", "kbet", "graph_connectivity"]


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


def compute_leiden_at_resolution(nn_15, labels, resolution):
    """Compute NMI/ARI using Leiden at a fixed resolution."""
    from scib_metrics import nmi_ari_cluster_labels_leiden
    try:
        result = nmi_ari_cluster_labels_leiden(
            nn_15, labels, optimize_resolution=False, resolution=resolution,
        )
        return {
            "nmi_leiden": float(result["nmi"]),
            "ari_leiden": float(result["ari"]),
        }
    except Exception as e:
        print(f"    WARNING: Leiden at resolution={resolution} failed: {e}")
        return {"nmi_leiden": None, "ari_leiden": None}


def compute_knn_metrics_at_k(X_emb, labels, batch, k):
    """Compute cLISI, iLISI, kBET, graph_connectivity at a specific k."""
    from scib_metrics import clisi_knn, ilisi_knn, kbet_per_label, graph_connectivity
    from scib_metrics.nearest_neighbors import pynndescent

    nn = pynndescent(X_emb, n_neighbors=k)
    results = {}

    try:
        results["clisi"] = float(np.nanmean(clisi_knn(nn, labels)))
    except Exception as e:
        print(f"    WARNING: cLISI at k={k} failed: {e}")
        results["clisi"] = None

    try:
        results["ilisi"] = float(np.nanmean(ilisi_knn(nn, batch)))
    except Exception as e:
        results["ilisi"] = None

    try:
        kbet_scores = kbet_per_label(nn, batch, labels)
        results["kbet"] = float(np.nanmean([v for v in kbet_scores.values()]))
    except Exception as e:
        results["kbet"] = None

    try:
        results["graph_connectivity"] = float(graph_connectivity(nn, labels))
    except Exception as e:
        results["graph_connectivity"] = None

    return results


def compute_kendall_tau(ranking_a, ranking_b):
    """Kendall's tau between two rankings, handling ties."""
    mask = ~(np.isnan(ranking_a) | np.isnan(ranking_b))
    if mask.sum() < 3:
        return None
    tau, p = stats.kendalltau(ranking_a[mask], ranking_b[mask])
    return {"tau": round(float(tau), 3), "p": round(float(p), 4)}


def _save_incremental(result):
    with open(INC_PATH, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def main():
    import cellxgene_census
    from scib_metrics.nearest_neighbors import pynndescent

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} Test 5: Hyperparameter Sensitivity of Rankings [CONFIRMATORY]")
    print(f"{_ts()} Prereg: PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md")
    print(f"{_ts()} Leiden resolutions: {LEIDEN_RESOLUTIONS}")
    print(f"{_ts()} kNN k values: {K_VALUES}")

    leiden_scores = {}
    knn_scores = {}

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

            n_src = src_adata.n_obs
            n_tgt = tgt_adata.n_obs

            embedding_dict = {}
            for emb_name in REAL_EMBEDDINGS:
                if emb_name in src_adata.obsm and emb_name in tgt_adata.obsm:
                    X_s = np.array(src_adata.obsm[emb_name])
                    X_t = np.array(tgt_adata.obsm[emb_name])
                    embedding_dict[emb_name] = np.vstack([X_s, X_t]).astype(np.float32)

            X_src_rp = random_projection(X_src_raw)
            X_tgt_rp = random_projection(X_tgt_raw)
            embedding_dict["random_projection"] = np.vstack([X_src_rp, X_tgt_rp]).astype(np.float32)

            X_src_ue = untrained_encoder(X_src_raw)
            X_tgt_ue = untrained_encoder(X_tgt_raw)
            embedding_dict["untrained_encoder"] = np.vstack([X_src_ue, X_tgt_ue]).astype(np.float32)

            X_src_bog, X_tgt_bog, bog_d = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            embedding_dict["bog_pca_512"] = np.vstack([X_src_bog, X_tgt_bog]).astype(np.float32)

            labels = np.concatenate([labels_src[:n_src], labels_tgt[:n_tgt]])
            batch = np.concatenate([np.full(n_src, "source"), np.full(n_tgt, "target")])

            for emb_name in EMBEDDINGS_ORDER:
                if emb_name not in embedding_dict:
                    continue
                X_emb = embedding_dict[emb_name]
                cond_key = f"{tissue_name}_{emb_name}"

                print(f"\n{_ts()} Leiden sweep: {cond_key}")
                nn_15 = pynndescent(X_emb, n_neighbors=15)

                for resolution in LEIDEN_RESOLUTIONS:
                    print(f"    {_ts()} resolution={resolution}")
                    res = compute_leiden_at_resolution(nn_15, labels, resolution)
                    leiden_scores.setdefault(cond_key, {})[resolution] = res
                    _save_incremental({
                        "test": "leiden", "condition": cond_key,
                        "resolution": resolution, "scores": res, "timestamp": _ts(),
                    })

                print(f"{_ts()} kNN sweep: {cond_key}")
                for k in K_VALUES:
                    print(f"    {_ts()} k={k}")
                    res = compute_knn_metrics_at_k(X_emb, labels, batch, k)
                    knn_scores.setdefault(cond_key, {})[k] = res
                    _save_incremental({
                        "test": "knn", "condition": cond_key,
                        "k": k, "scores": res, "timestamp": _ts(),
                    })

    print(f"\n{'='*60}")
    print(f"{_ts()} Computing ranking stability (Kendall's tau)...")

    leiden_stability = {}
    for metric in LEIDEN_METRICS:
        resolution_pairs = list(combinations(LEIDEN_RESOLUTIONS, 2))
        per_tissue = {}

        for tissue in TISSUES:
            for res_a, res_b in resolution_pairs:
                scores_a = []
                scores_b = []
                for emb in EMBEDDINGS_ORDER:
                    cond = f"{tissue}_{emb}"
                    va = leiden_scores.get(cond, {}).get(res_a, {}).get(metric)
                    vb = leiden_scores.get(cond, {}).get(res_b, {}).get(metric)
                    scores_a.append(va if va is not None else float("nan"))
                    scores_b.append(vb if vb is not None else float("nan"))

                tau_result = compute_kendall_tau(np.array(scores_a), np.array(scores_b))
                per_tissue.setdefault(tissue, {})[f"{res_a}_vs_{res_b}"] = tau_result

        extreme_taus = []
        for tissue in TISSUES:
            key = f"{LEIDEN_RESOLUTIONS[0]}_vs_{LEIDEN_RESOLUTIONS[-1]}"
            tau_val = per_tissue.get(tissue, {}).get(key, {})
            if tau_val and tau_val.get("tau") is not None:
                extreme_taus.append(tau_val["tau"])

        winner_changes = 0
        for tissue in TISSUES:
            ranks_at_low = []
            ranks_at_high = []
            for emb in EMBEDDINGS_ORDER:
                cond = f"{tissue}_{emb}"
                vl = leiden_scores.get(cond, {}).get(LEIDEN_RESOLUTIONS[0], {}).get(metric)
                vh = leiden_scores.get(cond, {}).get(LEIDEN_RESOLUTIONS[-1], {}).get(metric)
                ranks_at_low.append(vl if vl is not None else float("-inf"))
                ranks_at_high.append(vh if vh is not None else float("-inf"))
            winner_low = EMBEDDINGS_ORDER[np.argmax(ranks_at_low)]
            winner_high = EMBEDDINGS_ORDER[np.argmax(ranks_at_high)]
            if winner_low != winner_high:
                winner_changes += 1

        leiden_stability[metric] = {
            "per_tissue": per_tissue,
            "mean_extreme_tau": round(float(np.mean(extreme_taus)), 3) if extreme_taus else None,
            "extreme_taus": [round(t, 3) for t in extreme_taus],
            "winner_changes": f"{winner_changes}/{len(TISSUES)} tissues",
        }

    knn_stability = {}
    for metric in KNN_METRICS:
        k_pairs = list(combinations(K_VALUES, 2))
        per_tissue = {}

        for tissue in TISSUES:
            for ka, kb in k_pairs:
                scores_a = []
                scores_b = []
                for emb in EMBEDDINGS_ORDER:
                    cond = f"{tissue}_{emb}"
                    va = knn_scores.get(cond, {}).get(ka, {}).get(metric)
                    vb = knn_scores.get(cond, {}).get(kb, {}).get(metric)
                    scores_a.append(va if va is not None else float("nan"))
                    scores_b.append(vb if vb is not None else float("nan"))

                tau_result = compute_kendall_tau(np.array(scores_a), np.array(scores_b))
                per_tissue.setdefault(tissue, {})[f"{ka}_vs_{kb}"] = tau_result

        extreme_taus = []
        for tissue in TISSUES:
            key = f"{K_VALUES[0]}_vs_{K_VALUES[-1]}"
            tau_val = per_tissue.get(tissue, {}).get(key, {})
            if tau_val and tau_val.get("tau") is not None:
                extreme_taus.append(tau_val["tau"])

        winner_changes = 0
        for tissue in TISSUES:
            ranks_at_low = []
            ranks_at_high = []
            for emb in EMBEDDINGS_ORDER:
                cond = f"{tissue}_{emb}"
                vl = knn_scores.get(cond, {}).get(K_VALUES[0], {}).get(metric)
                vh = knn_scores.get(cond, {}).get(K_VALUES[-1], {}).get(metric)
                ranks_at_low.append(vl if vl is not None else float("-inf"))
                ranks_at_high.append(vh if vh is not None else float("-inf"))
            winner_low = EMBEDDINGS_ORDER[np.argmax(ranks_at_low)]
            winner_high = EMBEDDINGS_ORDER[np.argmax(ranks_at_high)]
            if winner_low != winner_high:
                winner_changes += 1

        knn_stability[metric] = {
            "per_tissue": per_tissue,
            "mean_extreme_tau": round(float(np.mean(extreme_taus)), 3) if extreme_taus else None,
            "extreme_taus": [round(t, 3) for t in extreme_taus],
            "winner_changes": f"{winner_changes}/{len(TISSUES)} tissues",
        }

    print(f"\n{'='*60}")
    print(f"{_ts()} SCORECARD: Ranking Stability (extreme hyperparameter tau)")
    print(f"{'Metric':<25} {'Mean tau':<12} {'Winner changes':<18} {'Predicted tau range'}")
    print("-" * 80)

    prereg_predictions = {
        "nmi_leiden": {"tau_range": [0.5, 0.7], "winner_tissues": 2},
        "ari_leiden": {"tau_range": [0.4, 0.6], "winner_tissues": "2-3"},
        "clisi": {"tau_range": [0.3, 0.6], "winner_tissues": "possibly"},
        "ilisi": {"tau_range": [0.2, 0.5], "winner_tissues": "yes"},
        "kbet": {"tau_range": [0.1, 0.4], "winner_tissues": "3-4"},
        "graph_connectivity": {"tau_range": [0.3, 0.6], "winner_tissues": "no"},
    }

    all_stability = {}
    for metric in LEIDEN_METRICS:
        data = leiden_stability[metric]
        tau = data["mean_extreme_tau"]
        pred = prereg_predictions.get(metric, {})
        pred_range = pred.get("tau_range", [None, None])
        in_range = (pred_range[0] is not None and tau is not None
                    and pred_range[0] <= tau <= pred_range[1])
        status = "IN RANGE" if in_range else "OUT OF RANGE" if tau is not None else "N/A"
        print(f"{metric:<25} {tau if tau is not None else 'N/A':<12} "
              f"{data['winner_changes']:<18} {pred_range} [{status}]")
        all_stability[metric] = {**data, "predicted_range": pred_range, "in_predicted_range": in_range}

    for metric in KNN_METRICS:
        data = knn_stability[metric]
        tau = data["mean_extreme_tau"]
        pred = prereg_predictions.get(metric, {})
        pred_range = pred.get("tau_range", [None, None])
        in_range = (pred_range[0] is not None and tau is not None
                    and pred_range[0] <= tau <= pred_range[1])
        status = "IN RANGE" if in_range else "OUT OF RANGE" if tau is not None else "N/A"
        print(f"{metric:<25} {tau if tau is not None else 'N/A':<12} "
              f"{data['winner_changes']:<18} {pred_range} [{status}]")
        all_stability[metric] = {**data, "predicted_range": pred_range, "in_predicted_range": in_range}

    output = {
        "test": "test5_hyperparameter_sensitivity",
        "status": "CONFIRMATORY",
        "prereg": "PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md",
        "freeze": "docs/frozen_prereg_scib_confirmatory/FROZEN_SUMMARY.json",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "census_version": CENSUS_VERSION,
        "seed": SEED,
        "leiden_resolutions": LEIDEN_RESOLUTIONS,
        "k_values": K_VALUES,
        "n_conditions": len(set(list(leiden_scores.keys()) + list(knn_scores.keys()))),
        "stability_results": all_stability,
        "raw_leiden_scores": {
            cond: {str(r): s for r, s in res.items()}
            for cond, res in leiden_scores.items()
        },
        "raw_knn_scores": {
            cond: {str(k): s for k, s in res.items()}
            for cond, res in knn_scores.items()
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{_ts()} Results saved to {OUTPUT_PATH}")

    return output


if __name__ == "__main__":
    main()
