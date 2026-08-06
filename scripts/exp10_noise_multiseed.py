"""Multi-seed noise dose-response for ALL embedding metric families.

Design choices:
  - Same-noise coupling: one standardized noise direction per seed,
    scaled by sigma. The dose-response is a smooth curve, not 7
    independent random draws.
  - 10 seeds per (tissue, model, sigma) cell.
  - All tissues auto-discovered from Census (same as spectral-gap panel).
  - ALL metric families computed at each noise level:
      * scIB bio-conservation (ARI, NMI, graph connectivity, silhouette, ...)
      * CKA on cell-type centroids (src vs tgt)
      * Procrustes similarity on cell-type centroids
      * kNN purity (cross-assay)
      * Transfer F1 (functional degradation control)
  - Primary test statistic: H = max_{sigma>0}[m(sigma) - m(0)].
    Reported per seed; seed-mean H tested for significance.
  - Key prediction: CKA/Procrustes/F1 degrade monotonically;
    scIB bio-conservation (ARI, NMI, graph connectivity) can improve.

Usage:
    python scripts/exp10_noise_multiseed.py [--n-seeds 10] [--tissues 4]
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.spatial import procrustes as scipy_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713
MIN_CELLS = 200
MIN_SHARED_TYPES = 8

SIGMAS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

OUTPUT_DIR = Path("results/noise_multiseed")

SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

CENSUS_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]

ORIGINAL_TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}

SCIB_PRIMARY = ["ari_leiden", "nmi_leiden", "graph_connectivity"]
SIMILARITY_METRICS = ["cell_type_cka", "procrustes_sim", "knn_purity"]
ALL_PRIMARY = SCIB_PRIMARY + SIMILARITY_METRICS + ["transfer_f1"]


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return None
    return float(hsic_xy / denom)


def cell_type_cka(X_src, y_src, X_tgt, y_tgt):
    shared = sorted(set(y_src) & set(y_tgt))
    if len(shared) < MIN_SHARED_TYPES:
        return None
    src_centroids = np.stack([X_src[y_src == ct].mean(0) for ct in shared])
    tgt_centroids = np.stack([X_tgt[y_tgt == ct].mean(0) for ct in shared])
    return linear_cka(src_centroids, tgt_centroids)


def procrustes_similarity(X_src, y_src, X_tgt, y_tgt, n_components=50):
    shared = sorted(set(y_src) & set(y_tgt))
    if len(shared) < MIN_SHARED_TYPES:
        return None
    src_centroids = np.stack([X_src[y_src == ct].mean(0) for ct in shared])
    tgt_centroids = np.stack([X_tgt[y_tgt == ct].mean(0) for ct in shared])
    k = min(n_components, len(shared) - 1, src_centroids.shape[1])
    if k < 2:
        return None
    pca = PCA(n_components=k, random_state=0)
    src_reduced = pca.fit_transform(src_centroids)
    tgt_reduced = PCA(n_components=k, random_state=0).fit_transform(
        tgt_centroids)
    _, _, disparity = scipy_procrustes(src_reduced, tgt_reduced)
    return float(1.0 - disparity)


def cross_assay_knn_purity(X_src, y_src, X_tgt, y_tgt, k=15):
    k_actual = min(k, len(X_tgt) - 1)
    if k_actual < 1:
        return None
    nn = NearestNeighbors(n_neighbors=k_actual, metric="cosine")
    nn.fit(X_tgt)
    _, indices = nn.kneighbors(X_src)
    hits = sum(1 for i, nbrs in enumerate(indices)
               for j in nbrs if y_tgt[j] == y_src[i])
    total = sum(len(nbrs) for nbrs in indices)
    src_to_tgt = hits / total if total > 0 else 0.0
    nn2 = NearestNeighbors(
        n_neighbors=min(k, len(X_src) - 1), metric="cosine")
    nn2.fit(X_src)
    _, indices2 = nn2.kneighbors(X_tgt)
    hits2 = sum(1 for i, nbrs in enumerate(indices2)
                for j in nbrs if y_src[j] == y_tgt[i])
    total2 = sum(len(nbrs) for nbrs in indices2)
    tgt_to_src = hits2 / total2 if total2 > 0 else 0.0
    return float((src_to_tgt + tgt_to_src) / 2)


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
    X_combined = np.log1p(np.vstack([X_src_raw, X_tgt_raw]).astype(np.float64))
    n_src = X_src_raw.shape[0]
    n_components = min(d_out, X_combined.shape[0], X_combined.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_combined).astype(np.float32)
    return X_pca[:n_src], X_pca[n_src:]


def compute_scib_metrics(adata, batch_key="batch", label_key="cell_type",
                         embed_key="X_emb"):
    from scib_metrics import (
        nmi_ari_cluster_labels_leiden,
        silhouette_label,
        silhouette_batch,
        isolated_labels,
        pcr_comparison,
        graph_connectivity,
        clisi_knn,
        ilisi_knn,
    )
    from scib_metrics.nearest_neighbors import pynndescent

    results = {}
    X_emb = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batch = adata.obs[batch_key].values

    nn_15 = pynndescent(X_emb, n_neighbors=15)
    nn_90 = pynndescent(X_emb, n_neighbors=90)

    try:
        nmi_ari = nmi_ari_cluster_labels_leiden(nn_15, labels,
                                                optimize_resolution=True)
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
        results["silhouette_batch"] = float(silhouette_batch(X_emb, labels,
                                                             batch))
    except Exception:
        results["silhouette_batch"] = None

    try:
        results["isolated_label_asw"] = float(isolated_labels(X_emb, labels,
                                                              batch))
    except Exception:
        results["isolated_label_asw"] = None

    try:
        results["graph_connectivity"] = float(graph_connectivity(nn_15,
                                                                 labels))
    except Exception:
        results["graph_connectivity"] = None

    try:
        X_pre = adata.X if not sp.issparse(adata.X) else adata.X.toarray()
        results["pcr_comparison"] = float(pcr_comparison(X_pre, X_emb, batch,
                                                         categorical=True))
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

    return results


def compute_transfer_f1(X_src, y_src, X_tgt, y_tgt):
    shared = sorted(set(y_src) & set(y_tgt))
    if len(shared) < MIN_SHARED_TYPES:
        return None
    le = LabelEncoder()
    le.fit(shared)
    mask_src = np.isin(y_src, shared)
    mask_tgt = np.isin(y_tgt, shared)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_src[mask_src], le.transform(y_src[mask_src]))
    y_pred = clf.predict(X_tgt[mask_tgt])
    return float(f1_score(le.transform(y_tgt[mask_tgt]), y_pred,
                          average='macro'))


def corrupt_embedding_coupled(X_emb, sigma, emb_std, z_direction):
    """Same-noise coupling: scale a fixed direction by sigma."""
    return X_emb + sigma * emb_std[np.newaxis, :] * z_direction


def discover_tissues(census):
    import cellxgene_census
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        column_names=["tissue", "tissue_ontology_term_id",
                      "assay_ontology_term_id", "cell_type",
                      "is_primary_data", "disease"],
        value_filter=(
            f"is_primary_data == True and disease == 'normal' and "
            f"assay_ontology_term_id in ['{SOURCE_ASSAY}', '{TARGET_ASSAY}']"
        ),
    )
    tissues = {}
    for tissue_name in obs_df['tissue'].unique():
        t_df = obs_df[obs_df['tissue'] == tissue_name]
        src = t_df[t_df['assay_ontology_term_id'] == SOURCE_ASSAY]
        tgt = t_df[t_df['assay_ontology_term_id'] == TARGET_ASSAY]
        if len(src) < MIN_CELLS or len(tgt) < MIN_CELLS:
            continue
        shared = set(src['cell_type'].unique()) & set(tgt['cell_type'].unique())
        if len(shared) < MIN_SHARED_TYPES:
            continue
        tid = t_df['tissue_ontology_term_id'].iloc[0]
        tissues[tissue_name] = tid
    return tissues


def load_tissue_pair(census, tissue_id, tissue_name):
    import cellxgene_census
    results = {}
    for side, assay_id, seed_offset in [("source", SOURCE_ASSAY, 0),
                                         ("target", TARGET_ASSAY, 1)]:
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
        print(f"  {_ts()} {tissue_name} {side}: {len(all_ids)} cells")
        if len(all_ids) == 0:
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
            obs_embeddings=CENSUS_EMBEDDINGS,
        )
        print(f"  {_ts()} {tissue_name} {side}: {adata.n_obs} cells, "
              f"{len(adata.obs['cell_type'].unique())} types")
        results[side] = adata

    src = results["source"]
    tgt = results["target"]
    shared = set(src.obs['cell_type'].values) & set(tgt.obs['cell_type'].values)
    if len(shared) < MIN_SHARED_TYPES:
        print(f"  {_ts()} Skipping {tissue_name}: {len(shared)} shared types")
        return None
    print(f"  {_ts()} {tissue_name}: shared={len(shared)}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--tissues", type=int, default=25,
                        help="Number of tissues (4=original, 25=all)")
    parser.add_argument("--tissue-index", type=int, default=None,
                        help="Process only the tissue at this index")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()
    n_seeds = args.n_seeds
    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} Multi-seed noise dose-response")
    print(f"{_ts()} Seeds: {n_seeds}, Tissues: {args.tissues}")
    print(f"{_ts()} Same-noise coupling: YES")
    print(f"{_ts()} Sigmas: {SIGMAS}")

    import cellxgene_census

    all_results = []
    inc_path = OUTPUT_DIR / "incremental.jsonl"

    done_keys = set()
    if inc_path.exists():
        with open(inc_path) as f:
            for line in f:
                row = json.loads(line)
                all_results.append(row)
                done_keys.add((row["tissue"], row["model"], row["seed_idx"]))
        print(f"{_ts()} Resuming: {len(done_keys)} results already done")

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        if args.tissues <= 4:
            tissues = ORIGINAL_TISSUES
        else:
            tissues = discover_tissues(census)
        tissue_list = list(tissues.items())
        if args.tissue_index is not None:
            if args.tissue_index >= len(tissue_list):
                print(f"{_ts()} tissue-index {args.tissue_index} out of range "
                      f"(only {len(tissue_list)} tissues)")
                return
            tissue_list = [tissue_list[args.tissue_index]]
        print(f"\n{_ts()} Running on {len(tissue_list)} tissues")

        for tissue_name, tissue_id in tqdm(tissue_list, desc="Tissues"):
            print(f"\n{_ts()} === {tissue_name} ===")
            pair = load_tissue_pair(census, tissue_id, tissue_name)
            if pair is None:
                continue

            src_adata = pair["source"]
            tgt_adata = pair["target"]
            labels_src = src_adata.obs["cell_type"].values.astype(str)
            labels_tgt = tgt_adata.obs["cell_type"].values.astype(str)
            donors_src = src_adata.obs["donor_id"].values.astype(str)
            donors_tgt = tgt_adata.obs["donor_id"].values.astype(str)

            X_src_raw = src_adata.X.toarray() if sp.issparse(src_adata.X) \
                else np.asarray(src_adata.X)
            X_tgt_raw = tgt_adata.X.toarray() if sp.issparse(tgt_adata.X) \
                else np.asarray(tgt_adata.X)

            embedding_dict = {}
            for emb_name in CENSUS_EMBEDDINGS:
                if emb_name in src_adata.obsm and emb_name in tgt_adata.obsm:
                    s = np.asarray(src_adata.obsm[emb_name])
                    t = np.asarray(tgt_adata.obsm[emb_name])
                    if np.all(np.isfinite(s)) and np.all(np.isfinite(t)):
                        embedding_dict[emb_name] = {
                            "src": s.astype(np.float32),
                            "tgt": t.astype(np.float32),
                        }

            s_rp = random_projection(X_src_raw)
            t_rp = random_projection(X_tgt_raw)
            embedding_dict["random_projection"] = {
                "src": s_rp, "tgt": t_rp,
            }

            s_ue = untrained_encoder(X_src_raw)
            t_ue = untrained_encoder(X_tgt_raw)
            embedding_dict["untrained_encoder"] = {
                "src": s_ue, "tgt": t_ue,
            }

            s_bog, t_bog = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            embedding_dict["bog_pca_512"] = {
                "src": s_bog, "tgt": t_bog,
            }

            n_src = src_adata.n_obs
            n_tgt = tgt_adata.n_obs
            X_raw_combined = sp.csr_matrix(np.vstack([X_src_raw, X_tgt_raw]))
            obs_base = pd.DataFrame({
                "cell_type": np.concatenate([labels_src, labels_tgt]),
                "batch": np.concatenate([np.full(n_src, "source"),
                                         np.full(n_tgt, "target")]),
                "donor_id": np.concatenate([donors_src, donors_tgt]),
            })

            for emb_name, emb_data in embedding_dict.items():
                X_src_clean = emb_data["src"]
                X_tgt_clean = emb_data["tgt"]
                X_clean = np.vstack([X_src_clean, X_tgt_clean])
                emb_std = X_clean.std(axis=0)
                emb_std = np.where(emb_std < 1e-8, 1e-8, emb_std)

                print(f"\n{_ts()} {tissue_name} / {emb_name} "
                      f"(d={X_clean.shape[1]})")

                for seed_idx in range(n_seeds):
                    if (tissue_name, emb_name, seed_idx) in done_keys:
                        continue
                    seed_val = SEED + seed_idx * 100000
                    rng = np.random.default_rng(seed_val)
                    z_direction = rng.standard_normal(X_clean.shape) \
                        .astype(np.float32)

                    seed_scores = {}

                    def _compute_all_metrics(X_emb, X_s, y_s, X_t, y_t,
                                             adata_obj):
                        scib = compute_scib_metrics(adata_obj)
                        scib["cell_type_cka"] = cell_type_cka(
                            X_s, y_s, X_t, y_t)
                        scib["procrustes_sim"] = procrustes_similarity(
                            X_s, y_s, X_t, y_t)
                        scib["knn_purity"] = cross_assay_knn_purity(
                            X_s, y_s, X_t, y_t)
                        scib["transfer_f1"] = compute_transfer_f1(
                            X_s, y_s, X_t, y_t)
                        return scib

                    # Baseline (sigma=0)
                    adata_base = ad.AnnData(X=X_raw_combined,
                                            obs=obs_base.copy())
                    adata_base.obsm["X_emb"] = X_clean
                    scores_0 = _compute_all_metrics(
                        X_clean, X_src_clean, labels_src,
                        X_tgt_clean, labels_tgt, adata_base)
                    seed_scores[0.0] = scores_0

                    for sigma in SIGMAS:
                        X_noisy = corrupt_embedding_coupled(
                            X_clean, sigma, emb_std, z_direction)
                        X_src_n = X_noisy[:n_src]
                        X_tgt_n = X_noisy[n_src:]

                        adata_n = ad.AnnData(X=X_raw_combined,
                                             obs=obs_base.copy())
                        adata_n.obsm["X_emb"] = X_noisy
                        scores = _compute_all_metrics(
                            X_noisy, X_src_n, labels_src,
                            X_tgt_n, labels_tgt, adata_n)
                        seed_scores[sigma] = scores

                    # Compute H statistic per metric
                    h_stats = {}
                    for metric in scores_0.keys():
                        baseline = seed_scores[0.0].get(metric)
                        if baseline is None:
                            continue
                        improvements = []
                        for sigma in SIGMAS:
                            val = seed_scores[sigma].get(metric)
                            if val is not None:
                                improvements.append(val - baseline)
                        if improvements:
                            h_stats[metric] = max(improvements)

                    result_row = {
                        "tissue": tissue_name,
                        "model": emb_name,
                        "seed_idx": seed_idx,
                        "seed_val": seed_val,
                        "scores_by_sigma": {
                            str(s): v for s, v in seed_scores.items()
                        },
                        "h_stats": h_stats,
                    }
                    all_results.append(result_row)

                    with open(inc_path, "a") as f:
                        f.write(json.dumps(result_row, default=str) + "\n")

                    h_ari = h_stats.get("ari_leiden", 0)
                    h_cka = h_stats.get("cell_type_cka", 0)
                    h_proc = h_stats.get("procrustes_sim", 0)
                    f1_0 = scores_0.get("transfer_f1")
                    f1_last = seed_scores[2.0].get("transfer_f1")
                    print(f"  seed {seed_idx:2d}: "
                          f"H(ARI)={h_ari:+.4f} "
                          f"H(CKA)={h_cka:+.4f} "
                          f"H(Proc)={h_proc:+.4f} "
                          f"F1: {f1_0:.3f}->{f1_last:.3f}"
                          if f1_0 and f1_last else
                          f"  seed {seed_idx:2d}: "
                          f"H(ARI)={h_ari:+.4f} "
                          f"H(CKA)={h_cka:+.4f} "
                          f"H(Proc)={h_proc:+.4f}")

    # === Analysis ===
    print(f"\n{'='*80}")
    print(f"{_ts()} === Analysis ===")

    all_metrics = list(set(k for r in all_results
                           for k in r["h_stats"].keys()))

    conditions = sorted(set((r["tissue"], r["model"]) for r in all_results))
    print(f"\nConditions: {len(conditions)}")

    summary = {
        "scib_metrics": {},
        "similarity_metrics": {},
        "all_metrics": {},
        "conditions": [],
    }

    def _analyze_metric(metric):
        h_values = []
        cond_results = []
        for tissue, model in conditions:
            seeds = [r for r in all_results
                     if r["tissue"] == tissue and r["model"] == model]
            hs = [s["h_stats"].get(metric) for s in seeds
                  if s["h_stats"].get(metric) is not None]
            if not hs:
                continue
            mean_h = np.mean(hs)
            frac_positive = np.mean([h > 0.01 for h in hs])
            h_values.append(mean_h)
            cond_results.append({
                "tissue": tissue, "model": model,
                "mean_h": mean_h, "std_h": float(np.std(hs)),
                "frac_positive": float(frac_positive),
                "n_seeds": len(hs),
            })
        n_positive = sum(1 for h in h_values if h > 0.01)
        frac = n_positive / len(h_values) if h_values else 0
        return {
            "n_conditions_with_mean_h_gt_001": n_positive,
            "n_conditions": len(h_values),
            "frac_conditions_positive": round(frac, 3),
            "mean_h_all": round(float(np.mean(h_values)), 4)
            if h_values else None,
            "per_condition": cond_results,
        }

    print("\n--- scIB bio-conservation metrics (predicted: non-monotonic) ---")
    for metric in sorted(all_metrics):
        if metric in SCIB_PRIMARY:
            entry = _analyze_metric(metric)
            summary["scib_metrics"][metric] = entry
            print(f"  [scIB] {metric}: "
                  f"{entry['n_conditions_with_mean_h_gt_001']}/"
                  f"{entry['n_conditions']} conditions with mean H > 0.01 "
                  f"({entry['frac_conditions_positive']:.1%}), "
                  f"overall mean H = {entry['mean_h_all']}")

    print("\n--- Similarity metrics (predicted: monotonically decreasing) ---")
    for metric in SIMILARITY_METRICS:
        if metric in all_metrics:
            entry = _analyze_metric(metric)
            summary["similarity_metrics"][metric] = entry
            print(f"  [SIM]  {metric}: "
                  f"{entry['n_conditions_with_mean_h_gt_001']}/"
                  f"{entry['n_conditions']} conditions with mean H > 0.01 "
                  f"({entry['frac_conditions_positive']:.1%}), "
                  f"overall mean H = {entry['mean_h_all']}")

    for metric in sorted(all_metrics):
        if metric not in SCIB_PRIMARY and metric not in SIMILARITY_METRICS:
            entry = _analyze_metric(metric)
            summary["all_metrics"][metric] = entry

    # Transfer F1 degradation control
    print(f"\n--- Transfer F1 degradation control ---")
    f1_mono_count = 0
    f1_total = 0
    for tissue, model in conditions:
        seeds = [r for r in all_results
                 if r["tissue"] == tissue and r["model"] == model]
        mean_f1_by_sigma = {}
        for sigma_key in ["0.0"] + [str(s) for s in SIGMAS]:
            vals = [s["scores_by_sigma"][sigma_key].get("transfer_f1")
                    for s in seeds
                    if s["scores_by_sigma"].get(sigma_key, {}).get(
                        "transfer_f1") is not None]
            if vals:
                mean_f1_by_sigma[sigma_key] = np.mean(vals)

        if len(mean_f1_by_sigma) == 8:
            f1_vals = [mean_f1_by_sigma[k] for k in
                       ["0.0"] + [str(s) for s in SIGMAS]]
            is_mono = all(a >= b - 0.005 for a, b in
                          zip(f1_vals, f1_vals[1:]))
            f1_mono_count += int(is_mono)
            f1_total += 1

    print(f"  F1 monotonically decreasing: {f1_mono_count}/{f1_total}")
    summary["f1_degradation_control"] = {
        "monotonic_decrease": f1_mono_count,
        "total": f1_total,
    }

    # Dissociation test
    scib_frac = np.mean([
        summary["scib_metrics"].get(m, {}).get("frac_conditions_positive", 0)
        for m in SCIB_PRIMARY if m in summary["scib_metrics"]
    ]) if summary["scib_metrics"] else 0
    sim_frac = np.mean([
        summary["similarity_metrics"].get(m, {}).get(
            "frac_conditions_positive", 0)
        for m in SIMILARITY_METRICS if m in summary["similarity_metrics"]
    ]) if summary["similarity_metrics"] else 0
    print(f"\n--- Dissociation ---")
    print(f"  scIB fraction with H > 0.01: {scib_frac:.1%}")
    print(f"  Similarity fraction with H > 0.01: {sim_frac:.1%}")
    print(f"  Dissociation = {scib_frac - sim_frac:+.1%}")
    summary["dissociation"] = {
        "scib_frac_positive": round(float(scib_frac), 3),
        "similarity_frac_positive": round(float(sim_frac), 3),
        "difference": round(float(scib_frac - sim_frac), 3),
    }

    # Save
    raw_path = OUTPUT_DIR / "noise_multiseed_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{_ts()} Saved raw to {raw_path}")

    summary_path = OUTPUT_DIR / "noise_multiseed_summary.json"
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    summary["n_seeds"] = n_seeds
    summary["n_tissues"] = len(set(r["tissue"] for r in all_results))
    summary["n_conditions"] = len(conditions)
    summary["n_total_evaluations"] = len(all_results)
    summary["same_noise_coupling"] = True
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"{_ts()} Saved summary to {summary_path}")
    print(f"{_ts()} Done.")


if __name__ == "__main__":
    main()
