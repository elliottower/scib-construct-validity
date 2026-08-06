"""Experiment 11: Cross-Tissue Second Ground Truth [CONFIRMATORY].

Prereg: preregistration_v11_cross_tissue_ground_truth.md

Extends the scIB inversion-rate analysis from the cross-assay primary
ground truth (Paper D) to a cross-tissue setting: same assay (10x 3' v3),
different tissues. Tests whether bio-conservation metrics invert pairwise
model rankings against cross-tissue transfer F1 at rates comparable to
the cross-assay finding (46-58%).

Hypotheses:
  H11.1 — Mean bio-metric inversion rate >= 0.40 (block bootstrap CI)
  H11.2 — scIB bio composite anti-predicts cross-tissue F1 (Spearman rho)
  H11.3 — [exploratory] Inversion rates by F1-gap stratum

Usage:
    python scripts/exp11_cross_tissue_validity.py
"""
import json
import itertools
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260801
N_PERMUTATIONS = 10_000
N_BOOTSTRAP = 10_000

OUTPUT_DIR = Path("results/exp11_cross_tissue_validity")

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
    "blood":  "UBERON:0000178",
}

ASSAY = "EFO:0009922"  # 10x 3' v3

CROSS_TISSUE_PAIRS = [
    ("lung", "brain"),
    ("lung", "kidney"),
    ("blood", "lung"),
    ("blood", "brain"),
    ("liver", "kidney"),
    ("blood", "liver"),
]

REAL_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]

CONTENDERS = ["geneformer", "scvi", "scgpt", "bog_pca_512"]

BIO_METRICS = [
    "nmi_leiden", "ari_leiden", "silhouette_label",
    "clisi", "isolated_label_asw",
]

V7_GENEFORMER_F1 = {
    ("lung", "brain"): 0.5323,
    ("lung", "kidney"): 0.4290,
    ("blood", "lung"): 0.1852,
    ("blood", "brain"): 0.6123,
    ("liver", "kidney"): 0.9824,
    ("blood", "liver"): 0.4127,
}


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def bag_of_genes_pca_combined(X_src_raw, X_tgt_raw, d_out=512):
    from sklearn.decomposition import PCA
    X_combined = np.log1p(np.vstack([X_src_raw, X_tgt_raw]).astype(np.float64))
    n_src = X_src_raw.shape[0]
    n_components = min(d_out, X_combined.shape[0], X_combined.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_combined).astype(np.float32)
    return X_pca[:n_src], X_pca[n_src:], n_components


def compute_scib_bio_metrics(adata, batch_key="batch", label_key="cell_type", embed_key="X_emb"):
    from scib_metrics import (
        nmi_ari_cluster_labels_leiden,
        silhouette_label,
        isolated_labels,
        clisi_knn,
    )
    from scib_metrics.nearest_neighbors import pynndescent

    results = {}
    X_emb = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    batch = adata.obs[batch_key].values

    print(f"    {_ts()} Computing kNN (k=15,90)...")
    nn_15 = pynndescent(X_emb, n_neighbors=15)
    nn_90 = pynndescent(X_emb, n_neighbors=90)

    try:
        nmi_ari = nmi_ari_cluster_labels_leiden(nn_15, labels, optimize_resolution=True)
        results["nmi_leiden"] = float(nmi_ari["nmi"])
        results["ari_leiden"] = float(nmi_ari["ari"])
    except Exception as e:
        print(f"    {_ts()} WARNING: NMI/ARI failed: {e}")
        results["nmi_leiden"] = None
        results["ari_leiden"] = None

    try:
        results["silhouette_label"] = float(silhouette_label(X_emb, labels))
    except Exception as e:
        print(f"    {_ts()} WARNING: silhouette_label failed: {e}")
        results["silhouette_label"] = None

    try:
        results["isolated_label_asw"] = float(isolated_labels(X_emb, labels, batch))
    except Exception as e:
        print(f"    {_ts()} WARNING: isolated_label_asw failed: {e}")
        results["isolated_label_asw"] = None

    try:
        results["clisi"] = float(np.nanmean(clisi_knn(nn_90, labels)))
    except Exception as e:
        print(f"    {_ts()} WARNING: cLISI failed: {e}")
        results["clisi"] = None

    return results


def run_transfer_probe(X_src, labels_src, X_tgt, labels_tgt):
    shared = sorted(set(labels_src) & set(labels_tgt))
    if len(shared) < 3:
        return {"f1_source": None, "f1_target": None, "n_shared": len(shared)}

    mask_src = np.isin(labels_src, shared)
    mask_tgt = np.isin(labels_tgt, shared)
    X_s = X_src[mask_src]
    X_t = X_tgt[mask_tgt]
    y_s = labels_src[mask_src]
    y_t = labels_tgt[mask_tgt]

    le = LabelEncoder()
    le.fit(shared)
    y_s_enc = le.transform(y_s)
    y_t_enc = le.transform(y_t)

    scaler = StandardScaler()
    X_s_sc = scaler.fit_transform(X_s)
    X_t_sc = scaler.transform(X_t)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_s_sc, y_s_enc)

    f1_src = f1_score(y_s_enc, clf.predict(X_s_sc), average="macro")
    f1_tgt = f1_score(y_t_enc, clf.predict(X_t_sc), average="macro")

    return {"f1_source": float(f1_src), "f1_target": float(f1_tgt), "n_shared": len(shared)}


def compute_pairwise_inversions(scores_by_model, f1_by_model, metric_name):
    models = sorted(scores_by_model.keys())
    pairs = list(itertools.combinations(models, 2))
    inversions = 0
    total = 0
    details = []
    for m_a, m_b in pairs:
        s_a = scores_by_model[m_a]
        s_b = scores_by_model[m_b]
        f_a = f1_by_model[m_a]
        f_b = f1_by_model[m_b]
        if s_a is None or s_b is None or f_a is None or f_b is None:
            continue
        metric_prefers = 1 if s_a > s_b else (-1 if s_a < s_b else 0)
        f1_prefers = 1 if f_a > f_b else (-1 if f_a < f_b else 0)
        if metric_prefers == 0 or f1_prefers == 0:
            continue
        inverted = metric_prefers != f1_prefers
        inversions += int(inverted)
        total += 1
        details.append({
            "model_a": m_a, "model_b": m_b,
            "metric_a": s_a, "metric_b": s_b,
            "f1_a": f_a, "f1_b": f_b,
            "inverted": inverted,
        })
    rate = inversions / total if total > 0 else None
    return {"rate": rate, "inversions": inversions, "total": total, "details": details}


def block_bootstrap_ci(per_tissue_rates, n_boot=N_BOOTSTRAP, alpha=0.05):
    rng = np.random.default_rng(SEED + 7)
    rates = np.array(per_tissue_rates, dtype=float)
    n = len(rates)
    if n == 0:
        return None, None, None
    boot_means = np.array([
        np.mean(rng.choice(rates, size=n, replace=True))
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(rates)), lo, hi


def permutation_p_value(observed_rho, x, y, n_perm=N_PERMUTATIONS):
    rng = np.random.default_rng(SEED + 42)
    count = 0
    for _ in range(n_perm):
        perm_y = rng.permutation(y)
        perm_rho, _ = spearmanr(x, perm_y)
        if abs(perm_rho) >= abs(observed_rho):
            count += 1
    return count / n_perm


def main():
    import cellxgene_census

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"

    print(f"{_ts()} Experiment 11: Cross-Tissue Second Ground Truth")
    print(f"{_ts()} Prereg: preregistration_v11_cross_tissue_ground_truth.md")
    print(f"{_ts()} Census: {CENSUS_VERSION}")
    print(f"{_ts()} Seed: {SEED}")
    print(f"{_ts()} Contenders: {CONTENDERS}")
    print(f"{_ts()} Cross-tissue pairs: {CROSS_TISSUE_PAIRS}")

    # Phase 1: Pull all tissue embeddings
    tissue_data = {}
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue_name, tissue_id in tqdm(TISSUES.items(), desc="Pulling tissues"):
            print(f"\n  {_ts()} Pulling {tissue_name}...")
            filter_str = (
                f"tissue_ontology_term_id == '{tissue_id}' "
                f"and is_primary_data == True "
                f"and assay_ontology_term_id == '{ASSAY}'"
            )
            obs_df = cellxgene_census.get_obs(
                census, ORGANISM,
                value_filter=filter_str,
                column_names=["soma_joinid", "donor_id"],
            )
            joinids = obs_df["soma_joinid"].values
            if len(joinids) == 0:
                print(f"    SKIPPED (no cells)")
                continue
            if len(joinids) > MAX_CELLS:
                rng = np.random.default_rng(SEED)
                idx = rng.choice(len(joinids), size=MAX_CELLS, replace=False)
                idx.sort()
                joinids = joinids[idx]

            adata = cellxgene_census.get_anndata(
                census,
                organism=ORGANISM,
                obs_value_filter=filter_str,
                obs_coords=joinids,
                obs_column_names=OBS_COLUMNS,
                obs_embeddings=REAL_EMBEDDINGS,
            )
            X_raw = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
            tissue_data[tissue_name] = {
                "adata": adata,
                "X_raw": X_raw,
                "labels": adata.obs["cell_type"].values,
                "donors": adata.obs["donor_id"].values,
            }
            print(f"    {adata.n_obs} cells, {len(np.unique(adata.obs['cell_type']))} cell types")

    # Phase 2: For each cross-tissue pair, compute embeddings + scIB + probe F1
    # scib_scores[(pair_id, model)] = {metric: score}
    # probe_f1s[(pair_id, model)] = float
    scib_scores = {}
    probe_f1s = {}
    pair_metadata = {}

    for src_tissue, tgt_tissue in tqdm(CROSS_TISSUE_PAIRS, desc="Cross-tissue pairs"):
        pair_id = f"{src_tissue}_to_{tgt_tissue}"
        print(f"\n{'='*60}")
        print(f"{_ts()} Pair: {src_tissue} -> {tgt_tissue}")

        if src_tissue not in tissue_data or tgt_tissue not in tissue_data:
            print(f"  SKIPPING: missing tissue data")
            continue

        src = tissue_data[src_tissue]
        tgt = tissue_data[tgt_tissue]

        # Build embedding dict for this pair
        embedding_dict = {}
        for emb_name in REAL_EMBEDDINGS:
            if emb_name in src["adata"].obsm and emb_name in tgt["adata"].obsm:
                X_s = np.array(src["adata"].obsm[emb_name])
                X_t = np.array(tgt["adata"].obsm[emb_name])
                embedding_dict[emb_name] = (X_s, X_t)
                print(f"  {emb_name}: d={X_s.shape[1]}")
            else:
                print(f"  {emb_name}: not available, skipping")

        X_s_bog, X_t_bog, bog_d = bag_of_genes_pca_combined(src["X_raw"], tgt["X_raw"])
        embedding_dict["bog_pca_512"] = (X_s_bog, X_t_bog)
        print(f"  bog_pca_512: d={bog_d}")

        pair_metadata[pair_id] = {
            "source_tissue": src_tissue,
            "target_tissue": tgt_tissue,
            "n_source": src["adata"].n_obs,
            "n_target": tgt["adata"].n_obs,
        }

        for model_name, (X_s, X_t) in tqdm(
            embedding_dict.items(), desc=f"  Models ({pair_id})", leave=False
        ):
            print(f"\n  {_ts()} {pair_id} / {model_name}")

            # scIB metrics on combined AnnData
            combined = ad.AnnData(
                X=sp.csr_matrix(np.zeros((X_s.shape[0] + X_t.shape[0], 1))),
                obs=pd.DataFrame({
                    "cell_type": np.concatenate([src["labels"], tgt["labels"]]),
                    "batch": np.concatenate([
                        np.full(X_s.shape[0], src_tissue),
                        np.full(X_t.shape[0], tgt_tissue),
                    ]),
                }),
            )
            combined.obsm["X_emb"] = np.vstack([X_s, X_t]).astype(np.float32)

            scores = compute_scib_bio_metrics(combined, batch_key="batch", label_key="cell_type")
            scib_scores[(pair_id, model_name)] = scores

            # Transfer probe F1
            probe = run_transfer_probe(X_s, src["labels"], X_t, tgt["labels"])
            probe_f1s[(pair_id, model_name)] = probe["f1_target"]

            n_shared = probe["n_shared"]
            pair_metadata[pair_id]["n_shared_cell_types"] = n_shared

            print(f"    Probe F1: src={probe['f1_source']}, tgt={probe['f1_target']}, shared={n_shared}")
            for m, v in scores.items():
                print(f"    {m}: {v}")

            # Sanity check: Geneformer F1 vs v7
            if model_name == "geneformer":
                v7_f1 = V7_GENEFORMER_F1.get((src_tissue, tgt_tissue))
                if v7_f1 is not None and probe["f1_target"] is not None:
                    delta = abs(probe["f1_target"] - v7_f1)
                    status = "OK" if delta < 0.05 else "MISMATCH"
                    pair_metadata[pair_id]["v7_sanity"] = {
                        "v7_f1": v7_f1,
                        "exp11_f1": probe["f1_target"],
                        "delta": float(delta),
                        "status": status,
                    }
                    if status == "MISMATCH":
                        pair_metadata[pair_id].setdefault("validity_warnings", []).append(
                            f"Geneformer F1 delta {delta:.4f} exceeds 0.05 threshold"
                        )
                    print(f"    v7 sanity check: v7={v7_f1:.4f}, now={probe['f1_target']:.4f}, delta={delta:.4f} [{status}]")

            result = {
                "pair_id": pair_id,
                "model": model_name,
                "scib_scores": scores,
                "probe_f1_target": probe["f1_target"],
                "probe_f1_source": probe["f1_source"],
                "n_shared_cell_types": n_shared,
                "timestamp": _ts(),
            }
            with open(inc_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")
                f.flush()

    # Phase 3: Hypothesis testing
    print(f"\n{'='*60}")
    print(f"{_ts()} HYPOTHESIS TESTING")

    # --- H11.1: Pairwise inversion rates ---
    print(f"\n{'-'*40}")
    print(f"{_ts()} H11.1: Cross-tissue inversion rates")

    # Per metric, per tissue pair: inversion rate
    # Then aggregate across tissue pairs with block bootstrap
    per_metric_results = {}
    for metric in BIO_METRICS:
        per_tissue_rates = []
        all_inversions = 0
        all_total = 0

        for src_tissue, tgt_tissue in CROSS_TISSUE_PAIRS:
            pair_id = f"{src_tissue}_to_{tgt_tissue}"
            scores_by_model = {}
            f1_by_model = {}
            for model in CONTENDERS:
                key = (pair_id, model)
                if key in scib_scores:
                    scores_by_model[model] = scib_scores[key].get(metric)
                if key in probe_f1s:
                    f1_by_model[model] = probe_f1s[key]

            inv = compute_pairwise_inversions(scores_by_model, f1_by_model, metric)
            if inv["rate"] is not None:
                per_tissue_rates.append(inv["rate"])
            all_inversions += inv["inversions"]
            all_total += inv["total"]

        overall_rate = all_inversions / all_total if all_total > 0 else None
        mean_rate, ci_lo, ci_hi = block_bootstrap_ci(per_tissue_rates)

        per_metric_results[metric] = {
            "overall_rate": overall_rate,
            "block_bootstrap_mean": mean_rate,
            "block_bootstrap_ci_lo": ci_lo,
            "block_bootstrap_ci_hi": ci_hi,
            "n_tissue_pairs": len(per_tissue_rates),
            "total_comparisons": all_total,
            "total_inversions": all_inversions,
        }
        print(f"  {metric}: rate={overall_rate:.3f}, "
              f"block-bootstrap mean={mean_rate:.3f} CI=[{ci_lo:.3f}, {ci_hi:.3f}]"
              if overall_rate is not None else f"  {metric}: N/A")

    # Aggregate across metrics
    metric_rates = [r["block_bootstrap_mean"] for r in per_metric_results.values()
                    if r["block_bootstrap_mean"] is not None]
    if metric_rates:
        grand_mean = float(np.mean(metric_rates))
        # Block bootstrap the grand mean: resample tissue pairs, recompute per-metric rates
        all_per_tissue = {}
        for metric in BIO_METRICS:
            tissue_rates = []
            for src_tissue, tgt_tissue in CROSS_TISSUE_PAIRS:
                pair_id = f"{src_tissue}_to_{tgt_tissue}"
                scores_by_model = {}
                f1_by_model = {}
                for model in CONTENDERS:
                    key = (pair_id, model)
                    if key in scib_scores:
                        scores_by_model[model] = scib_scores[key].get(metric)
                    if key in probe_f1s:
                        f1_by_model[model] = probe_f1s[key]
                inv = compute_pairwise_inversions(scores_by_model, f1_by_model, metric)
                tissue_rates.append(inv["rate"] if inv["rate"] is not None else np.nan)
            all_per_tissue[metric] = tissue_rates

        rng = np.random.default_rng(SEED + 11)
        n_pairs = len(CROSS_TISSUE_PAIRS)
        boot_grand_means = []
        for _ in range(N_BOOTSTRAP):
            pair_idx = rng.choice(n_pairs, size=n_pairs, replace=True)
            metric_means = []
            for metric in BIO_METRICS:
                rates = np.array(all_per_tissue[metric])
                boot_rates = rates[pair_idx]
                valid = boot_rates[~np.isnan(boot_rates)]
                if len(valid) > 0:
                    metric_means.append(np.mean(valid))
            if metric_means:
                boot_grand_means.append(np.mean(metric_means))

        boot_grand_means = np.array(boot_grand_means)
        grand_ci_lo = float(np.percentile(boot_grand_means, 2.5))
        grand_ci_hi = float(np.percentile(boot_grand_means, 97.5))

        # One-sided p-value: H0 is true mean <= 0.30 (metrics work).
        # p = fraction of bootstrap means that fall <= 0.30.
        h111_p = float(np.mean(boot_grand_means <= 0.30))
    else:
        grand_mean = None
        grand_ci_lo = None
        grand_ci_hi = None
        h111_p = None

    # H11.1 verdict (preliminary — Holm applied after H11.2)
    if grand_mean is not None:
        if grand_mean < 0.30:
            h111_verdict_pre_holm = "OVERTURNED"
        elif grand_mean >= 0.40:
            h111_verdict_pre_holm = "PENDING_HOLM"
        else:
            h111_verdict_pre_holm = "INDETERMINATE"
    else:
        h111_verdict_pre_holm = "N/A"

    h111_result = {
        "grand_mean_inversion_rate": grand_mean,
        "grand_ci_lo": grand_ci_lo,
        "grand_ci_hi": grand_ci_hi,
        "p_value_one_sided": h111_p,
        "null_hypothesis": "true mean inversion rate <= 0.30",
        "per_metric": per_metric_results,
        "verdict_pre_holm": h111_verdict_pre_holm,
    }
    print(f"\n  H11.1 Grand mean: {grand_mean:.3f} CI=[{grand_ci_lo:.3f}, {grand_ci_hi:.3f}], "
          f"p={h111_p:.4f}" if grand_mean is not None else "\n  H11.1: N/A")
    print(f"  H11.1 pre-Holm: {h111_verdict_pre_holm}")

    # --- H11.2: scIB composite anti-prediction ---
    print(f"\n{'-'*40}")
    print(f"{_ts()} H11.2: scIB composite anti-prediction")

    composites = []
    f1s = []
    condition_labels = []
    tissue_pair_indices = []

    for pair_idx, (src_tissue, tgt_tissue) in enumerate(CROSS_TISSUE_PAIRS):
        pair_id = f"{src_tissue}_to_{tgt_tissue}"
        for model in CONTENDERS:
            key = (pair_id, model)
            if key not in scib_scores or key not in probe_f1s:
                continue
            scores = scib_scores[key]
            f1 = probe_f1s[key]
            if f1 is None:
                continue
            bio_vals = [scores.get(m) for m in BIO_METRICS]
            bio_vals = [v for v in bio_vals if v is not None]
            if len(bio_vals) < 3:
                continue
            composite = float(np.mean(bio_vals))
            composites.append(composite)
            f1s.append(f1)
            condition_labels.append(f"{pair_id}/{model}")
            tissue_pair_indices.append(pair_idx)

    composites = np.array(composites)
    f1s = np.array(f1s)
    tissue_pair_indices = np.array(tissue_pair_indices)

    if len(composites) >= 4:
        rho, scipy_p = spearmanr(composites, f1s)
        perm_p = permutation_p_value(rho, composites, f1s)

        # Clustered bootstrap CI for rho — concatenate rows per sampled
        # tissue pair so duplicate draws are preserved (np.isin would
        # collapse them, understating variance).
        unique_pairs = np.unique(tissue_pair_indices)
        rows_by_pair = {p: np.where(tissue_pair_indices == p)[0] for p in unique_pairs}
        rng = np.random.default_rng(SEED + 22)
        boot_rhos = []
        for _ in range(N_BOOTSTRAP):
            sampled_pairs = rng.choice(unique_pairs, size=len(unique_pairs), replace=True)
            boot_idx = np.concatenate([rows_by_pair[p] for p in sampled_pairs])
            if len(boot_idx) < 4:
                continue
            r, _ = spearmanr(composites[boot_idx], f1s[boot_idx])
            if not np.isnan(r):
                boot_rhos.append(r)

        boot_rhos = np.array(boot_rhos)
        rho_ci_lo = float(np.percentile(boot_rhos, 2.5)) if len(boot_rhos) > 0 else None
        rho_ci_hi = float(np.percentile(boot_rhos, 97.5)) if len(boot_rhos) > 0 else None

        h112_result = {
            "spearman_rho": float(rho),
            "scipy_p": float(scipy_p),
            "permutation_p": float(perm_p),
            "ci_lo": rho_ci_lo,
            "ci_hi": rho_ci_hi,
            "n_conditions": len(composites),
            "verdict_pre_holm": "PENDING_HOLM",
        }
    else:
        perm_p = None
        h112_result = {
            "spearman_rho": None,
            "permutation_p": None,
            "n_conditions": len(composites),
            "verdict_pre_holm": "N/A (insufficient data)",
        }

    print(f"  rho={h112_result.get('spearman_rho')}, "
          f"perm_p={h112_result.get('permutation_p')}, "
          f"CI=[{h112_result.get('ci_lo')}, {h112_result.get('ci_hi')}]")
    print(f"  H11.2 pre-Holm: {h112_result['verdict_pre_holm']}")

    # --- Holm–Bonferroni correction over H11.1 and H11.2 ---
    print(f"\n{'-'*40}")
    print(f"{_ts()} Holm-Bonferroni correction (alpha=0.05, 2 tests)")

    holm_results = {}
    p_values = {}
    if h111_p is not None:
        p_values["H11.1"] = h111_p
    if perm_p is not None:
        p_values["H11.2"] = perm_p

    if len(p_values) == 2:
        sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
        k = len(sorted_tests)
        for rank, (test_name, p_val) in enumerate(sorted_tests):
            adjusted_alpha = 0.05 / (k - rank)
            rejected = p_val < adjusted_alpha
            holm_results[test_name] = {
                "raw_p": p_val,
                "holm_rank": rank + 1,
                "adjusted_alpha": adjusted_alpha,
                "rejected": rejected,
            }
            print(f"  Rank {rank+1}: {test_name}, p={p_val:.4f}, "
                  f"alpha={adjusted_alpha:.4f}, rejected={rejected}")

        # Apply Holm sequential logic: if rank-1 is not rejected, rank-2
        # cannot be rejected regardless of its p-value.
        first_test = sorted_tests[0][0]
        second_test = sorted_tests[1][0]
        if not holm_results[first_test]["rejected"]:
            holm_results[second_test]["rejected"] = False
            print(f"  (Holm sequential: {first_test} not rejected, "
                  f"so {second_test} also not rejected)")

        # Final verdicts
        if holm_results.get("H11.1", {}).get("rejected", False):
            if grand_mean >= 0.40:
                h111_verdict = "CONFIRMED"
            else:
                h111_verdict = "INDETERMINATE"
        elif h111_verdict_pre_holm == "OVERTURNED":
            h111_verdict = "OVERTURNED"
        else:
            h111_verdict = "INDETERMINATE"

        if holm_results.get("H11.2", {}).get("rejected", False):
            rho = h112_result.get("spearman_rho")
            if rho is not None and rho > 0:
                h112_verdict = "OVERTURNED"
            else:
                h112_verdict = "CONFIRMED"
        elif h112_result["verdict_pre_holm"] == "N/A (insufficient data)":
            h112_verdict = "N/A"
        else:
            # Holm did not reject — check direction
            rho = h112_result.get("spearman_rho")
            if rho is not None and rho <= 0:
                h112_verdict = "CONFIRMED"
            else:
                h112_verdict = "INDETERMINATE"
    elif len(p_values) == 1:
        test_name, p_val = list(p_values.items())[0]
        rejected = p_val < 0.05
        holm_results[test_name] = {
            "raw_p": p_val, "holm_rank": 1,
            "adjusted_alpha": 0.05, "rejected": rejected,
        }
        h111_verdict = h111_verdict_pre_holm if h111_verdict_pre_holm != "PENDING_HOLM" else "N/A"
        h112_verdict = h112_result["verdict_pre_holm"] if h112_result["verdict_pre_holm"] != "PENDING_HOLM" else "N/A"
    else:
        h111_verdict = "N/A"
        h112_verdict = "N/A"

    h111_result["verdict"] = h111_verdict
    h112_result["verdict"] = h112_verdict
    h111_result["holm"] = holm_results.get("H11.1")
    h112_result["holm"] = holm_results.get("H11.2")

    print(f"\n  H11.1 final verdict: {h111_verdict}")
    print(f"  H11.2 final verdict: {h112_verdict}")

    # --- H11.3: Inversion rate by F1-gap stratum [EXPLORATORY] ---
    print(f"\n{'-'*40}")
    print(f"{_ts()} H11.3: Inversion rate by F1-gap stratum [EXPLORATORY]")

    clear_gap_inversions = 0
    clear_gap_total = 0
    near_tie_inversions = 0
    near_tie_total = 0

    for src_tissue, tgt_tissue in CROSS_TISSUE_PAIRS:
        pair_id = f"{src_tissue}_to_{tgt_tissue}"
        for metric in BIO_METRICS:
            models = [m for m in CONTENDERS if (pair_id, m) in probe_f1s and (pair_id, m) in scib_scores]
            for m_a, m_b in itertools.combinations(models, 2):
                s_a = scib_scores[(pair_id, m_a)].get(metric)
                s_b = scib_scores[(pair_id, m_b)].get(metric)
                f_a = probe_f1s[(pair_id, m_a)]
                f_b = probe_f1s[(pair_id, m_b)]
                if any(v is None for v in [s_a, s_b, f_a, f_b]):
                    continue
                delta_f1 = abs(f_a - f_b)
                metric_prefers = 1 if s_a > s_b else (-1 if s_a < s_b else 0)
                f1_prefers = 1 if f_a > f_b else (-1 if f_a < f_b else 0)
                if metric_prefers == 0 or f1_prefers == 0:
                    continue
                inverted = metric_prefers != f1_prefers
                if delta_f1 > 0.10:
                    clear_gap_inversions += int(inverted)
                    clear_gap_total += 1
                else:
                    near_tie_inversions += int(inverted)
                    near_tie_total += 1

    h113_result = {
        "clear_gap_rate": clear_gap_inversions / clear_gap_total if clear_gap_total > 0 else None,
        "clear_gap_n": clear_gap_total,
        "near_tie_rate": near_tie_inversions / near_tie_total if near_tie_total > 0 else None,
        "near_tie_n": near_tie_total,
    }
    print(f"  Clear gap (|dF1|>0.10): {h113_result['clear_gap_rate']:.3f} "
          f"(n={h113_result['clear_gap_n']})"
          if h113_result["clear_gap_rate"] is not None else "  Clear gap: N/A")
    print(f"  Near tie (|dF1|<=0.10): {h113_result['near_tie_rate']:.3f} "
          f"(n={h113_result['near_tie_n']})"
          if h113_result["near_tie_rate"] is not None else "  Near tie: N/A")

    # Phase 4: Save results
    print(f"\n{'='*60}")
    print(f"{_ts()} Saving results...")

    # Build per-condition table
    conditions = []
    for src_tissue, tgt_tissue in CROSS_TISSUE_PAIRS:
        pair_id = f"{src_tissue}_to_{tgt_tissue}"
        for model in CONTENDERS:
            key = (pair_id, model)
            if key in scib_scores and key in probe_f1s:
                conditions.append({
                    "pair_id": pair_id,
                    "source_tissue": src_tissue,
                    "target_tissue": tgt_tissue,
                    "model": model,
                    "scib_scores": scib_scores[key],
                    "probe_f1_target": probe_f1s[key],
                })

    output = {
        "experiment": "exp11_cross_tissue_validity",
        "timestamp": timestamp,
        "prereg": "preregistration_v11_cross_tissue_ground_truth.md",
        "census_version": CENSUS_VERSION,
        "seed": SEED,
        "transfer_probe": "logistic_regression",
        "bog_pca_note": "combined source+target unsupervised PCA (transductive)",
        "n_conditions": len(conditions),
        "conditions": conditions,
        "pair_metadata": pair_metadata,
        "hypotheses": {
            "H11.1": h111_result,
            "H11.2": h112_result,
            "H11.3": h113_result,
        },
    }

    out_path = OUTPUT_DIR / "exp11_cross_tissue_validity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"{_ts()} Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"{_ts()} SUMMARY")
    print(f"  H11.1 (inversion rate >= 0.40): {h111_result['verdict']}")
    print(f"    Grand mean: {grand_mean}, CI: [{grand_ci_lo}, {grand_ci_hi}], p={h111_p}")
    if h111_result.get("holm"):
        print(f"    Holm: rank={h111_result['holm']['holm_rank']}, "
              f"adj_alpha={h111_result['holm']['adjusted_alpha']}, "
              f"rejected={h111_result['holm']['rejected']}")
    print(f"  H11.2 (composite anti-prediction): {h112_result['verdict']}")
    print(f"    rho: {h112_result.get('spearman_rho')}, p: {h112_result.get('permutation_p')}")
    if h112_result.get("holm"):
        print(f"    Holm: rank={h112_result['holm']['holm_rank']}, "
              f"adj_alpha={h112_result['holm']['adjusted_alpha']}, "
              f"rejected={h112_result['holm']['rejected']}")
    print(f"  H11.3 (F1-gap strata): clear={h113_result.get('clear_gap_rate')}, "
          f"near-tie={h113_result.get('near_tie_rate')}")
    print(f"\n{_ts()} Done.")


if __name__ == "__main__":
    main()
