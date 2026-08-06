"""Spectral-gap discriminative statistic on V3b panel.

Prereg: PREREGISTRATION_SPECTRAL_GAP.md
Null distributions: results/spectral_gap/null_distributions.json

This script reuses the V3b pipeline (same Census extraction, same tissue
discovery, same embedding computation) and adds SGR computation on source
centroid matrices.

Output: results/spectral_gap/spectral_gap_results.json

Usage:
    python scripts/exp_spectral_gap_census.py
    python scripts/exp_spectral_gap_census.py --phase2b-dir results/embeddings
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713
MIN_CELLS = 200
MIN_SHARED_TYPES = 8
N_PERMUTATIONS = 10_000
BOOTSTRAP_RESAMPLES = 10_000

OUTPUT_DIR = Path("results/spectral_gap")

SOURCE_ASSAY = "EFO:0009922"  # 10x 3' v3
TARGET_ASSAY = "EFO:0008931"  # Smart-seq2

CENSUS_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]
PHASE2B_MODELS = ["geneformer_v2_104m", "geneformer_v2_316m"]
BASELINE_EMBEDDINGS = ["bog_pca_512", "random_projection", "untrained_encoder"]
CONTENDER_MODELS = ["geneformer", "geneformer_v2_104m", "geneformer_v2_316m",
                    "scgpt", "scvi", "bog_pca_512"]
BASELINE_MODEL_NAMES = ["random_projection", "untrained_encoder"]

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

J_VALUES = [1, 2, None]  # None = floor((k-1)/3)
NULL_TRIALS = 5000
NULL_SEED = 42


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def spectral_gap_ratio(C, j):
    """Explained variance ratio of top-j singular values of centered C."""
    Cc = C - C.mean(axis=0)
    sv = np.linalg.svd(Cc, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 0.0, sv.tolist()
    return float(sv2[:j].sum() / total), sv.tolist()


def compute_null_sgr(k, d, j_values):
    """Compute null SGR distribution for a (k,d) pair not in the frozen file."""
    rng = np.random.default_rng(NULL_SEED)
    results = {}
    for j_raw in j_values:
        j = j_raw if j_raw is not None else max(1, (k - 1) // 3)
        j = min(j, k - 1)
        vals = np.empty(NULL_TRIALS)
        for i in range(NULL_TRIALS):
            C = rng.standard_normal((k, d))
            sgr, _ = spectral_gap_ratio(C, j)
            vals[i] = sgr
        key = f"j={j_raw}" if j_raw is not None else f"j=k/3={j}"
        results[key] = {
            'j': int(j),
            'mean': float(vals.mean()),
            'std': float(vals.std()),
        }
    return results


def compute_source_centroids(X_src, y_src, shared_types):
    """Compute mean source embedding per shared cell type."""
    centroids = []
    for ct in shared_types:
        mask = y_src == ct
        if mask.sum() > 0:
            centroids.append(X_src[mask].mean(axis=0))
    return np.stack(centroids) if centroids else None


def compute_transfer_f1(X_src, y_src, X_tgt, y_tgt):
    """Logistic regression transfer F1 on shared cell types."""
    shared = sorted(set(y_src) & set(y_tgt))
    if len(shared) < MIN_SHARED_TYPES:
        return None
    le = LabelEncoder()
    le.fit(shared)
    mask_src = np.isin(y_src, shared)
    mask_tgt = np.isin(y_tgt, shared)
    X_s = X_src[mask_src]
    y_s = le.transform(y_src[mask_src])
    X_t = X_tgt[mask_tgt]
    y_t = le.transform(y_tgt[mask_tgt])
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_s, y_s)
    y_pred = clf.predict(X_t)
    return float(f1_score(y_t, y_pred, average='macro'))


def linear_cka(X, Y):
    """Linear CKA on centered matrices."""
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
    """CKA on cell-type centroid matrices."""
    shared = sorted(set(y_src) & set(y_tgt))
    if len(shared) < MIN_SHARED_TYPES:
        return None
    src_c = np.stack([X_src[y_src == ct].mean(0) for ct in shared])
    tgt_c = np.stack([X_tgt[y_tgt == ct].mean(0) for ct in shared])
    return linear_cka(src_c, tgt_c)


def random_projection(X_raw, d_out=512):
    rng = np.random.default_rng(SEED)
    G = X_raw.shape[1]
    R = rng.standard_normal((G, d_out)) / np.sqrt(d_out)
    return np.log1p(np.abs(X_raw)) @ R


def untrained_encoder(X_raw, d_out=512, d_hidden=256):
    rng = np.random.default_rng(SEED)
    G = X_raw.shape[1]
    W1 = rng.standard_normal((G, d_hidden)) / np.sqrt(G)
    b1 = np.zeros(d_hidden)
    W2 = rng.standard_normal((d_hidden, d_out)) / np.sqrt(d_hidden)
    b2 = np.zeros(d_out)
    h = np.maximum(0, np.log1p(np.abs(X_raw)) @ W1 + b1)
    return h @ W2 + b2


def bag_of_genes_pca_combined(X_src_raw, X_tgt_raw, d_out=512):
    X_src_log = np.log1p(np.abs(X_src_raw))
    X_tgt_log = np.log1p(np.abs(X_tgt_raw))
    X_combined = np.vstack([X_src_log, X_tgt_log])
    pca = PCA(n_components=d_out, random_state=SEED)
    X_combined_pca = pca.fit_transform(X_combined)
    n_src = X_src_raw.shape[0]
    return X_combined_pca[:n_src], X_combined_pca[n_src:]


def partial_spearman(x, y, z):
    """Partial Spearman of x,y controlling for z via rank residuals."""
    rx = scipy_stats.rankdata(x)
    ry = scipy_stats.rankdata(y)
    rz = scipy_stats.rankdata(z)
    cx = np.polyfit(rz, rx, 1)
    cy = np.polyfit(rz, ry, 1)
    rx_resid = rx - np.polyval(cx, rz)
    ry_resid = ry - np.polyval(cy, rz)
    return scipy_stats.spearmanr(rx_resid, ry_resid)


def tissue_stratified_permutation(x, y, tissue_ids, z, n_perm=N_PERMUTATIONS, seed=42):
    """Tissue-stratified permutation p-value for partial Spearman."""
    rng = np.random.default_rng(seed)
    obs_rho, _ = partial_spearman(x, y, z)
    count = 0
    tissues_unique = list(set(tissue_ids))
    y_arr = np.asarray(y, dtype=float)
    for _ in range(n_perm):
        y_perm = y_arr.copy()
        for t in tissues_unique:
            mask = np.array([tid == t for tid in tissue_ids])
            y_perm[mask] = rng.permutation(y_perm[mask])
        perm_rho, _ = partial_spearman(x, y_perm, z)
        if abs(perm_rho) >= abs(obs_rho):
            count += 1
    return obs_rho, count / n_perm


def discover_tissues(census):
    """Auto-discover tissues with >= MIN_CELLS per assay, >= MIN_SHARED_TYPES."""
    import cellxgene_census
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        column_names=["tissue", "tissue_ontology_term_id", "assay_ontology_term_id",
                      "cell_type", "is_primary_data", "disease"],
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


def load_census_pair(census, tissue_id, tissue_name):
    """Load source/target adata pair from Census (matching V3b pipeline)."""
    import cellxgene_census

    results = {}
    for side, assay_id, seed_offset in [("source", SOURCE_ASSAY, 0), ("target", TARGET_ASSAY, 1)]:
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
        print(f"  {_ts()} {tissue_name} {side}: {len(all_ids)} cells available")
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
              f"{len(adata.obs['cell_type'].unique())} cell types")
        results[side] = adata

    src_adata = results["source"]
    tgt_adata = results["target"]

    shared = set(src_adata.obs['cell_type'].values) & set(tgt_adata.obs['cell_type'].values)
    if len(shared) < MIN_SHARED_TYPES:
        print(f"  {_ts()} Skipping {tissue_name}: only {len(shared)} shared types")
        return None

    print(f"  {_ts()} {tissue_name}: src={len(src_adata)}, tgt={len(tgt_adata)}, shared={len(shared)}")
    return results


def load_phase2b_embedding(model_name, tissue_name, side, phase2b_dir):
    """Load pre-computed phase2b embedding from disk."""
    p = Path(phase2b_dir) / f"{model_name}_{tissue_name}_{side}.npy"
    if p.exists():
        return np.load(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2b-dir", type=str, default="results/embeddings")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} === Spectral-Gap Discriminative Statistic ===")
    print(f"{_ts()} Prereg: PREREGISTRATION_SPECTRAL_GAP.md")

    null_path = OUTPUT_DIR / "null_distributions.json"
    if not null_path.exists():
        print(f"ERROR: null distributions not found at {null_path}")
        print("Run compute_sgr_null.py first.")
        return
    with open(null_path) as f:
        null_data = json.load(f)
    null_dists = null_data['null_distributions']
    print(f"{_ts()} Loaded null distributions for {null_data['n_kd_pairs']} (k, d) pairs")

    import cellxgene_census
    from scipy import sparse as sp

    all_results = []

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        tissues = discover_tissues(census)
        tissue_list = list(tissues.items())
        print(f"\n{_ts()} Running on {len(tissue_list)} tissues")

        for tissue_name, tissue_id in tqdm(tissue_list, desc="Tissues"):
            print(f"\n{_ts()} === {tissue_name} ===")

            pair = load_census_pair(census, tissue_id, tissue_name)
            if pair is None:
                continue
            src_adata = pair["source"]
            tgt_adata = pair["target"]

            src_labels = src_adata.obs["cell_type"].values.astype(str)
            tgt_labels = tgt_adata.obs["cell_type"].values.astype(str)

            X_src_raw = src_adata.X.toarray() if sp.issparse(src_adata.X) else np.asarray(src_adata.X)
            X_tgt_raw = tgt_adata.X.toarray() if sp.issparse(tgt_adata.X) else np.asarray(tgt_adata.X)

            models_to_run = {}

            for emb_name in CENSUS_EMBEDDINGS:
                if emb_name in src_adata.obsm and emb_name in tgt_adata.obsm:
                    src_emb = np.asarray(src_adata.obsm[emb_name])
                    tgt_emb = np.asarray(tgt_adata.obsm[emb_name])
                    if np.all(np.isfinite(src_emb)) and np.all(np.isfinite(tgt_emb)):
                        models_to_run[emb_name] = {
                            "source": src_emb, "target": tgt_emb, "d": src_emb.shape[1],
                        }

            bog_src, bog_tgt = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            models_to_run["bog_pca_512"] = {"source": bog_src, "target": bog_tgt, "d": 512}

            rp_src = random_projection(X_src_raw)
            rp_tgt = random_projection(X_tgt_raw)
            models_to_run["random_projection"] = {"source": rp_src, "target": rp_tgt, "d": 512}

            ue_src = untrained_encoder(X_src_raw)
            ue_tgt = untrained_encoder(X_tgt_raw)
            models_to_run["untrained_encoder"] = {"source": ue_src, "target": ue_tgt, "d": 512}

            for model_name in PHASE2B_MODELS:
                src_emb = load_phase2b_embedding(model_name, tissue_name, "source", args.phase2b_dir)
                tgt_emb = load_phase2b_embedding(model_name, tissue_name, "target", args.phase2b_dir)
                if src_emb is not None and tgt_emb is not None:
                    n_src = min(src_emb.shape[0], len(src_labels))
                    n_tgt = min(tgt_emb.shape[0], len(tgt_labels))
                    models_to_run[model_name] = {
                        "source": src_emb[:n_src], "target": tgt_emb[:n_tgt],
                        "d": src_emb.shape[1],
                    }

            for model_name, emb_data in tqdm(models_to_run.items(), desc=f"  {tissue_name}"):
                X_src = emb_data["source"]
                X_tgt = emb_data["target"]
                d = emb_data["d"]

                n_src = min(X_src.shape[0], len(src_labels))
                n_tgt = min(X_tgt.shape[0], len(tgt_labels))
                X_src = X_src[:n_src]
                X_tgt = X_tgt[:n_tgt]
                y_src = src_labels[:n_src]
                y_tgt = tgt_labels[:n_tgt]

                shared = sorted(set(y_src) & set(y_tgt))
                if len(shared) < MIN_SHARED_TYPES:
                    continue

                k = len(shared)
                print(f"  {_ts()} {model_name} (d={d}, k={k})")

                src_centroids = compute_source_centroids(X_src, y_src, shared)
                if src_centroids is None:
                    continue

                transfer_f1 = compute_transfer_f1(X_src, y_src, X_tgt, y_tgt)
                cka_val = cell_type_cka(X_src, y_src, X_tgt, y_tgt)

                null_key = f"k={k}_d={d}"
                null_entry = null_dists.get(null_key)
                if null_entry is None:
                    print(f"    {_ts()} Computing null for {null_key} (not in frozen file)...")
                    null_entry = compute_null_sgr(k, d, J_VALUES)
                    null_dists[null_key] = null_entry

                sgr_results = {}
                for j_spec in J_VALUES:
                    j = j_spec if j_spec is not None else max(1, (k - 1) // 3)
                    j = min(j, k - 1)
                    sgr_val, sv_list = spectral_gap_ratio(src_centroids, j)

                    z_score = None
                    if null_entry:
                        j_key = f"j={j_spec}" if j_spec is not None else f"j=k/3={j}"
                        if j_key in null_entry:
                            null_mean = null_entry[j_key]['mean']
                            null_std = null_entry[j_key]['std']
                            if null_std > 1e-12:
                                z_score = (sgr_val - null_mean) / null_std

                    label = f"j={j_spec}" if j_spec is not None else "j=k/3"
                    sgr_results[label] = {
                        'j': j,
                        'sgr': sgr_val,
                        'z_score': z_score,
                    }

                result = {
                    'tissue': tissue_name,
                    'model': model_name,
                    'd': d,
                    'k': k,
                    'transfer_f1': transfer_f1,
                    'cell_type_cka': cka_val,
                    'singular_values': sv_list,
                    'sgr': sgr_results,
                    'is_contender': model_name in CONTENDER_MODELS,
                }
                all_results.append(result)
                z1 = sgr_results['j=1']['z_score']
                z_str = f"{z1:.2f}" if z1 is not None else "N/A"
                f1_str = f"{transfer_f1:.4f}" if transfer_f1 is not None else "N/A"
                print(f"    SGR1={sgr_results['j=1']['sgr']:.4f}  z={z_str}  F1={f1_str}")

    # Save raw results
    raw_output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'prereg': 'PREREGISTRATION_SPECTRAL_GAP.md',
        'census_version': CENSUS_VERSION,
        'seed': SEED,
        'n_conditions': len(all_results),
        'conditions': all_results,
    }
    raw_path = OUTPUT_DIR / "spectral_gap_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(raw_output, f, indent=2, default=str)
    print(f"\n{_ts()} Saved raw results to {raw_path}")

    # Analysis
    print(f"\n{_ts()} === Analysis ===")
    contenders = [r for r in all_results if r['is_contender'] and r['transfer_f1'] is not None]
    baselines = [r for r in all_results if not r['is_contender'] and r['transfer_f1'] is not None]

    print(f"  Contenders: {len(contenders)}, Baselines: {len(baselines)}")

    # HS1: Discriminative validity
    hs1_pass = None
    p_mw = None
    print(f"\n  --- HS1: Discriminative validity ---")
    cont_z1 = [r['sgr']['j=1']['z_score'] for r in contenders if r['sgr']['j=1']['z_score'] is not None]
    base_z1 = [r['sgr']['j=1']['z_score'] for r in baselines if r['sgr']['j=1']['z_score'] is not None]
    if cont_z1 and base_z1:
        U, p_mw = scipy_stats.mannwhitneyu(cont_z1, base_z1, alternative='greater')
        print(f"  Contender z1: mean={np.mean(cont_z1):.3f}, median={np.median(cont_z1):.3f}")
        print(f"  Baseline z1:  mean={np.mean(base_z1):.3f}, median={np.median(base_z1):.3f}")
        print(f"  Mann-Whitney U={U:.0f}, p={p_mw:.6f} (one-sided)")
        hs1_pass = p_mw < 0.05
        print(f"  HS1 {'PASS' if hs1_pass else 'FAIL'}")
    else:
        print(f"  Insufficient data: {len(cont_z1)} contenders, {len(base_z1)} baselines")

    # HS2: Predictive validity
    hs2_pass = None
    obs_rho = None
    perm_p = None
    print(f"\n  --- HS2: Predictive validity ---")
    z1_vals = np.array([r['sgr']['j=1']['z_score'] for r in contenders if r['sgr']['j=1']['z_score'] is not None])
    f1_vals = np.array([r['transfer_f1'] for r in contenders if r['sgr']['j=1']['z_score'] is not None])
    d_vals = np.array([r['d'] for r in contenders if r['sgr']['j=1']['z_score'] is not None])
    tissue_ids = [r['tissue'] for r in contenders if r['sgr']['j=1']['z_score'] is not None]

    if len(z1_vals) >= 10:
        obs_rho, perm_p = tissue_stratified_permutation(z1_vals, f1_vals, tissue_ids, d_vals)
        print(f"  Partial rho (d): {obs_rho:.4f}")
        print(f"  Permutation p:   {perm_p:.6f}")
        hs2_pass = obs_rho > 0.20 and perm_p < 0.05
        print(f"  HS2 {'PASS' if hs2_pass else 'FAIL'} (threshold: rho > 0.20, p < 0.05)")
    else:
        print(f"  Insufficient data: {len(z1_vals)} conditions")

    # HS3: Incremental over CKA (conditional on HS2 pass)
    hs3_rho = None
    hs3_p = None
    cka_vals = np.array([r['cell_type_cka'] for r in contenders if r['sgr']['j=1']['z_score'] is not None and r['cell_type_cka'] is not None])
    if len(cka_vals) == len(z1_vals) and len(z1_vals) >= 10:
        rz1 = scipy_stats.rankdata(z1_vals)
        rf1 = scipy_stats.rankdata(f1_vals)
        rd = scipy_stats.rankdata(d_vals)
        rcka = scipy_stats.rankdata(cka_vals)
        covariates = np.column_stack([rd, rcka])
        beta_z1, _, _, _ = np.linalg.lstsq(np.column_stack([covariates, np.ones(len(rz1))]), rz1, rcond=None)
        beta_f1, _, _, _ = np.linalg.lstsq(np.column_stack([covariates, np.ones(len(rf1))]), rf1, rcond=None)
        resid_z1 = rz1 - covariates @ beta_z1[:2] - beta_z1[2]
        resid_f1 = rf1 - covariates @ beta_f1[:2] - beta_f1[2]
        hs3_rho, hs3_p = scipy_stats.spearmanr(resid_z1, resid_f1)
        print(f"\n  --- HS3: Incremental over CKA ---")
        print(f"  Partial rho (d, CKA): {hs3_rho:.4f}, p={hs3_p:.6f}")

    # Summary
    summary = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'prereg': 'PREREGISTRATION_SPECTRAL_GAP.md',
        'n_contenders': len(contenders),
        'n_baselines': len(baselines),
        'hs1_discriminative': {
            'contender_z1_mean': float(np.mean(cont_z1)) if cont_z1 else None,
            'baseline_z1_mean': float(np.mean(base_z1)) if base_z1 else None,
            'mann_whitney_p': float(p_mw) if p_mw is not None else None,
            'pass': bool(hs1_pass) if hs1_pass is not None else None,
        },
        'hs2_predictive': {
            'partial_rho_d': float(obs_rho) if obs_rho is not None else None,
            'perm_p': float(perm_p) if perm_p is not None else None,
            'pass': bool(hs2_pass) if hs2_pass is not None else None,
        },
        'hs3_incremental': {
            'partial_rho_d_cka': float(hs3_rho) if hs3_rho is not None else None,
            'p': float(hs3_p) if hs3_p is not None else None,
        },
    }
    sum_path = OUTPUT_DIR / "spectral_gap_summary.json"
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n{_ts()} Saved summary to {sum_path}")
    print(f"{_ts()} Done.")


if __name__ == '__main__':
    main()
