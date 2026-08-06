"""Experiment 4: Comparison to Existing Transferability Metrics.

Re-runs the same 10 pairs from Exp 0 and computes RankMe, MMD, C2ST
alongside the Preflight composite. Tests whether Preflight has a lower
false-certification rate than competing metrics.

Primary hypothesis (H4.1): Preflight has lower false-certification rate.
False certification = Tier >= 5 when relative_degradation > 0.30.

Usage:
    python scripts/exp4_metric_comparison.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
EMBEDDING = "geneformer"
MAX_CELLS = 2000

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

TISSUES = {
    "lung":   "UBERON:0002048",
    "heart":  "UBERON:0000948",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}

ASSAY_3P_V3 = "EFO:0009922"
ASSAY_5P_V2 = "EFO:0011025"

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v7/exp4_metric_comparison.json")
OUTPUT_DIR = Path("results/metric_comparison")
EXP0_RESULTS = Path("results/modal_results/composite_validation_v6/incremental.jsonl")


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _make_filter(tissue_id, assay_id):
    return (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{assay_id}'"
    )


def compute_rankme(X):
    """RankMe: effective rank via smoothed entropy of singular values."""
    sv = np.linalg.svd(X, compute_uv=False)
    sv = sv[sv > 1e-10]
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def compute_mmd(X_source, X_target, gamma=None):
    """MMD with Gaussian kernel (median heuristic bandwidth)."""
    if gamma is None:
        combined = np.vstack([X_source[:200], X_target[:200]])
        dists = cdist(combined, combined, metric="euclidean")
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * median_dist ** 2)

    def rbf_kernel(X, Y):
        dists_sq = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * dists_sq)

    n = X_source.shape[0]
    m = X_target.shape[0]
    K_ss = rbf_kernel(X_source, X_source)
    K_tt = rbf_kernel(X_target, X_target)
    K_st = rbf_kernel(X_source, X_target)
    mmd_sq = K_ss.sum() / (n * n) + K_tt.sum() / (m * m) - 2 * K_st.sum() / (n * m)
    return float(max(0, mmd_sq) ** 0.5)


def compute_c2st(X_source, X_target, n_splits=5):
    """C2ST: classifier two-sample test (logistic regression, 5-fold CV)."""
    X = np.vstack([X_source, X_target])
    y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_scaled[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf.predict(X_scaled[test_idx])))
    return float(np.mean(accs))


def rankme_shift(X_source, X_target):
    """RankMe difference as shift metric."""
    return abs(compute_rankme(X_source) - compute_rankme(X_target))


def define_pairs():
    """Same pair definitions as Exp 0."""
    from itertools import combinations
    pairs = []

    for tissue_name, tissue_id in TISSUES.items():
        pairs.append({
            "pair_id": f"cross_assay_{tissue_name}",
            "pair_type": "cross_assay",
            "tissue_source": tissue_name,
            "tissue_target": tissue_name,
            "source_filter": _make_filter(tissue_id, ASSAY_3P_V3),
            "target_filter": _make_filter(tissue_id, ASSAY_5P_V2),
            "source_seed": 42,
            "target_seed": 43,
        })

    tissue_names = sorted(TISSUES.keys())
    for t_a, t_b in combinations(tissue_names, 2):
        pairs.append({
            "pair_id": f"cross_tissue_{t_a}_to_{t_b}",
            "pair_type": "cross_tissue",
            "tissue_source": t_a,
            "tissue_target": t_b,
            "source_filter": _make_filter(TISSUES[t_a], ASSAY_3P_V3),
            "target_filter": _make_filter(TISSUES[t_b], ASSAY_3P_V3),
            "source_seed": 42,
            "target_seed": 43,
        })

    for tissue_name, tissue_id in TISSUES.items():
        pairs.append({
            "pair_id": f"neg_ctrl_{tissue_name}",
            "pair_type": "negative_control",
            "tissue_source": tissue_name,
            "tissue_target": tissue_name,
            "source_filter": _make_filter(tissue_id, ASSAY_3P_V3),
            "target_filter": _make_filter(tissue_id, ASSAY_3P_V3),
            "source_seed": 42,
            "target_seed": 99,
        })

    return pairs


def load_exp0_results():
    """Load the 10 pairs actually computed in Exp 0."""
    results = {}
    with open(EXP0_RESULTS) as f:
        for line in f:
            d = json.loads(line)
            results[d["pair_id"]] = d
    return results


def query_embeddings(census, cellxgene_census, filter_str, max_cells, seed):
    """Pull Geneformer embeddings from Census."""
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    joinids = obs_df["soma_joinid"].values
    if len(joinids) > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(joinids), size=max_cells, replace=False)
        idx.sort()
        joinids = joinids[idx]

    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=[EMBEDDING],
    )
    return adata.obsm[EMBEDDING]


def main():
    import cellxgene_census

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{_ts()} Experiment 4: Metric Comparison")
    print(f"{_ts()} Census: {CENSUS_VERSION}")

    exp0 = load_exp0_results()
    exp0_pair_ids = set(exp0.keys())
    print(f"{_ts()} Loaded {len(exp0)} Exp 0 pairs")

    all_pairs = define_pairs()
    pairs_to_run = [p for p in all_pairs if p["pair_id"] in exp0_pair_ids]
    print(f"{_ts()} Matching pairs: {len(pairs_to_run)}")

    results = []

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for pair in tqdm(pairs_to_run, desc="Pairs"):
            pair_id = pair["pair_id"]
            print(f"\n{_ts()} Processing: {pair_id}")

            X_src = query_embeddings(
                census, cellxgene_census,
                pair["source_filter"], MAX_CELLS, pair["source_seed"],
            )
            X_tgt = query_embeddings(
                census, cellxgene_census,
                pair["target_filter"], MAX_CELLS, pair["target_seed"],
            )
            print(f"  Source: {X_src.shape}, Target: {X_tgt.shape}")

            # Compute competing metrics
            rankme_diff = rankme_shift(X_src, X_tgt)
            mmd_val = compute_mmd(X_src, X_tgt)
            c2st_val = compute_c2st(X_src, X_tgt)

            # Get Preflight results from Exp 0
            exp0_pair = exp0[pair_id]
            composite_score = exp0_pair["composite_score"]
            composite_tier = exp0_pair["composite_tier"]
            relative_degradation = exp0_pair["relative_degradation"]

            result = {
                "pair_id": pair_id,
                "pair_type": pair["pair_type"],
                "relative_degradation": relative_degradation,
                "preflight_composite_score": composite_score,
                "preflight_tier": composite_tier,
                "rankme_diff": rankme_diff,
                "mmd": mmd_val,
                "c2st": c2st_val,
            }
            results.append(result)
            print(f"  Tier={composite_tier}, RankMe_diff={rankme_diff:.2f}, "
                  f"MMD={mmd_val:.4f}, C2ST={c2st_val:.3f}, "
                  f"rel_deg={relative_degradation:.3f}")

            # Save incrementally
            inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"
            with open(inc_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()

    # Hypothesis testing
    print(f"\n{'=' * 60}")
    print(f"HYPOTHESIS TESTING")
    print(f"{'=' * 60}")

    degradations = np.array([r["relative_degradation"] for r in results])
    composites = np.array([r["preflight_composite_score"] for r in results])
    tiers = np.array([r["preflight_tier"] for r in results])
    rankmes = np.array([r["rankme_diff"] for r in results])
    mmds = np.array([r["mmd"] for r in results])
    c2sts = np.array([r["c2st"] for r in results])

    # H4.1 (primary): False-certification rate comparison
    # False cert = Tier >= 5 (or metric equivalent) when degradation > 0.30
    deg_threshold = 0.30
    truly_degraded = degradations > deg_threshold

    # For competing metrics, "certification" means scoring below a threshold
    # that would indicate "safe to transfer". We use the negative-control median
    # as a natural threshold for each metric.
    neg_ctrl_mask = np.array([r["pair_type"] == "negative_control" for r in results])

    # Preflight false cert: Tier >= 5 on truly degraded pairs
    preflight_false_certs = np.sum((tiers >= 5) & truly_degraded)
    preflight_false_cert_rate = preflight_false_certs / max(1, truly_degraded.sum())

    # For RankMe/MMD/C2ST: "certifying" = below median of all values (lower = less shift)
    rankme_threshold = np.median(rankmes)
    mmd_threshold = np.median(mmds)
    c2st_threshold = np.median(c2sts)

    rankme_false_certs = np.sum((rankmes < rankme_threshold) & truly_degraded)
    mmd_false_certs = np.sum((mmds < mmd_threshold) & truly_degraded)
    c2st_false_certs = np.sum((c2sts < c2st_threshold) & truly_degraded)

    rankme_fc_rate = rankme_false_certs / max(1, truly_degraded.sum())
    mmd_fc_rate = mmd_false_certs / max(1, truly_degraded.sum())
    c2st_fc_rate = c2st_false_certs / max(1, truly_degraded.sum())

    print(f"\nH4.1 (PRIMARY): False-certification rate (degradation > {deg_threshold})")
    print(f"  Truly degraded pairs: {truly_degraded.sum()}")
    print(f"  Preflight (Tier >= 5): {preflight_false_certs}/{truly_degraded.sum()} = {preflight_false_cert_rate:.3f}")
    print(f"  RankMe (< median):    {rankme_false_certs}/{truly_degraded.sum()} = {rankme_fc_rate:.3f}")
    print(f"  MMD (< median):       {mmd_false_certs}/{truly_degraded.sum()} = {mmd_fc_rate:.3f}")
    print(f"  C2ST (< median):      {c2st_false_certs}/{truly_degraded.sum()} = {c2st_fc_rate:.3f}")

    h41_pass = preflight_false_cert_rate < min(rankme_fc_rate, mmd_fc_rate, c2st_fc_rate)
    print(f"  >>> H4.1: {'SUPPORTED' if h41_pass else 'REJECTED'}")

    # H4.2: Spearman rho with relative degradation
    shifted_mask = ~neg_ctrl_mask
    if shifted_mask.sum() >= 3:
        rho_composite, _ = stats.spearmanr(composites[shifted_mask], degradations[shifted_mask])
        rho_rankme, _ = stats.spearmanr(rankmes[shifted_mask], degradations[shifted_mask])
        rho_mmd, _ = stats.spearmanr(mmds[shifted_mask], degradations[shifted_mask])
        rho_c2st, _ = stats.spearmanr(c2sts[shifted_mask], degradations[shifted_mask])
    else:
        rho_composite = rho_rankme = rho_mmd = rho_c2st = 0.0

    print(f"\nH4.2: |Spearman rho| with relative degradation (shifted pairs only)")
    print(f"  Preflight: rho = {rho_composite:.3f} (|rho| = {abs(rho_composite):.3f})")
    print(f"  RankMe:    rho = {rho_rankme:.3f} (|rho| = {abs(rho_rankme):.3f})")
    print(f"  MMD:       rho = {rho_mmd:.3f} (|rho| = {abs(rho_mmd):.3f})")
    print(f"  C2ST:      rho = {rho_c2st:.3f} (|rho| = {abs(rho_c2st):.3f})")

    h42_pass = abs(rho_composite) > max(abs(rho_rankme), abs(rho_mmd), abs(rho_c2st))
    print(f"  >>> H4.2: {'SUPPORTED' if h42_pass else 'REJECTED'}")

    # H4.3: AUROC for shifted vs control classification
    labels = (~neg_ctrl_mask).astype(int)
    if labels.sum() > 0 and (1 - labels).sum() > 0:
        from sklearn.metrics import roc_auc_score
        auc_composite = roc_auc_score(labels, composites)
        auc_rankme = roc_auc_score(labels, rankmes)
        auc_mmd = roc_auc_score(labels, mmds)
        auc_c2st = roc_auc_score(labels, c2sts)
    else:
        auc_composite = auc_rankme = auc_mmd = auc_c2st = 0.5

    print(f"\nH4.3: AUROC for shifted-vs-control classification")
    print(f"  Preflight: {auc_composite:.3f}")
    print(f"  RankMe:    {auc_rankme:.3f}")
    print(f"  MMD:       {auc_mmd:.3f}")
    print(f"  C2ST:      {auc_c2st:.3f}")

    h43_pass = auc_composite > max(auc_rankme, auc_mmd, auc_c2st)
    print(f"  >>> H4.3: {'SUPPORTED' if h43_pass else 'REJECTED'}")

    # Save summary
    summary = {
        "experiment": "exp4_metric_comparison",
        "timestamp": timestamp,
        "prereg_sha": json.load(open(FROZEN_PREREG_PATH))["sha256"] if FROZEN_PREREG_PATH.exists() else "N/A",
        "census_version": CENSUS_VERSION,
        "n_pairs": len(results),
        "results": results,
        "hypotheses": {
            "H4.1_false_certification_rate": {
                "supported": bool(h41_pass),
                "preflight_rate": float(preflight_false_cert_rate),
                "rankme_rate": float(rankme_fc_rate),
                "mmd_rate": float(mmd_fc_rate),
                "c2st_rate": float(c2st_fc_rate),
                "n_truly_degraded": int(truly_degraded.sum()),
                "degradation_threshold": deg_threshold,
            },
            "H4.2_spearman_rho": {
                "supported": bool(h42_pass),
                "preflight_rho": float(rho_composite),
                "rankme_rho": float(rho_rankme),
                "mmd_rho": float(rho_mmd),
                "c2st_rho": float(rho_c2st),
            },
            "H4.3_auroc": {
                "supported": bool(h43_pass),
                "preflight_auroc": float(auc_composite),
                "rankme_auroc": float(auc_rankme),
                "mmd_auroc": float(auc_mmd),
                "c2st_auroc": float(auc_c2st),
            },
        },
    }

    output_path = OUTPUT_DIR / f"comparison_v7.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
