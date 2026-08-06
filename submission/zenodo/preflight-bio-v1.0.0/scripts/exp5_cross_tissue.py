"""Experiment 5: Cross-Tissue Shift.

Tests whether the Preflight composite detects cross-tissue domain shifts
(same assay, different tissue). Six cross-tissue pairs + 3 negative controls.

Usage:
    python scripts/exp5_cross_tissue.py
"""
import json
import networkx as nx
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
EMBEDDING = "geneformer"
MAX_CELLS = 2000
ASSAY = "EFO:0009922"  # 10x 3' v3

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

TISSUES = {
    "lung":   "UBERON:0002048",
    "blood":  "UBERON:0000178",
    "brain":  "UBERON:0000955",
    "kidney": "UBERON:0002113",
    "liver":  "UBERON:0002107",
}

CROSS_TISSUE_PAIRS = [
    ("lung", "brain"),
    ("lung", "kidney"),
    ("blood", "lung"),
    ("blood", "brain"),
    ("liver", "kidney"),
    ("blood", "liver"),
]

NEGATIVE_CONTROLS = ["lung", "blood", "brain"]

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v7/exp5_cross_tissue.json")
OUTPUT_DIR = Path("results/cross_tissue")


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _make_filter(tissue_id):
    return (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{ASSAY}'"
    )


def _build_knn_graph(X_source, X_target, k=10, max_nodes=500):
    X = np.vstack([X_source, X_target])
    n = X.shape[0]
    if n > max_nodes:
        rng = np.random.default_rng(0)
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


def run_probes(X_src, labels_src, X_tgt, labels_tgt):
    """Run cell-type classification probes."""
    le = LabelEncoder()
    shared = set(labels_src) & set(labels_tgt)
    if len(shared) < 2:
        return {"f1_source": 0.0, "f1_target": 0.0, "n_shared": len(shared)}

    mask_src = np.isin(labels_src, list(shared))
    mask_tgt = np.isin(labels_tgt, list(shared))
    X_s = X_src[mask_src]
    X_t = X_tgt[mask_tgt]
    y_s = labels_src[mask_src]
    y_t = labels_tgt[mask_tgt]

    le.fit(list(shared))
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


def query_tissue(census, cellxgene_census, tissue_name, seed):
    """Pull embeddings for a tissue."""
    tissue_id = TISSUES[tissue_name]
    filter_str = _make_filter(tissue_id)

    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    joinids = obs_df["soma_joinid"].values
    if len(joinids) > MAX_CELLS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(joinids), size=MAX_CELLS, replace=False)
        idx.sort()
        joinids = joinids[idx]
    elif len(joinids) == 0:
        return None, None

    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=[EMBEDDING],
    )
    X = adata.obsm[EMBEDDING]
    labels = adata.obs["cell_type"].values
    return X, labels


def main():
    import cellxgene_census
    from preflight.core.runner import run

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{_ts()} Experiment 5: Cross-Tissue Shift")
    print(f"{_ts()} Census: {CENSUS_VERSION}, Assay: {ASSAY}")

    # Pull all tissue embeddings once
    tissue_data = {}
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue_name in tqdm(TISSUES.keys(), desc="Pulling tissues"):
            print(f"  {_ts()} Pulling {tissue_name}...")
            X, labels = query_tissue(census, cellxgene_census, tissue_name, seed=42)
            if X is not None:
                tissue_data[tissue_name] = {"X": X, "labels": labels}
                print(f"    {X.shape[0]} cells")
            else:
                print(f"    SKIPPED (no cells)")

        # Also pull second random split for negative controls
        neg_ctrl_data = {}
        for tissue_name in tqdm(NEGATIVE_CONTROLS, desc="Neg ctrl splits"):
            print(f"  {_ts()} Neg ctrl {tissue_name} (seed=99)...")
            X, labels = query_tissue(census, cellxgene_census, tissue_name, seed=99)
            if X is not None:
                neg_ctrl_data[tissue_name] = {"X": X, "labels": labels}

    results = []
    inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"

    # Run cross-tissue pairs
    print(f"\n{_ts()} Running cross-tissue pairs...")
    for src_tissue, tgt_tissue in tqdm(CROSS_TISSUE_PAIRS, desc="Cross-tissue"):
        if src_tissue not in tissue_data or tgt_tissue not in tissue_data:
            print(f"  SKIPPING {src_tissue} -> {tgt_tissue}: missing data")
            continue

        X_src = tissue_data[src_tissue]["X"]
        X_tgt = tissue_data[tgt_tissue]["X"]
        labels_src = tissue_data[src_tissue]["labels"]
        labels_tgt = tissue_data[tgt_tissue]["labels"]

        metadata = {
            "source_labels": labels_src,
            "target_labels": labels_tgt,
            "graph": _build_knn_graph(X_src, X_tgt),
        }

        report = run(source=X_src, target=X_tgt, metadata=metadata, hyperparameters={"k": 5})
        probe_results = run_probes(X_src, labels_src, X_tgt, labels_tgt)

        result = {
            "pair_id": f"{src_tissue}_to_{tgt_tissue}",
            "pair_type": "cross_tissue",
            "source_tissue": src_tissue,
            "target_tissue": tgt_tissue,
            "n_source": X_src.shape[0],
            "n_target": X_tgt.shape[0],
            "composite_score": report.overall_score,
            "composite_tier": report.overall_tier,
            "module_scores": {k: {"score": v.score, "tier": v.tier} for k, v in report.module_results.items()},
            "probe_f1_source": probe_results["f1_source"],
            "probe_f1_target": probe_results["f1_target"],
            "relative_degradation": (probe_results["f1_source"] - probe_results["f1_target"]) / max(probe_results["f1_source"], 0.01),
            "n_shared_cell_types": probe_results["n_shared"],
        }
        results.append(result)
        with open(inc_path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

        print(f"  {src_tissue}->{tgt_tissue}: Tier {report.overall_tier}, "
              f"F1 {probe_results['f1_source']:.3f}->{probe_results['f1_target']:.3f}")

    # Run negative controls
    print(f"\n{_ts()} Running negative controls...")
    for tissue_name in tqdm(NEGATIVE_CONTROLS, desc="Neg controls"):
        if tissue_name not in tissue_data or tissue_name not in neg_ctrl_data:
            continue

        X_src = tissue_data[tissue_name]["X"]
        X_tgt = neg_ctrl_data[tissue_name]["X"]
        labels_src = tissue_data[tissue_name]["labels"]
        labels_tgt = neg_ctrl_data[tissue_name]["labels"]

        metadata = {
            "source_labels": labels_src,
            "target_labels": labels_tgt,
            "graph": _build_knn_graph(X_src, X_tgt),
        }

        report = run(source=X_src, target=X_tgt, metadata=metadata, hyperparameters={"k": 5})
        probe_results = run_probes(X_src, labels_src, X_tgt, labels_tgt)

        result = {
            "pair_id": f"neg_ctrl_{tissue_name}",
            "pair_type": "negative_control",
            "source_tissue": tissue_name,
            "target_tissue": tissue_name,
            "n_source": X_src.shape[0],
            "n_target": X_tgt.shape[0],
            "composite_score": report.overall_score,
            "composite_tier": report.overall_tier,
            "module_scores": {k: {"score": v.score, "tier": v.tier} for k, v in report.module_results.items()},
            "probe_f1_source": probe_results["f1_source"],
            "probe_f1_target": probe_results["f1_target"],
            "relative_degradation": (probe_results["f1_source"] - probe_results["f1_target"]) / max(probe_results["f1_source"], 0.01),
            "n_shared_cell_types": probe_results["n_shared"],
        }
        results.append(result)
        with open(inc_path, "a") as f:
            f.write(json.dumps(result) + "\n")
            f.flush()

        print(f"  neg_ctrl_{tissue_name}: Tier {report.overall_tier}")

    # Hypothesis testing
    print(f"\n{'=' * 60}")
    print("HYPOTHESIS TESTING")

    cross_tissue = [r for r in results if r["pair_type"] == "cross_tissue"]
    neg_ctrls = [r for r in results if r["pair_type"] == "negative_control"]

    ct_tiers = [r["composite_tier"] for r in cross_tissue]
    nc_tiers = [r["composite_tier"] for r in neg_ctrls]

    h51 = np.mean(ct_tiers) <= 3
    print(f"\nH5.1: Cross-tissue mean tier <= 3: {np.mean(ct_tiers):.2f} -> {'SUPPORTED' if h51 else 'REJECTED'}")

    h52 = all(t >= 5 for t in nc_tiers)
    print(f"H5.2: Neg controls all Tier >= 5: {nc_tiers} -> {'SUPPORTED' if h52 else 'REJECTED'}")

    ct_m1 = [r["module_scores"].get("m1_grassmannian", {}).get("score", 0) for r in cross_tissue]
    # Compare to Exp 0 cross-assay M1 values (hardcoded from results)
    exp0_cross_assay_m1 = [0.309, 0.290, 0.217]  # lung, liver, brain from Exp 0
    h53 = np.mean(ct_m1) > np.mean(exp0_cross_assay_m1)
    print(f"H5.3: Cross-tissue M1 ({np.mean(ct_m1):.3f}) > cross-assay M1 ({np.mean(exp0_cross_assay_m1):.3f}): {'SUPPORTED' if h53 else 'REJECTED'}")

    blood_pairs = [r for r in cross_tissue if "blood" in [r["source_tissue"], r["target_tissue"]]]
    non_blood = [r for r in cross_tissue if "blood" not in [r["source_tissue"], r["target_tissue"]]]
    blood_m1 = np.mean([r["module_scores"].get("m1_grassmannian", {}).get("score", 0) for r in blood_pairs]) if blood_pairs else 0
    non_blood_m1 = np.mean([r["module_scores"].get("m1_grassmannian", {}).get("score", 0) for r in non_blood]) if non_blood else 0
    h54 = blood_m1 > non_blood_m1
    print(f"H5.4: Blood pairs M1 ({blood_m1:.3f}) > non-blood M1 ({non_blood_m1:.3f}): {'SUPPORTED' if h54 else 'REJECTED'}")

    degs = [r["relative_degradation"] for r in results if r["pair_type"] != "negative_control"]
    scores = [r["composite_score"] for r in results if r["pair_type"] != "negative_control"]
    if len(degs) >= 3:
        from scipy import stats as sp_stats
        rho, _ = sp_stats.spearmanr(scores, degs)
    else:
        rho = 0
    h55 = rho < -0.3
    print(f"H5.5: Spearman rho(composite, degradation) = {rho:.3f}: {'SUPPORTED' if h55 else 'REJECTED'}")

    # Save summary
    summary = {
        "experiment": "exp5_cross_tissue",
        "timestamp": timestamp,
        "prereg_sha": json.load(open(FROZEN_PREREG_PATH))["sha256"] if FROZEN_PREREG_PATH.exists() else "N/A",
        "census_version": CENSUS_VERSION,
        "results": results,
        "hypotheses": {
            "H5.1": {"supported": bool(h51), "mean_tier": float(np.mean(ct_tiers))},
            "H5.2": {"supported": bool(h52), "neg_ctrl_tiers": nc_tiers},
            "H5.3": {"supported": bool(h53), "cross_tissue_m1_mean": float(np.mean(ct_m1))},
            "H5.4": {"supported": bool(h54), "blood_m1": float(blood_m1), "non_blood_m1": float(non_blood_m1)},
            "H5.5": {"supported": bool(h55), "rho": float(rho)},
        },
    }

    output_path = OUTPUT_DIR / "summary_v7.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
