"""Experiment 6: Cross-Disease Shift (EXPLORATORY).

Tests whether the Preflight composite detects disease-induced domain shifts.
Four tissue x disease pairs + 2 negative controls.

Status: EXPLORATORY — underpowered at n=4. H6.1/H6.5 confirmatory,
H6.2-H6.4 reported as directional observations only.

Usage:
    python scripts/exp6_cross_disease.py
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
ASSAY = "EFO:0009922"
MIN_CELLS = 500

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]

DISEASE_PAIRS = [
    {"tissue": "lung", "tissue_id": "UBERON:0002048", "disease": "pulmonary fibrosis", "label": "IPF"},
    {"tissue": "lung", "tissue_id": "UBERON:0002048", "disease": "lung adenocarcinoma", "label": "Cancer"},
    {"tissue": "brain", "tissue_id": "UBERON:0000955", "disease": "Alzheimer disease", "label": "AD"},
    {"tissue": "liver", "tissue_id": "UBERON:0002107", "disease": "hepatocellular carcinoma", "label": "HCC"},
]

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v7/exp6_cross_disease_exploratory.json")
OUTPUT_DIR = Path("results/cross_disease")


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


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


def query_cells(census, cellxgene_census, tissue_id, disease_filter, seed):
    """Pull embeddings with disease filter."""
    filter_str = (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{ASSAY}' "
        f"and {disease_filter}"
    )

    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    n_total = len(obs_df)
    if n_total == 0:
        return None, None, 0

    joinids = obs_df["soma_joinid"].values
    if n_total > MAX_CELLS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=MAX_CELLS, replace=False)
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
    X = adata.obsm[EMBEDDING]
    labels = adata.obs["cell_type"].values
    return X, labels, n_total


def main():
    import cellxgene_census
    from preflight.core.runner import run

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"{_ts()} Experiment 6: Cross-Disease Shift (EXPLORATORY)")
    print(f"{_ts()} Census: {CENSUS_VERSION}")

    results = []
    excluded = []
    inc_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for pair in tqdm(DISEASE_PAIRS, desc="Disease pairs"):
            tissue = pair["tissue"]
            tissue_id = pair["tissue_id"]
            disease = pair["disease"]
            label = pair["label"]

            print(f"\n{_ts()} {tissue} / {disease} ({label})")

            # Source: healthy
            print(f"  Pulling healthy {tissue}...")
            X_src, labels_src, n_src_total = query_cells(
                census, cellxgene_census, tissue_id,
                "disease == 'normal'", seed=42,
            )

            # Target: diseased
            print(f"  Pulling {disease}...")
            X_tgt, labels_tgt, n_tgt_total = query_cells(
                census, cellxgene_census, tissue_id,
                f"disease == '{disease}'", seed=43,
            )

            if X_src is None or X_tgt is None or n_tgt_total < MIN_CELLS:
                print(f"  EXCLUDED: src={n_src_total if X_src is not None else 0}, "
                      f"tgt={n_tgt_total} (min={MIN_CELLS})")
                excluded.append({"pair": f"{tissue}/{disease}", "reason": f"n_target={n_tgt_total}"})
                continue

            print(f"  Source: {X_src.shape[0]} cells, Target: {X_tgt.shape[0]} cells")

            metadata = {
                "source_labels": labels_src,
                "target_labels": labels_tgt,
                "graph": _build_knn_graph(X_src, X_tgt),
            }

            report = run(source=X_src, target=X_tgt, metadata=metadata, hyperparameters={"k": 5})
            probe_results = run_probes(X_src, labels_src, X_tgt, labels_tgt)

            result = {
                "pair_id": f"{tissue}_{label}",
                "pair_type": "cross_disease",
                "tissue": tissue,
                "disease": disease,
                "disease_label": label,
                "is_cancer": label in ["Cancer", "HCC"],
                "n_source": X_src.shape[0],
                "n_target": X_tgt.shape[0],
                "n_target_total_available": n_tgt_total,
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

            print(f"  Tier {report.overall_tier}, score={report.overall_score:.3f}, "
                  f"F1 {probe_results['f1_source']:.3f}->{probe_results['f1_target']:.3f}")

        # Negative controls (same tissue, both normal, random split)
        print(f"\n{_ts()} Negative controls...")
        for tissue_name in ["lung", "brain"]:
            tissue_id = {"lung": "UBERON:0002048", "brain": "UBERON:0000955"}[tissue_name]
            X_src, labels_src, _ = query_cells(
                census, cellxgene_census, tissue_id,
                "disease == 'normal'", seed=42,
            )
            X_tgt, labels_tgt, _ = query_cells(
                census, cellxgene_census, tissue_id,
                "disease == 'normal'", seed=99,
            )
            if X_src is None or X_tgt is None:
                continue

            metadata = {
                "source_labels": labels_src,
                "target_labels": labels_tgt,
                "graph": _build_knn_graph(X_src, X_tgt),
            }
            report = run(source=X_src, target=X_tgt, metadata=metadata, hyperparameters={"k": 5})

            result = {
                "pair_id": f"neg_ctrl_{tissue_name}",
                "pair_type": "negative_control",
                "tissue": tissue_name,
                "disease": "normal",
                "composite_score": report.overall_score,
                "composite_tier": report.overall_tier,
                "module_scores": {k: {"score": v.score, "tier": v.tier} for k, v in report.module_results.items()},
            }
            results.append(result)
            with open(inc_path, "a") as f:
                f.write(json.dumps(result) + "\n")
                f.flush()
            print(f"  neg_ctrl_{tissue_name}: Tier {report.overall_tier}")

    # Reporting
    print(f"\n{'=' * 60}")
    print("RESULTS (EXPLORATORY)")

    disease_results = [r for r in results if r["pair_type"] == "cross_disease"]
    neg_results = [r for r in results if r["pair_type"] == "negative_control"]

    if len(disease_results) < 3:
        print(f"\nUNDERPOWERED: only {len(disease_results)} disease pairs survived exclusions (need >= 3)")
        print(f"Excluded: {excluded}")
        print("No hypothesis testing performed.")
    else:
        disease_tiers = [r["composite_tier"] for r in disease_results]
        neg_tiers = [r["composite_tier"] for r in neg_results]

        h61 = np.mean(disease_tiers) <= 4
        print(f"\nH6.1: Disease pairs mean tier <= 4: {np.mean(disease_tiers):.2f} -> {'SUPPORTED' if h61 else 'REJECTED'}")

        h65 = all(t >= 5 for t in neg_tiers) if neg_tiers else False
        print(f"H6.5: Neg controls Tier >= 5: {neg_tiers} -> {'SUPPORTED' if h65 else 'REJECTED'}")

        # Observations (not hypothesis tests)
        cancer = [r for r in disease_results if r.get("is_cancer")]
        non_cancer = [r for r in disease_results if not r.get("is_cancer")]
        if cancer and non_cancer:
            cancer_scores = np.mean([r["composite_score"] for r in cancer])
            non_cancer_scores = np.mean([r["composite_score"] for r in non_cancer])
            print(f"\nH6.2 (observation): Cancer shift={cancer_scores:.3f}, "
                  f"non-cancer shift={non_cancer_scores:.3f}, "
                  f"delta={cancer_scores - non_cancer_scores:.3f}")

    # Save
    summary = {
        "experiment": "exp6_cross_disease_exploratory",
        "timestamp": timestamp,
        "status": "EXPLORATORY",
        "prereg_sha": json.load(open(FROZEN_PREREG_PATH))["sha256"] if FROZEN_PREREG_PATH.exists() else "N/A",
        "census_version": CENSUS_VERSION,
        "n_pairs_attempted": len(DISEASE_PAIRS),
        "n_pairs_completed": len(disease_results),
        "excluded": excluded,
        "results": results,
    }

    output_path = OUTPUT_DIR / "summary_v7.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
