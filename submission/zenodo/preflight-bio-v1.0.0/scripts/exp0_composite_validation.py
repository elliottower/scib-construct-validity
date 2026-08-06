"""Experiment 0: Composite Validation — does the tier predict real degradation?

Keystone experiment from preregistration_v4. Tests whether Preflight's
composite transportability score predicts actual downstream task degradation
across ~25 source->target pairs spanning four pair types:
  - Cross-assay, same tissue (5 pairs)
  - Cross-tissue, same assay (10 pairs)
  - Cross-dataset, same tissue+assay (5 pairs)
  - Negative control / random split (5 pairs)

Hypotheses (from preregistration_v4):
  H0.1: partial Spearman rho(composite, relative_degradation | F1_source) < -0.4 on shifted pairs
  H0.1b: raw rho across all ~25 pairs < -0.5
  H0.1c: rho within cross-tissue group alone < -0.3
  H0.2: Tier >= 5 pairs have mean relative_degradation < 0.15
  H0.3: Tier <= 2 pairs have mean relative_degradation > 0.40
  H0.4: M2 alone partial rho < -0.3 on shifted pairs
  H0.5: composite |partial rho| > M2 |partial rho| on shifted pairs

Usage (from the .venv-census environment):
    .venv-census/bin/python scripts/exp0_composite_validation.py
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
EMBEDDING = "geneformer"
MAX_CELLS = 2000
MIN_SHARED_CELL_TYPES = 3
MIN_CELLS_PER_TYPE = 10

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

ASSAY_3P_V3 = "EFO:0009922"   # 10x Chromium 3' v3
ASSAY_5P_V2 = "EFO:0011025"   # 10x Chromium 5' v2

OUTPUT_DIR = Path("results/composite_validation")
FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v6/exp0_composite_validation.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------

def _make_filter(tissue_id, assay_id):
    return (
        f"tissue_ontology_term_id == '{tissue_id}' "
        f"and is_primary_data == True "
        f"and assay_ontology_term_id == '{assay_id}'"
    )


def define_pairs():
    """Return list of pair dicts describing all ~25 source->target pairs."""
    pairs = []

    # Cross-assay: same tissue, 10x 3'v3 -> 10x 5'v2 (5 pairs)
    for tissue_name, tissue_id in TISSUES.items():
        pairs.append({
            "pair_id": f"cross_assay_{tissue_name}",
            "pair_type": "cross_assay",
            "tissue_source": tissue_name,
            "tissue_target": tissue_name,
            "source_filter": _make_filter(tissue_id, ASSAY_3P_V3),
            "target_filter": _make_filter(tissue_id, ASSAY_5P_V2),
            "source_desc": f"{tissue_name} 10x 3'v3",
            "target_desc": f"{tissue_name} 10x 5'v2",
            "source_seed": 42,
            "target_seed": 43,
        })

    # Cross-tissue: all C(5,2) pairs, same assay (10x 3'v3) (10 pairs)
    tissue_names = sorted(TISSUES.keys())
    for t_a, t_b in combinations(tissue_names, 2):
        pairs.append({
            "pair_id": f"cross_tissue_{t_a}_to_{t_b}",
            "pair_type": "cross_tissue",
            "tissue_source": t_a,
            "tissue_target": t_b,
            "source_filter": _make_filter(TISSUES[t_a], ASSAY_3P_V3),
            "target_filter": _make_filter(TISSUES[t_b], ASSAY_3P_V3),
            "source_desc": f"{t_a} 10x 3'v3",
            "target_desc": f"{t_b} 10x 3'v3",
            "source_seed": 42,
            "target_seed": 42,
        })

    # Cross-dataset: same tissue+assay, different dataset_ids (5 pairs)
    # These are discovered dynamically from Census in _resolve_cross_dataset_pairs
    for tissue_name, tissue_id in TISSUES.items():
        pairs.append({
            "pair_id": f"cross_dataset_{tissue_name}",
            "pair_type": "cross_dataset",
            "tissue_source": tissue_name,
            "tissue_target": tissue_name,
            "shared_filter": _make_filter(tissue_id, ASSAY_3P_V3),
            "source_filter": None,  # resolved dynamically
            "target_filter": None,
            "source_desc": f"{tissue_name} 10x 3'v3 dataset A",
            "target_desc": f"{tissue_name} 10x 3'v3 dataset B",
            "source_seed": 42,
            "target_seed": 43,
        })

    # Negative control: same pool, random split (5 pairs)
    for tissue_name, tissue_id in TISSUES.items():
        shared = _make_filter(tissue_id, ASSAY_3P_V3)
        pairs.append({
            "pair_id": f"neg_ctrl_{tissue_name}",
            "pair_type": "negative_control",
            "tissue_source": tissue_name,
            "tissue_target": tissue_name,
            "source_filter": shared,
            "target_filter": shared,
            "source_desc": f"{tissue_name} 10x 3'v3 split A",
            "target_desc": f"{tissue_name} 10x 3'v3 split B",
            "source_seed": 42,
            "target_seed": 99,
        })

    return pairs


# ---------------------------------------------------------------------------
# Census query helpers (same pattern as census_pilot_fast.py)
# ---------------------------------------------------------------------------

def _query_slice(census, cellxgene_census, filter_str, max_cells, seed):
    """Query cell IDs first (fast), subsample, then pull only those cells."""
    log.info("  Querying IDs: %s...", filter_str[:80])
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=filter_str,
        column_names=["soma_joinid"],
    )
    n_total = len(obs_df)
    log.info("  Total matching cells: %d", n_total)

    if n_total == 0:
        return None

    joinids = obs_df["soma_joinid"].values
    if n_total > max_cells:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_cells, replace=False)
        idx.sort()
        joinids = joinids[idx]
        log.info("  Subsampled to: %d cells", len(joinids))

    log.info("  Pulling embeddings for %d cells...", len(joinids))
    adata = cellxgene_census.get_anndata(
        census,
        organism=ORGANISM,
        obs_value_filter=filter_str,
        obs_coords=joinids,
        obs_column_names=OBS_COLUMNS,
        obs_embeddings=[EMBEDDING],
    )
    log.info("  AnnData: %d cells, obsm keys: %s", adata.n_obs, list(adata.obsm.keys()))
    return adata


def _resolve_cross_dataset_filters(census, cellxgene_census, shared_filter):
    """Find the two largest dataset_ids within a tissue+assay combo."""
    obs_df = cellxgene_census.get_obs(
        census, ORGANISM,
        value_filter=shared_filter,
        column_names=["soma_joinid", "dataset_id"],
    )
    if len(obs_df) == 0:
        return None, None

    counts = obs_df["dataset_id"].value_counts()
    if len(counts) < 2:
        return None, None

    top2 = counts.index[:2]
    ds_a, ds_b = str(top2[0]), str(top2[1])
    filter_a = shared_filter + f" and dataset_id == '{ds_a}'"
    filter_b = shared_filter + f" and dataset_id == '{ds_b}'"
    log.info("  Cross-dataset: dataset A = %s (%d cells), dataset B = %s (%d cells)",
             ds_a, counts.iloc[0], ds_b, counts.iloc[1])
    return filter_a, filter_b


# ---------------------------------------------------------------------------
# Metadata construction (same pattern as pilot)
# ---------------------------------------------------------------------------

def _build_m7_records(source_adata, target_adata):
    """Build M7 records: dataset_id as site, binarized disease as outcome."""
    records = []
    for adata in [source_adata, target_adata]:
        for _, row in adata.obs.iterrows():
            records.append({
                "site": str(row["dataset_id"]),
                "outcome": 0 if str(row.get("disease", "normal")) == "normal" else 1,
            })
    return records


def _build_m6_knn_graph(X_source, X_target, k=10, max_nodes=500):
    """Build k-NN graph on combined embeddings for M6 curvature."""
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


def _build_metadata(source_adata, target_adata, X_source, X_target, labels_source, labels_target):
    """Build metadata dict for preflight.run()."""
    metadata = {}
    if labels_source is not None:
        metadata["source_labels"] = labels_source
    if labels_target is not None:
        metadata["target_labels"] = labels_target

    m7_records = _build_m7_records(source_adata, target_adata)
    n_sites = len(set(r["site"] for r in m7_records))
    if n_sites >= 2:
        metadata["records"] = m7_records
        metadata["site_key"] = "site"
        metadata["outcome_key"] = "outcome"

    metadata["graph"] = _build_m6_knn_graph(X_source, X_target)
    return metadata


# ---------------------------------------------------------------------------
# F1 degradation measurement
# ---------------------------------------------------------------------------

def _get_shared_cell_types(source_labels, target_labels, min_cells=MIN_CELLS_PER_TYPE):
    """Find cell types present in both source and target with >= min_cells each."""
    from collections import Counter
    source_counts = Counter(source_labels)
    target_counts = Counter(target_labels)
    shared = set()
    for ct in source_counts:
        if ct in target_counts and source_counts[ct] >= min_cells and target_counts[ct] >= min_cells:
            shared.add(ct)
    return shared


def _measure_f1_source(X, labels, donors, n_folds=5):
    """Donor-stratified 5-fold CV macro F1 on source cells.

    Returns mean macro F1 across folds, or NaN if CV is not feasible.
    """
    labels = np.asarray(labels, dtype=str)
    donors = np.asarray(donors, dtype=str)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    unique_donors = np.unique(donors)
    n_donors = len(unique_donors)
    if n_donors < 2:
        return float("nan")

    # Donor-stratified splitting: assign each donor to a fold, then cells inherit
    k = min(n_folds, n_donors)
    rng = np.random.default_rng(0)
    donor_order = rng.permutation(n_donors)
    donor_to_fold = {}
    fold_sizes = np.zeros(k, dtype=int)
    donor_sizes = {d: int((donors == d).sum()) for d in unique_donors}

    # Greedy bin-packing: largest donors first
    sorted_donors = sorted(unique_donors, key=lambda d: -donor_sizes[d])
    for d in sorted_donors:
        fold = int(np.argmin(fold_sizes))
        donor_to_fold[d] = fold
        fold_sizes[fold] += donor_sizes[d]

    folds = np.array([donor_to_fold[d] for d in donors])

    fold_scores = []
    for fold_id in range(k):
        test_mask = folds == fold_id
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        y_train, y_test = y[train_mask], y[test_mask]
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if len(train_classes) < 2 or not test_classes.issubset(train_classes):
            continue

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_scores.append(f1)

    if len(fold_scores) == 0:
        return float("nan")
    return float(np.mean(fold_scores))


def _measure_f1_target(X_source, labels_source, X_target, labels_target):
    """Train on ALL source cells, evaluate on ALL target cells. Returns macro F1."""
    labels_source = np.asarray(labels_source, dtype=str)
    labels_target = np.asarray(labels_target, dtype=str)

    le = LabelEncoder()
    le.fit(np.concatenate([labels_source, labels_target]))
    y_source = le.transform(labels_source)
    y_target = le.transform(labels_target)

    if len(set(y_source)) < 2:
        return float("nan")

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_source)
    X_test = scaler.transform(X_target)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X_train, y_source)
    y_pred = clf.predict(X_test)
    return float(f1_score(y_target, y_pred, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _partial_spearman_rho(x, y, z):
    """Partial Spearman rho between x and y, controlling for z.

    Ranks all three variables, then computes partial correlation of ranks:
        r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2))

    Returns (rho, p_value). p_value is from a t-test with n-3 df.
    """
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")

    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)

    r_xy = np.corrcoef(rx, ry)[0, 1]
    r_xz = np.corrcoef(rx, rz)[0, 1]
    r_yz = np.corrcoef(ry, rz)[0, 1]

    denom = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-12:
        return float("nan"), float("nan")

    partial_rho = (r_xy - r_xz * r_yz) / denom

    # t-test for significance
    df = n - 3
    if df < 1:
        return float(partial_rho), float("nan")
    t_stat = partial_rho * np.sqrt(df / (1 - partial_rho**2 + 1e-12))
    p_value = 2 * stats.t.sf(abs(t_stat), df)
    return float(partial_rho), float(p_value)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    try:
        import cellxgene_census
        import cellxgene_census.experimental
    except ImportError:
        log.error("cellxgene-census not installed.")
        log.error("Run: uv pip install --python .venv-census/bin/python --only-binary :all: cellxgene-census")
        return 1

    from preflight.core.preregister import (
        DatasetSpec,
        compute_preregistration_sha,
        load_preregistration,
        verify_preregistration,
    )
    from preflight.core.runner import (
        ALL_MODULE_SOURCES,
        resolve_hyperparameters,
        run,
    )
    from preflight.extractors.anndata_loader import extract_arrays

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log.info("Experiment 0: Composite Validation")
    log.info("Timestamp: %s", timestamp)
    log.info("Census: %s, Embedding: %s, Max cells: %d", CENSUS_VERSION, EMBEDDING, MAX_CELLS)

    # ==================================================================
    # STEP 1: Load and verify frozen preregistration
    # ==================================================================
    log.info("[1/6] Loading frozen preregistration from %s", FROZEN_PREREG_PATH)

    if not FROZEN_PREREG_PATH.exists():
        log.error("Frozen preregistration not found at %s", FROZEN_PREREG_PATH)
        log.error("Run: python scripts/freeze_preregistration_v4.py")
        return 1

    frozen_record = load_preregistration(FROZEN_PREREG_PATH)
    frozen_sha = frozen_record.sha256
    log.info("  Frozen SHA: %s", frozen_sha[:32])

    dataset_spec = DatasetSpec(
        name="exp0_composite_validation",
        source_description="~25 source->target pairs across 4 pair types (cross-assay, cross-tissue, cross-dataset, negative control)",
        target_description="Same pairs — target side",
        n_source=MAX_CELLS,
        n_target=MAX_CELLS,
        extra={
            "census_version": CENSUS_VERSION,
            "embedding": EMBEDDING,
            "tissues": ["lung", "heart", "liver", "kidney", "brain"],
            "pair_types": ["cross_assay", "cross_tissue", "cross_dataset", "negative_control"],
            "n_pairs_approx": 25,
            "min_shared_cell_types": MIN_SHARED_CELL_TYPES,
            "primary_endpoint": "partial_spearman_rho_composite_vs_relative_degradation",
        },
    )

    hp, weights = resolve_hyperparameters({"k": 5})
    canonical_hp = {**hp, "weights": weights}

    current_record = compute_preregistration_sha(ALL_MODULE_SOURCES, canonical_hp, dataset_spec)
    if current_record.sha256 != frozen_sha:
        log.error("SHA MISMATCH: frozen=%s vs current=%s", frozen_sha[:16], current_record.sha256[:16])
        match, msg = verify_preregistration(frozen_record, ALL_MODULE_SOURCES, canonical_hp, dataset_spec)
        log.error("Details: %s", msg)
        return 1

    log.info("  SHA verified: scorer matches frozen preregistration")

    # ==================================================================
    # STEP 2: Define pairs
    # ==================================================================
    log.info("[2/6] Defining source->target pairs")
    pairs = define_pairs()
    log.info("  Defined %d pairs", len(pairs))
    for pt in ["cross_assay", "cross_tissue", "cross_dataset", "negative_control"]:
        n = sum(1 for p in pairs if p["pair_type"] == pt)
        log.info("    %s: %d", pt, n)

    # ==================================================================
    # STEP 3: Run all pairs
    # ==================================================================
    log.info("[3/6] Running all pairs (this will take a while)...")
    t_start = time.time()
    results = []
    dropped_pairs = []
    jsonl_path = OUTPUT_DIR / f"incremental_{timestamp}.jsonl"

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:

        # First pass: resolve cross-dataset filters
        for pair in pairs:
            if pair["pair_type"] == "cross_dataset" and pair["source_filter"] is None:
                log.info("  Resolving cross-dataset pair: %s", pair["pair_id"])
                filter_a, filter_b = _resolve_cross_dataset_filters(
                    census, cellxgene_census, pair["shared_filter"]
                )
                if filter_a is None:
                    log.warning("  SKIP %s: could not find 2 datasets", pair["pair_id"])
                    dropped_pairs.append({"pair_id": pair["pair_id"], "reason": "fewer than 2 datasets"})
                    pair["_skip"] = True
                else:
                    pair["source_filter"] = filter_a
                    pair["target_filter"] = filter_b

        # Main loop
        active_pairs = [p for p in pairs if not p.get("_skip", False)]
        for pair in tqdm(active_pairs, desc="Processing pairs"):
            pair_id = pair["pair_id"]
            log.info("--- Pair: %s (%s) ---", pair_id, pair["pair_type"])

            # Pull source
            log.info("  Pulling source: %s", pair["source_desc"])
            source_adata = _query_slice(
                census, cellxgene_census,
                pair["source_filter"], MAX_CELLS, pair["source_seed"],
            )
            if source_adata is None or source_adata.n_obs == 0:
                log.warning("  SKIP %s: no source cells", pair_id)
                dropped_pairs.append({"pair_id": pair_id, "reason": "no source cells"})
                continue

            # Pull target
            log.info("  Pulling target: %s", pair["target_desc"])
            target_adata = _query_slice(
                census, cellxgene_census,
                pair["target_filter"], MAX_CELLS, pair["target_seed"],
            )
            if target_adata is None or target_adata.n_obs == 0:
                log.warning("  SKIP %s: no target cells", pair_id)
                dropped_pairs.append({"pair_id": pair_id, "reason": "no target cells"})
                continue

            # Extract arrays
            X_source, labels_source, _ = extract_arrays(
                source_adata, obsm_key=EMBEDDING, obs_label_key="cell_type"
            )
            X_target, labels_target, _ = extract_arrays(
                target_adata, obsm_key=EMBEDDING, obs_label_key="cell_type"
            )

            if X_source.shape[1] != X_target.shape[1]:
                log.warning("  SKIP %s: dimension mismatch %d vs %d",
                            pair_id, X_source.shape[1], X_target.shape[1])
                dropped_pairs.append({"pair_id": pair_id, "reason": "dimension mismatch"})
                continue

            # Check shared cell types
            source_ct = source_adata.obs["cell_type"].values
            target_ct = target_adata.obs["cell_type"].values
            shared_types = _get_shared_cell_types(source_ct, target_ct)
            n_shared = len(shared_types)
            log.info("  Shared cell types (>= %d cells each): %d", MIN_CELLS_PER_TYPE, n_shared)

            if n_shared < MIN_SHARED_CELL_TYPES:
                log.warning("  SKIP %s: only %d shared cell types (need %d)",
                            pair_id, n_shared, MIN_SHARED_CELL_TYPES)
                dropped_pairs.append({
                    "pair_id": pair_id,
                    "reason": f"only {n_shared} shared cell types",
                })
                continue

            # Filter to shared cell types only
            source_mask = np.array([ct in shared_types for ct in source_ct])
            target_mask = np.array([ct in shared_types for ct in target_ct])
            X_src_filt = X_source[source_mask]
            X_tgt_filt = X_target[target_mask]
            ct_src_filt = source_ct[source_mask]
            ct_tgt_filt = target_ct[target_mask]
            donors_src_filt = source_adata.obs["donor_id"].values[source_mask]

            log.info("  After filtering: source=%d cells, target=%d cells",
                     X_src_filt.shape[0], X_tgt_filt.shape[0])

            # Run Preflight pipeline (on FULL arrays, not filtered)
            log.info("  Running Preflight pipeline...")
            metadata = _build_metadata(
                source_adata, target_adata, X_source, X_target,
                labels_source, labels_target,
            )

            report = run(
                source=X_source,
                target=X_target,
                metadata=metadata,
                hyperparameters={"k": 5},
            )

            # Measure F1 degradation (on FILTERED arrays — shared cell types only)
            log.info("  Measuring F1_source (donor-stratified CV)...")
            f1_source = _measure_f1_source(X_src_filt, ct_src_filt, donors_src_filt)
            log.info("  F1_source = %.3f", f1_source)

            log.info("  Measuring F1_target (train on source, eval on target)...")
            f1_target = _measure_f1_target(X_src_filt, ct_src_filt, X_tgt_filt, ct_tgt_filt)
            log.info("  F1_target = %.3f", f1_target)

            abs_degrad = f1_source - f1_target
            rel_degrad = abs_degrad / f1_source if f1_source > 1e-6 else float("nan")
            log.info("  Relative degradation = %.3f", rel_degrad)

            # Collect per-module scores
            module_scores = {}
            for mod_name, mod_result in report.module_results.items():
                module_scores[mod_name] = {
                    "score": float(mod_result.score),
                    "tier": mod_result.tier,
                }

            result = {
                "pair_id": pair_id,
                "pair_type": pair["pair_type"],
                "tissue_source": pair["tissue_source"],
                "tissue_target": pair["tissue_target"],
                "source_desc": pair["source_desc"],
                "target_desc": pair["target_desc"],
                "n_source": int(X_source.shape[0]),
                "n_target": int(X_target.shape[0]),
                "n_shared_cell_types": n_shared,
                "shared_cell_types": sorted(shared_types),
                "n_source_filtered": int(X_src_filt.shape[0]),
                "n_target_filtered": int(X_tgt_filt.shape[0]),
                "composite_score": float(report.overall_score),
                "composite_tier": report.overall_tier,
                "module_scores": module_scores,
                "f1_source": float(f1_source),
                "f1_target": float(f1_target),
                "abs_degradation": float(abs_degrad),
                "relative_degradation": float(rel_degrad),
            }
            results.append(result)
            with open(jsonl_path, "a") as _jf:
                _jf.write(json.dumps(result, default=str) + "\n")
                _jf.flush()
            log.info("  Composite: %.3f (Tier %d), F1: %.3f -> %.3f, rel_degrad: %.3f",
                     result["composite_score"], result["composite_tier"],
                     f1_source, f1_target, rel_degrad)

    elapsed = time.time() - t_start
    log.info("All pairs processed in %.1f minutes", elapsed / 60)
    log.info("  Valid pairs: %d, Dropped: %d", len(results), len(dropped_pairs))

    # ==================================================================
    # STEP 4: Hypothesis tests
    # ==================================================================
    log.info("[4/6] Running hypothesis tests")

    # Split into shifted and all
    shifted = [r for r in results if r["pair_type"] != "negative_control"]
    all_pairs = results

    # Filter to pairs with valid degradation values
    shifted = [r for r in shifted if not np.isnan(r["relative_degradation"])]
    all_valid = [r for r in all_pairs if not np.isnan(r["relative_degradation"])]

    hypotheses = {}

    # --- H0.1 (PRIMARY): partial rho on shifted pairs ---
    if len(shifted) >= 4:
        composite_scores = np.array([r["composite_score"] for r in shifted])
        rel_degradations = np.array([r["relative_degradation"] for r in shifted])
        f1_sources = np.array([r["f1_source"] for r in shifted])

        h01_rho, h01_p = _partial_spearman_rho(composite_scores, rel_degradations, f1_sources)
        h01_pass = h01_rho < -0.4 and h01_p < 0.05
        hypotheses["H0.1"] = {
            "description": "partial Spearman rho(composite, relative_degradation | F1_source) < -0.4 on shifted pairs",
            "partial_rho": h01_rho,
            "p_value": h01_p,
            "n_pairs": len(shifted),
            "threshold": -0.4,
            "passed": bool(h01_pass),
        }
        log.info("  H0.1: partial rho = %.3f (p=%.4f), n=%d -> %s",
                 h01_rho, h01_p, len(shifted), "PASS" if h01_pass else "FAIL")
    else:
        hypotheses["H0.1"] = {
            "description": "partial Spearman rho(composite, relative_degradation | F1_source) < -0.4 on shifted pairs",
            "error": f"too few shifted pairs ({len(shifted)})",
            "passed": None,
        }
        log.warning("  H0.1: SKIPPED — only %d shifted pairs", len(shifted))

    # --- H0.1b (SECONDARY): raw rho across all pairs ---
    if len(all_valid) >= 4:
        composite_all = np.array([r["composite_score"] for r in all_valid])
        rel_degrad_all = np.array([r["relative_degradation"] for r in all_valid])

        h01b_rho, h01b_p = stats.spearmanr(composite_all, rel_degrad_all)
        h01b_pass = h01b_rho < -0.5
        hypotheses["H0.1b"] = {
            "description": "raw Spearman rho(composite, relative_degradation) < -0.5 across all pairs",
            "rho": float(h01b_rho),
            "p_value": float(h01b_p),
            "n_pairs": len(all_valid),
            "threshold": -0.5,
            "passed": bool(h01b_pass),
        }
        log.info("  H0.1b: rho = %.3f (p=%.4f), n=%d -> %s",
                 h01b_rho, h01b_p, len(all_valid), "PASS" if h01b_pass else "FAIL")
    else:
        hypotheses["H0.1b"] = {
            "description": "raw Spearman rho(composite, relative_degradation) < -0.5 across all pairs",
            "error": f"too few valid pairs ({len(all_valid)})",
            "passed": None,
        }

    # --- H0.1c (SUBGROUP): rho within cross-tissue pairs ---
    cross_tissue = [r for r in shifted if r["pair_type"] == "cross_tissue"
                    and not np.isnan(r["relative_degradation"])]
    if len(cross_tissue) >= 4:
        ct_scores = np.array([r["composite_score"] for r in cross_tissue])
        ct_degrad = np.array([r["relative_degradation"] for r in cross_tissue])

        h01c_rho, h01c_p = stats.spearmanr(ct_scores, ct_degrad)
        h01c_pass = h01c_rho < -0.3
        hypotheses["H0.1c"] = {
            "description": "Spearman rho(composite, relative_degradation) < -0.3 within cross-tissue pairs",
            "rho": float(h01c_rho),
            "p_value": float(h01c_p),
            "n_pairs": len(cross_tissue),
            "threshold": -0.3,
            "passed": bool(h01c_pass),
        }
        log.info("  H0.1c: rho = %.3f (p=%.4f), n=%d -> %s",
                 h01c_rho, h01c_p, len(cross_tissue), "PASS" if h01c_pass else "FAIL")
    else:
        hypotheses["H0.1c"] = {
            "description": "Spearman rho(composite, relative_degradation) < -0.3 within cross-tissue pairs",
            "error": f"too few cross-tissue pairs ({len(cross_tissue)})",
            "passed": None,
        }

    # --- H0.2: Tier >= 5 pairs have mean relative_degradation < 0.15 ---
    tier_high = [r for r in results if r["composite_tier"] >= 5
                 and not np.isnan(r["relative_degradation"])]
    if len(tier_high) >= 4:
        mean_degrad_high = float(np.mean([r["relative_degradation"] for r in tier_high]))
        h02_pass = mean_degrad_high < 0.15
        hypotheses["H0.2"] = {
            "description": "Tier >= 5 pairs have mean relative_degradation < 0.15",
            "mean_relative_degradation": mean_degrad_high,
            "n_pairs": len(tier_high),
            "threshold": 0.15,
            "passed": bool(h02_pass),
        }
        log.info("  H0.2: mean rel_degrad = %.3f, n=%d -> %s",
                 mean_degrad_high, len(tier_high), "PASS" if h02_pass else "FAIL")
    else:
        hypotheses["H0.2"] = {
            "description": "Tier >= 5 pairs have mean relative_degradation < 0.15",
            "n_pairs": len(tier_high),
            "note": f"only {len(tier_high)} pairs in Tier >= 5 (need >= 4), reporting descriptively",
            "values": [r["relative_degradation"] for r in tier_high] if tier_high else [],
            "passed": None,
        }
        log.info("  H0.2: DESCRIPTIVE ONLY — %d pairs in Tier >= 5 (need >= 4)", len(tier_high))

    # --- H0.3: Tier <= 2 pairs have mean relative_degradation > 0.40 ---
    tier_low = [r for r in results if r["composite_tier"] <= 2
                and not np.isnan(r["relative_degradation"])]
    if len(tier_low) >= 4:
        mean_degrad_low = float(np.mean([r["relative_degradation"] for r in tier_low]))
        h03_pass = mean_degrad_low > 0.40
        hypotheses["H0.3"] = {
            "description": "Tier <= 2 pairs have mean relative_degradation > 0.40",
            "mean_relative_degradation": mean_degrad_low,
            "n_pairs": len(tier_low),
            "threshold": 0.40,
            "passed": bool(h03_pass),
        }
        log.info("  H0.3: mean rel_degrad = %.3f, n=%d -> %s",
                 mean_degrad_low, len(tier_low), "PASS" if h03_pass else "FAIL")
    else:
        hypotheses["H0.3"] = {
            "description": "Tier <= 2 pairs have mean relative_degradation > 0.40",
            "n_pairs": len(tier_low),
            "note": f"only {len(tier_low)} pairs in Tier <= 2 (need >= 4), reporting descriptively",
            "values": [r["relative_degradation"] for r in tier_low] if tier_low else [],
            "passed": None,
        }
        log.info("  H0.3: DESCRIPTIVE ONLY — %d pairs in Tier <= 2 (need >= 4)", len(tier_low))

    # --- H0.4: M2 alone partial rho < -0.3 on shifted pairs ---
    if len(shifted) >= 4:
        m2_scores = np.array([
            r["module_scores"].get("m2_domain_shift", {}).get("score", float("nan"))
            for r in shifted
        ])
        rel_deg_shifted = np.array([r["relative_degradation"] for r in shifted])
        f1_src_shifted = np.array([r["f1_source"] for r in shifted])

        valid_m2 = ~np.isnan(m2_scores)
        if valid_m2.sum() >= 4:
            h04_rho, h04_p = _partial_spearman_rho(
                m2_scores[valid_m2], rel_deg_shifted[valid_m2], f1_src_shifted[valid_m2]
            )
            h04_pass = h04_rho < -0.3
            hypotheses["H0.4"] = {
                "description": "M2 partial Spearman rho < -0.3 on shifted pairs",
                "partial_rho": h04_rho,
                "p_value": h04_p,
                "n_pairs": int(valid_m2.sum()),
                "threshold": -0.3,
                "passed": bool(h04_pass),
            }
            log.info("  H0.4: M2 partial rho = %.3f (p=%.4f), n=%d -> %s",
                     h04_rho, h04_p, int(valid_m2.sum()), "PASS" if h04_pass else "FAIL")
        else:
            hypotheses["H0.4"] = {
                "description": "M2 partial Spearman rho < -0.3 on shifted pairs",
                "error": "too few pairs with valid M2 scores",
                "passed": None,
            }
    else:
        hypotheses["H0.4"] = {
            "description": "M2 partial Spearman rho < -0.3 on shifted pairs",
            "error": f"too few shifted pairs ({len(shifted)})",
            "passed": None,
        }

    # --- H0.5: composite |partial rho| > M2 |partial rho| on shifted pairs ---
    h01_result = hypotheses.get("H0.1", {})
    h04_result = hypotheses.get("H0.4", {})
    if "partial_rho" in h01_result and "partial_rho" in h04_result:
        comp_abs = abs(h01_result["partial_rho"])
        m2_abs = abs(h04_result["partial_rho"])
        h05_pass = comp_abs > m2_abs
        hypotheses["H0.5"] = {
            "description": "composite |partial rho| > M2 |partial rho| on shifted pairs",
            "composite_abs_partial_rho": comp_abs,
            "m2_abs_partial_rho": m2_abs,
            "passed": bool(h05_pass),
        }
        log.info("  H0.5: |composite rho| = %.3f vs |M2 rho| = %.3f -> %s",
                 comp_abs, m2_abs, "PASS" if h05_pass else "FAIL")
    else:
        hypotheses["H0.5"] = {
            "description": "composite |partial rho| > M2 |partial rho| on shifted pairs",
            "error": "H0.1 or H0.4 could not be computed",
            "passed": None,
        }

    # --- Per-module partial rho (descriptive, not tested) ---
    per_module_rho = {}
    if len(shifted) >= 4:
        rel_deg_shifted = np.array([r["relative_degradation"] for r in shifted])
        f1_src_shifted = np.array([r["f1_source"] for r in shifted])

        all_module_names = set()
        for r in shifted:
            all_module_names.update(r["module_scores"].keys())

        for mod_name in sorted(all_module_names):
            mod_scores = np.array([
                r["module_scores"].get(mod_name, {}).get("score", float("nan"))
                for r in shifted
            ])
            valid = ~np.isnan(mod_scores)
            if valid.sum() >= 4:
                rho, p = _partial_spearman_rho(
                    mod_scores[valid], rel_deg_shifted[valid], f1_src_shifted[valid]
                )
                per_module_rho[mod_name] = {"partial_rho": rho, "p_value": p, "n": int(valid.sum())}
                log.info("  %s: partial rho = %.3f (p=%.4f)", mod_name, rho, p)

    # ==================================================================
    # STEP 5: Save results
    # ==================================================================
    log.info("[5/6] Saving results to %s", OUTPUT_DIR)

    full_results = {
        "experiment": "exp0_composite_validation",
        "timestamp": timestamp,
        "preregistration_sha": frozen_sha,
        "census_version": CENSUS_VERSION,
        "embedding": EMBEDDING,
        "max_cells_per_condition": MAX_CELLS,
        "min_shared_cell_types": MIN_SHARED_CELL_TYPES,
        "min_cells_per_type": MIN_CELLS_PER_TYPE,
        "n_pairs_attempted": len(pairs),
        "n_pairs_valid": len(results),
        "n_pairs_dropped": len(dropped_pairs),
        "dropped_pairs": dropped_pairs,
        "pair_results": results,
        "hypotheses": hypotheses,
        "per_module_partial_rho": per_module_rho,
        "elapsed_seconds": elapsed,
    }

    results_path = OUTPUT_DIR / f"results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    log.info("  Results JSON: %s", results_path)

    # ==================================================================
    # STEP 6: Generate markdown report
    # ==================================================================
    log.info("[6/6] Generating markdown report")

    md_lines = [
        "# Experiment 0: Composite Validation Results",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Preregistration SHA:** `{frozen_sha[:16]}...`",
        f"**Census:** {CENSUS_VERSION} | **Embedding:** {EMBEDDING}",
        f"**Pairs:** {len(results)} valid / {len(pairs)} attempted / {len(dropped_pairs)} dropped",
        f"**Elapsed:** {elapsed / 60:.1f} minutes",
        "",
        "## Pair Results",
        "",
        "| Pair | Type | Composite | Tier | F1 Source | F1 Target | Rel Degrad |",
        "|------|------|-----------|------|-----------|-----------|------------|",
    ]

    for r in sorted(results, key=lambda x: x["relative_degradation"], reverse=True):
        md_lines.append(
            f"| {r['pair_id']} | {r['pair_type']} | "
            f"{r['composite_score']:.3f} | {r['composite_tier']} | "
            f"{r['f1_source']:.3f} | {r['f1_target']:.3f} | "
            f"{r['relative_degradation']:.3f} |"
        )

    md_lines.extend([
        "",
        "## Hypothesis Results",
        "",
    ])

    for h_name in ["H0.1", "H0.1b", "H0.1c", "H0.2", "H0.3", "H0.4", "H0.5"]:
        h = hypotheses.get(h_name, {})
        status = "PASS" if h.get("passed") is True else ("FAIL" if h.get("passed") is False else "N/A")
        desc = h.get("description", "")
        md_lines.append(f"### {h_name}: {status}")
        md_lines.append(f"*{desc}*")
        md_lines.append("")

        if "partial_rho" in h:
            md_lines.append(f"- Partial rho: {h['partial_rho']:.3f}")
        if "rho" in h:
            md_lines.append(f"- Spearman rho: {h['rho']:.3f}")
        if "p_value" in h:
            md_lines.append(f"- p-value: {h['p_value']:.4f}")
        if "n_pairs" in h:
            md_lines.append(f"- N pairs: {h['n_pairs']}")
        if "mean_relative_degradation" in h:
            md_lines.append(f"- Mean relative degradation: {h['mean_relative_degradation']:.3f}")
        if "composite_abs_partial_rho" in h:
            md_lines.append(f"- |Composite rho|: {h['composite_abs_partial_rho']:.3f}")
            md_lines.append(f"- |M2 rho|: {h['m2_abs_partial_rho']:.3f}")
        if "error" in h:
            md_lines.append(f"- Note: {h['error']}")
        if "note" in h:
            md_lines.append(f"- Note: {h['note']}")
        if "values" in h and h["values"]:
            md_lines.append(f"- Individual values: {[f'{v:.3f}' for v in h['values']]}")
        md_lines.append("")

    if per_module_rho:
        md_lines.extend([
            "## Per-Module Partial Correlations (descriptive)",
            "",
            "| Module | Partial rho | p-value | N |",
            "|--------|-------------|---------|---|",
        ])
        for mod_name, vals in sorted(per_module_rho.items()):
            md_lines.append(
                f"| {mod_name} | {vals['partial_rho']:.3f} | {vals['p_value']:.4f} | {vals['n']} |"
            )
        md_lines.append("")

    if dropped_pairs:
        md_lines.extend([
            "## Dropped Pairs",
            "",
        ])
        for dp in dropped_pairs:
            md_lines.append(f"- **{dp['pair_id']}**: {dp['reason']}")
        md_lines.append("")

    report_md = "\n".join(md_lines)
    report_path = OUTPUT_DIR / f"report_{timestamp}.md"
    report_path.write_text(report_md)
    log.info("  Report MD: %s", report_path)

    # Print summary
    print("\n" + "=" * 70)
    print(report_md)
    print("=" * 70)

    # Decision summary
    print("\n--- DECISION SUMMARY ---")
    h01 = hypotheses.get("H0.1", {})
    if h01.get("passed") is True:
        print("H0.1 PASSED: The composite is a validated transfer-failure predictor.")
    elif h01.get("passed") is False:
        print("H0.1 FAILED: Composite weights need recalibration.")
        print("  Use the shifted pairs as training set for leave-one-out CV regression.")
    else:
        print("H0.1 COULD NOT BE EVALUATED.")

    h05 = hypotheses.get("H0.5", {})
    if h05.get("passed") is False:
        print("H0.5 FAILED: Composite adds no value over M2 alone.")
        print("  Consider simplifying to M2 + probes.")

    print(f"\nAll artifacts saved to {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
