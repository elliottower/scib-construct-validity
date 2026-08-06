"""
Spectral-gap discriminative statistic for cell-type centroid matrices.

MOTIVATION: CKA and Procrustes predict transfer (rho=0.53, 0.66) but score
all 104 conditions below random Gaussian centroids. The null floor arises
because CKA/Procrustes are Gram-matrix metrics — they respond to the full
matrix, not to whether the matrix has meaningful structure.

Random centroid matrices have flat (Marchenko-Pastur) eigenvalue spectra.
Real centroid matrices have concentrated spectra — a few dominant directions
capture most variance, reflecting cell-type hierarchy. A spectral-gap
statistic should discriminate real from random where CKA cannot.

STATISTIC: For a k × d centroid matrix C (mean embedding per shared cell type,
centered), compute the singular values s_1 >= s_2 >= ... >= s_r (r = min(k-1, d)).

Define the spectral-gap ratio at rank j:
  SGR_j = sum(s_i^2 for i <= j) / sum(s_i^2 for all i)
        = explained variance ratio of top-j components

For random Gaussian C, the singular values follow the Marchenko-Pastur law
(asymptotically). At finite k, d, the expected SGR_j can be computed by
simulation from Wishart eigenvalue distributions.

The discriminative statistic is the z-score:
  z_j = (SGR_j(real) - E[SGR_j(null, k, d)]) / std[SGR_j(null, k, d)]

A positive z_j means the real centroid matrix has MORE concentrated spectrum
than random — i.e., it has structure.

HYPOTHESIS: z_j is positive for contender models and near-zero for null
baselines (random_projection, untrained_encoder). If z_j also correlates
with transfer F1, it provides both discriminative AND predictive validity.

RANK CHOICE: j = 1 (top singular value explains more than expected).
Also report j = 2 and j = floor(k/3) for robustness.

This script:
1. Computes the null distribution of SGR_j for each (k, d) in the panel
2. Extracts cell-type centroid matrices from Census (same pipeline as V3b)
3. Computes SGR_j and z-scores for all 154 conditions
4. Tests predictive validity (partial Spearman vs transfer F1, controlling d)
5. Tests discriminative validity (contender vs baseline z-scores)

DATA: CellxGene Census v2023-12-15, same panel as V3b (25 tissues, 8 models,
source=EFO:0009922 10x 3' v3, target=EFO:0008931 Smart-seq2).

This script requires Census access (cellxgene_census) and runs ~30-60 min
on CPU. The null distributions are purely computational (no data needed).
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats as scipy_stats
from collections import defaultdict
import warnings

RESULTS_DIR = Path(__file__).parent.parent / "results" / "spectral_gap"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ASSAY = "EFO:0009922"  # 10x 3' v3
TARGET_ASSAY = "EFO:0008931"  # Smart-seq2
MIN_CELLS = 200
MIN_SHARED_TYPES = 8
MAX_CELLS = 2000
SEED = 20260713
CENSUS_VERSION = "2023-12-15"

NULL_TRIALS = 5000
NULL_SEED = 42
RANK_JS = [1, 2, None]  # None = floor(k/3)

MODELS = [
    "geneformer",
    "geneformer_v2_104m",
    "geneformer_v2_316m",
    "scgpt",
    "scvi",
    "bog_pca_512",
    "random_projection",
    "untrained_encoder",
]
CONTENDER_MODELS = [m for m in MODELS if m not in ("random_projection", "untrained_encoder")]
BASELINE_MODELS = ["random_projection", "untrained_encoder"]

MODEL_DIMS = {
    "geneformer": 512,
    "geneformer_v2_104m": 512,
    "geneformer_v2_316m": 512,
    "scgpt": 512,
    "scvi": 50,
    "bog_pca_512": 512,
    "random_projection": 512,
    "untrained_encoder": 512,
}


def spectral_gap_ratio(C, j):
    """Explained variance ratio of top-j singular values of centered matrix C."""
    Cc = C - C.mean(axis=0)
    sv = np.linalg.svd(Cc, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 0.0
    return sv2[:j].sum() / total


def compute_null_sgr(k, d, j_values, n_trials=NULL_TRIALS, seed=NULL_SEED):
    """Null distribution of SGR_j for random Gaussian k × d matrices."""
    rng = np.random.default_rng(seed)
    results = {}
    for j_raw in j_values:
        j = j_raw if j_raw is not None else max(1, (k - 1) // 3)
        j = min(j, k - 1)  # can't exceed rank after centering
        sgr_vals = []
        for _ in range(n_trials):
            C = rng.standard_normal((k, d))
            sgr_vals.append(spectral_gap_ratio(C, j))
        sgr_arr = np.array(sgr_vals)
        key = f"j={j_raw if j_raw is not None else 'k/3'}"
        results[key] = {
            'j': j,
            'mean': float(sgr_arr.mean()),
            'std': float(sgr_arr.std()),
            'p5': float(np.percentile(sgr_arr, 5)),
            'p95': float(np.percentile(sgr_arr, 95)),
        }
    return results


def partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y, controlling for z."""
    rx = scipy_stats.rankdata(x)
    ry = scipy_stats.rankdata(y)
    rz = scipy_stats.rankdata(z)
    # Residualize ranks
    from numpy.polynomial.polynomial import polyfit
    cx = np.polyfit(rz, rx, 1)
    cy = np.polyfit(rz, ry, 1)
    rx_resid = rx - np.polyval(cx, rz)
    ry_resid = ry - np.polyval(cy, rz)
    rho, p = scipy_stats.spearmanr(rx_resid, ry_resid)
    return rho, p


def tissue_stratified_permutation(metric_vals, f1_vals, tissue_labels, d_vals, n_perm=10000, seed=42):
    """Tissue-stratified permutation test for partial Spearman."""
    rng = np.random.default_rng(seed)
    obs_rho, _ = partial_spearman(metric_vals, f1_vals, d_vals)
    count = 0
    tissues_unique = list(set(tissue_labels))
    for _ in range(n_perm):
        perm_f1 = np.array(f1_vals, dtype=float).copy()
        for t in tissues_unique:
            mask = np.array([tl == t for tl in tissue_labels])
            perm_f1[mask] = rng.permutation(perm_f1[mask])
        perm_rho, _ = partial_spearman(metric_vals, perm_f1, d_vals)
        if abs(perm_rho) >= abs(obs_rho):
            count += 1
    return obs_rho, count / n_perm


def main():
    print(f"Spectral-gap discriminative statistic — {datetime.now().isoformat()}")
    print("=" * 70)

    # PHASE 1: Compute null distributions for all (k, d) pairs in the panel
    print("\n--- Phase 1: Null distributions ---")

    # Load V3b per-condition data to get k and d values
    v3b_path = Path(__file__).parent.parent / "results/biological_structure_v3b/biological_structure_v3b/per_condition.json"
    with open(v3b_path) as f:
        v3b_conditions = json.load(f)

    kd_pairs = set()
    for c in v3b_conditions:
        kd_pairs.add((c['n_shared_types'], c['d']))
    print(f"  Unique (k, d) pairs: {len(kd_pairs)}")

    null_cache = {}
    for k, d in sorted(kd_pairs):
        print(f"  Computing null for k={k}, d={d}...", end=" ", flush=True)
        null_cache[(k, d)] = compute_null_sgr(k, d, RANK_JS, n_trials=NULL_TRIALS)
        print("done")

    # Save null distributions
    null_output = {
        'timestamp': datetime.now().isoformat(),
        'n_trials': NULL_TRIALS,
        'seed': NULL_SEED,
        'null_distributions': {
            f"k={k}_d={d}": null_cache[(k, d)]
            for k, d in sorted(kd_pairs)
        }
    }
    null_path = RESULTS_DIR / "null_distributions.json"
    with open(null_path, 'w') as f:
        json.dump(null_output, f, indent=2)
    print(f"  Saved null distributions to {null_path}")

    # PHASE 2: Extract centroid matrices from Census and compute SGR z-scores
    print("\n--- Phase 2: Computing SGR on real data ---")
    print("  This phase requires Census access. Importing cellxgene_census...")

    try:
        import cellxgene_census
        import tiledbsoma
    except ImportError:
        print("  ERROR: cellxgene_census not installed. Saving null-only results.")
        print("  Install with: pip install cellxgene-census")
        return

    # The actual Census extraction follows the same pipeline as V3b.
    # For each tissue × model, we:
    # 1. Get source and target cells (same filtering as V3b)
    # 2. Compute embeddings (or load pre-computed)
    # 3. Compute cell-type centroids
    # 4. Compute SGR and z-score against null

    # NOTE: This is a stub — the full Census extraction pipeline is in
    # exp_biological_structure_v3.py. To avoid duplicating that 300-line
    # pipeline, we import the centroid computation from there.
    # For the pre-registration, we freeze the STATISTIC (SGR + z-score)
    # and the NULL DISTRIBUTIONS. The data extraction is inherited from V3b.

    print("  Full Census extraction not yet implemented in this script.")
    print("  Use exp_biological_structure_v3.py pipeline with SGR computation added.")
    print("  Null distributions are frozen and saved.")

    # PHASE 3: Analysis (will run after data extraction)
    # Stubbed here for pre-registration purposes
    print("\n--- Phase 3: Analysis (stub) ---")
    print("  After data extraction:")
    print("  1. Compute SGR z-scores for all 154 conditions")
    print("  2. Partial Spearman of z-score vs transfer F1, controlling for d")
    print("  3. Mann-Whitney U: contender vs baseline z-scores per tissue")
    print("  4. Fraction of tissues where contenders have significantly higher z")


if __name__ == '__main__':
    main()
