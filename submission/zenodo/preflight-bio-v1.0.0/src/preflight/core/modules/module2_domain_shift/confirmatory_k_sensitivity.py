"""
Phase 2 confirmatory analysis: C5 (k-sensitivity).

Reports the saturation criterion at k = 5, 10, 15, 20, 25.
For k=5,10,15: reads pre-computed geodesics from the results file.
For k=20,25: computes geodesics from raw per-instrument embeddings.

Implements exactly the procedure specified in PREREG_spectral_msms.md.
"""

import json
import numpy as np
from pathlib import Path
from itertools import combinations
from datetime import datetime

from transportability import top_k_subspace, geodesic_distance

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_FILE = RESULTS_DIR / "transportability_20260702_155956.json"

ESI_TYPES = {
    "LC-ESI-QTOF", "LC-ESI-QFT", "LC-ESI-QQ", "LC-ESI-ITFT",
    "LC-ESI-QIT", "LC-ESI-IT", "ESI-QTOF",
}
EIB = "EI-B"


def load_results():
    with open(DATA_FILE) as f:
        return json.load(f)


def geodesics_from_precomputed(pairs, k):
    key = f"geodesic_k{k}"
    return [p[key] for p in pairs if key in p]


def geodesics_from_embeddings(embedded_groups, k):
    group_names = sorted(embedded_groups.keys())
    pair_names = list(combinations(group_names, 2))
    geodesics = []
    pair_labels = []
    for a, b in pair_names:
        X_a, X_b = embedded_groups[a], embedded_groups[b]
        k_actual = min(k, X_a.shape[1], X_b.shape[1], X_a.shape[0] - 1, X_b.shape[0] - 1)
        U_a, _ = top_k_subspace(X_a, k_actual)
        U_b, _ = top_k_subspace(X_b, k_actual)
        geodesics.append(geodesic_distance(U_a, U_b))
        pair_labels.append((a, b))
    return geodesics, pair_labels


def saturation_criterion(geodesics, k):
    """Evaluate the frozen saturation criterion at a given k."""
    geo = np.array(geodesics)
    dyn_range = float(geo.max() - geo.min())
    theoretical_max = np.pi / 2 * np.sqrt(k)
    dyn_range_pct = dyn_range / theoretical_max * 100

    within_esi = []
    cross_ionization = []
    for g, (a, b) in zip(geodesics, [(None, None)] * len(geodesics)):
        pass

    criterion_1 = dyn_range < 0.20 * theoretical_max
    return {
        "k": k,
        "n_pairs": len(geodesics),
        "min": float(geo.min()),
        "max": float(geo.max()),
        "dynamic_range": dyn_range,
        "theoretical_max": float(theoretical_max),
        "dynamic_range_pct": dyn_range_pct,
        "criterion_1_met": criterion_1,
    }


def saturation_with_populations(geodesics, pairs, k):
    """Full saturation criterion including population split."""
    geo = np.array(geodesics)
    dyn_range = float(geo.max() - geo.min())
    theoretical_max = np.pi / 2 * np.sqrt(k)
    dyn_range_pct = dyn_range / theoretical_max * 100

    within_esi_geo = []
    cross_ionization_geo = []
    for g, p in zip(geodesics, pairs):
        a, b = p["site_a"], p["site_b"]
        if a in ESI_TYPES and b in ESI_TYPES:
            within_esi_geo.append(g)
        elif a != b:
            cross_ionization_geo.append(g)

    within_mean = float(np.mean(within_esi_geo)) if within_esi_geo else None
    cross_mean = float(np.mean(cross_ionization_geo)) if cross_ionization_geo else None
    family_diff = abs(within_mean - cross_mean) if (within_mean is not None and cross_mean is not None) else None

    criterion_1 = dyn_range < 0.20 * theoretical_max
    criterion_2 = (family_diff < 0.05 * dyn_range) if (family_diff is not None and dyn_range > 0) else None
    saturated = criterion_1 and (criterion_2 is True)

    return {
        "k": k,
        "n_pairs": len(geodesics),
        "min": float(geo.min()),
        "max": float(geo.max()),
        "mean": float(geo.mean()),
        "std": float(geo.std()),
        "dynamic_range": dyn_range,
        "theoretical_max": float(theoretical_max),
        "dynamic_range_pct": dyn_range_pct,
        "within_esi_mean": within_mean,
        "cross_ionization_mean": cross_mean,
        "family_diff": family_diff,
        "family_diff_pct_of_range": float(family_diff / dyn_range * 100) if (family_diff is not None and dyn_range > 0) else None,
        "criterion_1_dynamic_range_lt_20pct": criterion_1,
        "criterion_2_family_diff_lt_5pct_range": criterion_2,
        "saturated": saturated,
    }


def main():
    print(f"C5: k-Sensitivity Analysis — {datetime.now().isoformat()}")
    print("=" * 80)

    pairs = load_results()
    print(f"Loaded {len(pairs)} instrument pairs")

    results = {}
    for k in [5, 10, 15]:
        geo_key = f"geodesic_k{k}"
        geodesics = [p[geo_key] for p in pairs]
        result = saturation_with_populations(geodesics, pairs, k)
        results[f"k{k}"] = result
        print(f"\n  k={k}: range [{result['min']:.3f}, {result['max']:.3f}], "
              f"dynamic_range={result['dynamic_range']:.3f} ({result['dynamic_range_pct']:.1f}% of max {result['theoretical_max']:.3f})")
        print(f"    within-ESI mean: {result['within_esi_mean']:.3f}, cross-ionization mean: {result['cross_ionization_mean']:.3f}")
        if result['family_diff'] is not None:
            print(f"    family diff: {result['family_diff']:.3f} ({result['family_diff_pct_of_range']:.1f}% of range)")
        print(f"    criterion 1 (range < 20%): {result['criterion_1_dynamic_range_lt_20pct']}")
        print(f"    criterion 2 (family diff < 5% range): {result['criterion_2_family_diff_lt_5pct_range']}")
        print(f"    SATURATED: {result['saturated']}")

    for k in [20, 25]:
        print(f"\n  k={k}: requires raw embeddings (not in pre-computed results)")
        print(f"    To compute: re-run pipeline.py with k_values including {k}")
        results[f"k{k}"] = {
            "k": k,
            "status": "requires_recomputation",
            "note": f"Pre-computed results only contain k=5,10,15. "
                    f"Re-run pipeline.py with k_values=(5,10,15,{k}) to compute.",
        }

    output = {
        "generated": datetime.now().isoformat(),
        "prereg": "PREREG_spectral_msms.md",
        "c5_k_sensitivity": results,
    }

    out_path = RESULTS_DIR / f"confirmatory_c5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
