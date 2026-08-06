"""
Compute spectral-gap ratio null distributions for all (k, d) pairs in the V3b panel.
Runs locally (no Census needed). Output: null_distributions.json

Pre-registration: PREREGISTRATION_SPECTRAL_GAP.md
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

RESULTS_DIR = Path("results/spectral_gap")
NULL_TRIALS = 5000
NULL_SEED = 42


def spectral_gap_ratio(C, j):
    """Explained variance ratio of top-j singular values of centered C."""
    Cc = C - C.mean(axis=0)
    sv = np.linalg.svd(Cc, compute_uv=False)
    sv2 = sv ** 2
    total = sv2.sum()
    if total < 1e-12:
        return 0.0
    return float(sv2[:j].sum() / total)


def compute_null_for_kd(k, d, j_values, n_trials, rng):
    """Null distribution of SGR_j for random Gaussian k × d matrices."""
    results = {}
    for j_spec in j_values:
        j = j_spec if j_spec is not None else max(1, (k - 1) // 3)
        j = min(j, k - 1)
        vals = np.empty(n_trials)
        for i in range(n_trials):
            C = rng.standard_normal((k, d))
            vals[i] = spectral_gap_ratio(C, j)
        label = f"j={j_spec}" if j_spec is not None else f"j=k/3={j}"
        results[label] = {
            'j': int(j),
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'ci_95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
            'p5': float(np.percentile(vals, 5)),
            'p95': float(np.percentile(vals, 95)),
        }
    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    v3b_path = Path("results/biological_structure_v3b/biological_structure_v3b/per_condition.json")
    with open(v3b_path) as f:
        conditions = json.load(f)

    kd_pairs = sorted(set((c['n_shared_types'], c['d']) for c in conditions))
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} Computing SGR null for {len(kd_pairs)} (k, d) pairs")
    print(f"  Trials per pair: {NULL_TRIALS}")
    print(f"  j values: 1, 2, k/3")

    rng = np.random.default_rng(NULL_SEED)
    j_values = [1, 2, None]

    all_nulls = {}
    for k, d in tqdm(kd_pairs, desc="Null distributions"):
        key = f"k={k}_d={d}"
        all_nulls[key] = compute_null_for_kd(k, d, j_values, NULL_TRIALS, rng)

    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'prereg': 'PREREGISTRATION_SPECTRAL_GAP.md',
        'n_trials': NULL_TRIALS,
        'seed': NULL_SEED,
        'j_specs': ['1', '2', 'k/3'],
        'n_kd_pairs': len(kd_pairs),
        'null_distributions': all_nulls,
    }

    out_path = RESULTS_DIR / "null_distributions.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n--- Summary ---")
    print(f"{'k':>4s}  {'d':>5s}  {'SGR1_null_mean':>14s}  {'SGR1_null_std':>13s}  {'SGR2_null_mean':>14s}")
    for k, d in kd_pairs[:10]:
        key = f"k={k}_d={d}"
        s1 = all_nulls[key]['j=1']
        s2 = all_nulls[key]['j=2']
        print(f"{k:4d}  {d:5d}  {s1['mean']:14.6f}  {s1['std']:13.6f}  {s2['mean']:14.6f}")
    if len(kd_pairs) > 10:
        print(f"  ... and {len(kd_pairs) - 10} more pairs")


if __name__ == '__main__':
    main()
