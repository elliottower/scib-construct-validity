"""H-statistic null calibration for the noise experiment.

Permutes sigma labels within each condition to estimate how often
H > 0.01 arises from pure seed-to-seed jitter (no real noise effect).
Replaces the arbitrary 0.01 threshold with a per-metric significance test.

Reads: results/noise_multiseed/merged/noise_multiseed_raw.json
Writes: results/noise_multiseed/merged/h_null_calibration.json

Usage:
    uv run python scripts/h_stat_null_calibration.py
"""
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

RAW_PATH = Path("results/noise_multiseed/merged/noise_multiseed_raw.json")
OUTPUT_PATH = Path("results/noise_multiseed/merged/h_null_calibration.json")

N_PERMUTATIONS = 1000
SEED = 20260730
THRESHOLD = 0.01


def main():
    with open(RAW_PATH) as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} rows")

    conditions = defaultdict(list)
    for row in raw:
        key = (row["tissue"], row["model"])
        conditions[key].append(row)

    all_metrics = set()
    for row in raw:
        all_metrics.update(row.get("h_stats", {}).keys())
    all_metrics = sorted(all_metrics)
    print(f"Metrics: {all_metrics}")

    rng = np.random.default_rng(SEED)

    results_by_metric = {}

    for metric in tqdm(all_metrics, desc="Metrics"):
        n_raw_positive = 0
        n_calibrated_positive = 0
        n_conditions_valid = 0
        per_condition = []

        for (tissue, model), rows in sorted(conditions.items()):
            seed_h_values = []

            for row in rows:
                h = row.get("h_stats", {}).get(metric)
                if h is not None:
                    seed_h_values.append(h)

            if not seed_h_values:
                continue

            mean_h_observed = float(np.mean(seed_h_values))
            is_raw_positive = mean_h_observed > THRESHOLD

            per_seed_curves = []
            for row in rows:
                sbs = row.get("scores_by_sigma", {})
                curve = {}
                for sigma_str, metric_dict in sbs.items():
                    val = metric_dict.get(metric)
                    if val is not None:
                        curve[float(sigma_str)] = val
                if 0.0 in curve and len(curve) > 1:
                    per_seed_curves.append(curve)

            if per_seed_curves:
                unique_sigmas = sorted(set(s for c in per_seed_curves for s in c))

                null_h_distribution = []
                for _ in range(N_PERMUTATIONS):
                    perm_seed_hs = []
                    for curve in per_seed_curves:
                        sigmas_this_seed = sorted(curve.keys())
                        vals_this_seed = [curve[s] for s in sigmas_this_seed]
                        perm_vals = rng.permutation(vals_this_seed)
                        perm_curve = dict(zip(sigmas_this_seed, perm_vals))
                        baseline = perm_curve.get(0.0, 0.0)
                        h_perm = max(
                            perm_curve.get(s, baseline) - baseline
                            for s in sigmas_this_seed if s > 0
                        )
                        perm_seed_hs.append(h_perm)
                    null_h_distribution.append(float(np.mean(perm_seed_hs)))

                if null_h_distribution:
                    null_95 = float(np.percentile(null_h_distribution, 95))
                    p_value = float(np.mean([h >= mean_h_observed for h in null_h_distribution]))
                    is_calibrated_positive = mean_h_observed > null_95
                else:
                    null_95 = None
                    p_value = None
                    is_calibrated_positive = False
            else:
                null_95 = None
                p_value = None
                is_calibrated_positive = is_raw_positive

            n_conditions_valid += 1
            if is_raw_positive:
                n_raw_positive += 1
            if is_calibrated_positive:
                n_calibrated_positive += 1

            per_condition.append({
                "tissue": tissue,
                "model": model,
                "mean_h_observed": mean_h_observed,
                "null_95th": null_95,
                "perm_p_value": p_value,
                "raw_positive": is_raw_positive,
                "calibrated_positive": is_calibrated_positive,
            })

        frac_raw = n_raw_positive / n_conditions_valid if n_conditions_valid else 0
        frac_cal = n_calibrated_positive / n_conditions_valid if n_conditions_valid else 0

        results_by_metric[metric] = {
            "n_conditions": n_conditions_valid,
            "n_raw_positive": n_raw_positive,
            "frac_raw_positive": round(frac_raw, 3),
            "n_calibrated_positive": n_calibrated_positive,
            "frac_calibrated_positive": round(frac_cal, 3),
            "inflation_ratio": round(frac_raw / frac_cal, 2) if frac_cal > 0 else None,
            "per_condition": per_condition,
        }

        print(f"  {metric:25s}: raw={n_raw_positive}/{n_conditions_valid} "
              f"({frac_raw:.0%}), calibrated={n_calibrated_positive}/{n_conditions_valid} "
              f"({frac_cal:.0%})")

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_permutations": N_PERMUTATIONS,
        "threshold": THRESHOLD,
        "seed": SEED,
        "n_total_rows": len(raw),
        "n_conditions": len(conditions),
        "by_metric": results_by_metric,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
