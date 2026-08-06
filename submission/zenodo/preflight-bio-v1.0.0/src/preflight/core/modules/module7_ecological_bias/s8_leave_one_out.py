"""
Supplementary Analysis S8: Leave-One-Site-Out Stability

For each of the ~20 4CE sites, drops it and recomputes fixed-effects PMI
and random-effects PMI. Reports whether the 350-fold RE/FE divergence
is driven by a single site.
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

PMI_FILE = DATA_DIR / "dag_validation/data/4ce_neuro/site_pmi_effects.csv"
TIMEPOINT = "90"
EFFECT = "pmi.cns"


def load_pmi_data():
    sites = {}
    with open(PMI_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["timepoint"] == TIMEPOINT and row["effect_name"] == EFFECT:
                sites[row["site"]] = {
                    "log_pmi": float(row["log_pmi"]),
                    "se": float(row["se"]),
                    "pmi": float(row["pmi"]),
                }
    return sites


def fixed_effects_meta(site_data):
    weights = [1.0 / (s["se"] ** 2) for s in site_data.values()]
    estimates = [s["log_pmi"] for s in site_data.values()]
    w_sum = sum(weights)
    fe_estimate = sum(w * e for w, e in zip(weights, estimates)) / w_sum
    fe_se = 1.0 / np.sqrt(w_sum)
    return {
        "log_pmi": float(fe_estimate),
        "se": float(fe_se),
        "pmi": float(np.exp(fe_estimate)),
    }


def random_effects_meta(site_data):
    estimates = np.array([s["log_pmi"] for s in site_data.values()])
    variances = np.array([s["se"] ** 2 for s in site_data.values()])
    weights_fe = 1.0 / variances

    w_sum = np.sum(weights_fe)
    fe_est = np.sum(weights_fe * estimates) / w_sum

    q = np.sum(weights_fe * (estimates - fe_est) ** 2)
    k = len(estimates)
    c = w_sum - np.sum(weights_fe ** 2) / w_sum
    tau2 = max(0, (q - (k - 1)) / c)

    weights_re = 1.0 / (variances + tau2)
    w_sum_re = np.sum(weights_re)
    re_est = np.sum(weights_re * estimates) / w_sum_re
    re_se = 1.0 / np.sqrt(w_sum_re)

    return {
        "log_pmi": float(re_est),
        "se": float(re_se),
        "pmi": float(np.exp(re_est)),
        "tau2": float(tau2),
        "I2": float(max(0, (q - (k - 1)) / q)) if q > 0 else 0.0,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S8: Leave-One-Site-Out Stability  [{ts}]")
    print(f"  Timepoint: {TIMEPOINT}d, Effect: {EFFECT}")
    print("=" * 70)

    all_sites = load_pmi_data()
    print(f"  Loaded {len(all_sites)} sites")

    full_fe = fixed_effects_meta(all_sites)
    full_re = random_effects_meta(all_sites)
    full_ratio = full_re["pmi"] / full_fe["pmi"] if full_fe["pmi"] != 0 else float("inf")

    print(f"\n  Full dataset:")
    print(f"    FE PMI = {full_fe['pmi']:.4f}")
    print(f"    RE PMI = {full_re['pmi']:.4f}")
    print(f"    RE/FE ratio = {full_ratio:.1f}")
    print(f"    tau² = {full_re['tau2']:.4f}, I² = {full_re['I2']:.3f}")

    print(f"\n  Leave-one-out results:")
    loo_results = {}
    for dropped in sorted(all_sites.keys()):
        remaining = {k: v for k, v in all_sites.items() if k != dropped}
        fe = fixed_effects_meta(remaining)
        re = random_effects_meta(remaining)
        ratio = re["pmi"] / fe["pmi"] if fe["pmi"] != 0 else float("inf")

        loo_results[dropped] = {
            "fe_pmi": fe["pmi"],
            "re_pmi": re["pmi"],
            "re_fe_ratio": ratio,
            "tau2": re["tau2"],
            "I2": re["I2"],
        }
        print(f"    Drop {dropped:8s}:  FE={fe['pmi']:.4f}  RE={re['pmi']:.4f}  "
              f"ratio={ratio:.1f}  tau²={re['tau2']:.4f}")

    ratios = [r["re_fe_ratio"] for r in loo_results.values()]
    min_ratio_site = min(loo_results, key=lambda s: loo_results[s]["re_fe_ratio"])
    min_ratio = loo_results[min_ratio_site]["re_fe_ratio"]
    max_ratio_site = max(loo_results, key=lambda s: loo_results[s]["re_fe_ratio"])

    print(f"\n  RE/FE ratio range: {min(ratios):.1f} to {max(ratios):.1f}")
    print(f"  Most influential site: {min_ratio_site} "
          f"(dropping it gives ratio={min_ratio:.1f})")

    if min_ratio < 10:
        conclusion = (f"FALSIFICATION: Dropping {min_ratio_site} reduces "
                       f"RE/FE divergence to {min_ratio:.1f}-fold (below 10). "
                       f"The {full_ratio:.0f}-fold divergence is site-specific.")
    else:
        conclusion = (f"No single site drives the divergence below 10-fold. "
                       f"Minimum ratio after LOO = {min_ratio:.1f} "
                       f"(dropping {min_ratio_site}). The divergence is a "
                       f"general property of the data.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "timepoint": TIMEPOINT,
        "effect": EFFECT,
        "n_sites": len(all_sites),
        "full": {
            "fe_pmi": full_fe["pmi"],
            "re_pmi": full_re["pmi"],
            "re_fe_ratio": full_ratio,
            "tau2": full_re["tau2"],
            "I2": full_re["I2"],
        },
        "leave_one_out": loo_results,
        "most_influential_site": min_ratio_site,
        "min_ratio_after_loo": min_ratio,
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s8_leave_one_out.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
