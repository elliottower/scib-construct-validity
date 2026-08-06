"""
Supplementary Analysis S4: Ecological-Inference Bounds (Duncan-Davis)

Computes Duncan-Davis bounds on elderly (80+) mortality RATE using
only site-level data from each CDC pseudo-site. Also computes the
observed individual-level elderly mortality rate for comparison.
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_observed_elderly_mortality_rate():
    """Stream CDC CSV to compute actual 80+ mortality rate."""
    cache = DATA_DIR / "cdc_case_surveillance/results/age_group_mortality.json"
    if cache.exists():
        with open(cache) as f:
            age_mort = json.load(f)
        for ag, data in age_mort.items():
            if "80" in ag:
                return data["mortality_rate"], data["total"], data["died"]

    total_80plus = 0
    died_80plus = 0
    with open(DATA_DIR / "cdc_case_surveillance/cdc_full.csv",
              encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ag = row.get("age_group", "").strip()
            if "80" in ag:
                total_80plus += 1
                if row.get("death_yn", "").strip() == "Yes":
                    died_80plus += 1

    rate = died_80plus / total_80plus if total_80plus > 0 else 0
    return rate, total_80plus, died_80plus


def duncan_davis_bounds(total_mortality, elderly_prop):
    """Compute Duncan-Davis bounds on elderly mortality rate."""
    m = total_mortality
    p = elderly_prop

    if p <= 0 or p >= 1:
        return None, None

    lower = max(0.0, (m - (1 - p)) / p)
    upper = min(1.0, m / p)

    return lower, upper


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S4: Ecological-Inference Bounds (Duncan-Davis)  [{ts}]")
    print("=" * 70)

    with open(DATA_DIR / "cdc_case_surveillance/results/cdc_ecological_fallacy.json") as f:
        cdc = json.load(f)

    print("\nComputing observed elderly (80+) mortality rate...")
    observed_rate, n_elderly, n_died = compute_observed_elderly_mortality_rate()
    print(f"  Observed: {n_died}/{n_elderly} = {observed_rate:.4f} "
          f"({observed_rate*100:.2f}%)")

    print("\nDuncan-Davis bounds per site:")
    site_bounds = []
    for s in cdc["site_data"]:
        p = s["prop_80plus"]
        m = s["prop_died"]

        if p <= 0:
            print(f"  {s['site']:30s}  p_elderly=0 — bounds undefined (skipped)")
            site_bounds.append({
                "site": s["site"],
                "n": s["n"],
                "mortality_rate": m,
                "elderly_prop": p,
                "lower": None,
                "upper": None,
                "note": "elderly proportion is 0, bounds undefined",
            })
            continue

        lower, upper = duncan_davis_bounds(m, p)
        width = upper - lower
        contains_truth = lower <= observed_rate <= upper

        print(f"  {s['site']:30s}  m={m:.4f}  p={p:.4f}  "
              f"bounds=[{lower:.4f}, {upper:.4f}]  "
              f"width={width:.4f}  "
              f"{'contains' if contains_truth else 'EXCLUDES'} truth")

        site_bounds.append({
            "site": s["site"],
            "n": s["n"],
            "mortality_rate": m,
            "elderly_prop": p,
            "lower": lower,
            "upper": upper,
            "width": width,
            "contains_observed": contains_truth,
        })

    valid_bounds = [b for b in site_bounds if b["lower"] is not None]
    if valid_bounds:
        intersection_lower = max(b["lower"] for b in valid_bounds)
        intersection_upper = min(b["upper"] for b in valid_bounds)
        intersection_width = max(0, intersection_upper - intersection_lower)
        intersection_valid = intersection_lower <= intersection_upper
        intersection_contains = (intersection_valid and
                                 intersection_lower <= observed_rate <= intersection_upper)
    else:
        intersection_lower = None
        intersection_upper = None
        intersection_width = None
        intersection_valid = False
        intersection_contains = False

    print(f"\nIntersection of bounds across {len(valid_bounds)} valid sites:")
    if intersection_valid:
        print(f"  [{intersection_lower:.4f}, {intersection_upper:.4f}]  "
              f"width={intersection_width:.4f}")
        print(f"  Contains observed rate ({observed_rate:.4f}): "
              f"{'YES' if intersection_contains else 'NO'}")
    else:
        print("  Empty intersection — bounds from different sites are "
              "incompatible.")

    if intersection_width is not None and intersection_width < 0.05:
        if not intersection_contains:
            conclusion = ("FALSIFICATION: Bounds are tight (< 5pp) but "
                          "exclude the observed rate. Something is wrong.")
        else:
            conclusion = ("Bounds are tight (< 5pp) and contain the observed "
                          "rate. Ecological data is informative about elderly "
                          "mortality — a constructive finding.")
    elif intersection_valid:
        conclusion = (f"Bounds are wide ({intersection_width:.1%}). Ecological "
                      "data alone is uninformative about elderly mortality — "
                      "supports the paper's claims.")
    else:
        conclusion = ("Bounds have empty intersection. Individual sites "
                      "provide contradictory constraints, consistent with "
                      "substantial cross-site heterogeneity.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "observed_elderly_mortality_rate": observed_rate,
        "n_elderly_80plus": n_elderly,
        "n_died_80plus": n_died,
        "individual_or_80plus": cdc["individual_or_80plus"],
        "site_bounds": site_bounds,
        "intersection": {
            "lower": intersection_lower,
            "upper": intersection_upper,
            "width": intersection_width,
            "valid": intersection_valid,
            "contains_observed": intersection_contains,
        },
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s4_ecological_bounds.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
