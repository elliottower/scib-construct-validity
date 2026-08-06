"""
Supplementary Analysis S1: Power vs. Bias Decomposition

Question: Is the CDC ecological null (beta=+0.28, p=0.12) due to low power
(only 9 pseudo-sites) or genuine aggregation bias?

Method: Simulate individual-level data within each site from a logistic model
using the KNOWN individual-level ORs, then run ecological regression on the
site-level aggregates. Uses exact CDC 9-site compositions and sizes.

Conditions:
  (a) CDC-matched (n=9): exact compositions and sizes
  (b) CDC-resampled (n=20,50,100,500): resample from empirical CDC distribution
  (c) Independent confounders (n=9,20,50,100,500): all sites share
      population-average composition

See ANALYSIS_PROTOCOL.md for full specification and decision rules.
"""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

np.random.seed(20260707)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(DATA_DIR / "cdc_case_surveillance/results/cdc_ecological_fallacy.json") as f:
    cdc = json.load(f)

INDIVIDUAL_OR_AGE = cdc["individual_or_80plus"]  # 9.88
INDIVIDUAL_OR_COMORBIDITY = 1.5
INDIVIDUAL_OR_MALE = 1.7
BASELINE_INTERCEPT = -4.0

N_SIMS = 2000
N_SITES_RANGE = [9, 20, 50, 100, 500]

CDC_SITES = []
for s in cdc["site_data"]:
    CDC_SITES.append({
        "name": s["site"],
        "n": s["n"],
        "elderly": s["prop_80plus"],
        "comorbidity": s["prop_comorbidity"],
        "male": s["prop_male"],
    })

POPULATION_MEAN_ELDERLY = np.mean([s["elderly"] for s in CDC_SITES])
POPULATION_MEAN_COMORBIDITY = np.mean([s["comorbidity"] for s in CDC_SITES])
POPULATION_MEAN_MALE = np.mean([s["male"] for s in CDC_SITES])


def simulate_site(n, elderly_prop, comorbidity_prop, male_prop):
    is_elderly = np.random.binomial(1, elderly_prop, n)
    has_comorbidity = np.random.binomial(1, comorbidity_prop, n)
    is_male = np.random.binomial(1, male_prop, n)

    logit_p = (BASELINE_INTERCEPT
               + np.log(INDIVIDUAL_OR_AGE) * is_elderly
               + np.log(INDIVIDUAL_OR_COMORBIDITY) * has_comorbidity
               + np.log(INDIVIDUAL_OR_MALE) * is_male)
    p_death = 1.0 / (1.0 + np.exp(-logit_p))
    died = np.random.binomial(1, p_death, n)
    return elderly_prop, died.mean()


def run_ecological_regression(sites):
    elderly_props = np.array([s[0] for s in sites])
    mortalities = np.array([s[1] for s in sites])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        elderly_props, mortalities
    )
    return {
        "slope": float(slope),
        "p_value": float(p_value),
        "r_squared": float(r_value ** 2),
        "significant": bool(p_value < 0.05),
    }


def condition_a_cdc_matched():
    """Exact CDC 9-site compositions and sizes."""
    sites = []
    for s in CDC_SITES:
        sites.append(simulate_site(s["n"], s["elderly"], s["comorbidity"], s["male"]))
    return run_ecological_regression(sites)


def condition_b_cdc_resampled(n_sites):
    """Resample sites from empirical CDC distribution (with replacement)."""
    idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
    sites = []
    for i in idx:
        s = CDC_SITES[i]
        sites.append(simulate_site(s["n"], s["elderly"], s["comorbidity"], s["male"]))
    return run_ecological_regression(sites)


def condition_c_independent_confounders(n_sites):
    """Sites have elderly variation but confounders drawn independently."""
    elderly_vals = [s["elderly"] for s in CDC_SITES]
    comorbidity_vals = [s["comorbidity"] for s in CDC_SITES]
    male_vals = [s["male"] for s in CDC_SITES]
    size_vals = [s["n"] for s in CDC_SITES]

    elderly_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
    comorbidity_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
    male_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)
    size_idx = np.random.choice(len(CDC_SITES), size=n_sites, replace=True)

    sites = []
    for i in range(n_sites):
        sites.append(simulate_site(
            size_vals[size_idx[i]],
            elderly_vals[elderly_idx[i]],
            comorbidity_vals[comorbidity_idx[i]],
            male_vals[male_idx[i]],
        ))
    return run_ecological_regression(sites)


def run_condition(label, sim_fn, n_sims):
    sig_count = 0
    slopes = []
    p_values = []
    for _ in range(n_sims):
        res = sim_fn()
        if res["significant"]:
            sig_count += 1
        slopes.append(res["slope"])
        p_values.append(res["p_value"])

    power = sig_count / n_sims
    return {
        "power": power,
        "mean_slope": float(np.mean(slopes)),
        "median_slope": float(np.median(slopes)),
        "std_slope": float(np.std(slopes)),
        "mean_p": float(np.mean(p_values)),
        "median_p": float(np.median(p_values)),
        "n_sims": n_sims,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S1: Power vs. Bias Decomposition Simulation  [{ts}]")
    print(f"  Individual-level ORs: age={INDIVIDUAL_OR_AGE:.2f}, "
          f"comorbidity={INDIVIDUAL_OR_COMORBIDITY}, male={INDIVIDUAL_OR_MALE}")
    print(f"  N_SIMS = {N_SIMS} per condition")
    print(f"  CDC sites: {len(CDC_SITES)}")
    print(f"  Population means: elderly={POPULATION_MEAN_ELDERLY:.4f}, "
          f"comorbidity={POPULATION_MEAN_COMORBIDITY:.4f}, "
          f"male={POPULATION_MEAN_MALE:.4f}")
    print("=" * 70)

    results = {}

    print("\n[a] CDC-matched (n=9, exact compositions and sizes)")
    results["cdc_matched_9"] = run_condition(
        "CDC-matched n=9", condition_a_cdc_matched, N_SIMS
    )
    print(f"    power={results['cdc_matched_9']['power']:.3f}  "
          f"mean_slope={results['cdc_matched_9']['mean_slope']:+.4f}")

    for n_sites in N_SITES_RANGE:
        if n_sites == 9:
            continue
        label = f"cdc_resampled_{n_sites}"
        print(f"\n[b] CDC-resampled (n={n_sites})")
        results[label] = run_condition(
            label, lambda ns=n_sites: condition_b_cdc_resampled(ns), N_SIMS
        )
        print(f"    power={results[label]['power']:.3f}  "
              f"mean_slope={results[label]['mean_slope']:+.4f}")

    for n_sites in N_SITES_RANGE:
        label = f"independent_{n_sites}"
        print(f"\n[c] Independent confounders (n={n_sites})")
        results[label] = run_condition(
            label, lambda ns=n_sites: condition_c_independent_confounders(ns), N_SIMS
        )
        print(f"    power={results[label]['power']:.3f}  "
              f"mean_slope={results[label]['mean_slope']:+.4f}")

    print("\n" + "=" * 70)
    print("DECISION RULE CHECK")
    print("=" * 70)

    power_b_50 = results.get("cdc_resampled_50", {}).get("power", None)
    power_b_500 = results.get("cdc_resampled_500", {}).get("power", None)
    power_c_500 = results.get("independent_500", {}).get("power")
    power_a = results["cdc_matched_9"]["power"]

    print(f"\n  CDC-matched n=9:          power = {power_a:.3f}")
    if power_b_50 is not None:
        print(f"  CDC-resampled n=50:       power = {power_b_50:.3f}")
    if power_b_500 is not None:
        print(f"  CDC-resampled n=500:      power = {power_b_500:.3f}")
    if power_c_500 is not None:
        print(f"  Independent conf n=500:   power = {power_c_500:.3f}")

    if power_b_50 is not None and power_b_50 > 0.80:
        conclusion = ("FALSIFICATION TRIGGERED: Power > 80% at n<=50 under "
                       "CDC-matched confounding. The CDC null is primarily "
                       "a power problem, not bias.")
    elif (power_b_500 is not None and power_c_500 is not None
          and power_b_500 < 0.50 and power_c_500 > 0.80):
        conclusion = ("BIAS CONFIRMED: Power stays below 50% at n=500 with "
                       "CDC confounding but exceeds 80% with independent "
                       "confounders. The null is driven by correlated "
                       "confounding, not low power.")
    else:
        conclusion = ("MIXED: Neither clear falsification nor clear "
                       "confirmation. Both power and bias contribute.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "seed": 20260707,
        "individual_ors": {
            "age_80plus": INDIVIDUAL_OR_AGE,
            "comorbidity": INDIVIDUAL_OR_COMORBIDITY,
            "male": INDIVIDUAL_OR_MALE,
        },
        "n_sims": N_SIMS,
        "cdc_sites": CDC_SITES,
        "population_means": {
            "elderly": POPULATION_MEAN_ELDERLY,
            "comorbidity": POPULATION_MEAN_COMORBIDITY,
            "male": POPULATION_MEAN_MALE,
        },
        "results": results,
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s1_power_simulation.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
