"""
Supplementary Analysis S2: Within/Between Effect Decomposition (Mundlak)

Fits a Mundlak fixed-effects approximation on Mexico individual-level
data (3.99M patients, 32 states) to decompose the age effect into
within-site and between-site components.

Estimator: standard logistic with state dummies + within/between age
covariates (not a GLMM — avoids convergence issues at 4M rows).
"""

import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

MEXICO_CSV = DATA_DIR / "mexico_covid/COVID19MEXICO.csv"
AGE_THRESHOLD = 70


def load_mexico_data():
    print("  Loading Mexico individual-level data...")
    records = []
    with open(MEXICO_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                age = int(row["EDAD"])
            except (ValueError, KeyError):
                continue

            state = row.get("ENTIDAD_RES", "").strip()
            if not state:
                continue

            fecha_def = row.get("FECHA_DEF", "").strip()
            died = 0 if fecha_def in ("", "9999-99-99") else 1

            records.append({
                "state": state,
                "age": age,
                "died": died,
                "elderly": 1 if age >= AGE_THRESHOLD else 0,
            })

    print(f"  Loaded {len(records):,} records")
    return records


def compute_state_means(records):
    state_sums = {}
    state_counts = {}
    for r in records:
        s = r["state"]
        state_sums[s] = state_sums.get(s, 0) + r["elderly"]
        state_counts[s] = state_counts.get(s, 0) + 1
    return {s: state_sums[s] / state_counts[s] for s in state_sums}


def fit_mundlak(records, state_means):
    from statsmodels.discrete.discrete_model import Logit
    import pandas as pd

    states = sorted(set(r["state"] for r in records))
    state_to_idx = {s: i for i, s in enumerate(states)}

    n = len(records)
    n_states = len(states)

    print(f"  Building design matrix: {n:,} rows, {n_states} states...")

    y = np.zeros(n, dtype=np.float64)
    elderly_within = np.zeros(n, dtype=np.float64)
    elderly_between = np.zeros(n, dtype=np.float64)
    state_dummies = np.zeros((n, n_states - 1), dtype=np.float64)

    for i, r in enumerate(records):
        y[i] = r["died"]
        mean_elderly = state_means[r["state"]]
        elderly_within[i] = r["elderly"] - mean_elderly
        elderly_between[i] = mean_elderly
        idx = state_to_idx[r["state"]]
        if idx > 0:
            state_dummies[i, idx - 1] = 1.0

    X = np.column_stack([
        np.ones(n),
        elderly_within,
        elderly_between,
        state_dummies,
    ])

    col_names = (["const", "elderly_within", "elderly_between"] +
                 [f"state_{states[i]}" for i in range(1, n_states)])

    print("  Fitting logistic regression (this may take a few minutes)...")
    model = Logit(y, X)
    result = model.fit(method="newton", maxiter=100, disp=False)

    within_coef = result.params[1]
    between_coef = result.params[2]
    within_se = result.bse[1]
    between_se = result.bse[2]
    within_p = result.pvalues[1]
    between_p = result.pvalues[2]

    within_or = np.exp(within_coef)
    between_or = np.exp(between_coef)
    within_ci = (np.exp(within_coef - 1.96 * within_se),
                 np.exp(within_coef + 1.96 * within_se))
    between_ci = (np.exp(between_coef - 1.96 * between_se),
                  np.exp(between_coef + 1.96 * between_se))

    return {
        "within_coef": float(within_coef),
        "within_se": float(within_se),
        "within_or": float(within_or),
        "within_ci": [float(within_ci[0]), float(within_ci[1])],
        "within_p": float(within_p),
        "between_coef": float(between_coef),
        "between_se": float(between_se),
        "between_or": float(between_or),
        "between_ci": [float(between_ci[0]), float(between_ci[1])],
        "between_p": float(between_p),
        "difference_coef": float(between_coef - within_coef),
        "n_obs": n,
        "n_states": n_states,
        "converged": bool(result.mle_retvals.get("converged", True)),
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S2: Within/Between Effect Decomposition  [{ts}]")
    print(f"  Age threshold: {AGE_THRESHOLD}+")
    print("=" * 70)

    records = load_mexico_data()
    state_means = compute_state_means(records)

    print(f"\n  State elderly proportions (min-max): "
          f"{min(state_means.values()):.4f} - {max(state_means.values()):.4f}")

    result = fit_mundlak(records, state_means)

    print(f"\n  Within-site coefficient:  {result['within_coef']:+.4f} "
          f"(OR={result['within_or']:.2f}, "
          f"95% CI [{result['within_ci'][0]:.2f}, {result['within_ci'][1]:.2f}], "
          f"p={result['within_p']:.2e})")
    print(f"  Between-site coefficient: {result['between_coef']:+.4f} "
          f"(OR={result['between_or']:.2f}, "
          f"95% CI [{result['between_ci'][0]:.2f}, {result['between_ci'][1]:.2f}], "
          f"p={result['between_p']:.2e})")
    print(f"  Difference (aggregation bias): {result['difference_coef']:+.4f}")

    ci_overlap = (result["within_ci"][1] >= result["between_ci"][0] and
                  result["between_ci"][1] >= result["within_ci"][0])

    if ci_overlap:
        conclusion = ("FALSIFICATION: Within and between CIs overlap. "
                       "No statistically distinguishable aggregation bias.")
    else:
        conclusion = (f"Within and between coefficients differ by "
                       f"{result['difference_coef']:+.4f} with non-overlapping "
                       f"CIs. Aggregation bias is {result['between_or']/result['within_or']:.1f}x.")

    print(f"\n  {conclusion}")

    output = {
        "timestamp": ts,
        "age_threshold": AGE_THRESHOLD,
        "estimator": "Mundlak fixed-effects approximation (logistic + state dummies)",
        **result,
        "ci_overlap": ci_overlap,
        "conclusion": conclusion,
    }

    outpath = OUTPUT_DIR / "s2_mundlak_decomposition.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
