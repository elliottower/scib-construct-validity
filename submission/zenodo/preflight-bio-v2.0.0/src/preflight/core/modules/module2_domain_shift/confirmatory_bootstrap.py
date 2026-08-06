"""
Phase 2 confirmatory analyses: C1 (bootstrap CIs), C2 (confound test),
C3 (chordal distance robustness), C4 (minimum detectable effect).

Implements exactly the procedures specified in PREREG_spectral_msms.md.
Any deviation from the pre-registered procedure must be logged in the
DEVIATIONS section of the prereg with date and rationale.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.decomposition import PCA
from datetime import datetime

RESULTS_DIR = Path(__file__).parent.parent / "results"
DATA_FILE = RESULTS_DIR / "transportability_20260702_155956.json"

ESI_TYPES = {
    "LC-ESI-QTOF", "LC-ESI-QFT", "LC-ESI-QQ", "LC-ESI-ITFT",
    "LC-ESI-QIT", "LC-ESI-IT", "ESI-QTOF",
}
EIB = "EI-B"

K_VALUES = [5, 10, 15]
N_BOOTSTRAP = 10_000
N_PERMUTATIONS = 10_000
ALPHA = 0.05
N_METRICS = 6
N_POPULATIONS = 3
N_TESTS = N_METRICS * N_POPULATIONS * len(K_VALUES)  # 54
ALPHA_BONFERRONI = ALPHA / (N_TESTS - 1)  # 53 secondary tests


def load_results():
    with open(DATA_FILE) as f:
        return json.load(f)


def split_populations(pairs):
    all_pairs = pairs
    excl_eib = [p for p in pairs if p["site_a"] != EIB and p["site_b"] != EIB]
    within_esi = [p for p in pairs if p["site_a"] in ESI_TYPES and p["site_b"] in ESI_TYPES]
    return {
        "all_45": all_pairs,
        "excl_eib": excl_eib,
        "within_esi": within_esi,
    }


def bootstrap_spearman_ci(x, y, n_boot=N_BOOTSTRAP, alpha=ALPHA):
    rng = np.random.default_rng(seed=None)
    n = len(x)
    x, y = np.asarray(x), np.asarray(y)
    observed_rho = stats.spearmanr(x, y).statistic

    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic

    lo = np.percentile(rhos, 100 * alpha / 2)
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    se = np.std(rhos, ddof=1)

    return {
        "rho": float(observed_rho),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "bootstrap_se": float(se),
        "ci_includes_zero": bool(lo <= 0 <= hi),
        "n_bootstrap": n_boot,
    }


def chordal_distance(U_a, U_b):
    P_a = U_a @ U_a.T
    P_b = U_b @ U_b.T
    return np.linalg.norm(P_a - P_b, "fro") / np.sqrt(2)


def c1_bootstrap_cis(pairs, embeddings_by_instrument, k_values=K_VALUES):
    """C1: Bootstrap CIs for all metric-degradation correlations."""
    populations = split_populations(pairs)
    results = {}

    for pop_name, pop_pairs in populations.items():
        results[pop_name] = {}
        for k in k_values:
            geo_key = f"geodesic_k{k}"
            cen_key = f"centroid_dist_k{k}"
            dom_key = f"domain_auc_k{k}"

            geo = [p[geo_key] for p in pop_pairs]
            cen = [p[cen_key] for p in pop_pairs]
            dom = [p[dom_key] for p in pop_pairs]
            deg = [p["clf_degradation"] for p in pop_pairs]

            match_deg = [p["match_degradation"] for p in pop_pairs]

            metrics = {
                "geodesic": geo,
                "centroid": cen,
                "domain_auc": dom,
            }

            if embeddings_by_instrument is not None:
                chordal = []
                for p in pop_pairs:
                    a, b = p["site_a"], p["site_b"]
                    if a in embeddings_by_instrument and b in embeddings_by_instrument:
                        pca_a = PCA(n_components=k).fit(embeddings_by_instrument[a])
                        pca_b = PCA(n_components=k).fit(embeddings_by_instrument[b])
                        chordal.append(chordal_distance(
                            pca_a.components_.T, pca_b.components_.T
                        ))
                    else:
                        chordal.append(np.nan)
                if not any(np.isnan(c) for c in chordal):
                    metrics["chordal"] = chordal

            metrics["match_degradation"] = match_deg

            sheaf_h1 = [p.get("sheaf_h1_pairwise", np.nan) for p in pop_pairs]
            if not any(np.isnan(s) for s in sheaf_h1):
                metrics["sheaf_h1"] = sheaf_h1

            k_results = {}
            for metric_name, metric_vals in metrics.items():
                if metric_name == "match_degradation":
                    outcome = match_deg
                else:
                    outcome = deg

                ci_result = bootstrap_spearman_ci(metric_vals, outcome)
                ci_result["metric"] = metric_name
                ci_result["outcome"] = "clf_degradation" if metric_name != "match_degradation" else "match_degradation"
                ci_result["k"] = k
                ci_result["population"] = pop_name
                ci_result["n_pairs"] = len(pop_pairs)

                is_primary = (
                    metric_name == "geodesic"
                    and k == 10
                    and pop_name == "excl_eib"
                )
                ci_result["is_primary"] = is_primary
                ci_result["alpha"] = ALPHA if is_primary else ALPHA_BONFERRONI

                k_results[metric_name] = ci_result

            results[pop_name][f"k{k}"] = k_results

    return results


def c2_confound_test(pairs):
    """C2: Permutation-based confound test for EI-B."""
    rng = np.random.default_rng(seed=None)

    geo_all = np.array([p["geodesic_k10"] for p in pairs])
    deg_all = np.array([p["clf_degradation"] for p in pairs])

    is_eib = np.array([p["site_a"] == EIB or p["site_b"] == EIB for p in pairs])
    n_eib = int(is_eib.sum())
    n_total = len(pairs)

    rho_naive = stats.spearmanr(geo_all, deg_all).statistic
    rho_controlled = stats.spearmanr(geo_all[~is_eib], deg_all[~is_eib]).statistic
    observed_delta = abs(rho_naive) - abs(rho_controlled)

    null_deltas = np.empty(N_PERMUTATIONS)
    for i in range(N_PERMUTATIONS):
        fake_eib = rng.choice(n_total, n_eib, replace=False)
        mask = np.ones(n_total, dtype=bool)
        mask[fake_eib] = False
        rho_fake_naive = stats.spearmanr(geo_all, deg_all).statistic
        rho_fake_controlled = stats.spearmanr(geo_all[mask], deg_all[mask]).statistic
        null_deltas[i] = abs(rho_fake_naive) - abs(rho_fake_controlled)

    p_value = float(np.mean(null_deltas >= observed_delta))

    return {
        "rho_naive": float(rho_naive),
        "rho_controlled": float(rho_controlled),
        "observed_delta": float(observed_delta),
        "n_eib_pairs": n_eib,
        "n_total_pairs": n_total,
        "n_permutations": N_PERMUTATIONS,
        "p_value": p_value,
        "null_mean": float(np.mean(null_deltas)),
        "null_std": float(np.std(null_deltas)),
        "null_95th": float(np.percentile(null_deltas, 95)),
        "confound_significant": p_value < ALPHA,
    }


def c3_chordal_robustness(pairs, embeddings_by_instrument, k_values=K_VALUES):
    """C3: Chordal distance robustness check."""
    if embeddings_by_instrument is None:
        return {"status": "skipped", "reason": "no raw embeddings provided"}

    populations = split_populations(pairs)
    results = {}

    for pop_name, pop_pairs in populations.items():
        results[pop_name] = {}
        for k in k_values:
            chordal_vals = []
            deg_vals = []
            for p in pop_pairs:
                a, b = p["site_a"], p["site_b"]
                if a in embeddings_by_instrument and b in embeddings_by_instrument:
                    pca_a = PCA(n_components=k).fit(embeddings_by_instrument[a])
                    pca_b = PCA(n_components=k).fit(embeddings_by_instrument[b])
                    cd = chordal_distance(pca_a.components_.T, pca_b.components_.T)
                    chordal_vals.append(cd)
                    deg_vals.append(p["clf_degradation"])

            if len(chordal_vals) >= 5:
                ci = bootstrap_spearman_ci(chordal_vals, deg_vals)
                ci["k"] = k
                ci["population"] = pop_name
                ci["n_pairs"] = len(chordal_vals)

                dynamic_range = max(chordal_vals) - min(chordal_vals)
                max_chordal = np.sqrt(2 * k)
                ci["dynamic_range"] = float(dynamic_range)
                ci["dynamic_range_pct"] = float(dynamic_range / max_chordal * 100)
                ci["saturated"] = dynamic_range < 0.20 * max_chordal

                results[pop_name][f"k{k}"] = ci

    return results


def c4_minimum_detectable_effect():
    """C4: Minimum detectable Spearman rho at 80% power."""
    population_sizes = {"all_45": 45, "excl_eib": 36, "within_esi": 21}
    results = {}

    for pop_name, n in population_sizes.items():
        df = n - 2
        t_crit = t_dist.ppf(1 - ALPHA / 2, df)
        rho_min = t_crit / np.sqrt(df + t_crit**2)

        mc_verified = verify_mde_by_simulation(n, rho_min)

        results[pop_name] = {
            "n": n,
            "rho_min_asymptotic": float(rho_min),
            "alpha": ALPHA,
            "power_target": 0.80,
            "mc_power_at_rho_min": mc_verified["power"],
            "mc_n_replicates": mc_verified["n_replicates"],
        }

    return results


def verify_mde_by_simulation(n, rho_target, n_replicates=10_000):
    """Monte Carlo verification of minimum detectable effect."""
    rng = np.random.default_rng(seed=None)
    significant = 0

    for _ in range(n_replicates):
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        y = rho_target * x + np.sqrt(1 - rho_target**2) * z
        _, p = stats.spearmanr(x, y)
        if p < ALPHA:
            significant += 1

    return {"power": significant / n_replicates, "n_replicates": n_replicates}


def main():
    print(f"Phase 2 Confirmatory Analyses — {datetime.now().isoformat()}")
    print("=" * 80)

    pairs = load_results()
    embeddings = None

    print(f"\nLoaded {len(pairs)} instrument pairs")
    pops = split_populations(pairs)
    for name, pop in pops.items():
        print(f"  {name}: {len(pop)} pairs")

    print("\n\n--- C1: Bootstrap CIs ---")
    c1 = c1_bootstrap_cis(pairs, embeddings)
    primary = c1["excl_eib"]["k10"]["geodesic"]
    print(f"  PRIMARY (geodesic, k=10, excl EI-B):")
    print(f"    rho = {primary['rho']:+.3f}, 95% CI [{primary['ci_lo']:+.3f}, {primary['ci_hi']:+.3f}]")
    print(f"    CI includes zero: {primary['ci_includes_zero']}")

    for pop_name in ["all_45", "excl_eib", "within_esi"]:
        for k in K_VALUES:
            geo = c1[pop_name][f"k{k}"]["geodesic"]
            flag = " <-- PRIMARY" if geo["is_primary"] else ""
            print(f"  {pop_name} k={k}: rho={geo['rho']:+.3f} CI[{geo['ci_lo']:+.3f},{geo['ci_hi']:+.3f}] zero_in_CI={geo['ci_includes_zero']}{flag}")

    print("\n\n--- C2: Confound Test ---")
    c2 = c2_confound_test(pairs)
    print(f"  rho_naive = {c2['rho_naive']:+.3f}")
    print(f"  rho_controlled = {c2['rho_controlled']:+.3f}")
    print(f"  delta = {c2['observed_delta']:+.3f}")
    print(f"  permutation p = {c2['p_value']:.4f} (n_perm={c2['n_permutations']})")
    print(f"  confound significant: {c2['confound_significant']}")

    print("\n\n--- C3: Chordal Distance ---")
    c3 = c3_chordal_robustness(pairs, embeddings)
    print(f"  Status: {c3.get('status', 'requires raw embeddings; run with --embeddings flag')}")

    print("\n\n--- C4: Minimum Detectable Effect ---")
    c4 = c4_minimum_detectable_effect()
    for pop_name, mde in c4.items():
        print(f"  {pop_name} (n={mde['n']}): rho_min = {mde['rho_min_asymptotic']:.3f}, MC power = {mde['mc_power_at_rho_min']:.3f}")

    output = {
        "generated": datetime.now().isoformat(),
        "prereg": "PREREG_spectral_msms.md",
        "c1_bootstrap_cis": c1,
        "c2_confound_test": c2,
        "c3_chordal_robustness": c3,
        "c4_minimum_detectable_effect": c4,
    }

    out_path = RESULTS_DIR / f"confirmatory_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
