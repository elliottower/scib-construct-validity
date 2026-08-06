"""Extended construct-validity tests on scIB audit results.

Tests:
  1. Rank-inversion rate vs ground-truth F1 (within-tissue pairwise)
  2. Cross-tissue rank stability (Kendall's W)
  3. Metric-metric redundancy (pairwise Spearman across 24 conditions)
  4. Discriminative-vs-predictive dissociation

Input: results/exp10_scib_audit/summary.json
Output: results/exp10_scib_audit/extended_validity.json

Pre-registration: docs/PREREGISTRATION_SCIB_EXTENDED_VALIDITY.md
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path("results/exp10_scib_audit")
INPUT_PATH = RESULTS_DIR / "summary.json"
OUTPUT_PATH = RESULTS_DIR / "extended_validity.json"

BIO_METRICS = ["nmi_leiden", "ari_leiden", "silhouette_label", "clisi", "isolated_label_asw"]
BATCH_METRICS = ["silhouette_batch", "ilisi", "graph_connectivity", "pcr_comparison"]
ALL_METRICS = BIO_METRICS + BATCH_METRICS

TISSUES = ["lung", "liver", "kidney", "brain"]
EMBEDDINGS = ["geneformer", "scvi", "scgpt", "random_projection", "untrained_encoder", "bog_pca_512"]
CONTENDERS = ["geneformer", "scvi", "scgpt", "bog_pca_512"]
TRAINED_ONLY = ["geneformer", "scvi", "scgpt"]


def load_data():
    with open(INPUT_PATH) as f:
        data = json.load(f)
    return data["raw_scores"], data["raw_f1s"]


def get_metric_vector(raw_scores, metric):
    """Get metric values for all 24 conditions, ordered by tissue x embedding."""
    values = []
    for tissue in TISSUES:
        for emb in EMBEDDINGS:
            key = f"{tissue}_{emb}"
            val = raw_scores[key].get(metric)
            values.append(val)
    return values


def get_f1_vector(raw_f1s):
    values = []
    for tissue in TISSUES:
        for emb in EMBEDDINGS:
            key = f"{tissue}_{emb}"
            values.append(raw_f1s[key])
    return values


def _compute_inversions(raw_scores, raw_f1s, emb_list, metrics):
    """Count pairwise rank inversions vs F1, within each tissue."""
    results = {}
    for metric in metrics:
        total_concordant = 0
        total_discordant = 0
        total_ties = 0
        tissue_details = {}

        for tissue in TISSUES:
            concordant = 0
            discordant = 0
            ties = 0
            inversions = []

            for i, j in combinations(range(len(emb_list)), 2):
                key_i = f"{tissue}_{emb_list[i]}"
                key_j = f"{tissue}_{emb_list[j]}"

                m_i = raw_scores[key_i].get(metric)
                m_j = raw_scores[key_j].get(metric)
                f_i = raw_f1s[key_i]
                f_j = raw_f1s[key_j]

                if m_i is None or m_j is None:
                    ties += 1
                    continue

                metric_sign = np.sign(m_i - m_j)
                f1_sign = np.sign(f_i - f_j)

                if metric_sign == 0 or f1_sign == 0:
                    ties += 1
                elif metric_sign == f1_sign:
                    concordant += 1
                else:
                    discordant += 1
                    inversions.append({
                        "a": emb_list[i], "b": emb_list[j],
                        "metric_a": float(m_i), "metric_b": float(m_j),
                        "f1_a": float(f_i), "f1_b": float(f_j),
                    })

            tissue_details[tissue] = {
                "concordant": concordant,
                "discordant": discordant,
                "ties": ties,
                "inversions": inversions,
            }
            total_concordant += concordant
            total_discordant += discordant
            total_ties += ties

        denom = total_concordant + total_discordant
        rate = total_discordant / denom if denom > 0 else None
        results[metric] = {
            "inversion_rate": round(rate, 3) if rate is not None else None,
            "concordant": total_concordant,
            "discordant": total_discordant,
            "ties": total_ties,
            "per_tissue": tissue_details,
        }

    return results


def test1_rank_inversions(raw_scores, raw_f1s):
    """Count pairwise rank inversions vs F1 under three condition sets."""
    return {
        "all_6": _compute_inversions(raw_scores, raw_f1s, EMBEDDINGS, BIO_METRICS),
        "contenders_4": _compute_inversions(raw_scores, raw_f1s, CONTENDERS, BIO_METRICS),
        "trained_only_3": _compute_inversions(raw_scores, raw_f1s, TRAINED_ONLY, BIO_METRICS),
    }


def test2_cross_tissue_stability(raw_scores):
    """Kendall's W for embedding rankings across tissues."""
    results = {}
    for metric in ALL_METRICS:
        rankings = []
        for tissue in TISSUES:
            values = []
            for emb in EMBEDDINGS:
                key = f"{tissue}_{emb}"
                val = raw_scores[key].get(metric)
                values.append(val if val is not None else float("nan"))

            if all(np.isnan(v) if isinstance(v, float) else v is None for v in values):
                rankings.append(None)
                continue

            arr = np.array(values, dtype=float)
            ranked = stats.rankdata(arr, method="average", nan_policy="omit")
            rankings.append(ranked.tolist())

        valid_rankings = [r for r in rankings if r is not None]
        if len(valid_rankings) < 2:
            results[metric] = {"W": None, "note": "insufficient non-null tissues"}
            continue

        R = np.array(valid_rankings)
        k = R.shape[0]  # number of raters (tissues)
        n = R.shape[1]  # number of items (embeddings)
        col_sums = R.sum(axis=0)
        S = np.sum((col_sums - col_sums.mean()) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n))

        per_tissue_ranks = {}
        for i, tissue in enumerate(TISSUES):
            if rankings[i] is not None:
                rank_order = sorted(range(len(EMBEDDINGS)),
                                    key=lambda x: rankings[i][x], reverse=True)
                per_tissue_ranks[tissue] = [EMBEDDINGS[idx] for idx in rank_order]

        results[metric] = {
            "W": round(float(W), 3),
            "k_tissues": len(valid_rankings),
            "per_tissue_ranking": per_tissue_ranks,
        }

    return results


def test3_metric_redundancy(raw_scores):
    """Pairwise Spearman between all metrics across 24 conditions."""
    vectors = {}
    for metric in ALL_METRICS:
        vec = get_metric_vector(raw_scores, metric)
        vec_clean = [v if v is not None else float("nan") for v in vec]
        vectors[metric] = np.array(vec_clean)

    pairwise = {}
    for m1, m2 in combinations(ALL_METRICS, 2):
        v1, v2 = vectors[m1], vectors[m2]
        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() < 5:
            pairwise[f"{m1}_vs_{m2}"] = {"rho": None, "p": None, "n": int(mask.sum())}
            continue
        rho, p = stats.spearmanr(v1[mask], v2[mask])
        pairwise[f"{m1}_vs_{m2}"] = {
            "rho": round(float(rho), 3),
            "p": round(float(p), 4),
            "n": int(mask.sum()),
        }

    correlation_matrix = {}
    for m in ALL_METRICS:
        row = {}
        for m2 in ALL_METRICS:
            if m == m2:
                row[m2] = 1.0
            else:
                key = f"{m}_vs_{m2}" if f"{m}_vs_{m2}" in pairwise else f"{m2}_vs_{m}"
                row[m2] = pairwise.get(key, {}).get("rho")
        correlation_matrix[m] = row

    return {"pairwise": pairwise, "correlation_matrix": correlation_matrix}


def test1b_f1_gap_split(raw_scores, raw_f1s, gap_threshold=0.10):
    """Split contenders-4 inversions by F1 gap: clear winners vs near-ties."""
    results = {}
    for metric in BIO_METRICS:
        clear_conc, clear_disc = 0, 0
        tie_conc, tie_disc = 0, 0
        clear_inversions = []
        per_tissue = {}

        for tissue in TISSUES:
            t_clear_conc, t_clear_disc = 0, 0
            t_tie_conc, t_tie_disc = 0, 0

            for i, j in combinations(range(len(CONTENDERS)), 2):
                ki = f"{tissue}_{CONTENDERS[i]}"
                kj = f"{tissue}_{CONTENDERS[j]}"
                mi = raw_scores[ki].get(metric)
                mj = raw_scores[kj].get(metric)
                fi, fj = raw_f1s[ki], raw_f1s[kj]

                if mi is None or mj is None:
                    continue
                m_sign = np.sign(mi - mj)
                f_sign = np.sign(fi - fj)
                if m_sign == 0 or f_sign == 0:
                    continue

                f1_gap = abs(fi - fj)
                is_concordant = (m_sign == f_sign)
                is_clear = f1_gap > gap_threshold

                if is_clear:
                    if is_concordant:
                        clear_conc += 1
                        t_clear_conc += 1
                    else:
                        clear_disc += 1
                        t_clear_disc += 1
                        winner_f1 = CONTENDERS[i] if fi > fj else CONTENDERS[j]
                        winner_metric = CONTENDERS[i] if mi > mj else CONTENDERS[j]
                        clear_inversions.append({
                            "tissue": tissue,
                            "metric_picks": winner_metric,
                            "f1_picks": winner_f1,
                            "f1_gap": round(float(f1_gap), 4),
                        })
                else:
                    if is_concordant:
                        tie_conc += 1
                        t_tie_conc += 1
                    else:
                        tie_disc += 1
                        t_tie_disc += 1

            per_tissue[tissue] = {
                "clear_conc": t_clear_conc, "clear_disc": t_clear_disc,
                "tie_conc": t_tie_conc, "tie_disc": t_tie_disc,
            }

        clear_total = clear_conc + clear_disc
        tie_total = tie_conc + tie_disc
        results[metric] = {
            "clear_winners": {
                "inversion_rate": round(clear_disc / clear_total, 3) if clear_total > 0 else None,
                "concordant": clear_conc, "discordant": clear_disc, "n": clear_total,
                "inversions": clear_inversions,
            },
            "near_ties": {
                "inversion_rate": round(tie_disc / tie_total, 3) if tie_total > 0 else None,
                "concordant": tie_conc, "discordant": tie_disc, "n": tie_total,
            },
            "per_tissue": per_tissue,
            "gap_threshold": gap_threshold,
        }
    return results


def test1c_tissue_bootstrap_ci(raw_scores, raw_f1s, n_boot=10000):
    """Tissue-clustered bootstrap CI for contenders-4 inversion rates."""
    rng = np.random.default_rng(20260713)
    results = {}
    for metric in BIO_METRICS:
        tissue_rates = []
        for tissue in TISSUES:
            conc, disc = 0, 0
            for i, j in combinations(range(len(CONTENDERS)), 2):
                ki = f"{tissue}_{CONTENDERS[i]}"
                kj = f"{tissue}_{CONTENDERS[j]}"
                mi = raw_scores[ki].get(metric)
                mj = raw_scores[kj].get(metric)
                fi, fj = raw_f1s[ki], raw_f1s[kj]
                if mi is None or mj is None:
                    continue
                m_sign = np.sign(mi - mj)
                f_sign = np.sign(fi - fj)
                if m_sign == 0 or f_sign == 0:
                    continue
                if m_sign == f_sign:
                    conc += 1
                else:
                    disc += 1
            total = conc + disc
            tissue_rates.append(disc / total if total > 0 else 0)

        tissue_rates = np.array(tissue_rates)
        boot_means = np.array([
            tissue_rates[rng.choice(len(tissue_rates), size=len(tissue_rates), replace=True)].mean()
            for _ in range(n_boot)
        ])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        results[metric] = {
            "rate": round(float(tissue_rates.mean()), 3),
            "ci_95": [round(float(ci_lo), 3), round(float(ci_hi), 3)],
            "per_tissue": {t: round(float(r), 3) for t, r in zip(TISSUES, tissue_rates)},
        }
    return results


def test4_discriminative_vs_predictive(summary_data):
    """Correlate each metric's T1 delta with its T3 rho across bio metrics.

    If the correlation is near zero or negative, the suite's discriminative
    power (trained > null) is unrelated to its predictive validity (tracks F1).
    """
    verdicts = summary_data["verdicts"]
    t1_deltas = []
    t3_rhos = []
    metric_names = []

    for metric in BIO_METRICS:
        v = verdicts.get(metric, {})
        t1 = v.get("T1", {})
        t3 = v.get("T3", {})

        delta = t1.get("delta")
        rho = t3.get("rho")

        if delta is not None and rho is not None:
            t1_deltas.append(delta)
            t3_rhos.append(rho)
            metric_names.append(metric)

    if len(t1_deltas) < 3:
        return {"note": "insufficient metrics with both T1 delta and T3 rho"}

    t1_arr = np.array(t1_deltas)
    t3_arr = np.array(t3_rhos)

    rho_corr, p_val = stats.spearmanr(t1_arr, t3_arr)

    per_metric = []
    for i, m in enumerate(metric_names):
        per_metric.append({
            "metric": m,
            "t1_delta": round(float(t1_deltas[i]), 4),
            "t3_rho": round(float(t3_rhos[i]), 4),
        })

    return {
        "spearman_rho_delta_vs_rho": round(float(rho_corr), 3),
        "p_value": round(float(p_val), 4),
        "n_metrics": len(metric_names),
        "per_metric": per_metric,
        "interpretation": (
            "negative" if rho_corr < -0.1 else
            "near_zero" if abs(rho_corr) <= 0.1 else
            "positive"
        ),
    }


def main():
    raw_scores, raw_f1s = load_data()

    with open(INPUT_PATH) as f:
        summary_data = json.load(f)

    print("Test 1: Rank inversions vs F1 (three condition sets)...")
    t1 = test1_rank_inversions(raw_scores, raw_f1s)
    for cond_set, metrics_dict in t1.items():
        print(f"\n  --- {cond_set} ---")
        for metric, res in metrics_dict.items():
            print(f"    {metric}: inversion_rate={res['inversion_rate']}, "
                  f"concordant={res['concordant']}, discordant={res['discordant']}")

    print("\nTest 1b: F1-gap split (contenders-4, |dF1| threshold=0.10)...")
    t1b = test1b_f1_gap_split(raw_scores, raw_f1s)
    for metric, res in t1b.items():
        cw = res["clear_winners"]
        nt = res["near_ties"]
        print(f"  {metric}: clear={cw['discordant']}/{cw['n']}={cw['inversion_rate']}, "
              f"ties={nt['discordant']}/{nt['n']}={nt['inversion_rate']}")

    print("\nTest 1c: Tissue-clustered bootstrap CI (contenders-4)...")
    t1c = test1c_tissue_bootstrap_ci(raw_scores, raw_f1s)
    for metric, res in t1c.items():
        print(f"  {metric}: rate={res['rate']}, 95% CI={res['ci_95']}")

    print("\nTest 2: Cross-tissue rank stability (Kendall's W)...")
    t2 = test2_cross_tissue_stability(raw_scores)
    for metric, res in t2.items():
        print(f"  {metric}: W={res['W']}")

    print("\nTest 3: Metric-metric redundancy (top correlations)...")
    t3 = test3_metric_redundancy(raw_scores)
    sorted_pairs = sorted(t3["pairwise"].items(),
                          key=lambda x: abs(x[1]["rho"]) if x[1]["rho"] is not None else -1,
                          reverse=True)
    for pair, res in sorted_pairs[:10]:
        print(f"  {pair}: rho={res['rho']}, p={res['p']}")

    print("\nTest 4: Discriminative-vs-predictive dissociation...")
    t4 = test4_discriminative_vs_predictive(summary_data)
    print(f"  Spearman(T1_delta, T3_rho) = {t4.get('spearman_rho_delta_vs_rho')}, "
          f"p = {t4.get('p_value')}")
    for entry in t4.get("per_metric", []):
        print(f"    {entry['metric']:20s}  T1_delta={entry['t1_delta']:.4f}  T3_rho={entry['t3_rho']:.4f}")

    output = {
        "test1_rank_inversions": t1,
        "test1b_f1_gap_split": t1b,
        "test1c_tissue_bootstrap_ci": t1c,
        "test2_cross_tissue_stability": t2,
        "test3_metric_redundancy": t3,
        "test4_dissociation": t4,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
