"""Compute the full 12-metric LaTeX table from exp15 results.

Reads the summary JSON and prints LaTeX-ready table rows.
Also checks kNN saturation per the pre-registration guard.
"""
import json
from pathlib import Path

SUMMARY = Path("results/exp15_scc_stress_test/exp15_scc_stress_test/summary/exp15_summary.json")
ALL_ROWS = Path("results/exp15_scc_stress_test/exp15_scc_stress_test/summary/exp15_all_rows.json")

DISPLAY_ORDER = [
    "scc_logreg", "scc_knn", "scc_rf", "scc_svm",
    "mmd", "ccal",
    "rcs_baseline", "rcs_pca10", "rcs_trimmed", "rcs_normalized", "rcs_combined",
    "pad",
]

DISPLAY_NAMES = {
    "scc_logreg": "SCC (LR)",
    "scc_knn": "SCC (kNN)",
    "scc_rf": "SCC (RF)",
    "scc_svm": "SCC (SVM)",
    "mmd": "MMD",
    "ccal": "CCAL",
    "rcs_baseline": "RCS (baseline)",
    "rcs_pca10": "RCS (PCA-10)",
    "rcs_trimmed": "RCS (trimmed)",
    "rcs_normalized": "RCS (normalized)",
    "rcs_combined": "RCS (combined)",
    "pad": "PAD",
}


def fmt_p(p):
    if p < 0.001:
        return "$<0.001$"
    return f"${p:.3f}$"


def main():
    summary = json.loads(SUMMARY.read_text())
    print(f"=== {summary['n_conditions']} conditions across {summary['n_tissues']} tissues ===\n")

    print("LaTeX table rows (12 metrics):\n")
    for m in DISPLAY_ORDER:
        s = summary["metrics"][m]
        rho = s["spearman_rho"]
        ci_lo = s["bootstrap_ci_lo"]
        ci_hi = s["bootstrap_ci_hi"]
        p_bh_sp = s["bh_spearman_p"]
        pairwise = s["pairwise_accuracy"] * 100
        n_pos = s["n_positive_tau"]
        n_tot = s["n_tissues_tau"]
        p_bh_tau = s["bh_sign_p"]

        name = DISPLAY_NAMES[m]
        ci = f"$[{ci_lo:+.2f},\\;{ci_hi:+.2f}]$"
        print(f"{name:<20} & ${rho:+.3f}$ & {ci} & {fmt_p(p_bh_sp)} & {pairwise:.1f}\\% & {n_pos}/{n_tot} & {fmt_p(p_bh_tau)} \\\\")

    # Check kNN saturation
    print("\n\n=== kNN saturation check ===")
    all_rows = json.loads(ALL_ROWS.read_text())
    n_total = 0
    n_saturated = 0
    n_near_saturated = 0
    for row in all_rows:
        if "scc_knn" in row:
            n_total += 1
            if row["scc_knn"] == 1.0:
                n_saturated += 1
            if row["scc_knn"] >= 0.95:
                n_near_saturated += 1
    print(f"SCC-kNN = 1.0: {n_saturated}/{n_total} ({n_saturated/n_total*100:.1f}%)")
    print(f"SCC-kNN >= 0.95: {n_near_saturated}/{n_total} ({n_near_saturated/n_total*100:.1f}%)")

    # Pre-reg predictions check
    print("\n\n=== Pre-registration predictions ===")
    knn = summary["metrics"]["scc_knn"]
    print(f"P-1 (kNN): rho={knn['spearman_rho']:.3f} (need >=0.50): {'PASS' if knn['spearman_rho'] >= 0.50 else 'FAIL'}")
    knn_pct = knn["n_positive_tau"] / knn["n_tissues_tau"] * 100
    print(f"P-1 (kNN): tau+={knn['n_positive_tau']}/{knn['n_tissues_tau']} = {knn_pct:.0f}% (need >=70%): {'PASS' if knn_pct >= 70 else 'FAIL'}")

    rf = summary["metrics"]["scc_rf"]
    print(f"P-2 (RF):  rho={rf['spearman_rho']:.3f} (need >=0.40): {'PASS' if rf['spearman_rho'] >= 0.40 else 'FAIL'}")

    svm = summary["metrics"]["scc_svm"]
    print(f"P-3 (SVM): rho={svm['spearman_rho']:.3f} (need >=0.50): {'PASS' if svm['spearman_rho'] >= 0.50 else 'FAIL'}")

    lr = summary["metrics"]["scc_logreg"]
    print(f"P-4 (LR):  rho={lr['spearman_rho']:.3f} (need >=0.65): {'PASS' if lr['spearman_rho'] >= 0.65 else 'FAIL'}")

    # Which tissue was skipped?
    print("\n\n=== Tissue check ===")
    tissues_found = set()
    for row in all_rows:
        tissues_found.add(row.get("tissue", "unknown"))
    print(f"Tissues in results: {sorted(tissues_found)}")
    print(f"Count: {len(tissues_found)}")


if __name__ == "__main__":
    main()
