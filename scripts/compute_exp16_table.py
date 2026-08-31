"""Compute the 12-metric LaTeX table from exp16 (4-model) results.

Reads exp16_summary.json and writes the table to a JSON file,
then prints LaTeX-ready rows. This is the script that reproduces
Table 2 in paper_d_v39.tex (23 tissues, 92 conditions, 4 models:
Geneformer, scGPT, scVI, UCE).

Note: compute_exp15_table.py reads exp15 (3 models, 69 conditions),
which is the pre-registered stress test — a different experiment.
"""
import json
from pathlib import Path

SUMMARY = Path("results/exp16_expanded_models/exp16_expanded_models/summary/exp16_summary.json")
OUT = Path("results/exp16_expanded_models/exp16_expanded_models/summary")

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
    n_conditions = summary["n_conditions"]
    n_tissues = summary["n_tissues"]

    # Build structured output
    table_rows = {}
    for m in DISPLAY_ORDER:
        s = summary["metrics"][m]
        table_rows[m] = {
            "display_name": DISPLAY_NAMES[m],
            "spearman_rho": round(s["spearman_rho"], 3),
            "bootstrap_ci_lo": round(s["bootstrap_ci_lo"], 2),
            "bootstrap_ci_hi": round(s["bootstrap_ci_hi"], 2),
            "bh_spearman_p": s["bh_spearman_p"],
            "pairwise_accuracy_pct": round(s["pairwise_accuracy"] * 100, 1),
            "n_positive_tau": s["n_positive_tau"],
            "n_tissues_tau": s["n_tissues_tau"],
            "bh_sign_p": s["bh_sign_p"],
        }

    output = {
        "experiment": "exp16_expanded_models",
        "n_conditions": n_conditions,
        "n_tissues": n_tissues,
        "source": str(SUMMARY),
        "metrics": table_rows,
    }

    # Write before printing
    out_path = OUT / "exp16_table.json"
    json.dump(output, open(out_path, "w"), indent=2)
    print(f"Wrote {out_path}")

    # Print LaTeX
    print(f"\n=== {n_conditions} conditions across {n_tissues} tissues ===\n")
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


if __name__ == "__main__":
    main()
