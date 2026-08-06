"""Compute the full 9-metric table from exp14 results (plus bootstrap CIs).

Reads per-tissue JSON results, pools across all tissues x models,
computes Spearman rho, pairwise accuracy, within-tissue Kendall tau,
BH correction, and bootstrap 95% CIs.

Output: prints LaTeX table rows and summary stats.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results/exp14_transfer_metrics/exp14_transfer_metrics")

METRIC_NAMES = [
    "scc", "mmd", "cca",
    "rcs_baseline", "rcs_pca10", "rcs_trimmed", "rcs_normalized", "rcs_combined",
    "pad",
]

DISPLAY_NAMES = {
    "scc": "SCC",
    "mmd": "MMD",
    "cca": "CCAL",
    "rcs_baseline": "RCS (baseline)",
    "rcs_pca10": "RCS (PCA-10)",
    "rcs_trimmed": "RCS (trimmed)",
    "rcs_normalized": "RCS (normalized)",
    "rcs_combined": "RCS (combined)",
    "pad": "PAD",
}

EMBEDDINGS = ["geneformer", "scvi", "scgpt"]


def load_all():
    """Load all tissue results, return list of (tissue, model, metrics_dict)."""
    rows = []
    for tissue_dir in sorted(RESULTS_DIR.iterdir()):
        result_path = tissue_dir / "results.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        tissue = data["tissue"]
        for emb in EMBEDDINGS:
            key = f"metrics_{emb}"
            if key not in data:
                continue
            m = data[key]
            rows.append({"tissue": tissue, "model": emb, **m})
    return rows


def bh_correct(pvalues):
    """Benjamini-Hochberg step-up."""
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[sorted_idx]
    adjusted = np.zeros(n)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        val = sorted_p[i] * n / (i + 1)
        cummin = min(cummin, val)
        adjusted[sorted_idx[i]] = min(cummin, 1.0)
    return adjusted


def compute_stats(rows, metric_name):
    """Compute pooled Spearman, pairwise accuracy, within-tissue Kendall tau."""
    f1_vals = [r["f1"] for r in rows]
    metric_vals = [r[metric_name] for r in rows]

    rho, p_spearman = stats.spearmanr(metric_vals, f1_vals)

    tissues = sorted(set(r["tissue"] for r in rows))

    correct = 0
    total = 0
    for tissue in tissues:
        t_rows = [r for r in rows if r["tissue"] == tissue]
        for i in range(len(t_rows)):
            for j in range(i + 1, len(t_rows)):
                m_i, m_j = t_rows[i][metric_name], t_rows[j][metric_name]
                f_i, f_j = t_rows[i]["f1"], t_rows[j]["f1"]
                if f_i == f_j or m_i == m_j:
                    continue
                total += 1
                if (m_i > m_j) == (f_i > f_j):
                    correct += 1
    pairwise_acc = correct / total if total > 0 else 0.0

    p_pairwise = stats.binomtest(correct, total, 0.5).pvalue if total > 0 else 1.0

    tau_signs = []
    for tissue in tissues:
        t_rows = [r for r in rows if r["tissue"] == tissue]
        if len(t_rows) < 2:
            continue
        m_vals = [r[metric_name] for r in t_rows]
        f_vals = [r["f1"] for r in t_rows]
        tau, _ = stats.kendalltau(m_vals, f_vals)
        if np.isfinite(tau):
            tau_signs.append(1 if tau > 0 else 0)
    n_pos = sum(tau_signs)
    n_total = len(tau_signs)
    mean_tau = n_pos / n_total if n_total > 0 else 0
    p_tau_sign = stats.binomtest(n_pos, n_total, 0.5).pvalue if n_total > 0 else 1.0

    n_bootstrap = 10000
    rng = np.random.default_rng(42)
    n = len(f1_vals)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        m_boot = [metric_vals[i] for i in idx]
        f_boot = [f1_vals[i] for i in idx]
        r_boot, _ = stats.spearmanr(m_boot, f_boot)
        if np.isfinite(r_boot):
            boot_rhos.append(r_boot)
    ci_lo = np.percentile(boot_rhos, 2.5)
    ci_hi = np.percentile(boot_rhos, 97.5)

    return {
        "rho": rho,
        "p_spearman": p_spearman,
        "pairwise_acc": pairwise_acc * 100,
        "p_pairwise": p_pairwise,
        "n_pos_tau": n_pos,
        "n_total_tau": n_total,
        "mean_tau_frac": n_pos / n_total if n_total > 0 else 0,
        "p_tau_sign": p_tau_sign,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
    }


def main():
    rows = load_all()
    print(f"Loaded {len(rows)} conditions across {len(set(r['tissue'] for r in rows))} tissues")
    print()

    all_stats = {}
    raw_p_spearman = []
    raw_p_tau = []

    for m in METRIC_NAMES:
        s = compute_stats(rows, m)
        all_stats[m] = s
        raw_p_spearman.append(s["p_spearman"])
        raw_p_tau.append(s["p_tau_sign"])

    bh_spearman = bh_correct(raw_p_spearman)
    bh_tau = bh_correct(raw_p_tau)

    for i, m in enumerate(METRIC_NAMES):
        all_stats[m]["p_bh_spearman"] = bh_spearman[i]
        all_stats[m]["p_bh_tau"] = bh_tau[i]

    print("=" * 100)
    print(f"{'Metric':<20} {'rho':>8} {'95% CI':>16} {'p_BH':>8} {'Pairwise':>10} {'Tau(+/N)':>12} {'p_BH_tau':>10}")
    print("=" * 100)

    for m in METRIC_NAMES:
        s = all_stats[m]
        ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
        p_bh = f"<0.001" if s["p_bh_spearman"] < 0.001 else f"{s['p_bh_spearman']:.3f}"
        p_bh_tau = f"<0.001" if s["p_bh_tau"] < 0.001 else f"{s['p_bh_tau']:.3f}"
        print(f"{DISPLAY_NAMES[m]:<20} {s['rho']:+.3f}  {ci:>16}  {p_bh:>8}  {s['pairwise_acc']:>8.1f}%  {s['n_pos_tau']:>3}/{s['n_total_tau']:<3}       {p_bh_tau:>8}")

    print()
    print("LaTeX table rows:")
    print()
    for m in METRIC_NAMES:
        s = all_stats[m]
        p_bh = "$<0.001$" if s["p_bh_spearman"] < 0.001 else f"${s['p_bh_spearman']:.3f}$"
        p_bh_tau = "$<0.001$" if s["p_bh_tau"] < 0.001 else f"${s['p_bh_tau']:.3f}$"
        ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]"
        print(f"{DISPLAY_NAMES[m]:<20} & ${s['rho']:+.3f}$ & {ci} & {p_bh} & {s['pairwise_acc']:.1f}\\% & ${s['n_pos_tau']}/{s['n_total_tau']}$ & {p_bh_tau} \\\\")


if __name__ == "__main__":
    main()
