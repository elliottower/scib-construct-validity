"""Generate figures for Paper D (construct-validity paper).

Reads from results/exp10_scib_audit/ JSON files and produces publication-quality
figures in docs/figures/.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_DIR = Path("results/exp10_scib_audit")
FIGURES_DIR = Path("docs/figures")

METRIC_DISPLAY = {
    "nmi_leiden": "NMI",
    "ari_leiden": "ARI",
    "silhouette_label": "Silhouette (bio)",
    "silhouette_batch": "Silhouette (batch)",
    "isolated_label_asw": "Isolated label ASW",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
    "clisi": "cLISI",
    "ilisi": "iLISI",
    "kbet": "kBET",
}

EMB_DISPLAY = {
    "geneformer": "Geneformer",
    "scvi": "scVI",
    "scgpt": "scGPT",
    "bog_pca_512": "BoG-PCA",
    "random_projection": "Random proj.",
    "untrained_encoder": "Untrained enc.",
}

EMB_COLORS = {
    "geneformer": "#2166ac",
    "scvi": "#67a9cf",
    "scgpt": "#8c6bb1",
    "bog_pca_512": "#b2182b",
    "random_projection": "#999999",
    "untrained_encoder": "#cccccc",
}

TISSUE_DISPLAY = {
    "brain": "Brain",
    "kidney": "Kidney",
    "liver": "Liver",
    "lung": "Lung",
}


def load_noise_data():
    with open(RESULTS_DIR / "noise_dose_response.json") as f:
        return json.load(f)


def fig_noise_dose_response(data, out_path):
    """Panel of 6 metrics: 3 bio (fail) + 2 batch (pass) + graph_connectivity (dissociation).

    Each panel shows all 4 trained embeddings for one representative tissue (lung).
    """
    sigmas = data["sigmas"]
    raw = data["raw_results"]

    panels = [
        ("nmi_leiden", "Bio: NMI", "FAIL"),
        ("ari_leiden", "Bio: ARI", "FAIL"),
        ("graph_connectivity", "Bio: Graph connectivity", "FAIL"),
        ("silhouette_batch", "Batch: Silhouette", "PASS"),
        ("pcr_comparison", "Batch: PCR comparison", "PASS"),
        ("silhouette_label", "Bio: Silhouette", "FAIL"),
    ]

    trained_embs = ["geneformer", "scvi", "scgpt", "bog_pca_512"]
    tissue = "lung"

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)

    for idx, (metric, title, verdict) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]

        for emb in trained_embs:
            cond_key = f"{tissue}_{emb}"
            if cond_key not in raw:
                continue
            sbs = raw[cond_key]["scores_by_sigma"]
            values = [sbs[str(s)].get(metric) for s in sigmas]
            if any(v is None for v in values):
                continue
            ax.plot(
                sigmas, values,
                marker="o", markersize=3.5, linewidth=1.5,
                color=EMB_COLORS[emb], label=EMB_DISPLAY[emb],
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"Noise $\sigma$ (× embedding std)", fontsize=8)
        ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=9)
        ax.set_title(title, fontsize=9.5, fontweight="bold")

        verdict_color = "#d32f2f" if verdict == "FAIL" else "#388e3c"
        ax.text(
            0.97, 0.05, verdict,
            transform=ax.transAxes, fontsize=8, fontweight="bold",
            color=verdict_color, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=verdict_color, alpha=0.8),
        )

        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_noise_all_tissues(data, out_path):
    """4x1 panel: one representative bio metric (ARI) across all 4 tissues.

    Shows the universality of non-monotonicity.
    """
    sigmas = data["sigmas"]
    raw = data["raw_results"]
    metric = "ari_leiden"
    trained_embs = ["geneformer", "scvi", "scgpt", "bog_pca_512"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3), constrained_layout=True, sharey=True)

    for tidx, tissue in enumerate(["brain", "kidney", "liver", "lung"]):
        ax = axes[tidx]
        for emb in trained_embs:
            cond_key = f"{tissue}_{emb}"
            if cond_key not in raw:
                continue
            sbs = raw[cond_key]["scores_by_sigma"]
            values = [sbs[str(s)].get(metric) for s in sigmas]
            if any(v is None for v in values):
                continue
            ax.plot(
                sigmas, values,
                marker="o", markersize=3, linewidth=1.4,
                color=EMB_COLORS[emb], label=EMB_DISPLAY[emb],
            )
        ax.set_xscale("log")
        ax.set_title(TISSUE_DISPLAY[tissue], fontsize=10, fontweight="bold")
        ax.set_xlabel(r"$\sigma$", fontsize=9)
        if tidx == 0:
            ax.set_ylabel("ARI", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center",
        ncol=4, fontsize=8, frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_monotonicity_summary(data, out_path):
    """Bar chart: fraction of conditions non-monotonic per metric.

    Bio vs batch colored differently.
    """
    mv = data["monotonicity_verdicts"]

    bio_metrics = ["nmi_leiden", "ari_leiden", "silhouette_label", "clisi", "isolated_label_asw"]
    batch_metrics = ["silhouette_batch", "ilisi", "graph_connectivity", "pcr_comparison"]
    # kBET excluded (all null)

    ordered = bio_metrics + batch_metrics
    labels = [METRIC_DISPLAY.get(m, m) for m in ordered]
    fracs = []
    colors = []
    for m in ordered:
        v = mv[m]
        total = v["n_non_monotonic"] + v["n_monotonic"]
        frac = v["n_non_monotonic"] / total if total > 0 else 0
        fracs.append(frac)
        colors.append("#c62828" if m in bio_metrics else "#1565c0")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bars = ax.barh(range(len(ordered)), fracs, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Fraction of conditions non-monotonic", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.5, color="#666", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.invert_yaxis()

    for i, (frac, bar) in enumerate(zip(fracs, bars)):
        n_nm = mv[ordered[i]]["n_non_monotonic"]
        n_tot = mv[ordered[i]]["n_non_monotonic"] + mv[ordered[i]]["n_monotonic"]
        ax.text(frac + 0.02, i, f"{n_nm}/{n_tot}", va="center", fontsize=8, color="#333")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c62828", label="Bio-conservation"),
        Patch(facecolor="#1565c0", label="Batch-correction"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9, frameon=True)

    ax.set_title("Noise monotonicity: bio metrics fail, batch metrics survive", fontsize=10.5)
    ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_inversion_rates(out_path):
    """Bar chart of scIB pairwise inversion rates computed from raw data."""
    from itertools import combinations

    with open(RESULTS_DIR / "summary.json") as f:
        summary = json.load(f)

    raw_scores = summary["raw_scores"]
    raw_f1s = summary["raw_f1s"]
    contenders = ["geneformer", "scvi", "scgpt", "bog_pca_512"]
    tissues = ["lung", "liver", "kidney", "brain"]

    bio_metrics = ["nmi_leiden", "ari_leiden", "silhouette_label", "clisi", "isolated_label_asw"]
    batch_metrics = ["silhouette_batch", "ilisi", "graph_connectivity", "pcr_comparison"]
    ordered = bio_metrics + batch_metrics

    inv_rates = []
    counts = []
    for metric in ordered:
        concordant = 0
        discordant = 0
        for tissue in tissues:
            cond_scores = {}
            cond_f1s = {}
            for emb in contenders:
                key = f"{tissue}_{emb}"
                if key in raw_scores and metric in raw_scores[key]:
                    score = raw_scores[key][metric]
                    f1 = raw_f1s[key]
                    if score is not None and f1 is not None:
                        cond_scores[emb] = score
                        cond_f1s[emb] = f1
            for a, b in combinations(contenders, 2):
                if a in cond_scores and b in cond_scores:
                    if (cond_scores[a] > cond_scores[b]) == (cond_f1s[a] > cond_f1s[b]):
                        concordant += 1
                    else:
                        discordant += 1
        total = concordant + discordant
        inv_rates.append(discordant / total if total > 0 else 0)
        counts.append((discordant, total))

    labels = [METRIC_DISPLAY.get(m, m) for m in ordered]
    colors = ["#c62828" if m in bio_metrics else "#1565c0" for m in ordered]

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.bar(range(len(ordered)), inv_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0.5, color="#666", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Pairwise inversion rate vs F1", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("scIB metric ranking accuracy on contenders", fontsize=10.5)

    for i, (rate, (disc, tot)) in enumerate(zip(inv_rates, counts)):
        ax.text(i, rate + 0.015, f"{disc}/{tot}", ha="center", fontsize=7.5, color="#333")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c62828", label="Bio-conservation"),
        Patch(facecolor="#1565c0", label="Batch-correction"),
        plt.Line2D([0], [0], color="#666", linestyle="--", label="Coin flip (50%)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading noise dose-response data...")
    noise_data = load_noise_data()

    print("Generating figures...")
    fig_noise_dose_response(noise_data, FIGURES_DIR / "noise_dose_response.pdf")
    fig_noise_dose_response(noise_data, FIGURES_DIR / "noise_dose_response.png")

    fig_noise_all_tissues(noise_data, FIGURES_DIR / "noise_all_tissues.pdf")
    fig_noise_all_tissues(noise_data, FIGURES_DIR / "noise_all_tissues.png")

    fig_monotonicity_summary(noise_data, FIGURES_DIR / "monotonicity_summary.pdf")
    fig_monotonicity_summary(noise_data, FIGURES_DIR / "monotonicity_summary.png")

    fig_inversion_rates(FIGURES_DIR / "inversion_rates.pdf")
    fig_inversion_rates(FIGURES_DIR / "inversion_rates.png")

    print("\nDone. Generated figures:")
    for f in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"  {f}")
