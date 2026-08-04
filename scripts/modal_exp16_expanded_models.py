"""Modal: Exp16 — Expanded model panel (UCE + NMF added).

Same evaluation framework as exp15 (12 metrics, 4 SCC classifiers,
bootstrap CIs, BH correction) but with 5 embedding models instead of 3:
  - Geneformer (d=512, transformer)
  - scGPT (d=512, transformer)
  - scVI (d=128, VAE)
  - UCE (transformer, pretrained across species)
  - NMF (matrix factorization baseline)

Addresses reviewer concern about only 3 models per tissue. Increases
conditions from ~69 to ~115 and adds architectural diversity.

Usage:
    modal run --detach scripts/modal_exp16_expanded_models.py
    modal volume ls preflight-results exp16_expanded_models
    modal volume get preflight-results exp16_expanded_models results/exp16_expanded_models
"""
import modal

app = modal.App("preflight-exp16-expanded-models")

vol = modal.Volume.from_name("preflight-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.14.1",
        "scikit-learn==1.6.0",
        "anndata==0.11.4",
        "cellxgene-census==1.16.2",
        "tqdm==4.67.1",
        "pandas==2.2.3",
        "matplotlib==3.9.3",
    )
    .add_local_file("scripts/exp15_scc_stress_test.py", "/app/exp15_scc_stress_test.py", copy=True)
    .workdir("/app")
)

VOL_BASE = "/vol/exp16_expanded_models"

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"
EMBEDDINGS = ["geneformer", "scvi", "scgpt", "uce", "nmf"]
MAX_CELLS = 2000
MIN_SHARED_TYPES = 8
SEED = 20260801

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768)
def run_tissue(tissue_index, tissue_name, tissue_id):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import cellxgene_census
    import numpy as np

    from exp15_scc_stress_test import compute_all_metrics

    def _ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    vol_dir = Path(f"{VOL_BASE}/tissue_{tissue_index:03d}")
    result_path = vol_dir / "results.json"

    if vol_dir.exists() and result_path.exists():
        existing = json.loads(result_path.read_text())
        if all(f"metrics_{m}" in existing for m in EMBEDDINGS):
            if "scc_knn" in existing.get(f"metrics_{EMBEDDINGS[0]}", {}):
                print(f"[{tissue_index}] {tissue_name}: already done, skipping", flush=True)
                return existing

    print(f"[{tissue_index}] {_ts()} {tissue_name}: downloading from Census...", flush=True)

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        sides = {}
        for side, assay_id, seed_offset in [
            ("source", SOURCE_ASSAY, 0),
            ("target", TARGET_ASSAY, 1),
        ]:
            obs_filter = (
                f"tissue_ontology_term_id == '{tissue_id}' "
                f"and is_primary_data == True "
                f"and disease == 'normal' "
                f"and assay_ontology_term_id == '{assay_id}'"
            )
            obs_df = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=obs_filter,
                column_names=["soma_joinid", "donor_id"],
            )
            all_ids = obs_df["soma_joinid"].values
            print(f"  {_ts()} {side}: {len(all_ids)} cells available", flush=True)
            if len(all_ids) == 0:
                print(f"  {_ts()} No cells for {side}, skipping tissue", flush=True)
                return None

            rng = np.random.default_rng(SEED + seed_offset)
            if len(all_ids) > MAX_CELLS:
                idx = rng.choice(len(all_ids), size=MAX_CELLS, replace=False)
                idx.sort()
                all_ids = all_ids[idx]

            adata = cellxgene_census.get_anndata(
                census, organism=ORGANISM,
                obs_value_filter=obs_filter,
                obs_coords=all_ids,
                obs_column_names=OBS_COLUMNS,
                obs_embeddings=EMBEDDINGS,
            )
            print(f"  {_ts()} {side}: {adata.n_obs} cells, "
                  f"{len(adata.obs['cell_type'].unique())} types", flush=True)
            sides[side] = adata

    src = sides["source"]
    tgt = sides["target"]
    shared = sorted(set(src.obs["cell_type"].values) & set(tgt.obs["cell_type"].values))
    k = len(shared)
    if k < MIN_SHARED_TYPES:
        print(f"  {_ts()} Only {k} shared types, skipping", flush=True)
        return None

    print(f"  {_ts()} {k} shared cell types, computing 12 metrics + F1...", flush=True)

    src_labels = src.obs["cell_type"].values.astype(str)
    tgt_labels = tgt.obs["cell_type"].values.astype(str)

    result = {
        "tissue": tissue_name,
        "tissue_index": tissue_index,
        "k": k,
        "n_src": src.n_obs,
        "n_tgt": tgt.n_obs,
    }

    for emb_name in EMBEDDINGS:
        if emb_name not in src.obsm or emb_name not in tgt.obsm:
            print(f"  {_ts()} {emb_name}: missing from obsm, skipping", flush=True)
            continue

        X_src = np.asarray(src.obsm[emb_name])
        X_tgt = np.asarray(tgt.obsm[emb_name])

        if not (np.all(np.isfinite(X_src)) and np.all(np.isfinite(X_tgt))):
            print(f"  {_ts()} {emb_name}: non-finite values, skipping", flush=True)
            continue

        print(f"  {_ts()} {emb_name} (d={X_src.shape[1]}): computing 12 metrics + F1...",
              flush=True)

        metrics = compute_all_metrics(X_src, X_tgt, src_labels, tgt_labels, shared)
        result[f"metrics_{emb_name}"] = metrics

        print(f"  {_ts()} {emb_name}: F1={metrics['f1']:.3f}, "
              f"SCC_lr={metrics['scc_logreg']:.3f}, "
              f"SCC_knn={metrics['scc_knn']:.3f}, "
              f"SCC_rf={metrics['scc_rf']:.3f}, "
              f"SCC_svm={metrics['scc_svm']:.3f}, "
              f"MMD={metrics['mmd']:.3f}",
              flush=True)

    vol_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()

    print(f"  {_ts()} Done: {tissue_name}", flush=True)
    return result


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def compute_summary_stats(all_results):
    """Compute bootstrap CIs and full statistical summary from all tissue results."""
    import json
    from pathlib import Path

    import numpy as np
    from scipy import stats

    rows = []
    for r in all_results:
        if r is None:
            continue
        tissue = r["tissue"]
        for emb in EMBEDDINGS:
            key = f"metrics_{emb}"
            if key not in r:
                continue
            m = r[key]
            if m.get("f1") is None:
                continue
            row = {"tissue": tissue, "model": emb, "f1": m["f1"]}
            for metric_name in [
                "rcs_baseline", "rcs_pca10", "rcs_trimmed", "rcs_normalized",
                "rcs_combined", "pad", "scc_logreg", "scc_knn", "scc_rf",
                "scc_svm", "mmd", "ccal",
            ]:
                row[metric_name] = m.get(metric_name)
            rows.append(row)

    n_tissues = len(set(r["tissue"] for r in rows))
    n_conditions = len(rows)
    print(f"\n=== SUMMARY: {n_conditions} conditions across {n_tissues} tissues ===\n", flush=True)

    metric_names = [
        "scc_logreg", "scc_knn", "scc_rf", "scc_svm",
        "mmd", "ccal", "rcs_baseline", "rcs_pca10", "rcs_trimmed",
        "rcs_normalized", "rcs_combined", "pad",
    ]

    f1_vals = np.array([r["f1"] for r in rows])
    summary = {"n_tissues": n_tissues, "n_conditions": n_conditions, "metrics": {}}

    rng = np.random.default_rng(SEED)
    N_BOOTSTRAP = 10000

    for mname in metric_names:
        vals = np.array([r[mname] for r in rows if r[mname] is not None])
        f1_matched = np.array([r["f1"] for r in rows if r[mname] is not None])
        n_valid = len(vals)

        if n_valid < 5:
            print(f"  {mname}: only {n_valid} valid, skipping", flush=True)
            continue

        rho, p_rho = stats.spearmanr(vals, f1_matched)

        boot_rhos = np.zeros(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.choice(n_valid, size=n_valid, replace=True)
            br, _ = stats.spearmanr(vals[idx], f1_matched[idx])
            boot_rhos[b] = br
        ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

        tissues_unique = sorted(set(r["tissue"] for r in rows if r[mname] is not None))
        taus = []
        for tissue in tissues_unique:
            t_rows = [r for r in rows if r["tissue"] == tissue and r[mname] is not None]
            if len(t_rows) < 2:
                continue
            t_f1 = [r["f1"] for r in t_rows]
            t_metric = [r[mname] for r in t_rows]
            tau, _ = stats.kendalltau(t_metric, t_f1)
            if np.isfinite(tau):
                taus.append(tau)

        n_pos = sum(1 for t in taus if t > 0)
        n_tau = len(taus)
        mean_tau = float(np.mean(taus)) if taus else 0.0
        if n_tau > 0:
            sign_p = float(stats.binomtest(n_pos, n_tau, 0.5, alternative='greater').pvalue)
        else:
            sign_p = 1.0

        n_correct = 0
        n_pairs = 0
        for tissue in tissues_unique:
            t_rows = [r for r in rows if r["tissue"] == tissue and r[mname] is not None]
            for i in range(len(t_rows)):
                for j in range(i + 1, len(t_rows)):
                    f1_diff = t_rows[i]["f1"] - t_rows[j]["f1"]
                    m_diff = t_rows[i][mname] - t_rows[j][mname]
                    if f1_diff != 0 and m_diff != 0:
                        n_pairs += 1
                        if np.sign(f1_diff) == np.sign(m_diff):
                            n_correct += 1
        pairwise_acc = n_correct / n_pairs if n_pairs > 0 else 0.0
        if n_pairs > 0:
            pairwise_p = float(stats.binomtest(n_correct, n_pairs, 0.5, alternative='greater').pvalue)
        else:
            pairwise_p = 1.0

        entry = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "bootstrap_ci_lo": float(ci_lo),
            "bootstrap_ci_hi": float(ci_hi),
            "mean_kendall_tau": mean_tau,
            "n_positive_tau": n_pos,
            "n_tissues_tau": n_tau,
            "sign_test_p": sign_p,
            "pairwise_accuracy": pairwise_acc,
            "pairwise_n": n_pairs,
            "pairwise_p": pairwise_p,
            "n_valid": n_valid,
        }
        summary["metrics"][mname] = entry

        print(f"  {mname:20s}: rho={rho:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}], "
              f"tau={mean_tau:+.3f} ({n_pos}/{n_tau}), "
              f"pairwise={pairwise_acc:.1%} ({n_correct}/{n_pairs})", flush=True)

    all_sign_ps = [summary["metrics"][m]["sign_test_p"] for m in metric_names if m in summary["metrics"]]
    all_spearman_ps = [summary["metrics"][m]["spearman_p"] for m in metric_names if m in summary["metrics"]]
    all_pairwise_ps = [summary["metrics"][m]["pairwise_p"] for m in metric_names if m in summary["metrics"]]

    def bh_correct(pvals):
        n = len(pvals)
        if n == 0:
            return []
        order = np.argsort(pvals)
        sorted_p = np.array(pvals)[order]
        adjusted = np.zeros(n)
        cummin = 1.0
        for i in range(n - 1, -1, -1):
            val = sorted_p[i] * n / (i + 1)
            cummin = min(cummin, val)
            adjusted[order[i]] = min(cummin, 1.0)
        return adjusted.tolist()

    bh_sign = bh_correct(all_sign_ps)
    bh_spearman = bh_correct(all_spearman_ps)
    bh_pairwise = bh_correct(all_pairwise_ps)

    ordered_metrics = [m for m in metric_names if m in summary["metrics"]]
    for i, mname in enumerate(ordered_metrics):
        summary["metrics"][mname]["bh_sign_p"] = bh_sign[i]
        summary["metrics"][mname]["bh_spearman_p"] = bh_spearman[i]
        summary["metrics"][mname]["bh_pairwise_p"] = bh_pairwise[i]

    summary_dir = Path(f"{VOL_BASE}/summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "exp16_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(summary_dir / "exp16_all_rows.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    vol.commit()
    print(f"\n=== Summary saved to {summary_dir} ===", flush=True)
    return summary


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def orchestrate():
    import json
    from pathlib import Path

    import cellxgene_census

    print("[orchestrate] Discovering tissues...", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        obs_df = cellxgene_census.get_obs(
            census, ORGANISM,
            column_names=[
                "tissue", "tissue_ontology_term_id", "assay_ontology_term_id",
                "cell_type", "is_primary_data", "disease",
            ],
            value_filter=(
                f"is_primary_data == True and disease == 'normal' and "
                f"assay_ontology_term_id in ['{SOURCE_ASSAY}', '{TARGET_ASSAY}']"
            ),
        )
    tissues = {}
    for tissue_name in obs_df["tissue"].unique():
        t_df = obs_df[obs_df["tissue"] == tissue_name]
        src_df = t_df[t_df["assay_ontology_term_id"] == SOURCE_ASSAY]
        tgt_df = t_df[t_df["assay_ontology_term_id"] == TARGET_ASSAY]
        if len(src_df) < 100 or len(tgt_df) < 100:
            continue
        shared = set(src_df["cell_type"].unique()) & set(tgt_df["cell_type"].unique())
        if len(shared) < MIN_SHARED_TYPES:
            continue
        tissues[tissue_name] = t_df["tissue_ontology_term_id"].iloc[0]

    tissue_list = sorted(tissues.items())
    n = len(tissue_list)
    print(f"[orchestrate] Found {n} tissues:", flush=True)
    for i, (name, tid) in enumerate(tissue_list):
        print(f"  [{i:2d}] {name}", flush=True)

    print(f"\n[orchestrate] Launching {n} parallel containers...", flush=True)
    results = list(run_tissue.starmap(
        [(i, name, tid) for i, (name, tid) in enumerate(tissue_list)]
    ))

    n_ok = sum(1 for r in results if r is not None)
    n_skip = sum(1 for r in results if r is None)
    print(f"\n[orchestrate] {n_ok} completed, {n_skip} skipped", flush=True)

    all_rows = [r for r in results if r is not None]

    if all_rows:
        print("[orchestrate] Computing summary statistics...", flush=True)
        summary = compute_summary_stats.remote(all_rows)

    merged_dir = Path(f"{VOL_BASE}/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_dir / "exp16_all_tissues.json", "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    vol.commit()

    print(f"\n=== DONE === {n_ok} tissues completed, {n_skip} skipped", flush=True)


@app.local_entrypoint()
def main():
    orchestrate.remote()
