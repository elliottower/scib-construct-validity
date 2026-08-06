"""Modal: Exp14 — Transfer-aware embedding metrics.

Downloads Census embeddings per tissue, computes all 9 metrics + F1 for each
(tissue, model) condition, saves to Modal volume.

Usage:
    modal run --detach scripts/modal_exp14_transfer_metrics.py
    modal volume ls preflight-results exp14_transfer_metrics
    modal volume get preflight-results exp14_transfer_metrics results/exp14_transfer_metrics
"""
import modal

app = modal.App("preflight-exp14-transfer-metrics")

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
    .add_local_file("scripts/exp14_metrics.py", "/app/exp14_metrics.py", copy=True)
    .workdir("/app")
)

VOL_BASE = "/vol/exp14_transfer_metrics"

CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
SOURCE_ASSAY = "EFO:0009922"
TARGET_ASSAY = "EFO:0008931"
EMBEDDINGS = ["geneformer", "scvi", "scgpt"]
MAX_CELLS = 2000
MIN_SHARED_TYPES = 8
SEED = 20260727

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768)
def run_tissue(tissue_index, tissue_name, tissue_id):
    """Download Census data and compute all metrics for one tissue."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import cellxgene_census
    import numpy as np

    from exp14_metrics import compute_all_metrics

    def _ts():
        return datetime.now(timezone.utc).strftime("%H:%M:%S")

    vol_dir = Path(f"{VOL_BASE}/tissue_{tissue_index:03d}")
    result_path = vol_dir / "results.json"

    if vol_dir.exists() and result_path.exists():
        existing = json.loads(result_path.read_text())
        if all(f"metrics_{m}" in existing for m in EMBEDDINGS):
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

    print(f"  {_ts()} {k} shared cell types, computing metrics...", flush=True)

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

        print(f"  {_ts()} {emb_name} (d={X_src.shape[1]}): computing 9 metrics + F1...",
              flush=True)

        metrics = compute_all_metrics(X_src, X_tgt, src_labels, tgt_labels, shared)
        result[f"metrics_{emb_name}"] = metrics

        print(f"  {_ts()} {emb_name}: F1={metrics['f1']:.3f}, "
              f"RCS={metrics['rcs_baseline']:.3f}, "
              f"PAD={metrics['pad']:.3f}, "
              f"SCC={metrics['scc']:.3f}, "
              f"MMD={metrics['mmd']:.3f}, "
              f"CCA={metrics['cca']:.3f}",
              flush=True)

    vol_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    vol.commit()

    print(f"  {_ts()} Done: {tissue_name}", flush=True)
    return result


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def orchestrate():
    """Discover tissues and fan out."""
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

    print("[orchestrate] Merging...", flush=True)
    vol.reload()
    all_rows = [r for r in results if r is not None]

    if all_rows:
        merged_dir = Path(f"{VOL_BASE}/merged")
        merged_dir.mkdir(parents=True, exist_ok=True)
        with open(merged_dir / "exp14_all_metrics.json", "w") as f:
            json.dump(all_rows, f, indent=2, default=str)
        vol.commit()

    print(f"\n=== DONE === {len(all_rows)} tissues", flush=True)


@app.local_entrypoint()
def main():
    orchestrate.remote()
