"""Modal wrapper for multi-seed noise dose-response — PARALLELIZED.

Fans out one container per tissue (~25 tissues). Each runs 6 models x 10 seeds
x 8 sigma levels. Expected wall clock: 2-4 hours (vs weeks sequential).

Orchestration runs REMOTELY so --detach keeps everything alive.
Incremental saves to volume every 60s. Resumption on preemption.

Usage:
    modal run --detach scripts/modal_noise_multiseed.py
    modal volume ls preflight-results noise_multiseed
    modal volume get preflight-results noise_multiseed results/noise_multiseed
"""
import modal

app = modal.App("preflight-noise-multiseed-parallel")

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
        "scanpy==1.10.4",
        "scib-metrics==0.5.1",
        "pynndescent==0.5.13",
        "jax[cpu]==0.4.35",
        "matplotlib==3.9.3",
    )
    .add_local_dir("scripts", "/app/scripts", copy=True)
    .workdir("/app")
)

VOL_BASE = "/vol/noise_multiseed"


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768)
def run_single_tissue(tissue_index: int):
    import shutil
    import subprocess
    import sys
    import threading
    from pathlib import Path

    vol_dir = Path(f"{VOL_BASE}/tissue_{tissue_index:03d}")
    local_dir = Path(f"results/noise_multiseed/tissue_{tissue_index:03d}")
    local_dir.mkdir(parents=True, exist_ok=True)

    inc_local = local_dir / "incremental.jsonl"
    inc_vol = vol_dir / "incremental.jsonl"

    if vol_dir.exists() and inc_vol.exists():
        shutil.copy2(inc_vol, inc_local)
        n_lines = sum(1 for _ in open(inc_local))
        print(f"[tissue {tissue_index}] Resumed {n_lines} results from volume",
              flush=True)

    stop_sync = threading.Event()

    def sync_loop():
        while not stop_sync.wait(60):
            if inc_local.exists():
                vol_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(inc_local, inc_vol)
                vol.commit()

    syncer = threading.Thread(target=sync_loop, daemon=True)
    syncer.start()

    proc = subprocess.Popen(
        ["python", "-u", "scripts/exp10_noise_multiseed.py",
         "--n-seeds", "10", "--tissues", "25",
         "--tissue-index", str(tissue_index),
         "--output-dir", str(local_dir)],
        stdout=sys.stdout, stderr=sys.stderr,
    )
    rc = proc.wait()

    stop_sync.set()

    print(f"[tissue {tissue_index}] Script exited with code {rc}", flush=True)

    if local_dir.exists() and any(local_dir.iterdir()):
        vol_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, vol_dir, dirs_exist_ok=True)
        vol.commit()
        n_files = sum(1 for f in vol_dir.rglob("*") if f.is_file())
        print(f"[tissue {tissue_index}] Saved {n_files} files to volume",
              flush=True)

    return {"tissue_index": tissue_index, "exit_code": rc}


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=8192)
def orchestrate():
    """Runs remotely so --detach keeps everything alive."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    print("[orchestrate] Discovering tissues...", flush=True)
    proc = subprocess.run(
        ["python", "-u", "-c", (
            "import cellxgene_census, json, sys; "
            "sys.path.insert(0, '.'); "
            "from scripts.exp10_noise_multiseed import "
            "discover_tissues, CENSUS_VERSION, ORGANISM; "
            "census = cellxgene_census.open_soma("
            "census_version=CENSUS_VERSION); "
            "tissues = discover_tissues(census); "
            "census.close(); "
            "print(json.dumps(list(tissues.items())))"
        )],
        stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Tissue discovery failed")
    tissue_list = json.loads(proc.stdout.strip())
    n_tissues = len(tissue_list)
    print(f"[orchestrate] Found {n_tissues} tissues", flush=True)
    for i, (name, tid) in enumerate(tissue_list):
        print(f"  [{i:2d}] {name}", flush=True)

    print(f"\n[orchestrate] Launching {n_tissues} parallel containers...",
          flush=True)
    results = list(run_single_tissue.map(range(n_tissues)))

    n_ok = sum(1 for r in results if r and r.get("exit_code") == 0)
    n_fail = sum(1 for r in results if r and r.get("exit_code") != 0)
    n_none = sum(1 for r in results if r is None)
    print(f"\n[orchestrate] {n_ok} succeeded, {n_fail} failed, "
          f"{n_none} missing", flush=True)

    print("\n[orchestrate] Merging results...", flush=True)
    vol_base = Path(VOL_BASE)
    all_rows = []
    tissue_dirs = sorted(vol_base.glob("tissue_*"))
    for td in tissue_dirs:
        inc = td / "incremental.jsonl"
        if inc.exists():
            with open(inc) as f:
                for line in f:
                    all_rows.append(json.loads(line))

    print(f"[merge] {len(all_rows)} total rows from "
          f"{len(tissue_dirs)} tissue dirs", flush=True)

    if not all_rows:
        print("[merge] No results to merge!", flush=True)
        return

    import numpy as np

    scib_primary = ["ari_leiden", "nmi_leiden", "graph_connectivity"]
    similarity_metrics = ["cell_type_cka", "procrustes_sim", "knn_purity"]

    all_metrics = list(set(k for r in all_rows for k in r["h_stats"].keys()))
    conditions = sorted(set((r["tissue"], r["model"]) for r in all_rows))

    def _analyze_metric(metric):
        h_values = []
        cond_results = []
        for tissue, model in conditions:
            seeds = [r for r in all_rows
                     if r["tissue"] == tissue and r["model"] == model]
            hs = [s["h_stats"].get(metric) for s in seeds
                  if s["h_stats"].get(metric) is not None]
            if not hs:
                continue
            mean_h = float(np.mean(hs))
            frac_positive = float(np.mean([h > 0.01 for h in hs]))
            h_values.append(mean_h)
            cond_results.append({
                "tissue": tissue, "model": model,
                "mean_h": mean_h, "std_h": float(np.std(hs)),
                "frac_positive": frac_positive, "n_seeds": len(hs),
            })
        n_positive = sum(1 for h in h_values if h > 0.01)
        frac = n_positive / len(h_values) if h_values else 0
        return {
            "n_conditions_with_mean_h_gt_001": n_positive,
            "n_conditions": len(h_values),
            "frac_conditions_positive": round(frac, 3),
            "mean_h_all": round(float(np.mean(h_values)), 4)
            if h_values else None,
            "per_condition": cond_results,
        }

    summary = {"scib_metrics": {}, "similarity_metrics": {}, "all_metrics": {}}

    print("\n--- scIB bio-conservation (predicted: non-monotonic) ---",
          flush=True)
    for metric in sorted(all_metrics):
        if metric in scib_primary:
            entry = _analyze_metric(metric)
            summary["scib_metrics"][metric] = entry
            print(f"  {metric}: {entry['n_conditions_with_mean_h_gt_001']}/"
                  f"{entry['n_conditions']} positive "
                  f"({entry['frac_conditions_positive']:.1%}), "
                  f"mean H = {entry['mean_h_all']}", flush=True)

    print("\n--- Similarity (predicted: monotonic decrease) ---", flush=True)
    for metric in similarity_metrics:
        if metric in all_metrics:
            entry = _analyze_metric(metric)
            summary["similarity_metrics"][metric] = entry
            print(f"  {metric}: {entry['n_conditions_with_mean_h_gt_001']}/"
                  f"{entry['n_conditions']} positive "
                  f"({entry['frac_conditions_positive']:.1%}), "
                  f"mean H = {entry['mean_h_all']}", flush=True)

    for metric in sorted(all_metrics):
        if metric not in scib_primary and metric not in similarity_metrics:
            summary["all_metrics"][metric] = _analyze_metric(metric)

    from datetime import datetime, timezone
    summary["n_total_rows"] = len(all_rows)
    summary["n_tissues"] = len(set(r["tissue"] for r in all_rows))
    summary["n_conditions"] = len(conditions)
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()

    merged_dir = vol_base / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    raw_path = merged_dir / "noise_multiseed_raw.json"
    with open(raw_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    summary_path = merged_dir / "noise_multiseed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    vol.commit()

    print(f"\n=== DONE === {len(all_rows)} rows, "
          f"{summary['n_tissues']} tissues", flush=True)


@app.local_entrypoint()
def main():
    orchestrate.remote()
