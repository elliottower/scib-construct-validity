"""Modal wrapper for spectral-gap discriminative statistic.

Prereg: PREREGISTRATION_SPECTRAL_GAP.md

Usage:
    modal run --detach scripts/modal_spectral_gap.py
    modal volume get preflight-results spectral_gap results/spectral_gap
"""
import modal

app = modal.App("preflight-spectral-gap-discriminative")

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
    .add_local_dir("scripts", "/app/scripts", copy=True)
    .add_local_dir("results/spectral_gap", "/app/results/spectral_gap", copy=True)
    .add_local_dir("results/embeddings", "/app/results/embeddings", copy=True)
    .workdir("/app")
)


@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768)
def run_spectral_gap():
    import subprocess
    import shutil
    from pathlib import Path

    vol_emb = Path("/vol/embeddings")
    local_emb = Path("results/embeddings")
    if vol_emb.exists():
        shutil.copytree(vol_emb, local_emb, dirs_exist_ok=True)
        n = sum(1 for f in local_emb.rglob("*.npy"))
        print(f"Copied {n} .npy embedding files from volume")

    result = subprocess.run(
        ["python", "scripts/exp_spectral_gap_census.py",
         "--phase2b-dir", str(local_emb)],
        timeout=82800,
        capture_output=True,
        text=True,
    )

    local_results = Path("results/spectral_gap")
    log_path = local_results / "run_log.txt"
    with open(log_path, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}\n")
        f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

    print(f"STDOUT (last 5000 chars):\n{result.stdout[-5000:]}")
    if result.returncode != 0:
        print(f"Script failed with code {result.returncode}")
        print(f"STDERR (last 3000 chars): {result.stderr[-3000:]}")

    if local_results.exists() and any(local_results.iterdir()):
        vol_dest = Path("/vol/spectral_gap")
        shutil.copytree(local_results, vol_dest, dirs_exist_ok=True)
        vol.commit()
        n_files = sum(1 for _ in vol_dest.rglob("*") if _.is_file())
        print(f"Saved {n_files} result files to volume")


@app.local_entrypoint()
def main():
    run_spectral_gap.remote()
