"""Modal wrapper for Experiment 18: Label Granularity Robustness.

Prereg: scripts/exp18_label_granularity_prereg.md

Runs in-process rather than via subprocess so results land on the volume
incrementally — every record is written straight to /vol and committed, so a
killed container resumes from what it already wrote instead of losing the run.

Usage:
    modal run --detach scripts/modal_exp18_label_granularity.py
    modal volume get preflight-results exp18_label_granularity results/exp18_label_granularity
"""
import modal

app = modal.App("preflight-exp18-label-granularity")

vol = modal.Volume.from_name("preflight-results", create_if_missing=True)

# Dependency set from modal_cross_tissue_validity.py, a proven-working image for
# this exact metric stack. Do not "improve" it.
#
# ONE deliberate change: scipy is pinned. The proven image allows scipy>=1.10.0,
# and on scipy below ~1.14 `fcluster(Z, k, 'maxclust')` returns k-2 clusters at
# k=L in every trial tested, which would silently make the reference granularity
# level a two-merge coarsening rather than the leaf labelling (amendment A1).
# 1.15.3 verified clean against numpy 1.26.4: 0 cluster-count mismatches in 800
# draws. The code also asserts the achieved count, so a future drift aborts
# loudly rather than corrupting the ladder.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2.0",
        "scipy==1.15.3",
        "scikit-learn>=1.3.0",
        "anndata>=0.10.0",
        "cellxgene-census==1.16.2",
        "tqdm>=4.60.0",
        "pandas>=2.1.0",
        "scanpy>=1.10.0",
        "scib-metrics>=0.4.0",
        "pynndescent>=0.5.0",
        "jax[cpu]>=0.4.0,<0.5.0",
        "matplotlib>=3.7.0",
        "requests>=2.28.0",
    )
    .add_local_dir("src", "/app/src", copy=True)
    .add_local_dir("scripts", "/app/scripts", copy=True)
    .add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    .run_commands("cd /app && pip install .")
    .workdir("/app")
)

RESULT_DIR = "/vol/exp18_label_granularity"


# retries: this repo contains modal_noise_retry_and_aggregate.py solely because a
# Census S3 read timed out mid-run. Records already written are reusable and the
# resume path is stamp-checked, so a retry resumes rather than restarting.
@app.function(image=image, volumes={"/vol": vol}, timeout=86400, memory=32768, retries=3)
def run_label_granularity():
    import sys
    import traceback
    from pathlib import Path

    sys.path.insert(0, "/app/scripts")
    import exp18_label_granularity as exp18

    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    def checkpoint():
        vol.commit()

    try:
        results = exp18.run_experiment(out_dir=RESULT_DIR, checkpoint_cb=checkpoint)
    except Exception:
        # Preserve the traceback next to whatever records already landed, then
        # re-raise so the run is not silently reported as healthy.
        with open(f"{RESULT_DIR}/ERROR.txt", "w") as f:
            f.write(traceback.format_exc())
        vol.commit()
        raise

    vol.commit()
    n = sum(1 for _ in Path(RESULT_DIR).rglob("*") if _.is_file())
    print(f"Saved {n} files to volume at {RESULT_DIR}")
    return results


@app.local_entrypoint()
def main():
    run_label_granularity.remote()
