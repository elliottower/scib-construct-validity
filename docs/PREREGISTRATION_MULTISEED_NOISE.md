# Pre-registration: Multi-seed noise dose-response for metric validity

**Version**: 2  
**Supersedes**: Test 4 in `PREREGISTRATION_CONFIRMATORY_ROBUSTNESS.md` (single-seed, 4 tissues, scIB only)

## Scientific question

Do standard single-cell bio-conservation metrics behave like valid measures of embedding quality? Specifically: does adding noise to an embedding always decrease these scores, or can noise *improve* them?

## Design

### Noise model

For each (tissue, model, seed) triple:
1. Draw one standardized noise direction `z ~ N(0, I)` of the same shape as the combined (src + tgt) embedding matrix.
2. For each sigma in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0}, corrupt the embedding as `X_noisy = X + sigma * std_per_dim * z`.

This **same-noise coupling** means the dose-response is a smooth curve along one direction, not 7 independent random draws. Non-monotonicity reflects genuine metric behavior, not noise-direction jitter.

### Scope

- **Seeds**: 10 per condition (seed = 20260713 + seed_idx * 100000)
- **Tissues**: All tissues discoverable from Census 2023-12-15 with ≥200 cells per assay and ≥8 shared cell types (~25 tissues)
- **Models**: 6 embeddings (geneformer, scvi, scgpt, random_projection, untrained_encoder, bog_pca_512)
- **Sigmas**: 7 levels as above, plus baseline (sigma=0)

### Metrics computed at every (tissue, model, seed, sigma)

**scIB bio-conservation** (9 metrics):
- ARI (Leiden, resolution-optimized)
- NMI (Leiden, resolution-optimized)
- Graph connectivity
- Silhouette label
- Silhouette batch
- Isolated label ASW
- PCR comparison
- cLISI
- iLISI

**Similarity metrics** (3 metrics):
- Cell-type CKA (linear CKA on src/tgt cell-type centroids)
- Procrustes similarity (1 - disparity, on PCA-reduced centroids)
- Cross-assay kNN purity (symmetric, k=15, cosine)

**Functional control** (1 metric):
- Transfer F1 (logistic regression, macro-averaged, on shared cell types)

## Primary test statistic

For each (tissue, model, seed, metric), compute:

    H = max_{sigma > 0} [ m(sigma) - m(0) ]

H > 0 means the metric improved at some noise level relative to baseline. H is computed per seed; the seed-mean H is the condition-level summary.

## Predictions

### scIB bio-conservation metrics

**Prediction**: ARI, NMI, and graph connectivity will show H > 0.01 (seed-mean) in ≥50% of conditions. The single-seed experiment (Test 4) found non-monotonicity in 22/24 conditions for ARI and graph connectivity, 17/24 for NMI. We predict this replicates.

**Replication criterion**: Positive seed-mean H > 0.01 in ≥80% of conditions for at least 2 of {ARI, NMI, graph connectivity}.

### Similarity metrics

**Prediction**: CKA, Procrustes, and kNN purity will show seed-mean H ≤ 0.01 in ≥90% of conditions. These metrics operate on cell-type centroids or nearest-neighbor structure; additive noise degrades both. We predict monotonic degradation.

### Dissociation

**Prediction**: The fraction of conditions with H > 0.01 will be significantly higher for scIB bio-conservation metrics than for similarity metrics. This dissociation is the paper's central finding: metrics that are valid for *ranking* embeddings (they predict transfer F1) can still fail a basic *certification* test (they can improve under degradation).

### Functional degradation control

**Prediction**: Transfer F1 will decrease monotonically (within seed-mean ± 0.005 tolerance) in ≥90% of conditions. This confirms the noise actually degrades the embedding for the downstream task that matters.

## Analysis plan

1. Compute H per (tissue, model, seed, metric).
2. For each (tissue, model, metric), compute seed-mean H and fraction of seeds with H > 0.01.
3. For each metric, compute fraction of conditions with seed-mean H > 0.01.
4. Test the dissociation: compare the fraction-positive rates between scIB and similarity metric families.
5. Report transfer F1 monotonicity as a sanity check.
6. FDR correction (Benjamini-Hochberg) across the 3 primary scIB metrics for the replication test.

## Decision rule

- **If replication criterion holds** (≥80% conditions positive for ≥2/3 primary scIB metrics): scIB non-monotonicity is the paper's headline finding. Title becomes "Single-cell bio-conservation scores can improve as embeddings are degraded" or similar.
- **If replication fails**: fall back to CKA/Procrustes dimensionality-dependent null floor as headline. scIB result reported as exploratory single-seed finding only.

## Code

Script: `scripts/exp10_noise_multiseed.py`  
Modal wrapper: `scripts/modal_noise_multiseed.py`  
H-statistic null calibration: `scripts/h_stat_null_calibration.py`  
Output: `results/noise_multiseed/`

## Prior results being replicated

Single-seed (Test 4, `results/exp10_scib_audit/noise_dose_response.json`):
- ARI non-monotonic in 22/24 conditions
- Graph connectivity non-monotonic in 22/24 conditions
- NMI non-monotonic in 17/24 conditions
- 4 tissues, 6 embeddings, 1 seed

## Result and post-hoc null calibration

**Prereg decision**: 0/3 primary scIB metrics cleared 80% threshold. Headline is CKA dimensionality bound.

**Raw rates** (H > 0.01): ARI 70%, graph connectivity 48%, NMI 23%.

**Null-calibrated rates** (H > 95th percentile of per-seed sigma-permuted null, 1000 permutations):
ARI 7%, NMI 3%, graph connectivity 2%, transfer F1 2%, CKA 28%, Procrustes 31%, silhouette label 41%.

The raw 0.01 threshold was far below the seed-to-seed jitter floor of clustering-based metrics. ARI's apparent 70% non-monotonicity rate was inflated 10x by seed jitter. Clustering metrics (ARI, NMI, graph connectivity) have near-zero real noise sensitivity (2-7%). Centroid-based similarity metrics (CKA 28%, Procrustes 31%) have moderate real sensitivity. Transfer F1 at 2% confirms noise genuinely degrades functional quality.

The predicted dissociation (scIB up, similarity flat) failed, but a different dissociation emerged: centroid metrics near their dimensionality-dependent null floor are more noise-sensitive than clustering metrics. This supports the dimensionality-bound headline from a second, independent direction — operating near the analytic floor makes centroid metrics erratically sensitive to perturbation.

**Sigma-conditional patterns**: post-hoc patterns (e.g., ARI peaks spread across all sigma levels) were identified but are NOT reported as findings. See PREREGISTRATION_SCGRAPH_AND_RANDPROJ.md Appendix A.
