# Pre-registration: Random-projection dimensionality control + scGraph (exploratory)

**Version**: 2
**Supersedes**: v1 (which treated both experiments as confirmatory)

**Changes from v1**: Demoted scGraph to exploratory (citation-positioning, not load-bearing). Added Gaussian/orthogonal consistency check to Experiment B. Added analytic null validation at each reduced d. Added H-statistic null calibration for the noise experiment. Dropped sigma-conditional dissociation claim (HARKing — see Appendix A).

## Context: noise experiment result and the HARKing boundary

The multi-seed noise experiment (1,141 rows, 10 seeds, 22 tissues) tested whether scIB bio-conservation metrics improve under additive noise. The pre-registered decision rule required >= 80% of conditions positive for >= 2/3 primary scIB metrics. Result: 0/3 passed (ARI 70%, NMI 23%, graph connectivity 48%). Per the decision rule, the paper's headline is the CKA dimensionality bound, not noise non-monotonicity.

The noise result is reported as a clean negative: metric fragility under noise is general across both scIB and similarity metric families, not a scIB-specific pathology. The predicted dissociation (scIB up, similarity flat) did not hold — CKA and Procrustes improve in ~48% of conditions, comparable to ARI's 70%.

**What we do NOT claim**: any sigma-conditional pattern (e.g., "ARI peaks spread across all sigma levels while F1 peaks concentrate at low sigma") was identified post-hoc and is not reported as a finding of this paper. See Appendix A.

## Experiment B (confirmatory): Random-projection dimensionality control

### Scientific question

Is the CKA null floor driven by embedding dimensionality per se, or by PCA's signal-concentrating (denoising) effect?

### Motivation

PCA-reducing Geneformer's d=512 embeddings to d=50 causes CKA to clear the analytic null floor (0/22 tissues discriminable -> 18/22). A reviewer could object: PCA concentrates signal into the top components and discards noise. The improvement could reflect denoising rather than d-reduction. Random linear projection to d=50 preserves pairwise distances approximately (Johnson-Lindenstrauss) but does NOT concentrate signal. If it also clears the null floor, d is the causal driver. If only PCA clears it, the effect is denoising.

### Design

For each (tissue, model) pair with native d >= 256:
- **Models**: geneformer (d=512), scgpt (d=512), bog_pca_512 (d=512), random_projection (d=512), untrained_encoder (d=512)
- **Tissues**: All tissues from Census 2023-12-15 with >= 200 cells per assay and >= 8 shared cell types (~22 tissues)
- **Projections**:
  - Random Gaussian: W ~ N(0, 1/d_out), shape (d_native, d_out), d_out in {50, 200}
  - Random orthogonal: W = Q[:, :d_out] where Q is from QR decomposition of N(0,1) matrix
  - PCA to d_out (existing results, for comparison)
- **Seeds**: 10 random projection matrices per (tissue, model, d_out, method), to assess projection variance
- **Metrics**: Cell-type centroid CKA, transfer F1, Procrustes similarity

Projections are applied jointly to src and tgt cell-level embeddings (same joint fitting as PCA), then metrics are computed on the projected embeddings.

### Predictions

**P-B1 (primary)**: Random Gaussian projection to d=50 will cause CKA to clear the analytic null floor in >= 60% of (tissue, model) conditions (seed-mean CKA > null_cka(k, 50)). This threshold is weaker than PCA's 82% (18/22) because random projection preserves distances but does not concentrate variance.

**P-B2 (internal consistency)**: Gaussian and orthogonal random projections will produce the same clearance rate within 15 percentage points. Both test the same hypothesis (dimensionality alone drives the floor) via different random bases. Divergence beyond 15 pp would indicate that the choice of projection basis matters, which would itself be informative about whether the effect is purely geometric.

Quantitative criterion: |frac_clear(gaussian) - frac_clear(orthogonal)| < 0.15.

**P-B3 (analytic null validation)**: The observed CKA values after random projection to d_out will track the analytic null formula null_cka(k, d_out) = 1 - 1.06*(k-1)/(d_out + k). Specifically, the mean (CKA_observed - null_predicted) should be positive (real signal above the floor) and should NOT depend on d_out (the signal excess is constant; only the floor shifts). This validates the core analytic claim at new d values, strengthening the paper beyond a binary cleared/didn't-clear result.

Quantitative criterion: Spearman rho between d_out and (CKA_observed - null_predicted) is non-significant (p > 0.05).

**P-B4 (secondary)**: Transfer F1 after random projection to d=50 will decrease relative to native d (mean delta < -0.02), unlike after PCA reduction where F1 often improves (regularization). This confirms PCA's denoising, not d-reduction alone, drives F1 improvements under PCA.

**P-B5 (exploratory)**: The fraction of conditions clearing the null under random projection will increase monotonically from d_out=200 to d_out=50, paralleling the analytic prediction.

### Analysis plan

1. For each (tissue, model, d_out, projection_method, seed), project embeddings and compute CKA, F1, Procrustes.
2. For P-B1: compute seed-mean CKA per (tissue, model) at d_out=50, compare to null_cka(k, 50). Report fraction clearing null.
3. For P-B2: compare Gaussian vs orthogonal clearance rates. Report difference and its 95% CI (bootstrap).
4. For P-B3: compute signal_excess = CKA_observed - null_cka(k, d_out) at each d_out. Test rho(d_out, signal_excess).
5. For P-B4: compute delta_F1 = F1(projected) - F1(native). Report mean and sign test.
6. For P-B5: fraction-clearing-null at d_out=50 vs d_out=200.


## Experiment A (exploratory): scGraph evaluation

### Status: exploratory, not confirmatory

This experiment is a citation-positioning exercise for engaging with Wang et al. (2026, Nat Biotech), not a load-bearing result. If scGraph shows the same floor, that is a bonus finding for the discussion. If it does not run cleanly or gives ambiguous results, it is dropped rather than debugged under deadline pressure. It does NOT block submission.

### Scientific question

Does scGraph — a graph-based metric evaluating relational structure — show the same dimensionality-dependent null floor as centroid CKA?

### Design

Run scGraph on the same cross-assay embedding pairs:
- **Tissues**: Same ~22 tissues as Experiment B
- **Models**: geneformer (d=512), scgpt (d=512), scvi (d=128)
- **scGraph metrics**: Rank-PCA, Corr-PCA, Corr-Weights (primary = Corr-Weights)

For each (tissue, model) pair, compute scGraph at native d and at PCA-reduced d=50.

### Predictions (exploratory, no go/no-go thresholds)

- scGraph Corr-Weights will not show the same d-dependent floor as CKA (rank correlation between d and scGraph will be weaker than between d and CKA)
- PCA reduction will not substantially change scGraph scores (delta < 0.05)

### Analysis

Descriptive statistics and rank correlations only. Results go in discussion section alongside Wang et al. citation, not in the main results.


## Decision rule (Experiment B only — this drives the paper)

- **If P-B1 holds** (random projection clears null in >= 60% conditions) AND P-B2 holds (Gaussian/orthogonal agree): dimensionality is causal. Report as "the null floor is a function of d, not signal concentration."
- **If P-B1 holds but P-B2 fails** (methods disagree): the type of projection matters. Investigate which properties of the random basis drive the difference (variance structure, orthogonality, effective rank). Report as partial support — d matters but the projection method also contributes.
- **If P-B1 fails** (< 60% clear under random projection) but PCA still works: the effect is denoising, not d alone. Reframe the recommendation from "reduce d" to "apply PCA before computing CKA" and explain the distinction.


## H-statistic null calibration (addendum to noise experiment)

### Problem

The noise experiment reports "70% of conditions have H > 0.01 for ARI." But the null behavior of H under pure seed-to-seed sampling noise is unknown. If Leiden resolution-optimized ARI has enough intrinsic jitter across seeds, many conditions could exceed H = 0.01 by chance alone, making the 70% rate an artifact of the threshold, not evidence of noise-sensitivity.

### Null calibration procedure

For each (tissue, model) condition:
1. Pool all 10 seeds x 8 sigma levels (80 metric values).
2. Permute sigma labels 1000 times (preserving the number of measurements per sigma level).
3. Recompute H_permuted = max_{sigma > 0}[m_permuted(sigma) - m_permuted(0)] for each permutation.
4. Compute the empirical null distribution of H for each metric.
5. Report the fraction of conditions where H_observed > 95th percentile of H_null.

This replaces the arbitrary 0.01 threshold with a per-metric, per-condition significance test. If the calibrated fraction drops substantially below the raw fraction, the original result was inflated by seed jitter.

### Code

Add to existing analysis script (no new Modal run required — operates on `results/noise_multiseed/merged/noise_multiseed_raw.json`).


## Appendix A: What we do NOT claim (sigma-conditional dissociation)

After the noise prereg failed, a pattern was identified in the data: ARI peaks spread across all sigma levels (including 16% at sigma=2.0), while transfer F1 peaks concentrate at low sigma and F1 reliably degrades at sigma >= 0.5. This pattern was NOT pre-registered. The sigma=2.0 threshold was chosen after seeing the data. There is no correction for testing a new hypothesis on the same dataset that failed the original one.

This pattern is noted here as hypothesis-generating material for a potential follow-up study. If pursued, it requires:
1. A new pre-registration with the specific sigma-conditional prediction
2. Fresh tissues not used in the original experiment (holdout validation)
3. Explicit labeling as a follow-up to a failed pre-registered prediction

It is NOT reported in the current paper as a finding.


## Code

Experiment B script: `scripts/exp13_randproj_control.py`
Experiment A script: `scripts/exp12_scgraph_eval.py`
Modal wrappers: `scripts/modal_randproj_control.py`, `scripts/modal_scgraph_eval.py`
Output: `results/randproj_control/`, `results/scgraph_eval/`

## Prior results motivating these experiments

- PCA reduction (d=512 -> d=50): Geneformer 0/22 -> 18/22 tissues clear CKA null; scGPT 0/22 -> 8/22
- Analytic null: E[CKA] = 1 - 1.06*(k-1)/(d+k). At d=512, k=10: null = 0.98. At d=50, k=10: null = 0.84.
- Noise experiment (1,141 rows, 10 seeds): ARI improves under noise in 70% of conditions; CKA improves in 48%. Prereg criterion not met (0/3 primary >= 80%); dimensionality bound is headline.
- Wang et al. 2026: Islander games scIB; scGraph captures relational structure. Complementary critique.
