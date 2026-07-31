# Pre-registration: scGraph evaluation and random-projection dimensionality control

**Version**: 1

## Overview

Two confirmatory experiments addressing reviewer-anticipatable critiques of our dimensionality-dependent null floor finding. Experiment A tests whether a graph-based metric (scGraph, from Wang et al. 2025) avoids the centroid-CKA null floor. Experiment B disambiguates PCA signal concentration from dimensionality by using random projections instead of PCA.

## Experiment A: scGraph evaluation on cross-assay embeddings

### Scientific question

Does scGraph — a graph-based metric that evaluates relational structure rather than centroid geometry — show the same dimensionality-dependent null floor as centroid CKA?

### Motivation

Wang et al. (2025, Nature Biotechnology) introduced scGraph to address qualitative failures of scIB metrics (the "drifting islands" problem: a neural network can game scIB while distorting cell-type relationships). scGraph computes weighted rank correlations on k-nearest-neighbor graphs, capturing relational structure rather than centroid positions. If the dimensionality-dependent null floor is specific to centroid-based metrics (as our analytic formula E[CKA] = 1 - 1.06*(k-1)/(d+k) predicts), scGraph should NOT exhibit the same floor. If it does, that would indicate the problem extends beyond centroid geometry to graph-based approaches.

### Design

Run scGraph on the same cross-assay (10x Chromium v3 vs Smart-seq2) embedding pairs used throughout the paper:
- **Tissues**: All tissues from Census 2023-12-15 with >=200 cells per assay and >=8 shared cell types (~22 tissues)
- **Models**: geneformer (d=512), scgpt (d=512), scvi (d=128), bog_pca_512 (d=512), random_projection (d=512), untrained_encoder (d=512)
- **scGraph metrics**: Rank-PCA, Corr-PCA, Corr-Weights (primary = Corr-Weights, following Wang et al.)

For each (tissue, model) pair, compute:
1. scGraph Corr-Weights score on native-dimensionality embeddings
2. scGraph Corr-Weights score on PCA-reduced (d=50) embeddings
3. Cell-type centroid CKA on the same pairs (for direct comparison)

### Predictions

**P-A1 (primary)**: scGraph Corr-Weights will NOT show the dimensionality-dependent floor. Specifically, high-d foundation models (geneformer, scgpt at d=512) will achieve scGraph scores comparable to or better than low-d models (scvi at d=128), unlike CKA where high-d models are trapped near the null floor.

Quantitative criterion: the rank correlation between embedding dimensionality and scGraph Corr-Weights across conditions will be rho < 0.3 (non-significant or weakly positive). For CKA, we have already shown strong negative rank correlation with d.

**P-A2 (secondary)**: scGraph scores will correlate more strongly with transfer F1 than CKA does, at native dimensionality. This validates Wang et al.'s claim that graph-based metrics better capture functional embedding quality.

Quantitative criterion: Spearman rho(scGraph, transfer_F1) > rho(CKA, transfer_F1) across all (tissue, model) conditions.

**P-A3 (exploratory)**: PCA-reducing high-d embeddings to d=50 before computing scGraph will NOT change scGraph scores substantially (delta < 0.05), because scGraph operates on neighbor-graph structure which is approximately preserved under PCA.

### Analysis plan

1. Compute scGraph (all three sub-scores) for every (tissue, model) pair at native d.
2. Compute scGraph for PCA-reduced (d=50) embeddings.
3. Compute CKA at native d and at d=50 (reuse existing results where available).
4. For P-A1: rank-correlate dimensionality with scGraph Corr-Weights and with CKA across conditions.
5. For P-A2: compute Spearman rho(scGraph, F1) and rho(CKA, F1) across conditions; test difference via bootstrap.
6. For P-A3: paired Wilcoxon signed-rank test on scGraph(native) vs scGraph(d=50).


## Experiment B: Random-projection dimensionality control

### Scientific question

Is the CKA null floor driven by embedding dimensionality per se, or by PCA's signal-concentrating (denoising) effect?

### Motivation

Our main finding is that PCA-reducing Geneformer's d=512 embeddings to d=50 causes CKA to clear the analytic null floor (0/22 tissues discriminable -> 18/22). A reviewer could object: PCA doesn't just reduce d; it concentrates signal into the top components and discards noise. The improvement could reflect denoising rather than d-reduction. If a random linear projection to d=50 (which preserves pairwise distances approximately but does NOT concentrate signal) also clears the null floor, then d is the causal driver. If only PCA clears it, the effect is denoising, not dimensionality.

### Design

For each (tissue, model) pair with native d >= 256:
- **Models**: geneformer (d=512), scgpt (d=512), bog_pca_512 (d=512), random_projection (d=512), untrained_encoder (d=512)
- **Tissues**: Same ~22 tissues as Experiment A
- **Projections**:
  - Random Gaussian: W ~ N(0, 1/d_out), shape (d_native, d_out), d_out in {50, 200}
  - Random orthogonal: W = Q[:, :d_out] where Q is from QR decomposition of N(0,1) matrix
  - PCA to d_out (existing results, for comparison)
- **Seeds**: 10 random projection matrices per (tissue, model, d_out, method), to assess variance from the projection itself
- **Metrics**: Cell-type centroid CKA, transfer F1, Procrustes similarity

For each projection, apply it jointly to src and tgt cell-level embeddings (same joint fitting as PCA), then compute metrics on the projected embeddings.

### Predictions

**P-B1 (primary)**: Random Gaussian projection to d=50 will cause CKA to clear the analytic null floor in >= 60% of (tissue, model) conditions (seed-mean CKA > null + 2*sigma_null). This is weaker than PCA's 82% (18/22) because random projection preserves distances but doesn't concentrate variance, so some signal-to-noise will be lost.

Quantitative criterion: fraction of conditions where seed-mean CKA > null_cka(k, 50) is >= 60%.

**P-B2 (secondary)**: Random orthogonal projection will perform comparably to random Gaussian (within 0.05 CKA), confirming that the choice of random basis doesn't matter — only the target dimensionality does.

Quantitative criterion: |mean CKA(orthogonal) - mean CKA(gaussian)| < 0.05 across conditions.

**P-B3 (secondary)**: Transfer F1 after random projection to d=50 will decrease relative to native d (mean delta < -0.02), unlike after PCA reduction where F1 often improves (regularization). This confirms that PCA's denoising, not dimensionality reduction alone, is responsible for F1 improvements under PCA.

**P-B4 (exploratory)**: The fraction of conditions clearing the null under random projection will increase monotonically from d_out=200 to d_out=50, paralleling the analytic prediction null(k, d_out).

### Analysis plan

1. For each (tissue, model, d_out, projection_method, seed), project embeddings and compute CKA, F1, Procrustes.
2. For P-B1: compute seed-mean CKA per (tissue, model) at d_out=50, compare to null_cka(k, 50). Report fraction clearing null.
3. For P-B2: paired comparison of Gaussian vs orthogonal CKA across conditions (Wilcoxon signed-rank).
4. For P-B3: compute delta_F1 = F1(projected) - F1(native) per condition. Report mean and sign test.
5. For P-B4: compute fraction-clearing-null at d_out=50 and d_out=200, compare to analytic prediction.

## Decision rule

- **If P-B1 holds** (random projection clears null in >= 60% conditions): dimensionality is causal. Report in paper as "the null floor is a function of d, not signal concentration." If P-A1 also holds, report scGraph as a dimensionality-robust alternative alongside the analytic fix.
- **If P-B1 fails** (< 60% clear under random projection) but PCA still works: the effect is denoising, not d alone. Reframe the paper's recommendation from "reduce d" to "apply PCA before computing CKA" and explain the distinction. Still cite scGraph as orthogonal validation.
- **If P-A1 fails** (scGraph also shows d-dependent floor): report as a finding about graph metrics' limitations. The dimensionality problem is broader than centroid geometry.

## Code

Experiment A script: `scripts/exp12_scgraph_eval.py`
Experiment B script: `scripts/exp13_randproj_control.py`
Modal wrappers: `scripts/modal_scgraph_eval.py`, `scripts/modal_randproj_control.py`
Output: `results/scgraph_eval/`, `results/randproj_control/`

## Prior results motivating these experiments

- PCA reduction (d=512 -> d=50): Geneformer 0/22 -> 18/22 tissues clear CKA null; scGPT 0/22 -> 8/22
- Analytic null: E[CKA] = 1 - 1.06*(k-1)/(d+k). At d=512, k=10: null = 0.98. At d=50, k=10: null = 0.84.
- Noise experiment (1,141 rows, 10 seeds): ARI improves under noise in 70% of conditions; CKA improves in 48%. Prereg criterion not met (0/3 primary scIB metrics >= 80%); dimensionality bound is headline.
- Wang et al. 2025: Islander games scIB; scGraph captures relational structure. Complementary critique.
