# Pre-registration: M3 Direction Stability as Cross-Assay Transfer Predictor (Confirmatory, n=104)

**Status:** FROZEN — M3-vs-F1 correlation has not been computed on the 25-tissue panel
**Author:** Elliot Tower (ORCiD 0000-0001-7004-8884)
**Parent studies:**
- PREREGISTRATION_BIOLOGICAL_STRUCTURE_V2.md (V3b panel: 25 tissues, 104 contenders)
- PREREGISTRATION_GEOMETRIC_PREDICTORS.md (geometric metrics on 4-tissue panel, n=12)
- SHARED_FINDING_geometric_confounds_v5.md (construct validity audit)

## Freeze chain

- This document SHA-256:            dd66b9a83931e7170e27173a87242f5868be99692ef1caeda9d6c72882858327
- Parent V3b prereg SHA-256:        b018262f2f4d056eaca638058b38af0c14ef3f5521741a482c26c24376cd40b9
- Geometric prereg SHA-256:         fcad6893f1cc57509eb5a513b175c6cbc41f80e77b578197dc28441a63ed1f9a
- Shared finding v5 last updated:   2026-07-13

---

## 1. Motivation

The construct validity audit (SHARED_FINDING_geometric_confounds_v5.md)
eliminated five geometric transfer metrics for identifiable reasons:
M1 and M6 are JL-confounded, M4 has dimension-dependent normalization
artifacts, M2 anti-predicts transfer. M3 (direction stability) is the
sole survivor: it discriminates learned embeddings from random
projections (+1.5 tier gap) and does not anti-predict transfer.

M3's predictive value remains underpowered. At n=12 (4 tissues, 3
contender models at 512d), M3 correlates with transfer F1 at
rho = +0.30, CI [-0.34, +0.80] — statistically indeterminate. The
shared finding explicitly states: "A future composite anchored by M3
+ probes remains possible but requires a model panel with n >> 12 at
fixed dimension to establish M3's predictive value."

The V3b biological-structure panel now provides 104 contender conditions
(25 tissues, 6 contender models, dimensionality range 50-1152). This is
the powered follow-up the audit called for.

Separately, cell-type CKA was found to have predictive validity
(partial rho = +0.536, p = 4.58e-9) but fails discriminative validity:
all 104 contender conditions score below the k-matched random-centroid
null. CKA tracks distance from an isotropic-random reference — a
property of null geometry, not biology. If M3 both predicts transfer
AND discriminates against random, it would be the first geometric
metric to pass both validity axes simultaneously.

---

## 2. Design

### Panel (inherited from V3b, frozen)

- Census version: 2023-12-15
- Source assay: EFO:0009922 (10x 3' v3)
- Target assay: EFO:0008931 (10x 5')
- Inclusion: MIN_CELLS = 200 per assay, MIN_SHARED_TYPES = 8
- MAX_CELLS: 2000 per assay per tissue
- SEED: 20260713
- 8 embedding models: geneformer, scvi, scgpt, geneformer_v2_104m,
  geneformer_v2_316m, bog_pca_512, random_projection, untrained_encoder
- Contenders exclude random_projection and untrained_encoder
- Ground truth: logistic-regression macro F1 on shared cell types

### M3 metric (G3: bootstrap direction stability)

Computed on **source embeddings only** (no target labels required).

1. Compute PCA on full source embedding matrix X_src (n_src, d).
   k = min(20, d/2) components. Store U_full = (d, k) subspace basis.
2. Draw 50 bootstrap resamples of X_src (sample with replacement).
3. For each resample, compute PCA with same k. Store U_boot.
4. Compute Grassmannian geodesic distance between U_full and each
   U_boot: d_G = sqrt(sum(theta_i^2)), where theta_i are principal
   angles between the subspaces.
5. M3 score = mean Grassmannian distance across 50 resamples.
   Lower = more stable directions = predicted better transfer.

This is identical to G3 in PREREGISTRATION_GEOMETRIC_PREDICTORS.md
and `bootstrap_direction_stability()` in exp_geometric_cross_model.py.

### Negative reference: M4 (participation ratio)

Included to demonstrate that the construct validity audit's elimination
holds on the expanded panel. PR = (sum sigma_i^2)^2 / sum(sigma_i^4),
computed on mean-centered source centroid matrix. M4 was killed because
it breaks across dimensionalities (UCE at 1280d floors 3/4 tissues;
BoG-PCA floors 4/4 after PCA leakage fix). On this panel with d
ranging 50-1152, M4 should fail the cross-dimensionality robustness
check. Including it as a pre-registered negative reference validates
the audit methodology.

---

## 3. Hypotheses

**HM1 (primary, confirmatory):** M3 direction stability negatively
correlates with cross-assay transfer F1 among contender models.
Lower instability = better transfer. Test: partial Spearman controlling
for d, tissue-stratified permutation (10,000 draws). Pass: partial
rho < -0.20 with permutation p < 0.05.

The threshold is rho < -0.20 (weaker than CKA's +0.536) because M3
operates on the raw embedding space, not on cell-type centroids. The
prior estimate (+0.30 at n=12) had sign uncertainty; the direction
is pre-specified as negative (lower instability = more robust = better
transfer), matching the drug-transport and atlas findings.

**HM2 (primary, discriminative):** M3 discriminates trained embeddings
from random baselines. Among conditions at the same d, random_projection
and untrained_encoder should have higher M3 (more instability) than
contender models. Test: for each tissue with >= 3 contenders and >= 1
baseline at the same d, compute Mann-Whitney U between contender and
baseline M3 values. Report fraction of tissues where contenders have
significantly lower M3 (one-sided p < 0.05). Pass: > 50% of eligible
tissues show significant discrimination.

**HM3 (secondary, incremental):** Among G0-passing metrics (those
with both predictive and discriminative validity), M3 adds predictive
variance beyond CKA. Test: partial Spearman of M3 vs F1, controlling
for d AND CKA. Only evaluated if HM1 passes.

**HM4 (negative reference):** M4 participation ratio does NOT
significantly predict transfer F1 after controlling for d among
contender models (partial Spearman permutation p > 0.05). M4 is
included as a pre-committed negative reference to validate the
construct validity audit's elimination.

**Positive control:** CKA partial rho reproduces the V3b estimate
(+0.536) within the V3b bootstrap CI [0.371, 0.672]. Failure indicates
a data or pipeline discrepancy.

---

## 4. Statistical analysis

### Primary tests

- **Partial Spearman rank correlation** controlling for d (rank
  residuals from OLS on ranked variables), on contender models only.
- **Tissue-stratified permutation** (10,000 iterations): within each
  tissue, permute F1 values among models independently. Report
  proportion of permutations with |rho_perm| >= |rho_obs|.
- **Tissue-block bootstrap 95% CI** (10,000 resamples): resample
  tissues with replacement, take all conditions within each resampled
  tissue. Report percentile CI.

### Additional controls

- Partial Spearman controlling for d AND k (n_shared_types)
- Partial Spearman controlling for d AND n_src (source sample size)
- Partial Spearman controlling for d, k, AND min_per_type_count
  (minimum cells per shared type in source)
- All-conditions (including baselines) partial Spearman — reported
  separately, not used for hypothesis testing

### Discrimination analysis (HM2)

- Compute M3 for all 8 models (including baselines) at each tissue
- Within each d stratum (models sharing the same d), compare contender
  vs baseline M3 distributions
- If insufficient within-d comparisons (< 3 tissues with matched-d
  contender-baseline pairs), report HM2 as untestable and fall back
  to: across all conditions, random_projection M3 vs contender M3
  pooled comparison with bootstrap CI on the difference

### Multiple comparisons

Two primary hypotheses (HM1, HM2) testing complementary aspects of
M3 (predictive and discriminative validity). No correction. HM3 is
conditional and secondary. HM4 is a negative reference.

---

## 5. Decision rules

- **HM1 confirmed AND HM2 confirmed:** M3 is the first geometric
  metric to pass both predictive and discriminative validity. Report
  as the recommended geometric preflight metric, alongside downstream
  probes.
- **HM1 confirmed, HM2 fails:** M3 predicts transfer but does not
  discriminate against random (same problem as CKA). Report
  predictive-but-non-discriminative.
- **HM1 fails, HM2 confirmed:** M3 discriminates (geometric structure
  is real) but does not predict (geometric structure is not what drives
  transfer). Report as a construct validity finding.
- **Both fail:** No geometric metric predicts cross-assay transfer.
  The paper reports this as a definitive negative alongside the
  M1-M6 construct validity diagnoses.
- **HM4 (M4) unexpectedly significant:** If M4 predicts transfer
  despite the cross-dimensionality confound, investigate whether the
  25-tissue panel's d distribution creates a spurious correlation.
  Report but do not claim M4 is rehabilitated without resolving the
  UCE kidney counterexample.

---

## 6. Failure modes acknowledged

- M3 is computed on source PCA subspaces. If the relevant biological
  structure is in a supervised subspace (LDA), PCA-based M3 may miss
  it. The runner uses LDA when labels are available, but the
  cross-model scripts use PCA. This prereg specifies PCA to match
  the existing G3 implementation.
- At MAX_CELLS=2000 and with 50 bootstrap resamples, each resample
  has 2000 cells with replacement. Tissues with fewer source cells
  (near MIN_CELLS=200) will have more bootstrap variability,
  potentially inflating M3. Controlling for n_src addresses this.
- The V3b panel uses a different target assay (10x 5') than the n=12
  panel (Smart-seq2). If M3's predictive value is assay-pair-specific,
  the confirmatory result may not replicate.
- M3 at k = min(20, d/2) has different effective subspace rank across
  d values. At d=50, k=20; at d=512, k=20; at d=1152, k=20. This is
  uniform for d >= 40, so the panel's d range (50-1152) should not
  create a k-dependent artifact.

---

## 7. Frozen parameters

All V3b panel parameters inherited. New parameters:
- PCA components for M3: k = min(20, d/2)
- Bootstrap resamples: 50 per condition
- Bootstrap seed: SEED + 42 (20260755)
- Permutation seed: 42 (matching V3b)
- Permutation iterations: 10,000
- Bootstrap CI resamples: 10,000
- M4 computed on mean-centered source centroid matrix (negative reference)
