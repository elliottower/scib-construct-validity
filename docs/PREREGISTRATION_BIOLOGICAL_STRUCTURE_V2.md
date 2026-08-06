# Pre-registration: Biological-Structure Metrics as Cross-Assay Transfer Predictors (V2, confound-free)

Supersedes: PREREGISTRATION_BIOLOGICAL_STRUCTURE.md (SHA `2bbd7e22`, now exploratory pilot).

## Motivation

Geometric transportability metrics (Grassmannian distance, domain shift, curvature)
anti-predict cross-assay transfer: higher geometric divergence correlates with HIGHER
transfer F1 across 10 metrics (4 significant, all wrong sign; 6 null). This occurs
because geometric metrics confound model expressiveness with transferability — good
models learn assay-specific representations (large subspace divergence) while
preserving cell-type identity (good transfer).

Biological-structure metrics operate on aligned cell-type topology rather than raw
embedding geometry. They measure whether the arrangement of cell-type identities is
preserved across assays, which directly determines classifier generalization.

## Design

### Inclusion criteria (tissues)

All tissues in CellxGene Census (version 2023-12-15) satisfying:
- At least 200 cells with assay 10x 3' v3 (EFO:0009922)
- At least 200 cells with assay 10x 5' (EFO:0008931)
- **At least 8 cell types shared between assays** (raised from 3 to avoid
  unreliable small-matrix alignment in CKA/Procrustes)
- is_primary_data == True

### Embedding models

Per tissue, up to 8 embedding models (where available):
- Census embeddings: geneformer, scvi, scgpt
- Phase 2b embeddings: geneformer_v2_104m, geneformer_v2_316m
- Baselines: bog_pca_512, random_projection, untrained_encoder

### Outcome variable

Cross-assay transfer F1 (macro-averaged): logistic regression trained on source
assay embeddings, evaluated on target assay embeddings. Shared cell types only.

### Metrics

**Primary metrics (inferential):**

1. **Cell-type CKA** (`cell_type_cka`): Linear CKA (Kornblith et al. 2019) on
   cell-type centroid matrices. Compute the mean embedding for each shared cell
   type in source and target, form (n_types, d) centroid matrices, compute linear
   CKA. Range [0, 1]. Aggregates to centroids and loses per-cell label structure,
   so shares only cell-type grouping (not individual cell labels) with the outcome.

2. **Procrustes similarity** (`procrustes_sim`): 1 minus Procrustes disparity
   (Gower 1975) on PCA-reduced cell-type centroids. Centroids projected to
   min(50, n_types-1, d) dimensions via PCA before alignment. Same centroid
   aggregation as CKA — shares cell-type grouping with outcome but not per-cell
   structure.

**Positive control (declared near-tautological):**

3. **Cross-assay kNN purity** (`knn_purity`): For each cell in source, find k=15
   nearest neighbors in target (cosine distance). Fraction sharing the same cell
   type, symmetrized (average of source-to-target and target-to-source).
   **This metric is near-tautological with the F1 outcome**: a cell whose
   cross-assay neighbors share its label is approximately a cell the classifier
   will classify correctly. It is included as a positive control (expected to
   correlate), not as independent evidence.

### Statistical analysis

**Primary test:** Partial Spearman rank correlation (controlling for embedding
dimensionality d) between each primary metric and transfer F1, computed on
**contender models only** (excluding random_projection and untrained_encoder).
The trained/random divide would manufacture a positive correlation regardless of
metric validity.

**Permutation scheme:** Tissue-stratified permutation test (10,000 iterations).
Within each tissue, permute F1 values among models independently. This preserves
within-tissue correlation structure and tests whether the metric-F1 relationship
holds beyond tissue-level confounds. Conditions cluster by tissue (shared cell
populations), so unstratified permutation overstates degrees of freedom.

**Secondary analyses (reported for transparency):**
- Raw Spearman (not controlling d) on contenders
- All-conditions (including baselines) partial Spearman
- kNN purity correlation (expected significant; non-informative if so)
- 95% bootstrap CI (tissue-block bootstrap: resample tissues with replacement,
  take all conditions within each resampled tissue, 10,000 resamples)

### Hypotheses

**HB1 (primary):** Cell-type CKA has positive partial Spearman correlation
(controlling d) with transfer F1 among contender models (tissue-stratified
permutation p < 0.05).

**HB2 (primary):** Procrustes similarity has positive partial Spearman correlation
(controlling d) with transfer F1 among contender models (tissue-stratified
permutation p < 0.05).

**HB3 (positive control):** kNN purity has positive partial Spearman correlation
with transfer F1 (expected significant; informative only if it FAILS, which would
indicate a data quality problem).

**HB4 (concordance):** Both CKA and Procrustes show positive partial-rho.
Concordance counts only across the two primary metrics; kNN purity does not
contribute to concordance assessment.

### Multiple comparisons

Two primary hypotheses (HB1, HB2) testing the same theoretical prediction on two
operationalizations. No correction applied. If only one is significant, the
non-significant metric's failure is discussed (may reflect small centroid matrices
in low-type tissues). Concordance (HB4) is the robustness check.

### Failure modes acknowledged

- CKA and Procrustes on centroid matrices with < 8 types may align even for
  unrelated embeddings (observed in synthetic tests). MIN_SHARED_TYPES = 8
  mitigates but does not eliminate this.
- If correlation holds only with baselines included but vanishes for contenders,
  the result shows "trained beats random" (trivial), not metric validity.
- Tissue-stratified permutation is conservative; may miss a real effect at current
  sample size.

### Parameters (frozen)

- Census version: 2023-12-15
- Source assay: EFO:0009922 (10x 3' v3)
- Target assay: EFO:0008931 (10x 5')
- MAX_CELLS: 2000 per assay per tissue
- SEED: 20260713
- kNN k: 15
- Procrustes PCA components: min(50, n_types-1, d)
- MIN_CELLS: 200 per assay
- **MIN_SHARED_TYPES: 8**
- N_PERMUTATIONS: 10,000
- Classifier: LogisticRegression(max_iter=2000, C=1.0)
- F1: macro-averaged over shared cell types
