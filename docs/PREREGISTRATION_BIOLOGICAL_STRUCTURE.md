# Pre-registration: Biological-Structure Metrics as Cross-Assay Transfer Predictors

## Motivation

Geometric transportability metrics (Grassmannian distance, domain shift measures,
Ollivier-Ricci curvature) operate on raw embedding geometry — subspace angles,
distribution distances, graph curvature. Across 10 such metrics tested on 4 tissues
(32 conditions), 4 reached significance but ALL predicted in the wrong direction:
higher geometric divergence correlated with HIGHER transfer F1, not lower.

We hypothesize this anti-prediction occurs because geometric metrics confound model
expressiveness with transferability. Good models learn assay-specific representations
(large subspace divergence) while preserving cell-type identity (good transfer).
Bad models produce geometry-flat embeddings (small divergence, poor transfer).

Biological-structure metrics operate on aligned cell-type topology rather than raw
embedding geometry. They measure whether the arrangement of cell-type identities is
preserved across assays, which directly determines whether a classifier trained on
one assay generalizes to another.

## Design

### Inclusion criteria (tissues)

All tissues in CellxGene Census (version 2023-12-15) satisfying:
- At least 200 cells with assay 10x 3' v3 (EFO:0009922)
- At least 200 cells with assay 10x 5' (EFO:0008931)
- At least 3 cell types shared between assays
- is_primary_data == True

The qualifying tissue list is determined by the Census data, not hand-picked.

### Embedding models

Per tissue, 8 embedding models (where available):
- Census embeddings: geneformer, scvi, scgpt
- Phase 2b embeddings: geneformer_v2_104m, geneformer_v2_316m
- Baselines: bog_pca_512, random_projection, untrained_encoder

### Outcome variable

Cross-assay transfer F1 (macro-averaged): logistic regression trained on source
assay embeddings, evaluated on target assay embeddings. Shared cell types only.
Identical to prior experiments.

### Metrics

Three biological-structure metrics, all measuring cell-type-level similarity
between source and target embeddings:

1. **Cell-type CKA** (`cell_type_cka`): Linear CKA (Kornblith et al. 2019) on
   cell-type centroid matrices. Compute the mean embedding for each shared cell
   type in source and target, form (n_types, d) matrices, compute linear CKA.
   Range [0, 1], higher = more similar cell-type arrangement.

2. **Procrustes similarity** (`procrustes_sim`): 1 minus Procrustes disparity
   (Gower 1975) on PCA-reduced cell-type centroids. Centroids are projected to
   min(50, n_types-1, d) dimensions via PCA before alignment. Higher = more
   similar.

3. **Cross-assay kNN purity** (`knn_purity`): For each cell in source, find its
   k=15 nearest neighbors in target (cosine distance). Fraction sharing the same
   cell type = source-to-target purity. Symmetrize by averaging with target-to-source.
   Higher = cell types more consistently placed across assays.

### Statistical analysis

- Spearman rank correlation (rho) between each metric and transfer F1 across all
  conditions (tissues x models)
- Two-sided permutation test (10,000 permutations) for significance
- 95% bootstrap confidence interval (10,000 resamples)
- Partial Spearman controlling for embedding dimensionality d
- Contender-only analysis excluding random_projection and untrained_encoder

### Hypotheses

**HB1 (primary)**: Cell-type CKA positively correlates with cross-assay transfer F1
(rho > 0, permutation p < 0.05).

**HB2 (primary)**: Procrustes similarity positively correlates with cross-assay
transfer F1 (rho > 0, permutation p < 0.05).

**HB3 (primary)**: Cross-assay kNN purity positively correlates with cross-assay
transfer F1 (rho > 0, permutation p < 0.05).

**HB4 (secondary)**: Among contender models only (excluding random baselines), at
least one biological-structure metric retains significant positive correlation —
demonstrating the relationship is not driven solely by the trained/random divide.

### Multiple comparisons

Three primary hypotheses testing the same theoretical prediction (biological-structure
similarity predicts transfer) on three operationalizations. No correction applied to
individual tests. Concordance across metrics is itself informative: if all three show
the same sign, the pattern is robust to operationalization choice.

### Parameters (frozen)

- Census version: 2023-12-15
- Source assay: EFO:0009922 (10x 3' v3)
- Target assay: EFO:0008931 (10x 5')
- MAX_CELLS: 2000 per assay per tissue
- SEED: 20260713
- kNN k: 15
- Procrustes PCA components: min(50, n_types-1, d)
- MIN_CELLS: 200 per assay
- MIN_SHARED_TYPES: 3
- N_PERMUTATIONS: 10,000
- Classifier: LogisticRegression(max_iter=2000, C=1.0)
- F1: macro-averaged over shared cell types
