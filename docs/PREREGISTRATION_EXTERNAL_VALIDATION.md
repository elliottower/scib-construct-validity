# Pre-registration: External Validation of Biological-Structure Metrics

Follows: PREREGISTRATION_BIOLOGICAL_STRUCTURE_V2.md (V3b Census experiment).
This experiment tests whether CKA and Procrustes generalize to completely
different datasets, tissues, and cross-technology comparisons.

## Motivation

V3b showed that cell-type CKA (partial-rho = +0.536) and Procrustes similarity
(partial-rho = +0.657) predict cross-assay transfer F1 across 25 Census tissues
and 104 contender conditions. That result used a single cross-assay comparison
(10x 3' v3 vs 10x 5') and pre-computed foundation model embeddings.

External validation requires:
1. Different datasets (not CellxGene Census)
2. Different technologies (not just 10x 3'/5' variants)
3. Different embedding methods (computed from raw counts, not pre-computed)
4. Comparison against field-standard baseline metrics

## Datasets

### D1: Pancreas multi-protocol (Luecken et al. 2022, scIB benchmark)

Source: FigShare (https://figshare.com/ndownloader/files/24539828)
Format: h5ad, `celltype` column for annotations, `tech` column for technology
Technologies: CEL-seq, CEL-seq2, Smart-seq2, inDrop, Fluidigm C1, SMARTER-seq (6)
Cross-assay pairs: C(6,2) = 15 ordered pairs (each direction counted separately = 30)
Cell types: Pancreatic islet types (alpha, beta, delta, ductal, acinar, etc.)

### D2: Tabula Sapiens (The Tabula Sapiens Consortium, Science 2022)

Source: CZ CELLxGENE / FigShare
Format: h5ad, Cell Ontology annotations, `method` column for technology
Technologies: 10x (3') and Smart-seq2 for each organ
Cross-assay pairs: 1 per qualifying organ (10x → Smart-seq2), both directions = 2
Qualifying organs: all organs with >= 200 cells per technology AND >= 8 shared
cell types between technologies
Cell types: 475 distinct types with Cell Ontology terms

### D3: PBMC benchmark (Ding et al. 2020, Nature Biotechnology)

Source: Single Cell Portal SCP424 / GEO GSE132044
Format: h5ad or count matrix + metadata
Technologies: 10x Chromium v2, 10x Chromium v3, Drop-seq, inDrops, Seq-Well,
CEL-Seq2, Smart-seq2 (7)
Cross-assay pairs: C(7,2) = 21 ordered pairs (each direction = 42)
Cell types: 9 annotated PBMC types (CD4+ T, cytotoxic T, NK, CD16+ monocyte,
CD14+ monocyte, megakaryocyte, B cell, dendritic cell, plasmacytoid DC)

## Embedding models

For each dataset, embeddings are computed from normalized log-transformed counts
using these methods:

**Contenders (primary analysis):**
1. `pca_full_50` — PCA on all genes, d=50
2. `pca_hvg_50` — PCA on top 2000 highly variable genes, d=50
3. `pca_full_200` — PCA on all genes, d=200
4. `pca_hvg_200` — PCA on top 2000 highly variable genes, d=200

**Baselines (excluded from primary, included in secondary):**
5. `random_projection` — Random Gaussian projection, d=200
6. `untrained_encoder` — Two-layer untrained network, d=200

### Preprocessing

Per dataset:
1. Filter to shared genes between all technologies in the dataset
2. Library-size normalize to median counts per cell
3. Log1p transform
4. For HVG variants: select top 2000 highly variable genes (Seurat v3 method via scanpy)
5. PCA/projections computed on the combined source+target data per tech pair
   (joint PCA ensures shared coordinate system)

## Outcome variable

Cross-assay transfer F1 (macro-averaged): logistic regression (C=1.0, max_iter=2000)
trained on source technology embeddings, evaluated on target technology embeddings.
Shared cell types only. Same protocol as V3b.

Direction: for each unordered pair, compute both directions (A→B and B→A) as
separate conditions. This doubles the sample size and tests whether the metrics
predict well regardless of which technology is source vs target.

## Metrics

### Our metrics (cell-type structural alignment):

1. **Cell-type CKA** (`cell_type_cka`): Linear CKA on cell-type centroid matrices.
   Same implementation as V3b.

2. **Procrustes similarity** (`procrustes_sim`): 1 - Procrustes disparity on
   PCA-reduced centroids. Same implementation as V3b.

### Field baselines (commonly used evaluation metrics):

3. **Silhouette score (source)** (`silhouette_src`): Mean silhouette width on
   cell type labels in source embeddings. Higher = better-separated cell types.
   Standard scIB bio-conservation metric applied to source only.

4. **Silhouette score (target)** (`silhouette_tgt`): Same, applied to target
   embeddings only.

5. **MMD** (`mmd`): Maximum mean discrepancy between source and target embedding
   distributions using RBF kernel (bandwidth = median pairwise distance).
   Standard domain adaptation metric. Lower = more similar distributions.

6. **Domain classifier AUC** (`domain_auc`): AUC of logistic regression trained
   to distinguish source from target cells. Lower = more similar distributions.
   Standard domain shift metric.

## Statistical analysis

### Primary analysis (per dataset)

For each dataset independently:
- Compute Spearman rank correlation between each metric and transfer F1
- Contender models only (exclude random_projection and untrained_encoder)
- Permutation test (10,000 iterations): permute F1 across conditions within
  each technology-pair stratum (analogous to tissue-stratified permutation in V3b)
- 95% bootstrap CI: tech-pair-block bootstrap (resample tech pairs with replacement)

No partial correlation controlling d: the contender models differ in both method
(full vs HVG) and dimensionality (50 vs 200), so d is not a pure confound but
part of the model identity. Raw Spearman is the appropriate test.

### Cross-dataset concordance

After per-dataset analysis, report whether each metric shows positive significant
correlation across all 3 datasets. A metric "generalizes" if positive in all 3.

### Secondary analyses

- All conditions (including baselines): Spearman + permutation
- Per-metric head-to-head: for each field baseline, test whether CKA/Procrustes
  has significantly higher correlation than the baseline (Williams' test for
  dependent correlations)

## Hypotheses

**HE1:** Cell-type CKA has positive Spearman correlation with transfer F1 among
contender models in each of the 3 datasets (tech-pair-stratified permutation
p < 0.05).

**HE2:** Procrustes similarity has positive Spearman correlation with transfer F1
among contender models in each of the 3 datasets (tech-pair-stratified permutation
p < 0.05).

**HE3 (field baseline prediction):** Silhouette (source), Silhouette (target),
MMD, and Domain classifier AUC each FAIL to positively predict transfer F1 in at
least 2 of 3 datasets (either null or wrong sign). This tests our claim that
within-assay quality and domain shift metrics do not predict cross-assay transfer.

**HE4 (concordance):** Both CKA and Procrustes show positive correlation in all
3 datasets simultaneously.

## Multiple comparisons

Six metrics tested per dataset, but the analysis is structured as: 2 confirmatory
(CKA, Procrustes — testing V3b replication) + 4 exploratory (field baselines —
characterizing what doesn't work). No correction applied. Pre-registration
specifies which are confirmatory vs exploratory.

## Failure modes acknowledged

- Pancreas has only ~14k cells across 6 technologies; some pairs may have few
  shared cell types (< 8) and will be excluded.
- PBMC has only 9 cell types; centroid matrices are small. CKA/Procrustes on
  9-point configurations may be unreliable. MIN_SHARED_TYPES = 6 for PBMC
  (relaxed from 8, since 9 is the maximum possible).
- Tabula Sapiens cross-technology differences (10x vs Smart-seq2) are large;
  transfer F1 may have low variance if all methods perform poorly.
- PCA-based embeddings may not provide enough quality spread to detect metric-F1
  correlations. If all PCA variants perform similarly, the correlation will be
  driven by noise.
- Some technologies in the pancreas dataset have very few cells (< 200); pairs
  involving these technologies will be excluded.

## Parameters (frozen)

- SEED: 20260724
- MIN_CELLS: 200 per technology per tech pair
- MIN_SHARED_TYPES: 8 (pancreas, Tabula Sapiens), 6 (PBMC)
- MAX_CELLS: 2000 per technology (subsample if larger)
- N_PERMUTATIONS: 10,000
- Classifier: LogisticRegression(max_iter=2000, C=1.0, random_state=0)
- F1: macro-averaged over shared cell types
- HVG: top 2000 (Seurat v3 via scanpy.pp.highly_variable_genes)
- MMD kernel: RBF, bandwidth = median pairwise distance (subsample 500 per group)
- Domain classifier: LogisticRegression(max_iter=1000, C=1.0), 5-fold CV AUC
- Silhouette: sklearn.metrics.silhouette_score with metric="euclidean"
- kNN k: 15 (for any kNN-based metric)
- Procrustes PCA: min(50, n_types-1, d)
