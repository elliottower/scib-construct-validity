# Exp14: Transfer-Aware Embedding Metrics — Registered Analysis Plan

## Context

Exploratory analysis of RCS and official scGraph (exp12, exp13, official scGraph Modal run)
showed both metrics have weak construct validity for predicting cross-assay transfer F1.
RCS achieves marginal significance (pairwise binomial p=0.005, within-tissue sign test
p=0.027) while official scGraph is indistinguishable from chance. Metric design below is
informed by understanding WHY RCS fails: it measures rank-order preservation of centroid
distances but is invariant to absolute cluster separability.

## Part A: Ablations isolating RCS vs official scGraph differences

Official scGraph differs from RCS in four ways simultaneously. We isolate each:

| Ablation | What changes vs RCS baseline | Tests |
|----------|------------------------------|-------|
| A1: RCS-PCA10 | Reduce embeddings to PCA-10 before computing RCS | Dimensionality |
| A2: RCS-trimmed | Use 5% trimmed-mean centroids instead of simple mean | Centroid method |
| A3: RCS-normalized | Column-max normalize distance matrix before Spearman | Normalization |
| A4: RCS-PCA10-trimmed-normalized | All three changes (= RCS with scGraph's preprocessing) | Combined |

Baseline: RCS centroid (simple mean, native dim, raw distances).

## Part B: Four new transfer-aware metrics

### B1: Proxy A-distance (PAD)

From Ben-David et al. (2006). Train a linear classifier (logistic regression, same
hyperparameters as F1 classifier: max_iter=1000) to distinguish source vs target cells
in embedding space. Metric = 2(1 - 2*error_rate). Low PAD = similar distributions =
better transfer expected. **Negate for correlation with F1** (low PAD → high F1).

### B2: Source classifier confidence (SCC)

Train cell-type classifier on source (same logistic regression as F1 baseline).
Apply to target cells. Metric = mean(max(predicted_probabilities)) across target cells.
High confidence = model thinks it knows the answer. No target labels needed.

### B3: Maximum Mean Discrepancy (MMD)

Gaussian kernel MMD between source and target embedding distributions.
Kernel bandwidth = median pairwise distance (median heuristic).
**Negate for correlation with F1** (low MMD → high F1).

### B4: Class-conditional alignment (CCA)

For each shared cell type t:
  - Compute source centroid c_s(t) and target centroid c_t(t)
  - Compute source within-class std sigma_s(t)
  - alignment(t) = 1 - ||c_s(t) - c_t(t)|| / (sigma_s(t) + epsilon)

Metric = mean over shared cell types. High CCA = clusters align well.
epsilon = 1e-8 to avoid division by zero.

Note: CCA uses source labels + target labels (for centroid computation).
In deployment you wouldn't have target labels, but our evaluation framework
has them. We flag this as a limitation.

## Evaluation framework

Same 21 tissues x 3 models (geneformer, scvi, scgpt) = 63 conditions.
Ground truth: macro-averaged transfer F1 (logistic regression, source→target).

### Primary tests (per metric):
1. Pooled Spearman correlation with F1 (n=63)
2. Pairwise model discrimination accuracy with binomial test (n=63 pairs)
3. Within-tissue Kendall tau with sign test (n=21 independent observations)

### Multiple comparison correction:
- Benjamini-Hochberg across all metrics (4 ablations + 4 new + 1 RCS baseline = 9 metrics)
- Applied separately to each of the 3 test types
- Primary success criterion: BH-corrected p < 0.05 on within-tissue Kendall tau sign test

### Secondary analyses:
- Per-pair breakdown (gf-scvi, gf-scgpt, scvi-scgpt)
- Partial correlations controlling for k (number of shared cell types)
- Bootstrap 95% CI for best-performing metric's Spearman rho (10000 draws)

## Implementation

- All metrics computed from same Census download (same cells, same random seed 20260727)
- Same train/test split as previous experiments (source=10x, target=Smart-seq2)
- MAX_CELLS=2000 per side, MIN_SHARED_TYPES=8
- MMD computed on PCA-50 reduced embeddings (native dim too expensive for kernel)
- PAD and SCC use native-dim embeddings

## What this plan does NOT pre-register

- Which metric will win (we have strong intuitions from exploratory analysis)
- Effect size thresholds for "good enough"
- Whether to pursue any metric further based on results
