# Exp17: Marker Gene Recovery — Registered Analysis Plan

## Context

Exp16 found that source classifier confidence (SCC) predicts cross-assay
transfer F1 with Spearman rho=0.673 across 92 conditions (23 tissues,
4 Census embeddings). A reviewer objection is that SCC and F1 share
classification machinery: both rely on a trained classifier applied to
embedded cells. The correlation may reflect classifier transferability
rather than an embedding-space property that generalizes to
non-classification tasks.

This experiment tests whether SCC also predicts a ground truth that
involves no classifier: marker gene recovery (MGR). MGR is scored
entirely in gene space on raw counts — the embedding enters only via
Leiden cluster assignment, and no classifier is trained or applied.

## Ground truth: marker gene recovery

Algorithm:
1. **Reference markers.** For each shared cell type, compute the top-50
   upregulated marker genes by Wilcoxon rank-sum test on raw counts in
   the source domain (true labels, 10x Chromium 3' v3).
2. **Embedding clusters.** Cluster target cells (Smart-seq2) by Leiden
   community detection on the embedding's kNN graph (k=15 neighbors,
   resolution=1.0, seed=20260801).
3. **Recovered markers.** For each Leiden cluster, compute the top-50
   upregulated marker genes by Wilcoxon rank-sum test on raw counts in
   the target domain (cluster labels).
4. **Cluster-type matching.** Match each cluster to its majority cell
   type among the shared types.
5. **Score.** For each cell type, take the **maximum Jaccard** overlap
   across all clusters mapping to that type (not the union of cluster
   marker sets, which mechanically penalizes over-clustering). MGR =
   mean of these per-type best-cluster Jaccard values. Types with no
   matching cluster score 0.

## Ceiling control

Reference markers come from source cells, recovered markers from target
cells, so Jaccard absorbs cross-assay expression shift on top of
embedding quality. To separate these effects:

Compute MGR using **true target labels** in place of Leiden clusters.
This gives the maximum achievable MGR for each tissue given domain
shift alone, with the embedding removed from the picture. Computed
once per tissue (not per embedding).

Report both raw MGR and ceiling-normalized MGR (raw / ceiling) for
each condition. Normalized MGR is the fraction of achievable marker
recovery that the embedding captures.

## Design

The same condition pool as exp16: 4 Census v2023-12-15 embeddings ×
qualifying tissues, same cell subsampling (n=2000 per side, seed
20260801), same shared-type threshold (≥8). Each condition produces
one MGR score, one ceiling-normalized MGR score, and the same 12
metric scores + F1 from the exp15/exp16 framework.

The primary analysis correlates each of the 12 metrics with MGR,
using Spearman rank correlation with block-bootstrap CIs (resampling
tissues, not conditions, to respect within-tissue clustering).

## Precondition: ground-truth independence

**P-0:** The Spearman correlation between F1 and raw MGR across
conditions falls within [0.25, 0.80].

- Below 0.25 → MGR is unrelated to transfer quality, so "SCC predicts
  MGR" is uninformative even if true.
- Above 0.80 → MGR is redundant with F1, so it does not constitute
  independent evidence.

This precondition is checked first. If it fails, P-5 through P-7 are
reported but the convergent validity claim is not made.

## Primary prediction

**P-5 (convergent validity):** SCC computed with logistic regression
achieves Spearman rho with raw MGR that is positive and significant
(BH-corrected p < 0.05).

## Secondary predictions

**P-6 (relative ranking):** SCC-LR ranks in the top 3 among the 12
metrics by Spearman rho with raw MGR. (SCC-LR specifically, not
best-of-4 SCC variants, to avoid cherry-picking.)

**P-7 (within-tissue concordance):** Within-tissue Kendall tau between
SCC-LR and MGR is positive in ≥ 60% of tissues with ≥ 3 qualifying
conditions. (With 4 models per tissue, tau is coarse but provides a
tissue-level replication check beyond pooled correlation.)

## Decision rules

- P-0 PASS, P-5 PASS → SCC captures embedding-space properties beyond
  classifier transferability. The convergent validity argument stands:
  SCC predicts both classification performance (F1) and
  non-classification biological signal recovery (MGR).

- P-0 PASS, P-5 FAIL, P-6 PASS → SCC-LR's correlation with MGR is
  not individually significant after correction, but SCC is still
  among the best available metrics for predicting MGR. Report as
  suggestive rather than confirmatory.

- P-0 PASS, P-5 FAIL, P-6 FAIL → SCC's validity is limited to
  classification-adjacent tasks. Report this limitation.

- P-0 FAIL (rho < 0.25) → MGR does not track transfer quality in this
  assay pair. Report the null and note that convergent validity
  requires a ground truth that is related to but distinct from F1.

- P-0 FAIL (rho > 0.80) → MGR and F1 are too similar to constitute
  independent evidence. Report the correlation but treat MGR as a
  robustness check, not convergent validity.

## Statistical methods

**Block bootstrap.** 10,000 bootstrap resamples at the tissue level
(not condition level) for each metric's Spearman rho with MGR. Each
resample draws 23 tissues with replacement and includes all conditions
from each drawn tissue. This respects the within-tissue clustering
(4 embeddings per tissue are not independent). Report 95% CIs from
the block-bootstrap distribution.

**Multiple comparison correction.** Same BH family as exp15/exp16:
all 12 metrics enter the correction for pooled Spearman p-values
against MGR.

**Within-tissue Kendall tau.** Computed per tissue for all 12 metrics
against MGR. Sign test: fraction of tissues with positive tau. Tissues
with < 3 qualifying conditions or constant metric/MGR values are
excluded from the count.

## Tissue and model inclusion

Identical to exp16: all tissues with ≥8 shared cell types between
source (10x Chromium 3' v3) and target (Smart-seq2), ≥100 cells per
side in Census v2023-12-15, after subsampling to n=2000 per side with
seed 20260801. Embeddings: geneformer, scvi, scgpt, uce (the 4
pretrained embeddings with complete cell coverage). If the qualifying
tissue set differs from exp16's 23, report the discrepancy and use
the exp17 set for MGR analysis.

## Hyperparameter sensitivity

Leiden resolution (1.0), kNN neighbors (15), and top-N marker genes
(50) are fixed at scanpy ecosystem defaults. These were not tuned on
any data from this study. Sensitivity to these parameters is not
pre-registered but may be reported as supplementary robustness checks.

## What this plan does NOT pre-register

- The exact number of qualifying tissues/conditions (determined by the
  inclusion rule)
- Whether specific embedding models will have higher or lower MGR
  (this experiment tests metric validity, not model quality)
- Sensitivity analyses for Leiden resolution, kNN k, or top-N genes
- Whether ceiling-normalized MGR produces different rankings than raw
  MGR (both are reported; raw is primary for predictions)

## Amendment history

- **v1** (SHA `d6b9b7ebd68973cce07951ba7c8b8f131ae1e8980ae54224b93b9bda7dbcd719`):
  Initial draft. Superseded due to: union-of-clusters Jaccard bug,
  no ceiling control, condition-level bootstrap (not block), no
  within-tissue test, arbitrary rho threshold, cherry-picked
  best-of-4 SCC variants.
- **v2** (current): Fixed Jaccard scoring (max-over-clusters). Added
  ceiling control. Block bootstrap (tissues). Precondition P-0 with
  two-sided window. Significance-based P-5 threshold (not arbitrary
  rho). Fixed P-6 to SCC-LR specifically. Added within-tissue
  Kendall tau (P-7).

Implementation: `exp17_marker_gene_recovery.py` and
`modal_exp17_marker_gene_recovery.py` (committed before results are
computed).
