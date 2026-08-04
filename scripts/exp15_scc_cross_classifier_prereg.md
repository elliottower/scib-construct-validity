# Exp15: SCC Cross-Classifier Robustness — Registered Analysis Plan

## Context

Exp14 found that source classifier confidence (SCC) predicts cross-assay
transfer F1 with Spearman rho=0.733 (BH-corrected p < 0.001, 21/21
tissues concordant). A reviewer objection is that SCC and F1 share the
same logistic-regression classifier, so the correlation partly reflects
shared classifier capacity rather than an embedding-space property.

This experiment tests whether SCC computed with a non-logistic-regression
classifier still predicts logistic-regression F1, breaking the
shared-machinery link.

## Design

Four classifiers compute SCC on source embeddings applied to target cells:

| Classifier | Why it breaks the link |
|---|---|
| kNN (k=15) | Non-parametric, no linear decision boundary |
| Random forest (100 trees, default) | Ensemble of non-linear splits |
| Linear SVM (Platt scaling) | Linear but different optimization objective |
| Logistic regression (original) | Control — same machinery as F1 |

F1 ground truth always uses logistic regression (max_iter=1000), matching
the main evaluation panel.

## Tissue inclusion

All tissues qualifying under: ≥8 shared cell types between source (10x
Chromium 3' v3) and target (Smart-seq2), ≥100 cells per side in Census
(v2023-12-15), after subsampling to n=2000 per side with seed 20260801.
Tissues failing this threshold are reported by name with the failure
reason. The tissue count is determined by the inclusion rule, not by a
target number.

## Primary prediction

**P-1 (cross-classifier transfer):** SCC computed with kNN (k=15)
achieves pooled Spearman rho ≥ 0.50 with logistic-regression F1, and
within-tissue Kendall tau is positive in ≥ 70% of tissues.

kNN is primary because it shares no machinery with logistic regression:
non-parametric, no learned decision boundary, no optimization objective.

## Secondary predictions

**P-2 (random forest):** SCC-RF achieves Spearman rho ≥ 0.40 with LR-F1.

**P-3 (linear SVM):** SCC-SVM achieves Spearman rho ≥ 0.50 with LR-F1.
(Linear SVM shares the linear-boundary property with LR, so high
correlation here is less informative than P-1.)

**P-4 (original replication):** SCC-LR on the exp15 tissue set achieves
Spearman rho ≥ 0.65. (Replicates exp14 with different seed and
potentially different tissue count.)

## Decision rules

- P-1 PASS → SCC measures an embedding-space property (cluster
  separability / decision-boundary transfer), not a logistic-regression
  artifact. Recommend SCC as a general transfer-prediction metric.

- P-1 FAIL, P-4 PASS → SCC's predictive power is partly
  classifier-specific. Report SCC as a within-classifier diagnostic
  (useful when the downstream task uses the same classifier family)
  rather than a general embedding metric. Revise practical
  recommendations accordingly.

- P-1 FAIL, P-4 FAIL → Exp14 result does not replicate under different
  seed/tissue set. Investigate and report the discrepancy.

## Multiple comparison correction

Benjamini-Hochberg correction across all 12 metrics (5 RCS variants +
4 SCC variants + MMD + CCAL + PAD), applied separately to each of three
test types (pooled Spearman, pairwise binomial, within-tissue Kendall
tau sign test). All 12 metrics enter the same BH family — the
conservative choice.

## Bootstrap confidence intervals

10,000 bootstrap resamples (with replacement) of the pooled condition
set for each metric's Spearman rho. Report 95% CIs for all 12 metrics.

## Exp14-vs-exp15 primacy rule

Exp15 uses a different seed (20260801 vs 20260727) and computes 12
metrics (vs exp14's 9). The tissue inclusion rule is identical, so the
qualifying tissue set may differ slightly due to subsampling.

- If exp15's qualifying tissue count ≥ exp14's 21: exp15 results
  **replace** exp14 in the paper entirely. The expanded metric set
  (12 metrics with bootstrap CIs) supersedes exp14's 9-metric table.
  Exp14 results are retained in the supplementary as a replication
  check under different seed.

- If exp15's qualifying tissue count < 21: report both tissue sets.
  The paper's primary table uses the union of qualifying tissues
  (metrics computed on whichever set each tissue qualifies for).
  Note the discrepancy and attribute it to subsampling variability.

In either case, SCC-LR from exp15 is compared against exp14's SCC-LR
to quantify seed sensitivity (P-4).

## kNN saturation guard

kNN (k=15) can assign probability 1.0 when all 15 nearest neighbors
share a class, creating a ceiling effect that compresses SCC variance.

- Report the fraction of (tissue, model) conditions where
  SCC-kNN = 1.0 (exact saturation) and SCC-kNN ≥ 0.95 (near-saturation).

- If ≥ 50% of conditions saturate at SCC-kNN = 1.0: P-1 results are
  reported with a caveat that ceiling compression may attenuate the
  correlation. The Kendall tau criterion (≥ 70% positive) is more
  robust to compression than Spearman rho, so it becomes the primary
  discriminator.

- Saturation itself is informative: SCC-kNN = 1.0 means source
  neighborhoods transfer perfectly, which should predict high F1. The
  concern is only when saturation removes variance needed for
  rank-ordering.

## What this plan does NOT pre-register

- Exact tissue count (determined by the inclusion rule, not known before
  Census query)
- Hyperparameter sensitivity of kNN (k=15 is fixed; varying k is a
  future experiment)
- Whether SVM results add information beyond kNN (if P-1 and P-3 both
  pass, the paper reports both but does not claim independent evidence
  since both are supervised classifiers)

## Amendment history

- **v1** (SHA `2d96e84975a43812319ae0ec2165d943f8f20bcebef448f94184b6a4aea7fe2f`):
  Original registration.
- **v2** (SHA `727ec26285603bbd13dc8851a6f5fe58d5cf0a1279b647a3c3a2c7f3be76e899`):
  Added exp14-vs-exp15 primacy rule and kNN saturation
  guard. Removed ambiguous "whether exp15 results will replace exp14"
  from the non-registered list (now explicitly registered above).
  Added SVM redundancy note.
- **v3** (current): Expanded embedding pool from 3 models (geneformer,
  scgpt, scvi) to all 4 pretrained embeddings with complete cell
  coverage in Census v2023-12-15 (adding UCE). Analysis framework,
  metrics, statistical tests, and thresholds unchanged. NMF excluded
  due to incomplete cell coverage on target assay (Smart-seq2),
  consistent with CZI benchmark exclusion. Exp16 results supersede
  exp15 (92 conditions across 23 tissues vs 69 conditions across 23
  tissues).
