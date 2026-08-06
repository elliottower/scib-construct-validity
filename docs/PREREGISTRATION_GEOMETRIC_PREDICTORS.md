# PREREGISTRATION: Geometric Transportability Metrics as Cross-Assay Transfer Predictors

**Status:** FROZEN — no geometric-metric-vs-F1 correlations computed on expanded panel
**Prereg version:** v1
**Author:** Elliot Tower (ORCiD 0000-0001-7004-8884)
**Date frozen:** 2026-07-24
**Parent study:** Paper D — scIB construct-validity audit
**Depends on:** PREREGISTRATION_SCIB_EXPANDED_PANEL.md (scIB metrics + F1 ground truth)

## Freeze chain

- This document SHA-256:            fcad6893f1cc57509eb5a513b175c6cbc41f80e77b578197dc28441a63ed1f9a
- Geometric scorer script SHA-256:  eef5cdf5c4787bdc404478ee25b3378625f0eab6527068a56ded885f380330b0
- Parent prereg SHA-256:            cab868661c4f7697ab9ff9c834a486793921c8a97f06b3385fd382fa39771dea

No hypothesis below has been evaluated on V2-104M, V2-316M, or any
cross-model geometric-vs-F1 correlation. Prior single-model results
(Geneformer only, n=6 tissue pairs) exist in
results/modal_results/composite_validation_v6/results.json — these
informed metric selection but NOT the cross-model hypotheses.

---

## 1. Motivation

The parent study shows scIB bio-conservation metrics do not predict
cross-assay transfer F1 among competitive embeddings (46–58% inversion
rate). The matched-assay replication localizes the failure: scIB predicts
cross-tissue transfer (rho=+0.63) but not cross-assay transfer.

Grassmannian transportability metrics from separate studies predict
cross-domain transfer in other settings:
- Grassmannian geodesic distance predicts cross-cohort classifier
  degradation (Paper 1: biomedical-cohort-transfer)
- Direction instability predicts cross-cell-line drug transport
  (Paper 5: drug-perturbation-geometry)
- Standard MS/MS benchmarks fail to predict cross-instrument transfer
  (Paper 6: msms-subspace-collapse)

This pre-registration tests whether geometric transportability metrics
predict cross-assay transfer F1 ACROSS MODELS — i.e., can geometric
distance between source and target embedding subspaces predict which
model will transfer best?

---

## 2. Object of study

### Geometric metrics under test

Three geometric transportability metrics, computed per (model, tissue)
on source (10x 3' v3) and target (Smart-seq2) embeddings:

**G1 — Grassmannian geodesic distance.**
PCA top-k subspaces of source and target embeddings (k = min(20, d/2)).
Geodesic = L2 norm of principal angles. Lower distance = more preserved
geometry = predicted better transfer.

**G2 — Subspace overlap.**
Mean cosine of principal angles between source and target PCA subspaces.
Higher overlap = predicted better transfer.

**G3 — Direction stability (bootstrap eigenvector stability).**
Draw 50 bootstrap resamples of source embeddings; compute PCA on each;
measure Grassmannian distance between each bootstrap subspace and the
full-sample subspace. Report mean distance. Lower instability = more
robust structure = predicted better transfer.

### Model panel

All models from PREREGISTRATION_SCIB_EXPANDED_PANEL.md that
successfully produce embeddings:

**Contenders (expected):** Geneformer V1, V2-104M, V2-316M, scGPT,
scVI, BoG-PCA-512.

**Nulls:** Random projection, untrained encoder.

**Conditional:** UCE, scPRINT, Nicheformer, CellPLM — included only
if embeddings were generated successfully.

### Data

Identical to parent study: CellxGene Census v2023-12-15, 4 tissues,
n=2,000 per side, source=10x 3' v3, target=Smart-seq2.

### Ground truth

Cross-assay cell-type transfer F1 (macro-averaged kNN or logistic
regression F1 from source to target), as computed in the parent study.

---

## 3. Pre-committed hypotheses

**HG1 (primary):** Grassmannian geodesic distance (G1) negatively
correlates with cross-assay transfer F1 across all (model, tissue)
conditions. Test: Spearman rho with 10,000-permutation exact p-value.
Pass: rho < -0.30 with p < 0.05. Expected direction: negative
(smaller distance = better transfer).

**HG2 (secondary):** Subspace overlap (G2) positively correlates with
cross-assay transfer F1. Test: Spearman rho, 10,000-permutation p.
Pass: rho > +0.30 with p < 0.05.

**HG3 (exploratory):** Direction stability (G3) negatively correlates
with cross-assay transfer F1. Test: Spearman rho, 10,000-permutation p.
This is exploratory because the single-model prior (M3 rho=+0.30 on
n=12) was underpowered and the cross-model relationship is untested.

**HG4 (comparative):** For each geometric metric that passes (rho
significant at p < 0.05), compare its |rho| against the best scIB bio
metric's |rho| on the same conditions. Report whether the geometric
metric has stronger correlation with transfer F1 than scIB.

**HG5 (partial correlation):** For each significant geometric metric,
report the partial Spearman rho controlling for embedding dimensionality.
This tests whether the geometric prediction is confounded by d.

---

## 4. Statistical methods

- Spearman rank correlation with 10,000-permutation exact p-values.
- Bootstrap 95% CIs (10,000 resamples, percentile method).
- Partial Spearman via rank residuals (controlling for d).
- No multiplicity correction on the primary hypothesis (HG1). BH
  q-values reported for HG1–HG3 as sensitivity analysis.
- Power note: with 6 contenders x 4 tissues = 24 conditions, the
  minimum detectable |rho| at alpha=0.05 (two-sided) is ~0.40.
  With nulls included (8 models x 4 = 32), ~0.35.

---

## 5. Analysis specification

For each (model, tissue) condition:
1. Load source and target embeddings (Census or .npy from volume).
2. Compute PCA on source (k = min(20, d/2)) and target separately.
3. Compute G1 (geodesic), G2 (overlap), G3 (bootstrap stability, 50 resamples).
4. Load transfer F1 from parent study results.
5. Aggregate into (n_conditions,) arrays for correlation.

Conditions with failed embeddings or missing F1 are excluded with
disclosure.

---

## 6. Kill / integrity rules

1. If fewer than 3 contender models produce valid embeddings on all
   4 tissues, the analysis is underpowered and reported as such.
2. No metric is added after freeze.
3. No threshold is changed after freeze.
4. If all three geometric metrics fail (no significant rho), the paper
   reports this as a negative result alongside the scIB failure.
5. Results are saved to JSON before any interpretation or plotting.
