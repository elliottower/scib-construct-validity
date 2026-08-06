# Preregistration v11: Cross-Tissue Second Ground Truth

**Status**: FROZEN — SHA-256 hash computed before any cross-tissue
results for scVI, scGPT, or BoG-PCA are generated.
**Scorer code**: `compute_scib_metrics` from `exp10_scib_audit.py` +
probe-F1 from `exp5_cross_tissue.py` (both frozen; hashes below).
**Census version**: 2023-12-15 (same as primary ground truth).
**RNG seed**: 20260801 (distinct from primary-GT seed 20260713).
**Bootstrap**: 10,000 resamples for every CI and p-value.
**Date**: 2026-07-13.

---

## 0. Motivation and scope

Paper D (v11) establishes that scIB bio-conservation metrics invert
46–58% of pairwise contender rankings against cross-assay transfer F1
(10x → Smart-seq2). A Genome Biology reviewer concern is that this
finding may be specific to the cross-assay transfer setting rather than
a general property of geometric metrics applied to model ranking.

This preregistration defines a confirmatory test using a **second,
independent ground truth**: cross-tissue cell-type transfer F1 at
matched assay (10x 3' v3 only, different tissues). Cross-tissue and
cross-assay transfer stress mechanistically distinct axes: tissue shift
tests whether embeddings preserve biological signal across different
cell-type compositions and transcriptional programs, while assay shift
tests robustness to protocol artifacts (capture efficiency, UMI
distributions, gene detection sensitivity). A metric that inverts
rankings against both is failing across independent failure modes, which
is stronger evidence than failing on one. If the inversion rates
reproduce on cross-tissue transfer, the finding generalizes beyond the
cross-assay setting. If they do not, the paper's claim must be scoped
to cross-assay transfer specifically.

**What already exists.** Experiment 5 (v7) produced cross-tissue
composite scores and probe F1 for **Geneformer only** (6 cross-tissue
pairs + 3 negative controls). This preregistration extends the analysis
to scVI, scGPT, and BoG-PCA-512 so pairwise inversions can be computed.
No results for these three models on cross-tissue pairs exist at the
time of freezing.

---

## 1. Design

### 1.1 Embedding set (contenders-4)

Same contender set as the primary ground truth:

| Model | Type | d |
|-------|------|---|
| Geneformer | trained FM | 512 |
| scVI | trained VAE | 50 |
| scGPT | trained FM | 512 |
| BoG-PCA-512 | no-learned-parameter baseline | 512 |

Null baselines (random projection, untrained encoder) are excluded from
the inversion analysis (same rationale as primary GT: "is it trained?"
is not the deployment question).

### 1.2 Tissue pairs

Cross-tissue pairs at matched assay (10x 3' v3 source → 10x 3' v3
target, different tissues, zero donor overlap). Pairs selected from the
6 that produced valid Geneformer results in v7:

| Pair | Source | Target | Geneformer probe F1 (v7) |
|------|--------|--------|-------------------------|
| lung → brain | lung | brain | 0.532 |
| lung → kidney | lung | kidney | 0.429 |
| blood → lung | blood | lung | 0.185 |
| blood → brain | blood | brain | 0.612 |
| liver → kidney | liver | kidney | 0.982 |
| blood → liver | blood | liver | 0.413 |

n = 2,000 cells per side. Donor-stratified subsampling.

Note: kidney appears as target in 2 pairs (lung→kidney, liver→kidney).
This is critical because the primary GT's sharpest clear-gap inversion
is BoG-PCA outperforming scVI/scGPT on kidney transfer while every bio
metric ranks it lower. The cross-tissue panel preserves this test.

### 1.3 Ground truth

Cross-tissue cell-type transfer F1: macro-averaged logistic-regression
cell-type classification F1 (same probe as the primary GT and v7
cross-tissue analysis) trained on source tissue cells, evaluated on
target tissue cells, restricted to cell types present in both tissues.

### 1.4 Metrics evaluated

Same 5 bio-conservation scIB metrics from the primary analysis: NMI,
ARI, silhouette-label, cLISI, isolated-label ASW. Same hyperparameters
(Leiden r = 1.0, kNN k = 15). Metrics are computed on the combined
source+target AnnData with `batch_key="tissue"` and
`label_key="cell_type"`, mirroring the primary GT setup (which uses
`batch_key="assay"`).

---

## 2. Hypotheses

### H11.1 [CONFIRMATORY]: Cross-tissue inversion rates reproduce

**Claim**: scIB bio-conservation metrics invert pairwise contender
rankings against cross-tissue transfer F1 at rates comparable to the
cross-assay primary ground truth (46–58%).

**Test**: For each of the 5 bio metrics, compute the pairwise inversion
rate against cross-tissue F1 across all contenders-4 pairs within each
tissue pair (C(4,2) = 6 model pairs × 6 tissue pairs = 36 comparisons
per metric). The 36 comparisons are not independent: only 6 unique
model-pair contrasts exist, each observed across 6 tissue contexts.
Bootstrap CIs are therefore clustered by tissue pair (block bootstrap,
10,000 resamples) to avoid overstating precision.

**Null hypothesis**: The true mean bio-metric inversion rate ≤ 0.30
(i.e., metrics retain meaningful predictive validity for cross-tissue
transfer). One-sided block-bootstrap p-value: fraction of 10,000
bootstrap grand-mean resamples that fall ≤ 0.30.

**Confirm rule**: Reject H₀ at Holm-adjusted α (see §3). The finding
reproduces if the mean inversion rate is ≥ 0.40 **and** the one-sided
p-value survives Holm correction.

**Overturn rule**: The finding is overturned if the mean inversion rate
is < 0.30 (point estimate below the null boundary).

**Indeterminate**: If the point estimate is ≥ 0.40 but the p-value does
not survive Holm correction, the result is reported as "indeterminate —
point estimate consistent with replication but insufficient power to
reject that metrics have modest predictive validity in the cross-tissue
setting." This is not confirmation.

### H11.2 [CONFIRMATORY]: scIB composite anti-prediction holds on cross-tissue GT

**Claim**: The scIB bio-conservation composite (unweighted mean of the
5 bio metrics in §1.4 — not the Preflight m1–m6 composite from v7)
anti-predicts (or fails to predict) cross-tissue transfer F1, consistent
with the cross-assay finding (ρ = −0.76 on primary GT).

**Test**: Spearman correlation between the scIB bio composite and
cross-tissue probe F1 across all model × tissue-pair conditions where
both are computable (up to 4 models × 6 pairs = 24 conditions).
10,000-resample bootstrap 95% CI (clustered by tissue pair).
10,000-permutation exact p-value.

**Confirm rule**: The composite's Spearman ρ against cross-tissue F1 is
negative or the 95% CI includes zero — consistent with anti-prediction
or no predictive power.

**Overturn rule**: The composite achieves ρ > 0 with 95% CI excluding
zero — meaning the composite positively predicts cross-tissue transfer,
contradicting the primary GT finding.

### H11.3 [EXPLORATORY]: Inversion rate by F1-gap stratum

Compute the same F1-gap split (clear winners |ΔF1| > 0.10 vs.
near-ties |ΔF1| ≤ 0.10) on the cross-tissue data. Report rates per
stratum. No preregistered threshold; reported with uncorrected CIs and
labeled exploratory.

---

## 3. Multiple testing

Two confirmatory hypotheses, each producing a p-value.
Holm–Bonferroni correction within this family at α = 0.05:
1. Order the two p-values (H11.1 one-sided bootstrap, H11.2
   permutation) smallest first.
2. Compare the smallest to α/2 = 0.025; if rejected, compare the
   second to α/1 = 0.05.
H11.3 is exploratory and excluded from the correction family.

---

## 4. Analysis plan

1. Generate embeddings for scVI, scGPT, and BoG-PCA-512 on all 6
   cross-tissue pairs (n = 2,000 per side). Regenerate Geneformer
   embeddings on the same cells (v7 stored only summary statistics,
   not raw embeddings; the scIB metrics need the full embedding
   matrices). Verify Geneformer probe F1 matches v7 values within
   bootstrap noise (sanity check).
2. Compute 5 scIB bio-conservation metrics on all 4 models × 6 pairs
   (combined source+target AnnData, batch_key="tissue").
3. Compute cross-tissue probe F1 (logistic regression, same as v7
   and primary GT) for all 4 models × 6 pairs.
4. Compute pairwise inversion rates (H11.1).
5. Compute composite-vs-F1 Spearman correlation (H11.2).
6. Compute F1-gap stratified rates (H11.3).
7. Save all results to `results/exp11_cross_tissue_validity.json`.

### 4.1 Handling missing data

If a tissue pair produces < 3 shared cell types for a model (too few
for meaningful classification), that condition is excluded and reported.
If > 50% of conditions are excluded for any model, that model is dropped
from the analysis and the exclusion is reported.

### 4.2 scVI dimension handling

scVI (d = 50) is included in the inversion analysis (same as primary GT)
but excluded from any matched-dimensionality comparison, consistent with
the v11 paper's stated policy.

---

## 5. Deliverables

- `results/exp11_cross_tissue_validity.json`: all scores, F1s, inversions
- Update to paper_d_v12.tex with cross-tissue results
- Comparison table: primary GT inversion rates vs. cross-tissue rates

---

## 6. Falsification

If H11.1 is overturned (mean inversion rate < 0.30), the paper must:
1. Add "for cross-assay transfer" to every claim about scIB validity
2. Discuss why the metrics may work for cross-tissue but not cross-assay
3. This is a genuine scope reduction, not a failure of the paper

If H11.2 is overturned (composite positively predicts cross-tissue F1),
the paper must:
1. Report the finding as evidence that the composite has predictive
   validity in the cross-tissue setting
2. Investigate whether the anti-prediction on cross-assay GT is driven
   by the assay-specific variance mechanism (which would not apply to
   cross-tissue at matched assay)

---

## 7. SHA-256 freeze

This document and the analysis code are hashed before any
scVI/scGPT/BoG-PCA cross-tissue embeddings are generated.

Hash of this document (pre-insertion version): c1add08688b2fd13f3a655ba5c173a3b3bb862f2429774ef21d460070bfc43f8
Hash of exp11_cross_tissue_validity.py: a49524424d7577d8d009dc6571ea3e5ac2dda9f73eca3766d5ce90487ab8f041
Hash of exp10_scib_audit.py (scIB computation): 8c006b5050d9181bb80632279943adb10df861387846381b88d9bd0aeae7866a
Hash of exp5_cross_tissue.py (probe + Census queries): 827137b5cc0a0fb95c610e0417eca575efa731394b5965555d2aaab8ee830e47
