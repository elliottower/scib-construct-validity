# Pre-registration: Null-corrected relational structure metrics

**Version**: 2

**Correction from v1**: scVI native dimensionality is d=50 in our data, not d=128 as stated in PREREGISTRATION_SCGRAPH_CONFIRMATORY.md. The original prereg inherited d=128 from the scVI architecture specification, but our Census embeddings were extracted at d=50. All P-5 analyses used the correct d=50 data; only the text was wrong.

## Scientific question

The paper's central thesis is that you cannot interpret a metric score without characterizing its null. CKA on cell-type centroids has a closed-form analytic null E[CKA] = 1 - 1.06*(k-1)/(d+k) that produces null saturation at high d. We showed this analytically for CKA and empirically for Leiden-dependent scIB metrics (via permutation-calibrated noise tests).

Our relational structure metrics passed their primary pre-registered predictions: dimensionality-stable (P-1 PASS, median delta 0.017) and discriminating embedding quality (P-2 PASS, geneformer > scgpt in 77% of tissues). However, two secondary predictions failed:

- **P-3 FAIL**: Spearman rho(RCS centroid_dist, transfer F1) = 0.24 across all 66 conditions, below the 0.30 threshold. Within-model tissue-level prediction is absent.
- **P-5 FAIL**: scVI vs geneformer ordering agreement on (RCS, F1) = 50%, below the 60% threshold. Diagnosis: scVI at d=50 has lower-dimensional embeddings where kNN structure is better preserved, inflating RCS independently of embedding quality.

The P-5 failure diagnosis is a null-dependence claim. If the paper's thesis is correct, null-correcting relational metrics should fix this inflation.

### Implementation divergence

Two distinct metrics are evaluated in this experiment:

1. **Official scGraph** (`scgraph-eval` v0.1.2, github.com/Genentech/Islander): Wang et al. 2026. Computes PCA (10 components) on highly variable genes within each batch, trimmed mean centroids (5% trim), pairwise centroid distances normalized by column max, then scores via Rank-PCA (Spearman), Corr-PCA (Pearson), and Corr-Weighted (weighted Pearson). Compares each batch embedding to a multi-batch consensus. No kNN component.

2. **Relational consistency score (RCS)**: Our variant, used in all prior experiments (PREREGISTRATION_SCGRAPH_CONFIRMATORY.md). Computes three Spearman correlations on upper-triangular entries of k-by-k cell-type relationship matrices between source and target embeddings at native d: centroid distance correlation, kNN overlap (k=15, cosine distance), and weighted affinity (mean 1/(1+dist) for cross-type kNN edges). No PCA step; pairwise comparison rather than consensus.

The official scGraph includes a PCA reduction step that may already absorb d-dependent null effects. Our RCS operates at native d and has demonstrated d-dependence (P-5 FAIL). This experiment tests null correction on both implementations.

Wang et al. do not discuss null correction or dimensionality dependence of their metrics.

## Design

### Data

**In-sample (cross-assay panel):**
- 66 conditions: 22 tissues x 3 models (geneformer d=512, scgpt d=512, scvi d=50)
- Assay pairs: 10x 3' v3 (source) vs Smart-seq2 (target)
- RCS scores already computed; official scGraph scores to be computed fresh.
- This panel was used to develop and test original RCS predictions. It is in-sample for the correction.

**Holdout (cross-tissue panel):**
- 24 conditions: tissue-to-tissue transfer pairs across 4 models (geneformer d=512, scgpt d=512, scvi d=50, bog_pca d=512)
- scIB scores and transfer F1 available; neither scGraph nor RCS was ever computed on this data.
- Source: `results/exp11_cross_tissue_validity/exp11_cross_tissue_validity.json`
- Genuine holdout: different evaluation regime (cross-tissue vs cross-assay), never touched by metric development.

### Three experimental arms

| Arm | Metric | Implementation | Role |
|-----|--------|---------------|------|
| A | Official scGraph | `scgraph-eval` package, unmodified | Reference: does the citable metric have d-dependence? |
| B | Official scGraph + null correction | Same + S_corr formula | Confirmatory: does correction improve the official metric? |
| C | RCS + null correction | Our variant + S_corr formula | Robustness: does the correction principle transfer across implementations? |

### Null correction

For each metric S and each (k, d) pair, define the null-corrected score:

    S_corr = (S_obs - E[S_null]) / (1 - E[S_null])

where E[S_null] is the expected metric under random Gaussian centroid matrices (k x d, entries ~ N(0,1)).

**Null estimation procedure:**
- For each unique (k, d) pair in the data, generate 1000 independent pairs of random Gaussian centroid matrices (k x d).
- Compute each metric on each pair.
- E[S_null] = mean across the 1000 draws.
- Compute separately for official scGraph and RCS (different metric definitions produce different nulls).
- If a closed-form expression is derivable for any metric, use it and verify against Monte Carlo within 1%.

### What null correction captures

Relational structure metrics compute correlations between k-by-k matrices derived from cell-type centroids. Although the final correlation operates in k-by-k space, intermediate steps (computing centroids, pairwise distances, kNN graphs) operate in d-dimensional space. At low d, random Gaussian centroids produce distance matrices with more structured pairwise relationships, yielding higher expected scores under the null.

The official scGraph includes a PCA reduction to 10 components, which may already normalize d-dependence. If so, the null correction will have minimal effect on Arm A/B — which is itself an informative result (PCA absorption of the null effect).

The correction does NOT address the resolution limit identified in P-3 (within-model tissue-level prediction absent at rho = 0.24). That limitation is likely structural: k-by-k matrices discard within-model, cross-tissue variation by design. We do not attempt to fix P-3.

## Predictions

### P-6a (primary): Official scGraph shows d-dependence

Official scGraph scores differ between native d and PCA-reduced d=50 by more than a minimal threshold.

**Criterion**: Median |scGraph_official(native_d) - scGraph_official(d=50)| > 0.05 across d=512 conditions.

**Rationale**: If the official scGraph's internal PCA step already normalizes d-dependence, we expect minimal change and this prediction FAILS. A failure here means the correction is unnecessary for the official metric — the contribution narrows to explaining why implementation choices (PCA vs no PCA) determine whether null correction is needed. Either outcome is informative and publishable.

### P-6b (primary): Null correction fixes scVI ranking on at least one implementation

After null correction, scVI vs geneformer ordering agreement on (corrected metric, F1) >= 60% on either official scGraph (Arm B) or RCS (Arm C) or both.

**Criterion**: For each tissue where both scVI and geneformer are available (n=22), compare whether corrected scores and F1 agree on which model is better. Agreement rate >= 60% (the threshold uncorrected RCS missed at 50% in P-5). Test on both implementations; report both.

**Rationale**: The P-5 failure was diagnosed as d-dependent null inflation. If the diagnosis is correct, correction should fix the ordering on at least the implementation that operates at native d (RCS). If the official scGraph already absorbs d-effects via PCA, its uncorrected ordering may already exceed 60%.

### P-7 (primary): Null correction does not degrade passing predictions

On both implementations where applicable:

- **P-7a (P-1 retest, RCS only)**: Median |corrected_RCS(native_d) - corrected_RCS(d=50)| < 0.10 for centroid distance correlation across all d > 50 conditions.
- **P-7b (P-2 retest, both)**: Corrected metric for geneformer > scgpt in >= 70% of tissues (centroid distance for RCS; Rank-PCA for official scGraph).

**Rationale**: Correction should not degrade existing successes. For P-7b, both geneformer and scgpt are at d=512, so the correction shifts both by the same E[S_null](k, 512), preserving their ordering.

### P-8 (secondary): Corrected metrics improve F1 correlation

Spearman correlation between corrected metric and transfer F1 across all 66 cross-assay conditions >= 0.30, on at least one implementation.

**Criterion**: rho >= 0.30 (the threshold uncorrected RCS missed at 0.24 in P-3). Test on both implementations.

**Rationale**: Removing the d-dependent baseline should tighten the relationship between corrected scores and F1 by placing all three models on the same scale.

### P-9 (secondary, holdout): Metrics generalize to cross-tissue evaluation

On the cross-tissue panel (24 conditions, never used for either metric), corrected metric >= 0.20 in >= 80% of conditions, on at least one implementation.

**Criterion**: Fraction of cross-tissue conditions where corrected centroid-based score >= 0.20. Test on both implementations.

### P-10 (secondary, holdout): Corrected metrics predict cross-tissue transfer

On the cross-tissue panel, Spearman correlation between corrected metric and transfer F1 >= 0.25 on at least one implementation.

**Criterion**: rho >= 0.25 with p < 0.05 (one-sided). Test on both implementations.

**Rationale**: The cross-tissue panel has 24 conditions across 4 models, so power is limited. The 0.25 threshold accounts for this.

## Analysis plan

1. Install `scgraph-eval` and compute official scGraph on all 66 cross-assay conditions and all 24 cross-tissue conditions.
2. For each unique (k, d) pair across both panels, estimate E[S_null] via 1000 Monte Carlo draws, separately for official scGraph metrics and RCS metrics.
3. Compute corrected scores: S_corr = (S_obs - E[S_null]) / (1 - E[S_null]) for all metrics on all three arms.
4. **P-6a**: Compute |scGraph_official(native) - scGraph_official(d50)| for d=512 conditions. Report median.
5. **P-6b**: For 22 tissues with scVI and geneformer, compare corrected ordering to F1 ordering. Report agreement rate for both Arm B and Arm C.
6. **P-7a**: Compute |corrected_RCS(native) - corrected_RCS(d50)| for d > 50 conditions. Report median and IQR.
7. **P-7b**: Paired comparison of corrected metric (geneformer vs scgpt) per tissue, both implementations. Report fraction.
8. **P-8**: Spearman correlation of corrected metric with F1 across 66 conditions, both implementations. Report rho and p.
9. **P-9**: Fraction of 24 cross-tissue conditions with corrected score >= 0.20, both implementations.
10. **P-10**: Spearman correlation of corrected metric with F1 on 24 cross-tissue conditions, both implementations. Report rho and p.
11. Report uncorrected vs corrected scores side-by-side for all predictions, both implementations.
12. Report E[S_null] as a function of (k, d) for both implementations and assess whether a closed-form approximation is feasible.
13. Compare official scGraph and RCS directly: do they agree on model rankings? Report Spearman correlation between implementations.

## Decision rules

- **If P-6a FAILS (official scGraph shows no d-dependence)**: The official scGraph's PCA step already absorbs d-effects. The contribution becomes: (a) explaining why implementation choices determine null sensitivity, and (b) null-correcting RCS as a demonstration of the thesis. Report official scGraph's d-immunity as a finding — PCA reduction is itself a form of implicit null correction.
- **If P-6a PASSES and P-6b PASSES on Arm B**: Null correction improves the official, citable metric. Strongest result. The paper's thesis (characterize the null to fix the metric) gains a third confirmatory test applied to someone else's published metric.
- **If P-6b PASSES on Arm C only**: Correction works on RCS but not needed for official scGraph. Still validates the thesis on our implementation; report that the official metric avoids the problem by design (PCA).
- **If P-6b FAILS on both arms**: The d-dependent inflation in P-5 is NOT a null-dependence artifact. The resolution limit is structural. Report as a third diagnosable failed repair.
- **If P-7 FAILS**: Correction degrades existing performance. Investigate which sub-prediction failed and why. Report as a mixed result.
- **If P-9 or P-10 fail (holdout)**: Relational metrics do not generalize to cross-tissue evaluation. Report as a scope boundary.

## Code

Script: `scripts/exp13_null_corrected_relational.py`
Modal wrapper: `scripts/modal_null_corrected_relational.py`
Output: `results/null_corrected_relational/`

## Prior results motivating this experiment

- **P-5 FAIL** (PREREGISTRATION_SCGRAPH_CONFIRMATORY.md): scVI vs geneformer ordering agreement = 50%, below 60% threshold. Diagnosis: scVI at d=50 has inflated RCS scores because lower-dimensional embeddings produce better-preserved kNN structure independent of embedding quality. This is a null-dependence problem.
- **P-3 FAIL**: rho(RCS, F1) = 0.24 across 66 conditions, below 0.30 threshold. Within-model tissue prediction absent. Likely structural (k-by-k matrices discard within-model variation). NOT targeted by null correction.
- **Implementation divergence**: Official scGraph (Wang et al. 2026) includes PCA reduction and consensus scoring. Our RCS operates at native d with pairwise Spearman. The PCA step may implicitly correct for d-dependence. This experiment tests both.
- **Paper thesis**: You cannot interpret a metric score without characterizing its null. CKA null saturation is the analytic proof. Leiden-dependent null inflation is the empirical proof. Null-correcting relational metrics is the constructive test.
- **Two prior failed repairs**: Spectral-gap statistic inverted. Z-score normalization degraded ranking. Both diagnosed and informative. A success here would demonstrate the thesis is constructive.
- **Chance-corrected agreement precedent**: S_corr = (S_obs - E[S_null]) / (1 - E[S_null]) is the standard form of Cohen's kappa. Applying it to relational metrics extends an established measurement-theory principle to single-cell embedding evaluation.
- **No existing null characterization**: Wang et al. do not discuss null correction, expected values under random embeddings, or dimensionality dependence of scGraph. This gap is what the experiment fills.
