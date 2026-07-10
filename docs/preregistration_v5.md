# Preflight Bio — Experiment Preregistration (v5)

**Status:** FROZEN — SHA computed after scorer changes, before re-run  
**Date:** 2026-07-10  
**Author:** Elliot Tower  
**Changes from v4:** Post-hoc corrections based on v4 exp0 results (8 valid pairs). Three changes, all documented transparently:

1. **M7 removed from default weights.** M7 (ecological bias) measures between-study disease-rate variation via CV of site-level outcome rates. On Census data, datasets mix healthy and diseased tissue (e.g., lung: 48% normal, 34% COVID, 18% cancer across 9 datasets), producing extreme CV and score=0.0 (Tier 1) for most pairs. This is real ecological heterogeneity but irrelevant to embedding transfer diagnostics. M7 remains available when users explicitly pass records.

2. **Worst-tier gate removed.** The gate capped overall tier at 3 whenever any module scored Tier ≤2. Since M2 correctly scores low on real shift pairs (high domain AUC = low transportability score), the gate systematically crushed all shift pairs to Tier 3 regardless of composite score. Negative controls with composite 0.85+ were also capped at Tier 3 when M7 triggered. Removing the gate restored clean separation: shifts at T3-4, controls at T7.

3. **Shared cell-type threshold relaxed from 5 to 3.** In v4, 17 of 25 pairs were dropped for having <5 shared cell types with ≥10 cells each. Cross-tissue and cross-dataset pairs were disproportionately affected. Relaxing to ≥3 types retains more pairs for the correlation test while still ensuring meaningful classification probes.

**Transparency note:** These changes were motivated by v4 results. The v4 raw Spearman ρ = −0.786 (p = 0.021, n=8) is reported as-is under the v4 preregistration. The v5 re-run tests the same hypotheses with the corrected scorer and relaxed threshold. Both v4 and v5 results are reported side-by-side.

**Prior versions:** v4 (original), v3 (H0.2/H0.3 bin-size gate), v2 (Perplexity review fixes)

---

## Overview

Six experiments validating the Preflight Bio transportability diagnostics platform on real CELLxGENE Census data. **Experiment 0 is the keystone**: it tests whether Preflight's composite score predicts actual downstream task degradation across diverse source→target pairs. Experiments 1–5 are supporting experiments that validate individual components and establish baselines.

All hypotheses are registered BEFORE any data access. After review and finalization, scorer source code + hyperparameters + dataset specs are SHA-256 hashed and frozen.

**Available embeddings on Census (v2023-12-15):**
- `geneformer` — 512 dimensions (transformer, pretrained on ~30M cells)
- `scvi` — 50 dimensions (VAE, dataset-specific)
- Raw expression `X` — 60,664 genes (bag-of-genes baseline, log1p-normalized)

**Available metadata columns:** cell_type, tissue, assay, disease, dataset_id, donor_id, is_primary_data

---

## Experiment 0: Composite Validation — Does the Tier Predict Real Degradation?

### Motivation

Preflight currently demonstrates that its modules *fire* on known distribution shifts and *don't fire* on negative controls. What it has not demonstrated is that the composite transportability score predicts how badly a downstream task actually degrades. Until this link is established, the composite is seven validated-in-isolation modules with unvalidated weights. This experiment closes that gap.

### Design

Construct ~25 source→target embedding pairs spanning a wide range of expected degradation magnitudes. For each pair, measure both the Preflight composite score and the actual downstream probe degradation.

**Pair types (all from Census, Geneformer embeddings, 2000 cells per condition):**

| Type | Construction | Expected shift | N pairs |
|------|-------------|----------------|---------|
| **Cross-assay, same tissue** | Tissue X, 10x 3'v3 → 10x 5'v2 | Moderate | 5 |
| **Cross-tissue, same assay** | Tissue A 3'v3 → Tissue B 3'v3 | Large | 10 |
| **Cross-dataset, same tissue+assay** | Tissue X, dataset A → dataset B | Small | 5 |
| **Negative control (random split)** | Tissue X, same pool, seed A → seed B | None | 5 |

**Tissues:** Lung (UBERON:0002048), Heart (UBERON:0000948), Liver (UBERON:0002107), Kidney (UBERON:0002113), Brain (UBERON:0000955)

**Cross-tissue pairs (10):** All unordered pairs from {Lung, Heart, Liver, Kidney, Brain} = C(5,2) = 10 pairs, using 10x 3'v3 where available.

**Metrics per pair:**
- Preflight composite score (weighted mean of active modules, 0–1)
- Preflight composite tier (1–7)
- Per-module scores (M1, M2, M3, M4, M6; M7 only when records explicitly provided)
- F1_source: cell-type probe macro F1 via donor-stratified 5-fold CV on source
- F1_target: train probe on ALL source cells, evaluate on ALL target cells
- Absolute degradation = F1_source − F1_target
- **Relative degradation = (F1_source − F1_target) / F1_source** (PRIMARY endpoint)

### Hypotheses

**Primary analysis is on shifted pairs only (cross-assay + cross-tissue + cross-dataset, ~20 pairs).** Negative-control pairs are excluded from the primary correlation because they anchor at (high score, ~0 degradation) and inflate ρ trivially. The with-negative-controls analysis is reported as a secondary check.

**H0.1 (PRIMARY):** Preflight composite score and relative degradation are negatively correlated across shifted pairs, after partialling out F1_source (partial Spearman ρ < −0.4, p < 0.05).  
*Rationale:* Higher composite score means better predicted transportability, which should correspond to less relative degradation. The partial correlation controls for source quality — a pair with a weak source probe mechanically cannot degrade much (floor effect), which would inflate a raw correlation. The −0.4 threshold is conservative for n ≈ 20 shifted pairs.

**H0.1b (SECONDARY):** Raw Spearman ρ(composite, relative degradation) across all ~25 pairs including negative controls is < −0.5.  
*Rationale:* Descriptive robustness check using a different estimator (raw ρ, not partial). H0.1 and H0.1b use different correlation methods (partial vs raw) and different sample compositions (shifted-only vs all), so discrepancies between them are expected and do not indicate a problem. H0.1b is not a pass/fail gate — it is reported for transparency.

**H0.1c (SUBGROUP):** Within the cross-tissue group alone (~10 pairs), Spearman ρ(composite, relative degradation) < −0.3.  
*Rationale:* The cross-tissue pairs will cluster at "large shift / high degradation." If the correlation holds within this group (not just from the contrast between groups), the composite discriminates among large-shift scenarios, not just between "shift" and "no shift."

**H0.2:** Pairs with composite Tier ≥ 5 have mean relative degradation < 0.15 (F1 drops by < 15% of source F1).  
*Rationale:* "Acceptable" tier should correspond to acceptable real-world performance.  
*Minimum bin size:* H0.2 is only evaluated if ≥ 4 pairs fall in Tier ≥ 5. If fewer than 4 pairs land in this bin, report the individual values descriptively without testing the hypothesis. A mean over n < 4 is not a reliable estimator and should not be treated as a pass/fail result.

**H0.3:** Pairs with composite Tier ≤ 2 have mean relative degradation > 0.40 (F1 drops by > 40% of source F1).  
*Rationale:* "Very Poor" / "Failure" tier should correspond to severe performance collapse.  
*Minimum bin size:* H0.3 is only evaluated if ≥ 4 pairs fall in Tier ≤ 2. Same reasoning as H0.2 — report descriptively if the bin is too small.

**H0.4:** M2 domain shift AUC alone achieves partial ρ < −0.3 with relative degradation (controlling for F1_source) on shifted pairs.  
*Rationale:* M2 is the a-priori expected strongest single predictor based on prior MSMS results. This establishes the single-module baseline.

**H0.5:** The composite outperforms M2 specifically: composite |partial ρ| > M2 |partial ρ| on shifted pairs.  
*Rationale:* The comparison is against M2 by name (the a-priori favorite), not against the best-of-6 post hoc. If the composite doesn't beat M2, the weighting scheme adds no value over a single domain classifier and should be replaced. Per-module ρ values for all other modules are reported descriptively but not tested against the composite.

### Decision criteria

- **H0.1 passes:** The composite is a validated transfer-failure predictor. This is the fundable, publishable, product-ready claim.
- **H0.1 fails (partial ρ > −0.4):** The composite weights need recalibration. Use the ~20 shifted pairs as a training set to learn weights via leave-one-out cross-validated regression, then report the CV ρ.
- **H0.5 fails:** The composite adds no value over M2 alone. Simplify the product to M2 + probes and drop the composite scoring.
- **Both H0.2 and H0.3 pass:** The tier scale has real meaning — "Tier 5" and "Tier 2" correspond to qualitatively different outcomes.
- **H0.1c fails:** The composite distinguishes shift-vs-no-shift but not among shift severities. The correlation is driven by cluster contrast, not smooth prediction. Report this honestly; it limits the product claim to binary go/no-go rather than graded risk.

### Analysis plan

1. Enumerate all ~25 pairs (tissue × assay × pair-type combinations)
2. For each pair:
   a. Pull source and target from Census (2000 cells each)
   b. Run `preflight.run()` → composite score, per-module scores
   c. Train cell-type probe on source (donor-stratified 5-fold CV) → F1_source
   d. Train probe on ALL source cells, evaluate on ALL target cells → F1_target
   e. Compute relative degradation = (F1_source − F1_target) / F1_source
3. **Primary:** Compute partial Spearman ρ(composite_score, relative_degradation | F1_source) on shifted pairs only (~20 pairs)
4. **Secondary:** Compute raw Spearman ρ across all ~25 pairs including negative controls
5. **Subgroup:** Compute Spearman ρ within cross-tissue pairs only (~10 pairs)
6. Compute partial ρ per module (M1, M2, M3, M4, M6) on shifted pairs — report all, test H0.5 against M2 only
7. Test H0.2 and H0.3 by grouping pairs by tier, using relative degradation
8. Plot: composite score vs relative degradation, with pair-type as color, F1_source as point size
9. Save all results to results/composite_validation/

### Contingencies

- Some tissue × assay combinations may have too few cells or too few cell types for a meaningful probe. Pairs with < 3 shared cell types (present in both source and target with ≥ 10 cells each) are dropped. Record the number of dropped pairs.
- If < 15 pairs survive filtering, the correlation test is underpowered. In that case, relax cross-tissue pairs to include cross-assay within the same tissue for additional tissues (e.g., blood, pancreas).

---

## Experiment 1: Bag-of-Genes Baseline

### Motivation

Without a non-learned baseline, we cannot distinguish "the model is good" from "any embedding works." Bag-of-genes (log1p-normalized raw expression, PCA-reduced) provides this control. Also needed as the third embedding for Experiment 0 replication.

### Design

- **Source:** Lung, 10x 3' v3 (EFO:0009922), 2000 cells
- **Target:** Lung, 10x 5' v2 (EFO:0011025), 2000 cells
- **Embeddings compared:**
  - Geneformer (512d, from Census obsm)
  - scVI (50d, from Census obsm)
  - Bag-of-genes (raw X, log1p, PCA to 512d and separately to 50d)
- **Metrics:** All active modules + cell-type probe (donor-stratified, k=5)

### Hypotheses

**H1.1:** Geneformer cell-type probe macro F1 > bag-of-genes F1.  
*Rationale:* A pretrained model should encode cell-type-discriminative features better than raw expression.

**H1.2:** Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC.  
*Rationale:* Learned embeddings may absorb assay-specific batch effects that raw expression does not. If Geneformer's domain AUC is *higher* than bag-of-genes, the model has learned domain-specific (non-transportable) features.

**H1.3:** Bag-of-genes M1 Grassmannian distance ≤ Geneformer M1 distance.  
*Rationale:* If the learned model distorts the subspace under assay shift more than raw expression does, the model is adding domain-specific structure.

### Decision criteria

- If H1.1 fails (bag-of-genes ≥ Geneformer on F1): the model adds no value over raw expression for this tissue/assay pair. Report as a negative finding.
- If H1.2 fails (bag-of-genes domain AUC ≥ Geneformer): raw expression is *more* domain-shifted than the model, which would be surprising.
- If H1.3 fails: the model preserves subspace structure better than raw expression under shift.

### Analysis plan

1. Pull source + target from Census (same filters as pilot)
2. Extract geneformer, scvi from obsm
3. Extract raw X, log1p-normalize, PCA to 512d and 50d
4. Run preflight.run() on each embedding pair (source, target)
5. Run preflight.probes.run_probes() with all three embeddings
6. Compare module scores and probe results
7. Paired Wilcoxon on fold-level F1 scores (Geneformer vs bag-of-genes)

---

## Experiment 2: Multi-Model Sweep Across Tissues

### Motivation

A single tissue could be an outlier. Testing across 5 tissues reveals whether transportability patterns are tissue-dependent and whether model rankings are stable. This also generates the cross-assay pairs needed for Experiment 0.

### Design

- **Tissues:** Lung, Heart, Liver, Kidney, Brain (same UBERON IDs as Exp 0)
- **Shift type:** Cross-assay (10x 3' v3 → 10x 5' v2) where both assays have ≥500 cells
- **Fallback:** If a tissue lacks both assays, use cross-dataset split (different dataset_ids, same assay)
- **Cells per condition:** 2000
- **Embeddings:** Geneformer, scVI, bag-of-genes (PCA 512d)
- **Metrics:** All active modules + cell-type probe per tissue × model

### Hypotheses

**H2.1:** Model rankings on cell-type probe F1 are consistent across ≥3 of 5 tissues (same model wins).  
*Rationale:* A robust foundation model should consistently outperform alternatives.

**H2.2:** M2 domain shift AUC varies by tissue (≥0.1 spread across tissues for at least one model).  
*Rationale:* Some tissues have more assay-dependent expression programs than others.

**H2.3:** M1 Grassmannian distance and M2 domain AUC are positively correlated across the 15 (tissue × model) conditions (Spearman ρ > 0.3).  
*Rationale:* Subspace divergence and domain separability should co-occur.

**H2.4:** At least one tissue shows Tier ≥ 5 for Geneformer.  
*Rationale:* If the model fails everywhere, it's not transportable at all.

### Decision criteria

- If H2.1 fails: model rankings are tissue-dependent. Report per-tissue rankings.
- If H2.3 fails: M1 and M2 measure independent failure modes (this is informative, not a bug).
- If H2.4 fails: strong negative finding for Geneformer.

### Analysis plan

1. For each tissue: query Census for both assays, subsample to 2000 cells
2. Extract all three embedding types
3. Run full pipeline + probes per (tissue, model) pair
4. Aggregate into 5×3 score matrix
5. Test hypotheses as stated
6. Save to results/sweep/

---

## Experiment 3: Sample Size Sensitivity

### Motivation

The 200-cell pilot is a smoke test. We need to know how many cells produce stable module scores.

### Design

- **Tissue:** Lung, 10x 3' v3 → 10x 5' v2
- **Embedding:** Geneformer only
- **Cell counts:** 200, 500, 1000, 2000, 5000, 10000
- **Replicates:** 3 per cell count (different random seeds)

### Hypotheses

**H3.1:** Module scores stabilize (SD across 3 replicates < 0.05 for all modules) at ≤ 2000 cells.

**H3.2:** Cell-type probe F1 increases monotonically with cell count up to at least 2000 cells.

**H3.3:** M6 curvature score is stable (SD < 0.05) at ≤ 1000 cells despite 500-node subsampling.

### Decision criteria

- If H3.1 fails at 2000: set minimum to smallest n where SD < 0.05.
- If H3.3 fails: increase ORC node cap or change graph construction.

### Analysis plan

1. For each cell count × replicate: pull, run pipeline + probe
2. Compute mean and SD per module per cell count
3. Plot score ± SD vs cell count
4. Identify convergence threshold

---

## Experiment 4: Negative Control with Full Pipeline

### Motivation

Validate calibration: same tissue + same assay should produce high tier (≥5). Confirms the framework doesn't false-alarm.

### Design

- **Source:** Lung, 10x 3' v3, random split A, 2000 cells (seed=42)
- **Target:** Lung, 10x 3' v3, random split B, 2000 cells (seed=99)
- **Embeddings:** Geneformer, scVI, bag-of-genes
- **Metrics:** All active modules + cell-type probe

### Hypotheses

**H4.1:** Overall tier ≥ 5 for all three embeddings.

**H4.2:** M2 domain AUC ∈ [0.45, 0.55] for all embeddings.

**H4.3:** *(Deferred — M7 removed from defaults in v5.)* If M7 is run with explicit records, ecological bias score ≥ 0.5 (low bias). Disease prevalence genuinely varies by dataset, so this is expected to fail on Census data.

**H4.4:** Cell-type probe F1 does not differ between Geneformer and scVI by more than 0.1 (paired Wilcoxon p > 0.05).

### Decision criteria

- H4.1 fails: scoring miscalibration. Investigate.
- H4.2 fails: embedding contains within-assay batch effects. Finding about the model.
- H4.3 fails: acceptable. Disease prevalence variation across datasets is real.

### Analysis plan

1. Pull both splits from Census
2. Run full pipeline on all three embeddings
3. Compare against Experiment 1 cross-assay results

---

## Experiment 5: Gene-Level Regulatory Probes

### Motivation

Cell-level probes test whether the model encodes cell types. Gene-level probes test whether it encodes regulatory structure.

### Design

- **Gene embeddings:** Extract from Geneformer model weights (token embedding layer). NOT on Census.
- **Regulatory edges:** OmniPath REST API (DoRothEA + CollecTRI)
- **Probes:** tf_target, hub_tf, rsa

### Hypotheses

**H5.1:** TF-target mean AUC > 0.6 for Geneformer gene embeddings.

**H5.2:** Hub-TF AUC > 0.6.

**H5.3:** RSA Spearman ρ > 0 (p < 0.05).

### Blockers

- Gene embeddings are NOT on Census. Need to load Geneformer model weights (~2GB from HuggingFace). If blocked, defer this experiment.
- Regulatory edges from OmniPath: no blocker.

### Analysis plan

1. Load Geneformer, extract token embedding matrix
2. Map gene IDs to Ensembl
3. Pull regulatory edges from OmniPath
4. Run tf_target_probe, hub_tf_probe, rsa_probe
5. Report with n_tfs, n_edges, intersection size

---

## Preregistration Procedure

After review and hypothesis finalization:

1. Commit this document + all scorer source code
2. Compute SHA-256 of scorer modules + hyperparameters + dataset specs (per experiment)
3. Save preregistration JSON per experiment
4. **Then** access data and run experiments
5. Report results alongside frozen SHA

**Scorer modules covered by SHA:** m1, m2, m3, m4, m6, m7, runner, probes.

**Hyperparameters:** k=5 (PCA rank, CV folds), ORC alpha=0.5, max_nodes=500 (M6), min_cells_per_class=10 (probes).

---

## Summary Table

| Exp | Name | Purpose | Models | Pairs/Tissues | New code |
|-----|------|---------|--------|---------------|----------|
| **0** | **Composite validation** | **Does tier predict degradation?** | GF | ~25 pairs | degradation measurement script |
| 1 | Bag-of-genes baseline | Non-learned control | 3 | 1 (lung) | bag-of-genes extractor |
| 2 | Multi-model sweep | Cross-tissue stability | 3 | 5 tissues | sweep script |
| 3 | Sample size sensitivity | Convergence threshold | 1 (GF) | 1 (lung) | sensitivity script |
| 4 | Negative control (full) | Calibration check | 3 | 1 (lung) | minor update |
| 5 | Gene-level probes | Regulatory structure | 1 (GF) | N/A | gene embedding extractor |

**Experiment 0 is the keystone.** If H0.1 passes (partial ρ < −0.4 on shifted pairs, controlling for source quality), the composite is a validated transfer predictor. If it fails, the weights need recalibration using the same pairs as training data. Either outcome is informative and publishable.

**Statistical notes:**
- All Experiment 0 correlations use relative degradation (not absolute) to control for floor effects in low-F1 sources.
- Primary analysis excludes negative-control pairs to avoid trivially inflating ρ. Negative controls are tested separately in Experiment 4.
- H0.5 tests composite vs M2 by name (a-priori), not vs best-of-6 post hoc, to avoid multiple-comparisons inflation.

**Total Census queries:** ~80 (across all experiments)  
**Estimated wall time:** 2-4 hours local (no GPU needed, except Exp 5 gene embedding extraction)  
**New code:** bag-of-genes extractor, composite validation script, sweep script, sensitivity script, gene embedding extractor
