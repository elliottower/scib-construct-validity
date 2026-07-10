# Preflight Bio — Experiment Preregistration

**Status:** DRAFT — pending review before SHA freeze  
**Date:** 2026-07-10  
**Author:** Elliot Tower  
**Reviewer:** (Perplexity / external)

---

## Overview

Five experiments evaluating the Preflight Bio transportability diagnostics platform on real CELLxGENE Census data. Each experiment has explicit hypotheses, analysis plans, and decision criteria registered BEFORE any data access.

**Available embeddings on Census (v2023-12-15):**
- `geneformer` — 512 dimensions (transformer, pretrained on ~30M cells)
- `scvi` — 50 dimensions (VAE, dataset-specific)
- Raw expression `X` — 60,664 genes (bag-of-genes baseline, log1p-normalized)

**Available metadata columns:** cell_type, tissue, assay, disease, dataset_id, donor_id, is_primary_data

---

## Experiment 1: Bag-of-Genes Baseline

### Motivation

Without a non-learned baseline, we cannot distinguish "the model is good" from "any embedding works." Bag-of-genes (log1p-normalized raw expression, PCA-reduced to match embedding dimensionality) provides this control.

### Design

- **Source:** Lung, 10x 3' v3 (EFO:0009922), 2000 cells
- **Target:** Lung, 10x 5' v2 (EFO:0011025), 2000 cells
- **Embeddings compared:**
  - Geneformer (512d, from Census obsm)
  - scVI (50d, from Census obsm)
  - Bag-of-genes (raw X, log1p, PCA to 512d and separately to 50d)
- **Metrics:** All 7 modules + cell-type probe (donor-stratified, k=5)

### Hypotheses

**H1.1:** Geneformer cell-type probe macro F1 > bag-of-genes F1.  
*Rationale:* A pretrained model should encode cell-type-discriminative features better than raw expression projected to the same dimensionality.

**H1.2:** Bag-of-genes M2 domain shift AUC < Geneformer M2 AUC.  
*Rationale:* Learned embeddings may absorb assay-specific batch effects that raw expression does not. If Geneformer's domain AUC is *higher* than bag-of-genes, the model has learned domain-specific (non-transportable) features.

**H1.3:** Bag-of-genes M1 Grassmannian distance ≤ Geneformer M1 distance.  
*Rationale:* If the learned model distorts the subspace under assay shift more than raw expression does, the model is adding domain-specific structure rather than generalizable structure.

### Decision criteria

- If H1.1 fails (bag-of-genes ≥ Geneformer on F1): the model adds no value over raw expression for this tissue/assay pair. Report as a negative finding.
- If H1.2 fails (bag-of-genes domain AUC ≥ Geneformer): raw expression is *more* domain-shifted than the model, which would be surprising and worth investigating.
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

A single tissue (lung) could be an outlier. Testing across 5 tissues reveals whether the transportability pattern is tissue-dependent and whether model rankings are stable.

### Design

- **Tissues:** Lung (UBERON:0002048), Heart (UBERON:0000948), Liver (UBERON:0002107), Kidney (UBERON:0002113), Brain (UBERON:0000955)
- **Shift type:** Cross-assay (10x 3' v3 → 10x 5' v2) where both assays have ≥500 cells
- **Fallback:** If a tissue lacks both assays, use cross-dataset split (different dataset_ids, same assay)
- **Cells per condition:** 2000
- **Embeddings:** Geneformer, scVI, bag-of-genes (PCA 512d)
- **Metrics:** All 7 modules + cell-type probe per tissue × model

### Hypotheses

**H2.1:** Model rankings on cell-type probe F1 are consistent across ≥3 of 5 tissues (same model wins).  
*Rationale:* A robust foundation model should consistently outperform alternatives.

**H2.2:** M2 domain shift AUC varies by tissue (≥0.1 spread across tissues for at least one model).  
*Rationale:* Some tissues have more assay-dependent expression programs than others.

**H2.3:** M1 Grassmannian distance and M2 domain AUC are positively correlated across the 15 (tissue × model) conditions (Spearman rho > 0.3).  
*Rationale:* Subspace divergence and domain separability should co-occur.

**H2.4:** At least one tissue shows Tier ≥ 5 for Geneformer (the model works somewhere).  
*Rationale:* If the model fails everywhere, it's not transportable at all.

### Decision criteria

- If H2.1 fails: model rankings are tissue-dependent, and "which scFM is best" has no single answer. Report per-tissue rankings.
- If H2.3 fails: M1 and M2 measure different failure modes. This is informative (independent diagnostics are more valuable than redundant ones).
- If H2.4 fails: Geneformer embeddings do not transport across any assay shift. This would be a strong negative finding for the field.

### Analysis plan

1. For each tissue: query Census for both assays, subsample to 2000 cells
2. Extract all three embedding types
3. Run full pipeline + probes per (tissue, model) pair
4. Aggregate results into a 5×3 score matrix
5. Test H2.1 by checking pairwise model rankings per tissue
6. Test H2.3 by computing Spearman correlation on the 15-row (M1, M2) pairs
7. Save all results to results/sweep/ with timestamps

---

## Experiment 3: Sample Size Sensitivity

### Motivation

The 200-cell pilot is a smoke test. We need to know how many cells produce stable module scores so we don't over- or under-sample in future experiments.

### Design

- **Tissue:** Lung (same filters as pilot)
- **Shift:** 10x 3' v3 → 10x 5' v2
- **Embedding:** Geneformer only
- **Cell counts:** 200, 500, 1000, 2000, 5000, 10000
- **Replicates:** 3 per cell count (different random seeds)

### Hypotheses

**H3.1:** Module scores stabilize (SD across 3 replicates < 0.05 for all modules) at ≤ 2000 cells.  
*Rationale:* Geometric properties of the embedding space should converge with moderate sample sizes.

**H3.2:** Cell-type probe F1 increases monotonically with cell count up to at least 2000 cells.  
*Rationale:* More training data improves linear probe performance.

**H3.3:** M6 curvature score is stable (SD < 0.05) at ≤ 1000 cells despite subsampling to 500 nodes for ORC.  
*Rationale:* The 500-node cap should not introduce excessive variability if the graph structure is consistent.

### Decision criteria

- If H3.1 fails at 2000: we need more cells for stable results. Set minimum to the smallest n where SD < 0.05.
- If H3.3 fails: increase the ORC node cap or use a different graph construction.

### Analysis plan

1. For each cell count × replicate: pull from Census, run pipeline + probe
2. Compute mean and SD of each module score across 3 replicates per cell count
3. Plot score ± SD vs cell count for each module
4. Identify convergence threshold

---

## Experiment 4: Negative Control with Full Pipeline

### Motivation

Validate calibration: same tissue + same assay should produce high tier (≥5). The earlier negative control ran without M6/M7. This one runs the full pipeline.

### Design

- **Source:** Lung, 10x 3' v3, random split A, 2000 cells (seed=42)
- **Target:** Lung, 10x 3' v3, random split B, 2000 cells (seed=99)
- **Embeddings:** Geneformer, scVI, bag-of-genes
- **Metrics:** All 7 modules + cell-type probe

### Hypotheses

**H4.1:** Overall tier ≥ 5 for all three embeddings.  
*Rationale:* No distribution shift exists, so the framework should score high.

**H4.2:** M2 domain AUC ∈ [0.45, 0.55] for all embeddings.  
*Rationale:* Source and target are from the same distribution; a domain classifier should perform at chance.

**H4.3:** M7 ecological bias score ≥ 0.5 (low bias).  
*Rationale:* With same tissue and assay, disease prevalence should not vary systematically across dataset_ids. (This hypothesis is weaker — disease prevalence genuinely varies by dataset even within same tissue/assay, so M7 might still flag bias.)

**H4.4:** Cell-type probe F1 does not differ between Geneformer and scVI by more than 0.1 (paired Wilcoxon p > 0.05).  
*Rationale:* On identical distributions, both models should perform comparably on cell-type classification.

### Decision criteria

- If H4.1 fails: scoring is miscalibrated. Investigate which module(s) false-alarm and adjust weights or thresholds.
- If H4.2 fails: the embedding contains batch effects even within the same assay. This is a finding about the model, not a framework bug.
- If H4.3 fails: acceptable — disease prevalence variation across datasets is real, not a calibration error. Note it and move on.

### Analysis plan

1. Pull both splits from Census (same filter, different seeds)
2. Run full pipeline on all three embeddings
3. Check tier scores and domain AUC
4. Compare against Experiment 1 (cross-assay) results — the difference is the assay-shift signal

---

## Experiment 5: Gene-Level Regulatory Probes

### Motivation

Cell-level probes test whether the model encodes cell types. Gene-level probes test whether the model encodes regulatory structure — a deeper form of biological knowledge.

### Design

- **Gene embeddings:** Extract from Geneformer model weights (token embedding layer). NOT from Census — Census hosts cell embeddings, not gene embeddings.
- **Regulatory edges:** Pull from OmniPath REST API (DoRothEA + CollecTRI)
- **Probes:** tf_target, hub_tf, rsa

### Hypotheses

**H5.1:** TF-target mean AUC > 0.6 for Geneformer gene embeddings.  
*Rationale:* If the model's token embeddings encode gene function, targets of the same TF should cluster.

**H5.2:** Hub-TF AUC > 0.6.  
*Rationale:* Hub TFs (high out-degree) should occupy a geometrically distinct region in the gene embedding space.

**H5.3:** RSA Spearman rho > 0 (p < 0.05).  
*Rationale:* Gene embedding similarity should correlate with regulatory adjacency.

### Blockers and contingencies

- **Gene embeddings are NOT on Census.** We need to load Geneformer model weights and extract the token embedding matrix. This requires either:
  - (a) `pip install geneformer` + download model weights (~2GB), or
  - (b) Download from HuggingFace directly
- If gene embedding extraction is blocked, this experiment is deferred.
- **Regulatory edges:** OmniPath REST API is public and free. ~50K edges from DoRothEA + CollecTRI. No blocker here.

### Analysis plan

1. Load Geneformer model, extract token embedding matrix (n_genes × 512)
2. Map gene IDs to Ensembl IDs
3. Pull regulatory edges from OmniPath
4. Intersect gene IDs between embeddings and regulatory network
5. Run tf_target_probe, hub_tf_probe, rsa_probe
6. Report results with n_tfs, n_edges, intersection size

---

## Preregistration Procedure

After Perplexity review and hypothesis finalization:

1. Commit this document + all scorer source code
2. Compute SHA-256 of scorer modules + hyperparameters + dataset specs
3. Save preregistration JSON per experiment
4. **Then** access data and run experiments
5. Report results alongside frozen SHA

**Scorer modules covered by SHA:** m1 (Grassmannian), m2 (domain shift), m3 (direction stability), m4 (domain validity), m6 (curvature), m7 (ecological bias), runner, probes.

**Hyperparameters:** k=5 (PCA subspace rank, CV folds), ORC alpha=0.5, max_nodes=500 (M6), min_cells_per_class=10 (probes).

---

## Summary Table

| Exp | Name | Models | Tissues | Cells | New code needed |
|-----|------|--------|---------|-------|-----------------|
| 1 | Bag-of-genes baseline | 3 | 1 (lung) | 2000 | bag-of-genes extractor |
| 2 | Multi-model sweep | 3 | 5 | 2000 each | sweep script |
| 3 | Sample size sensitivity | 1 (GF) | 1 (lung) | 200–10K | sensitivity script |
| 4 | Negative control (full) | 3 | 1 (lung) | 2000 | minor update to existing |
| 5 | Gene-level probes | 1 (GF) | N/A | N/A | gene embedding extractor |

**Total Census queries:** ~50 (across all experiments)  
**Estimated wall time:** 1-2 hours local (no GPU needed)  
**New code:** bag-of-genes extractor, sweep script, sensitivity script, gene embedding extractor
