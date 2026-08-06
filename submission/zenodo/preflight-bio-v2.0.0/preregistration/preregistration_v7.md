# Preregistration v7: Extended Validation Experiments

**Status**: DRAFT v2 — revised per external review (Perplexity, July 2026). Ready for final read before SHA freeze.

**Scorer version**: v6 (same as experiments 0–3; no scorer changes).  
**Census version**: 2023-12-15 (confirmed: all Census pulls use this version).  
**Git branch**: main (commit TBD after freeze)

---

## Overview

Five new experiments extending the v6 validation suite. Priority ordering reflects external review (Perplexity, July 2026): defend the core claim first, broaden second.

**Priority 1 — Defend the core claim (load-bearing):**
1. **Exp 4**: Comparison to existing transferability metrics (RankMe, MMD, C2ST) — primary hypothesis: false-certification rate (absolute tier cutoff)
2. **Exp 4b**: Degenerate embedding characterization — Part A is confirmatory re-analysis of existing v6 data; Part B is a counterfactual simulation of a proposed gate (scorer unchanged)

**Priority 2 — Broaden the panel (additive):**
3. **Exp 5**: Cross-tissue shift (same assay, different tissue) — confirmatory
4. **Exp 6**: Cross-disease shift (same tissue + assay, healthy vs. diseased) — **EXPLORATORY** (underpowered at n=4)
5. **Exp 7**: scGPT embeddings (third foundation model) — confirmatory

All experiments use the frozen v6 scorer. No modifications to scoring logic, composite weights, or tier thresholds. The only new code is experiment harness scripts that call the existing scorer. Any proposed scorer changes (e.g., the M4 worst-module gate in Exp 4b Part B) are evaluated counterfactually only — actual implementation deferred to a separate v7 scorer preregistration cycle.

---

## Experiment 4: Comparison to Existing Transferability Metrics

### Motivation

The ML literature has established transferability estimation metrics (LEEP, LogME, H-Score, NCE) that require target labels, and unsupervised embedding quality metrics (RankMe, MMD, C2ST) that do not. Preflight's composite should be compared to these unsupervised alternatives on the same calibration pairs from Exp 0.

### Design

Run three competing metrics on the same 10 pairs from Exp 0:

- **RankMe** (Garrido et al., ICML 2023): Effective rank of the embedding matrix via smoothed entropy of singular values. Computed separately on source and target; report the absolute difference as a shift metric.
- **MMD** (Maximum Mean Discrepancy): Gaussian kernel MMD between source and target embeddings. Standard two-sample test for distribution shift.
- **C2ST** (Classifier Two-Sample Test): Train a classifier to distinguish source from target; report accuracy. Equivalent to our M2 module but as a standalone metric.

Each metric produces a single scalar per pair. We compute Spearman correlation with relative degradation (same as Exp 0) and tier-separation ability (AUROC for shifted vs. control classification).

### Hypotheses

The central claim is not "we rank better" but "we abstain where they falsely certify." The composite's value is knowing when not to trust a ranking.

**Primary hypothesis:**

- **H4.1**: The Preflight composite has a lower false-certification rate than any competing metric. "False certification" is defined as: assigning Tier ≥ 5 to a pair whose relative degradation exceeds 0.30. (Absolute tier cutoff; not relative to sample median.)

**Secondary hypotheses:**

- **H4.2**: Preflight composite achieves higher |Spearman rho| with relative degradation than RankMe, MMD, or C2ST individually.
- **H4.3**: Preflight composite achieves higher AUROC for shifted-vs-control classification than any single competing metric.
- **H4.4**: C2ST performance is similar to Preflight's M2 module alone (since they measure the same thing), but the full composite outperforms C2ST by incorporating subspace geometry (M1) and dimensionality (M4).
- **H4.5**: On pairs where the BoG-PCA embedding scores Tier ≥ 5 (degenerate M4), competing metrics (RankMe, MMD) also produce passing scores — i.e., they falsely certify a degenerate embedding as transferable. Preflight's M4 module flags the degeneracy that single-metric approaches miss.

### Implementation

- Use the same 10 pairs from `results/modal_results/composite_validation_v6/incremental.jsonl`
- Re-use saved embeddings (no Census re-pull needed)
- RankMe: `np.exp(entropy(sv / sv.sum()))` where `sv` = singular values
- MMD: Gaussian kernel with median heuristic bandwidth
- C2ST: Logistic regression with 5-fold CV (identical to M2)
- All computation is CPU-only, runs locally in minutes

### Output

`results/metric_comparison/comparison_v7.json`:
```json
{
  "pairs": [...],
  "metrics": {
    "preflight_composite": {"rho": ..., "auroc": ...},
    "rankme_diff": {"rho": ..., "auroc": ...},
    "mmd": {"rho": ..., "auroc": ...},
    "c2st": {"rho": ..., "auroc": ...}
  }
}
```

---

## Experiment 4b: Degenerate Embedding Characterization

### Motivation

The Exp 1 and Exp 2 results show that bag-of-genes PCA-512 receives Tier 5–6 overall despite having the worst probe F1. This is the most dangerous failure mode for a naive user: a degenerate embedding receives a passing grade. This experiment has two distinct parts:

**Part A** is a confirmatory re-analysis of already-observed data (H4b.1 and H4b.3). These are not true preregistered hypotheses — the data were collected under v6 experiments — but formalizing the check here documents the reasoning and makes the claim testable against future datasets.

**Part B** is a prospective hypothesis (H4b.2) that tests a proposed improvement: what would happen to the tier assignments if we applied a worst-module gate. This gate is NOT part of the frozen v6 scorer. It is a proposed v7 modification that we evaluate counterfactually on existing data, with any actual scorer change deferred until after this preregistration is complete.

### Design

Using the already-computed Exp 1 and Exp 2 results:

**Part A (confirmatory re-analysis, data already observed):**
1. Tabulate BoG-512's M4 participation ratio across all conditions.
2. Tabulate M4 for Geneformer and scVI across all conditions.

**Part B (prospective counterfactual, no scorer modification):**
3. Simulate: if a "worst-module gate" were applied (any module ≤ Tier 1 caps overall tier to ≤ Tier 3), what would BoG-512's reclassified tier be?
4. Report whether such a gate would change tiers for any non-degenerate embedding.

### Hypotheses

**Part A — Confirmatory re-analysis (data already collected under Exp 1/2):**

- **H4b.1**: BoG-512 has M4 participation ratio < 0.01 in all tissues tested (Exp 1 + Exp 2 data). *(Already observed in Exp 1; confirmed here across Exp 2 tissues.)*
- **H4b.3**: No non-degenerate embedding (Geneformer, scVI) has M4 < 0.05 in any condition tested. *(Confirms the M4 threshold separates degenerate from non-degenerate embeddings.)*

**Part B — Prospective counterfactual (scorer unchanged, gate simulated):**

- **H4b.2**: Under a worst-module gate (any module ≤ Tier 1 caps overall to ≤ Tier 3), BoG-512 would be reclassified from Tier 5–6 to ≤ Tier 3 in all conditions, while no non-degenerate embedding is reclassified.

### Note on scorer modifications

The v6 scorer is frozen for all experiments in this preregistration. If H4b.2 is confirmed, the worst-module gate will be proposed as a v7 scorer change in a separate document, with its own preregistration cycle. No rule changes are made here.

### Implementation

Pure analysis of existing results — no new data pull. Check M4 values from:
- `results/modal_results/bag_of_genes_v6/bag_of_genes_baseline/summary_20260711_122910.json`
- `results/modal_results/sweep_v6/sweep/summary_20260711_221653.json`

### Output

`results/degeneracy_check/degeneracy_v7.json` with per-condition M4 values and gate-simulation results.

---

## Experiment 5: Cross-Tissue Shift

### Motivation

All experiments 0–3 use cross-assay shifts within a single tissue. Cross-tissue transfer (deploying a model trained on one tissue to another tissue) is a distinct and arguably more important failure mode: the dominant variance directions differ between tissues (immune subtypes in blood vs. epithelial-stromal gradients in solid tissue).

### Design

Six cross-tissue pairs, all using Geneformer embeddings and the same assay (10x 3' v3, EFO:0009922):

| Source | Target | Biological rationale |
|--------|--------|---------------------|
| Lung | Brain | Epithelial vs. neural |
| Lung | Kidney | Epithelial vs. tubular |
| Blood | Lung | Immune-dominated vs. parenchymal |
| Blood | Brain | Immune vs. neural |
| Liver | Kidney | Hepatic vs. renal (both metabolic) |
| Blood | Liver | Immune vs. metabolic |

Plus 3 negative controls: same tissue, same assay, random split (lung, blood, brain).

All pairs: 2,000 cells per side, Geneformer embeddings.

Blood uses `tissue_general == "blood"` (UBERON:0000178).

### Hypotheses

- **H5.1**: Cross-tissue pairs score Tier ≤ 3 on average (strong shift detected).
- **H5.2**: Negative controls score Tier ≥ 5 (no false alarms).
- **H5.3**: M1 geodesic distance is larger for cross-tissue pairs than for cross-assay pairs from Exp 0 (subspace reorganization is more severe across tissues than across assays).
- **H5.4**: Pairs involving blood (immune-dominated) vs. solid tissue show the largest M1 distances (reflecting the immune/parenchymal axis difference).
- **H5.5**: The composite still correlates with probe F1 degradation across pairs (Spearman rho < -0.3).

### Output

`results/cross_tissue/summary_v7.json` with per-pair tiers, scores, module breakdowns, and probe F1.

---

## Experiment 6: Cross-Disease Shift (EXPLORATORY)

**Status**: Exploratory. With only 4 disease pairs (likely fewer after exclusions), this experiment is underpowered for confirmatory hypothesis testing. Results will be reported descriptively to guide future work, not as confirmatory evidence.

### Motivation

The pharmaceutical use case: a model trained on healthy tissue is deployed to characterize disease states. The embedding space may reorganize under disease because the cell-type composition changes (e.g., immune infiltration in tumors) and individual cell states shift.

### Design

Four tissue × disease pairs, all using Geneformer + 10x 3' v3:

| Tissue | Source (healthy) | Target (disease) | Disease |
|--------|-----------------|-----------------|---------|
| Lung | `disease == "normal"` | `disease == "pulmonary fibrosis"` | IPF |
| Lung | `disease == "normal"` | `disease == "lung adenocarcinoma"` | Cancer |
| Brain | `disease == "normal"` | `disease == "Alzheimer disease"` | AD |
| Liver | `disease == "normal"` | `disease == "hepatocellular carcinoma"` | HCC |

Plus 2 negative controls: same tissue, same disease status (both normal), random split.

All pairs: 2,000 cells per side (or maximum available if fewer than 2,000 disease cells exist).

### Hypotheses

**Confirmatory (if ≥ 3 pairs survive exclusions):**

- **H6.1**: Cross-disease pairs score Tier ≤ 4 (shift detected, but potentially milder than cross-assay since the biology is related).
- **H6.5**: Negative controls score Tier ≥ 5.

**Exploratory observations (descriptive only, n too small for confirmatory testing):**

- **H6.2** (observation): Cancer pairs (lung adenocarcinoma, HCC) show larger shifts than non-cancer disease pairs (IPF, AD). Reported as directional observation; n=2 per group precludes statistical comparison.
- **H6.3** (observation): M2 domain AUC is lower for cross-disease than for cross-assay shifts (disease shifts are subtler than technical batch effects).
- **H6.4** (observation): M3 direction stability is lower for disease pairs than negative controls (the discriminant directions change under disease).

### Contingency

If Census has fewer than 500 cells for a disease condition, that pair is dropped and reported as excluded. If fewer than 3 disease pairs remain after exclusions, the experiment is underpowered and marked as such (no hypothesis testing on < 3 pairs).

### Output

`results/cross_disease/summary_v7.json` with per-pair tiers, scores, module breakdowns, cell counts, and disease metadata.

---

## Experiment 7: scGPT Embeddings

### Motivation

Experiments 0–3 use Geneformer and scVI. Adding scGPT (the other major hosted embedding in Census) tests whether the scorer's findings generalize across foundation models. scGPT uses a different architecture (transformer with gene-token attention) and different pretraining data composition.

### Design

Re-run the Exp 2 multi-model sweep with scGPT added as a fourth embedding. Four tissues (lung, liver, kidney, brain) × 4 embeddings (Geneformer, scVI, bag-of-genes PCA-512, scGPT). All cross-assay shifts (10x 3' v3 → 10x 5' v2).

### Hypotheses

- **H7.1**: scGPT achieves intermediate tiers between Geneformer and BoG-PCA (Tier 3–4), because it carries batch signal but potentially less than Geneformer.
- **H7.2**: scGPT probe F1 is within 0.10 of Geneformer (both are foundation models optimized for cell-type information).
- **H7.3**: The M1-M2 correlation across all 16 conditions (4 tissues × 4 embeddings) remains strong (rho > 0.5).
- **H7.4**: The tier ordering Geneformer < scGPT < scVI < BoG is consistent across ≥ 3 of 4 tissues.

### Contingency

If Census does not host scGPT embeddings for a tissue (check `obsm` keys after pull), that tissue is excluded for scGPT. If fewer than 2 tissues have scGPT available, the experiment is dropped entirely and reported as "embeddings unavailable."

### Output

`results/sweep_v7/summary_v7.json` with per-condition tiers, scores, and probe F1. Same format as Exp 2 with an additional embedding column.

---

## Preregistration Procedure

1. This document is finalized after external review.
2. `scripts/freeze_preregistration_v7.py` computes SHA-256 over:
   - v6 scorer source code (same as experiments 0–3)
   - This document (experiment specifications + hypotheses)
   - All new experiment scripts (exp4–7)
   - Census version string
3. Hashes are written to `docs/frozen_prereg_v7/`.
4. Git commit with hash in commit message.
5. Only then are experiments executed.

---

## Analysis Plan

- All hypothesis tests use the same thresholds defined above.
- No post-hoc hypothesis modifications after data access.
- **Confirmatory hypotheses** (Exp 4 H4.1 primary, H4.2–H4.5 secondary; Exp 5 H5.1–H5.5; Exp 6 H6.1/H6.5; Exp 7 H7.1–H7.4): tested at stated thresholds, reported as pass/fail. Exp 4 H4.1 is the pre-registered primary endpoint; H4.2–H4.5 are secondary and do not gate the overall claim.
- **Confirmatory re-analysis** (Exp 4b Part A H4b.1/H4b.3, Exp 4b Part B H4b.2): data already observed under Exp 1/2; formalized here for completeness but acknowledged as non-blind. H4b.2 is a counterfactual simulation on known M4 values — the outcome is deterministic given the gate rule, so it tests the rule's design adequacy rather than a blind prediction. If Part A fails, it contradicts previously reported results and triggers a data integrity check.
- **Exploratory observations** (Exp 6 H6.2–H6.4): reported descriptively with effect sizes but no confirmatory pass/fail verdict. n too small for statistical comparison.
- Exploratory analyses beyond stated hypotheses (correlations, module breakdowns) are labeled as exploratory.
- If a hypothesis is rejected, we report why and what it reveals.
- Results from Exp 4 (metric comparison) will be incorporated into the paper's Related Work section as a quantitative comparison table.
