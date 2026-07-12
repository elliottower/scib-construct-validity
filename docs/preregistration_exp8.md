# Preregistration: Experiment 8 — Expanded Model Panel (UCE + multi-task)

**Status**: DRAFT — final read before SHA freeze.
**Scorer version**: v7 (FROZEN; unchanged from Experiments 4-7. Weights [M1=2,M2=2,M3=1,M4=1,M6=0.5];
tier thresholds [0.85,0.70,0.55,0.40,0.25,0.10]).
**Experiment script**: scripts/exp8_uce_expanded.py (hashed at freeze).
**Census version**: 2023-12-15.
**Git commit**: TBD after freeze.

---

## Frozen configuration (must match code exactly)

- **Organism**: homo_sapiens.
- **Embeddings**: geneformer (512d), scvi (50d), scgpt (512d), uce (1280d)
  (+ bag-of-genes PCA-512 baseline, computed locally).
- **Tissues**: lung (UBERON:0002048), liver (UBERON:0002107),
  kidney (UBERON:0002113), brain (UBERON:0000955).
- **Shift**: cross-assay. Source assay EFO:0009922 (10x 3' v3),
  target assay EFO:0011025 (10x 5' v2).
- **Max cells per side**: 2000.
- **Downstream tasks** (three):
  1. Logistic-regression transfer, macro-F1,
     LogisticRegression(max_iter=1000, random_state=42).
  2. kNN transfer, KNeighborsClassifier(n_neighbors=min(5, n_source-1),
     metric="cosine"), macro-F1. k drops below 5 only when a split has
     fewer than 6 source cells; any such reduction is logged per condition.
  3. Silhouette score (unsupervised diagnostic, not a "downstream task"
     in the hypothesis-testing sense), sample_size=min(1000, n),
     random_state=42.
- **kNN graph (M6 curvature input)**: k=10, max_nodes=500, seed=0.
  This is DISTINCT from the kNN probe (k=5).
- **Cell subsampling seeds**: source seed=42, target seed=43.
- **Degradation**: relative = 1 - (target_metric / max(source_metric, 1e-8)),
  computed per task.

## Model-availability policy (RESOLVES THE SILENT-FALLBACK RISK)

The script's original embedding-pull fallback to [geneformer, scvi] on
exception has been **replaced with a strict abort** for this preregistration.

- [x] **STRICT**: a failed pull of any of {geneformer, scvi, scgpt, uce}
  ABORTS the run with RuntimeError. No silent 2-model result is possible.
  The 4-model claim is guaranteed or the run stops.
- [ ] ~~CONTINGENCY~~: not selected.

A run in which any tissue silently reports fewer than the four registered
embeddings is a PROTOCOL VIOLATION and is discarded.

---

## Confirmatory vs. exploratory

Confirmatory (Holm-Bonferroni, family alpha=0.05): H8.1, H8.2.
Exploratory (uncorrected, labeled "exploratory"): H8.3, H8.4.

## Falsification criterion

If H8.1 fails (no tissue-consistent shifted<control separation for the real
models under the strict availability policy), the "validated across scFM
families" claim is NOT supported and the paper reports the panel result as-is
without that framing.

---

## Hypotheses

### H8.1 [CONFIRMATORY]

For every real foundation model that passes the strict availability check
(geneformer, scvi, scgpt, uce), the composite assigns strictly lower tier
to cross-assay shifted pairs than to same-assay controls in >= 3 of 4
tissues. Pass = holds for ALL four real models.

### H8.2 [CONFIRMATORY]

Composite tier correlates negatively with relative degradation pooled
across the >= 16 (model x tissue) conditions and BOTH probe tasks
(logistic regression and kNN); Spearman rho <= -0.4, 95% bootstrap CI
(10,000 resamples) excluding 0.

### H8.3 [EXPLORATORY]

The transportability-vs-probe-F1 dissociation (high probe F1 but low
tier, or vice versa) appears in >= 1 real model beyond Geneformer.

### H8.4 [EXPLORATORY]

The negative tier-vs-degradation correlation holds separately for BOTH
probe tasks (logreg and kNN), not just one.

## Analysis conventions

- 10,000-resample bootstrap for all CIs/p-values.
- Confirmatory family Holm-Bonferroni corrected; exploratory uncorrected and
  explicitly labeled "exploratory (uncorrected)".
- Donor-stratified splits where donor metadata available.
- Heart excluded a priori (Census returns 0 target cells; documented in v6/v7).
- Incremental JSONL saves; final results in results/exp8_uce_expanded/.
- Silhouette score is an unsupervised diagnostic complement, not a
  "downstream task" for hypothesis testing. The paper refers to "two
  independent downstream tasks" (logistic regression and kNN transfer);
  silhouette is reported separately.

## Preregistration mechanics

SHA-256 over: (1) frozen v7 scorer source (hash must match v7 records at
docs/frozen_prereg_v7/FROZEN_SUMMARY.json), (2) scripts/exp8_uce_expanded.py
AND scripts/modal_experiments_v8.py, (3) this document, (4) config constants
above. Store to disk; verify four-way match after the run (record hash,
report hash, disk hash, fresh recompute).
