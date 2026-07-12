# Preregistration v8: A+ Validation Program

**Status**: DRAFT — ready for final read before SHA freeze.
**Scorer version**: v6 (FROZEN — identical to Exp 0–3 and v7; no scoring logic,
composite weights [2,2,1,1,0.5], or tier thresholds [0.85,0.70,0.55,0.40,0.25,0.10]
are changed by anything in this document).
**Census version**: 2023-12-15 (all Census pulls). External dataset in Exp 12
is NOT Census and is specified below.
**Depends on**: v7 (Exp 4/4b/5/6/7). This document does NOT re-register those.
**Git branch**: main (freeze commit TBD).
**RNG seed**: 20260712 (all stochastic steps: CV splits, bootstrap, MMD subsampling,
random-projection matrices).
**Bootstrap**: 10,000 resamples for every CI and p-value.

---

## 0. Scope and epistemic discipline

This program takes Preflight from B+/A- (2 real models, 1 tissue family, 1 task,
n=10 calibration pairs) to A+ (6 embeddings incl. 2 null controls, 4+ tissues,
4 downstream task types, graduated ground-truth shift, an external non-Census
dataset, and one held-out prospective prediction).

**The v6 scorer is frozen.** Every experiment below only adds harness scripts
that CALL the frozen scorer. No experiment modifies scoring. Any scorer change
is out of scope and deferred to a separate scorer-prereg cycle.

**Confirmatory vs. exploratory.** Hypotheses are tagged [CONFIRMATORY] (hard
pass/fail, Holm-Bonferroni corrected within the confirmatory family) or
[EXPLORATORY] (reported with uncorrected p-values, explicitly labeled, no
significance claimed). Tagging is fixed here, before data access.

**Confirmatory family** (Holm-Bonferroni, family-wise alpha = 0.05):
H8.1, H9.1, H10.1, H11.1, H12.1, H13.1. All others exploratory.

---

## 1. Falsification criteria (kill conditions)

The Preflight thesis — *the composite measures structural transportability, and
this predicts downstream failure independent of representation quality* — is
NOT SUPPORTED if ANY of the following occur in the confirmatory set:

1. **K1 (confound)**: A random-projection OR untrained-encoder embedding
   (Exp 9) reproduces the shifted-vs-control tier ordering of the real
   foundation models with Spearman rho >= 0.7 against the real-model tiers.
   → the composite tracks raw distributional difference, not learned structure.
2. **K2 (no predictive validity)**: Across the pooled multi-task set (Exp 10),
   composite tier does NOT correlate with degradation (pooled Spearman
   95% CI includes 0).
3. **K3 (not monotonic)**: On the graduated-shift ladder (Exp 11), composite
   tier is NOT monotonically non-increasing with true shift magnitude
   (Spearman rho(tier, shift-level) > -0.5 fails).
4. **K4 (composite adds nothing)**: In leave-one-module-out (Exp 13), the full
   composite does NOT beat its best single module on false-certification rate
   (H13.1 fails) — i.e., the six-way decomposition is unjustified.

If any single criterion fires, the paper is reframed (not spun) to report the
negative result honestly.

---

## Experiment 8: Full foundation-model panel [CONFIRMATORY]

### Motivation
The paper cites five scFMs (Geneformer, scGPT, UCE, scFoundation, Nicheformer)
but v6 tested two (Geneformer, scVI); v7 adds scGPT. This experiment completes
the panel so the claim is "validated across scFM families," not two models.

### Design
Embeddings: Geneformer, scVI, scGPT (from v7), **UCE**, **scFoundation**
(+ Nicheformer if spatial-compatible target available, else EXPLORATORY).
Baselines: bag-of-genes PCA-512, PCA-50 (from v6).
Grid: {all embeddings} x {lung, liver, kidney, brain} x {cross-assay shift +
same-assay control}. 2,000 cells/side, donor-stratified. Reuse Census pulls
where cached; new pulls use Census 2023-12-15.

### Hypotheses
- **H8.1** [CONFIRMATORY]: For every real foundation model, the composite assigns
  strictly lower tier to cross-assay shifted pairs than to same-assay controls
  in >= 3 of 4 tissues. Pass = holds for ALL tested real models.
- **H8.2** [EXPLORATORY]: The transportability-vs-probe-F1 dissociation (a model
  ranks high on probe F1 but low on tier, or vice versa) replicates in >= 1
  additional model beyond Geneformer.
- **H8.3** [EXPLORATORY]: UCE (cross-species pretraining) shows higher M1
  subspace alignment across tissues than within-species models.

---

## Experiment 9: Null-embedding confound controls [CONFIRMATORY — LOAD-BEARING]

### Motivation
The single most dangerous reviewer objection: "your tier just tracks how
different the two raw distributions are, not learned structure." This is the
metric-confound question. It must be settled BEFORE investing in the rest.
RUN THIS FIRST.

### Design
Two null embeddings on the SAME lung cross-assay pairs from Exp 0:
- **Random projection**: Gaussian random matrix R in R^(genes x 512), applied
  to log-normalized counts. No learning. Seed 20260712.
- **Untrained encoder**: Randomly initialized 2-layer ReLU MLP
  (genes -> 256 hidden -> 512 output), matching Geneformer's output
  dimensionality. Xavier-normal initialization, no pretraining, forward pass
  only. This tests whether network nonlinearity alone (without learned weights)
  produces meaningful tier separation.
Both scored with the frozen v6 scorer, identical pipeline to the real models.
Same-assay control pairs (10x 3' v3 source vs 10x 3' v3 target, different
donors) also scored, to compute the shifted-vs-control tier gap.

### Hypotheses
- **H9.1** [CONFIRMATORY]: The composite does NOT rank null embeddings' shifted-
  vs-control separation as strongly as real models. Operationally: the mean
  tier gap (control minus shifted) for real foundation models exceeds the null-
  embedding tier gap by >= 1.0 tier, 95% bootstrap CI excluding 0.
  **Failure fires kill criterion K1.**
- **H9.2** [EXPLORATORY]: Null embeddings collapse on M4 (participation ratio)
  and/or M3 (direction stability) specifically, identifying which modules carry
  the learned-structure signal vs. the raw-distribution signal.

---

## Experiment 10: Multi-task predictive validity [CONFIRMATORY]

### Motivation
v6's causal claim rests on n=10 pairs and ONE cell-type probe (rho=-0.709).
A+ requires the tier to predict degradation across task TYPES, not one probe.

### Design
For each (model x tissue x shift) condition, measure degradation on four
downstream tasks trained on source, evaluated on target:
1. Cell-type annotation (macro-F1) — as in v6.
2. Batch-integration quality (scIB kBET / iLISI on the merged pair).
3. Disease-state classification where disease labels exist (else task dropped
   for that condition, logged).
4. Marker-gene / cluster-preservation (ARI of target clustering vs. reference).
"Relative degradation" = (source-metric minus target-metric) / source-metric,
per task, clipped to [0,1].

### Hypotheses
- **H10.1** [CONFIRMATORY]: Pooled across all conditions and tasks, composite
  tier correlates negatively with relative degradation, Spearman rho <= -0.4,
  95% bootstrap CI excluding 0. **Failure fires K2.**
- **H10.2** [EXPLORATORY]: The negative correlation holds within EACH task type
  separately (per-task rho < 0).
- **H10.3** [EXPLORATORY]: M1 (subspace) predicts linear-probe degradation best;
  M6 (curvature) predicts clustering/ARI degradation best (module-task specificity).

---

## Experiment 11: Graduated ground-truth shift ladder [CONFIRMATORY]

### Motivation
Prove the composite measures the AMOUNT of reorganization, not just its
presence — the "graduated control" logic that makes a metric an instrument.

### Design
Construct controlled mixtures: target = assay-B fraction f of cells mixed into
assay-A, for f in {0.0, 0.25, 0.50, 0.75, 1.0}. Lung, Geneformer + one more
real model, 2,000 cells/side, 3 replicates per level (seed 20260712).
True shift magnitude is the known mixing fraction f.

### Hypotheses
- **H11.1** [CONFIRMATORY]: Composite tier is monotonically non-increasing in f;
  Spearman rho(tier, f) <= -0.5 for every tested model.
  **Failure fires K3.**
- **H11.2** [EXPLORATORY]: The composite is approximately linear in f
  (R^2 of linear fit > 0.7), not merely a step function at f>0.

---

## Experiment 12: External non-Census validation [EXPLORATORY->CONFIRMATORY if powered]

### Motivation
Everything is CELLxGENE Census. Show the diagnostic is not Census-specific.

### Design
One independent atlas NOT from Census (pre-declared choice: Human Cell Atlas
lung, or Tabula Sapiens; final pick fixed here: **Tabula Sapiens**, version
recorded at freeze). Construct source/target pairs by cross-donor and cross-
compartment splits. Score with frozen v6 scorer. Requires no scorer change; only
a new data loader (harness code, hashed).

### Hypotheses
- **H12.1** [CONFIRMATORY, contingent on >= 20 pairs]: Composite separates
  shifted from control pairs with AUROC >= 0.75 on Tabula Sapiens.
  If < 20 pairs obtainable, DOWNGRADE to [EXPLORATORY] and report without
  significance claim (pre-committed contingency, not post-hoc).
- **H12.2** [EXPLORATORY]: Tier distributions on Tabula Sapiens are consistent
  with Census (no systematic offset > 1 tier for matched shift types).

---

## Experiment 13: Leave-one-module-out & robustness [CONFIRMATORY]

### Motivation
Justify the six-module complexity: the composite must beat every single module.

### Design
Recompute the composite with each module removed in turn (frozen weights among
remaining modules, renormalized). Compare to full composite and to each module
alone, on the false-certification metric from v7 Exp 4 (Tier>=5 assigned to a
pair with degradation > 0.30), pooled over Exp 8 + Exp 10 conditions.
Also: sensitivity to k (M1 subspace dim in {5,10,20}) and to tier thresholds
(+/- 0.05).

### Hypotheses
- **H13.1** [CONFIRMATORY]: Full composite has a false-certification rate strictly
  lower than its best single module. **Failure fires K4.**
- **H13.2** [EXPLORATORY]: No single module removal degrades composite AUROC
  (Exp 10) by more than 0.10 — i.e., no single point of failure.
- **H13.3** [EXPLORATORY]: Tier assignments are stable (>= 90% unchanged) under
  k in {5,10,20} and threshold perturbation +/- 0.05.

---

## Experiment 14: Held-out prospective prediction [SHOWPIECE — SEPARATE SUB-FREEZE]

### Motivation
The highest-value use of preregistration: forecast, then verify. This is the
one experiment that turns "we scored things" into "we built an instrument."

### Protocol (strict ordering, enforced by two SHAs)
1. After Exp 8–13 pipeline is final, SELECT one (model x tissue x assay-shift)
   combination that has NOT been run at any point in v6/v7/v8.
   Pre-declared choice fixed here: **scFoundation x pancreas x
   (10x 3' v3 -> 10x 5' v2)**.
2. Using only the Exp 0–13 calibration relationship, WRITE a numeric predicted
   tier (integer 1–7) and predicted relative-degradation bin (low/med/high)
   into this file BEFORE pulling pancreas data.
   **Registered prediction: Tier = 3, degradation = HIGH.**
   (Rationale: cross-assay shift on an unseen tissue/model; calibration predicts
   Tier 2–3. Recorded now; a hit requires |predicted - observed tier| <= 1.)
3. SHA-freeze this file (freeze SHA #1).
4. Pull pancreas data, run frozen scorer + downstream tasks.
5. Record observed tier and degradation; commit (freeze SHA #2).

### Hypotheses
- **H14.1** [EXPLORATORY, but prospectively registered]: Observed tier is within
  +/- 1 of the registered prediction (3), AND observed degradation bin matches
  the registered bin (HIGH). Reported as a single prospective hit/miss; no
  correction, no re-selection if it misses.

---

## Execution order (dependency-aware)

1. **Exp 9 (confound controls) FIRST** — gates everything. If K1 fires, stop
   and reframe before spending compute.
2. Exp 8 (model panel) — the bulk of GPU time.
3. Exp 10 (multi-task) — depends on Exp 8 conditions existing.
4. Exp 11 (graduated ladder), Exp 13 (LOMO) — cheap once Exp 8/10 exist.
5. Exp 12 (external) — independent, parallelizable.
6. Exp 14 (prospective) LAST — register prediction only after 8–13 frozen.

---

## Analysis conventions (frozen)

- Confirmatory family {H8.1, H9.1, H10.1, H11.1, H12.1, H13.1}: Holm-Bonferroni,
  family-wise alpha = 0.05.
- Exploratory: uncorrected p-values, labeled "exploratory (uncorrected)" in every
  table; no significance claims.
- All CIs/p-values: 10,000-resample bootstrap, seed 20260712.
- Degradation clipped to [0,1]; AUROC computed with shifted=positive class.
- Donor-stratified CV throughout (no donor spans folds).
- Any pre-declared contingency (Exp 12 downgrade) is honored exactly; deviations
  are logged as deviations, never relabeled.

## Preregistration mechanics

Compute SHA-256 over: (1) frozen v6 scorer source (unchanged — hash should match
v6/v7 records), (2) all Exp 8–14 harness scripts, (3) this document, (4) the
dataset specs (Census 2023-12-15; Tabula Sapiens version string). Store to disk.
Exp 14 uses a two-stage freeze (prediction SHA, then result SHA) as above.
