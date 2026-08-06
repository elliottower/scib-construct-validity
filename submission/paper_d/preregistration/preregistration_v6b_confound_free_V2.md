# Preregistration v6b: Post-Confound Revision

**Status**: DRAFT — freeze before running validation.
**Parent**: v6 (FAILED confound gate K1 on 2026-07-12; see Exp 9 results).
**Motivation**: v6 composite uses modules M1 (Grassmannian, weight=2) and M6
(curvature, weight=0.5) that are anti-discriminative against null embeddings.
This revision removes them based on a JL-preservation diagnosis, then validates
on held-out tissues that did not inform the module selection.

---

## 1. What happened to v6

Experiment 9 (preregistered in v8) tested the frozen v6 composite against two
null baselines (random Gaussian projection, untrained 2-layer MLP) on lung
cross-assay pairs. Result:

- Random projection achieved Tier 3 shifted / Tier 2 control (gap = -1)
- Geneformer achieved Tier 3 shifted / Tier 2 control (gap = -1)
- **The null baseline reproduced the real model's tier pattern exactly.**

Kill criterion K1 fired: the v6 composite tracks raw distributional geometry,
not learned structure. This is the honest, preregistered headline.

## 2. Diagnosis: Johnson-Lindenstrauss preservation

Two modules are anti-discriminative (null scores HIGHER than real):

- **M1 (Grassmannian subspace alignment)**: A random linear map preserves
  covariance/subspace structure. If source and target share gene-program
  covariance (which lung 10x cells do across assays), any linear map shows
  high subspace alignment. M1 cannot distinguish "alignment because the model
  learned transferable biology" from "alignment because linear maps preserve
  covariance."
- **M6 (Ollivier-Ricci curvature)**: JL guarantees pairwise distance
  preservation to (1 +/- epsilon) in 512d. The kNN graph is therefore
  approximately invariant under random projection. Curvature is a function
  of graph structure, so it is preserved.

Two modules DO discriminate (real scores higher than null by >= 1.5 tiers):

- **M3 (direction stability)**: Learned embeddings have stable, biologically
  meaningful principal directions across domains. Random projections have
  random directions that are unstable across independent samples.
  Real mean tier = 4.0, Null mean tier = 2.5 (gap = +1.5).
- **M4 (participation ratio)**: Learned embeddings compress to task-relevant
  dimensions (structured spectrum). Random projections have ~uniform spectra.
  Real mean tier = 6.5, Null mean tier = 5.0 (gap = +1.5).

One module weakly discriminates:

- **M2 (domain shift / classifier AUC)**: Real mean = 2.0, Null mean = 1.5
  (gap = +0.5). Correct direction but small gap; the domain classifier can
  slightly better distinguish domains in learned space, but this module is
  near-floor for both.

## 3. v6b composite definition

**Modules**: M2 (domain shift), M3 (direction stability).
**Weights**: M2 = 1.0, M3 = 2.0.
**Tier thresholds**: Unchanged from v6 [0.85, 0.70, 0.55, 0.40, 0.25, 0.10].

Rationale for weights: M3 carries the strongest conceptual signal (stable
learned directions vs unstable random directions) and showed the clearest
continuous-score separation on shifted pairs in Exp 9 (real mean tier = 4.0,
null mean tier = 2.5, gap = +1.5). M2 is retained despite weak discrimination
(real = 2.0, null = 1.5, gap = +0.5) because domain separability is
conceptually distinct from direction stability and provides a
classification-based signal complementary to M3's spectral signal.

**Dropped (confirmed confounded)**:
- M1 (Grassmannian): anti-discriminative (null > real). Confounded by JL
  preservation of covariance structure under random linear maps.
- M6 (curvature): anti-discriminative (null > real). Confounded by JL
  preservation of pairwise distances preserving kNN graph structure.

**Held out (pending diagnostic — NOT in confirmatory composite)**:
- M4 (participation ratio): Discriminates real from null on shifted pairs
  (Exp 9: Geneformer 0.845 vs random projection 0.581, gap +0.26). However,
  M4 produces hard zeros (score = 0.0) on 4 of 20 model×tissue cells in
  Exp 8, and the floor-trigger pattern is unexplained:
  - BoG at 512d hits the PR<0.01 floor in lung (n_shared=48)
  - UCE at 1280d passes the floor in lung but hits it in liver/kidney/brain
  - This inversion contradicts the "high-d normalization" explanation
  - The "cell-type diversity" hypothesis was falsified by regression
    (Spearman rho = -0.08, p = 0.74; kidney at n_shared=4 has highest
    non-zero tissue mean)
  - The raw PR values (pr_source, pr_target) for the floor-triggering
    cases have never been measured — only floored scores were observed.
  A raw-PR diagnostic (scripts/exp9b_m4_raw_pr_diagnostic.py) will resolve
  the mechanism. If the floor-trigger is characterizable and the limitation
  accurately statable, M4 may be added via a separate mini-prereg with its
  own held-out validation. Until then, it is excluded.

## 4. Validation plan (confirmatory — must pass before any v6b claim)

### 4a. Held-out tissue sanity check: real vs random (CONFIRMATORY — floor)

The module selection was performed on LUNG data. v6b must first pass a
basic sanity check on tissues that did not inform the selection.

**Tissues**: liver (UBERON:0002107), kidney (UBERON:0002113).

**Design**: Cross-assay shifted pairs only. For each tissue, score:
- Real embeddings: Geneformer (512d), scVI (50d), scGPT (512d)
- Random null baselines: random Gaussian projection (512d), untrained
  2-layer ReLU MLP (genes → 256 hidden → 512 output, Xavier-normal init)

Each embedding is scored on the shifted condition (source = 10x 3' v3,
target = 10x 5' v2). No control condition is used.

**Why shifted-only (no gap test)**: The control condition (same-assay,
different-donor) is invalid for this experiment. Census donor_id overlap
between 10x 3' v3 and 10x 5' v2 is ZERO in all four tissues
(check_donor_overlap.py, run 2026-07-12). This means "same-assay control"
compares entirely different cohorts — the control introduces cohort variation
that may exceed assay variation. The Exp 9 result confirmed this: controls
scored WORSE than shifted for all embeddings, making the gap uninterpretable.
The valid comparison is real-vs-null on the same (shifted) pairs.

**Test statistic**: For each tissue separately:
  delta_random = mean(real_scores) - mean(random_null_scores)
where real_scores = [v6b(Geneformer), v6b(scVI), v6b(scGPT)] and
random_null_scores = [v6b(random_proj), v6b(untrained_encoder)].

**Inference**: 10,000 bootstrap resamples. Report the 95% percentile CI.
**Seed**: 20260712.

**H-v6b.1** [CONFIRMATORY — FLOOR]: On liver AND kidney separately, the
95% bootstrap CI of delta_random excludes 0.

This test establishes that v6b distinguishes trained embeddings from
random projections — a necessary but NOT sufficient condition for a
transportability instrument. Passing H-v6b.1 alone does not support the
"confound-free" claim (trained models trivially have more structure than
random noise). It guards only against the case where v6b is *so broken*
that even random > real on held-out tissues.

### 4b. Held-out tissue discrimination: real vs structured null (CONFIRMATORY — primary)

The load-bearing test. BoG (bag-of-genes PCA 512d) has real biological
structure (gene-expression covariance preserved by PCA) but no learned
cross-domain transport. If v6b cannot separate foundation models from BoG,
then v6b measures "has structure" rather than "has learned transportability."

**Design**: Same tissues (liver, kidney), same shifted condition.
- Real embeddings: Geneformer (512d), scVI (50d), scGPT (512d)
- Structured null: bag_of_genes_pca512 (512d)

**Test statistic**: For each tissue separately:
  delta_structured = mean(real_scores) - v6b(BoG)
where real_scores = [v6b(Geneformer), v6b(scVI), v6b(scGPT)].

**Inference**: 10,000 bootstrap resamples of (real_scores, BoG_score).
Report the 95% percentile CI of delta_structured.
**Seed**: 20260712.

**H-v6b.2** [CONFIRMATORY — PRIMARY]: On liver AND kidney separately, the
95% bootstrap CI of delta_structured excludes 0.

This is the test with teeth. BoG is not random — it preserves biological
covariance and gene-program structure. v6b must score trained foundation
models higher than this structured baseline to claim it measures learned
transport rather than raw biological geometry.

### Pass and kill criteria

**Pass** = H-v6b.1 AND H-v6b.2 both hold on BOTH liver AND kidney (all
four tests pass). Only then does v6b earn the "confound-free instrument"
claim.

**Kill criterion K-v6b**: Fires if ANY of the four CIs includes zero, OR
if any point estimate of delta is negative. Report honestly and revise.

**Partial pass** = H-v6b.1 passes but H-v6b.2 fails: v6b distinguishes
trained-from-random but not trained-from-structured. Report as: "v6b
measures embedding quality above the random floor, but cannot distinguish
learned transport from raw biological structure." This is still a
publishable (negative) result — it establishes the limits of purely
geometric diagnostics.

### 4c. Donor-matched control (DROPPED — infeasible)

Originally planned: score donor-matched pairs (same donor_id appearing in
both assays) to isolate assay effects from cohort effects. This test is
DROPPED because Census shows zero donor overlap between 10x 3' v3 and
10x 5' v2 in all four tissues (lung, liver, kidney, brain). The control
condition as designed is impossible to construct.

This is reported as a limitation: the shifted-only test cannot distinguish
"the model captures transferable biology across assays" from "the model
produces geometrically stable embeddings regardless of input quality."
The donor-matched design would have separated these interpretations but
is infeasible in Census 2023-12-15.

### 4d. Exploratory analyses (not confirmatory — reported separately)

The following are computed and reported but do NOT gate the v6b claim:

1. **Per-module breakdown** (M2 and M3 separately) on liver and kidney —
   which module drives the separation, if any?
2. **Dimensionality robustness check**: scVI operates at 50d while all
   other models (Geneformer, scGPT, BoG, random projection) operate at
   512d. Do M2/M3 scores for scVI fall outside the range of same-d models?
   This checks whether M2/M3 are themselves dimension-confounded — the same
   class of problem that knocked out M4. If scVI's v6b score is an outlier
   relative to Geneformer/scGPT despite similar task-probe performance,
   this is flagged as a potential dimension sensitivity in M2/M3.
3. **Comparison to Exp 8 task-probe correlations** (logreg F1, kNN F1,
   silhouette) on the same tissues — does v6b track downstream utility?
4. **Per-model rank ordering**: Does v6b's ranking of models match the
   task-probe ranking? (Spearman over models within each tissue.)

## 5. What this means for the paper

The paper reports four things:

1. **v6 failed its own preregistered confound gate.** The prereg had teeth.
   This is the primary result of Exp 9.
2. **The failure mode is Johnson-Lindenstrauss preservation** — subspace
   alignment and graph curvature are invariant under random linear maps,
   so they measure raw distributional geometry, not learned structure.
   This is a field-relevant warning for geometric evaluation methods.
3. **v6b (M2+M3, post-hoc revision) validated on held-out tissues against
   a structured null (bag-of-genes)** — OR it wasn't, and the limitation
   is that purely geometric diagnostics can distinguish trained-from-random
   but not trained-from-structured-biological-baselines.
4. **M4 (participation ratio) remains under investigation.** It discriminates
   real-from-null on shifted pairs but produces unexplained zeros on ~20% of
   model×tissue cells. Two hypothesized mechanisms (high-d normalization,
   cell-type diversity) were both falsified. The raw-PR diagnostic will
   either resolve this (enabling a future M4 inclusion) or reveal a deeper
   issue with PR-based metrics on heterogeneous biological samples.

The JL finding connects to the broader theme: naive geometric metrics on
embeddings are necessary but not sufficient for transportability claims.
Learned structure shows up in direction stability (M3) — not in subspace
alignment, graph curvature, or (provisionally) spectral compression ratios.

## 6. Preregistration mechanics

Freeze: SHA-256 over this document + the v6b scorer variant source.
Commit before pulling liver/kidney data.
The v6b scorer is a configuration change (weights only), not new code —
implemented by passing custom weights to the existing `run()` function.
