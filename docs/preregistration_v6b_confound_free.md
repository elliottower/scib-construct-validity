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

**Modules**: M2 (domain shift), M3 (direction stability), M4 (participation ratio).
**Weights**: M2 = 1.0, M3 = 2.0, M4 = 1.0.
**Tier thresholds**: Unchanged from v6 [0.85, 0.70, 0.55, 0.40, 0.25, 0.10].

Rationale for weights: M3 carries the strongest conceptual signal (stable
learned directions vs unstable random directions) and showed the clearest
continuous-score separation. M4 discriminates equally well on tiers but its
continuous gap is partially explained by dimensionality matching (both source
and target are high-dimensional in learned space). M2 is retained despite weak
discrimination because domain separability is conceptually distinct from the
other two and provides a classification-based signal complementary to the
spectral signals of M3/M4.

**Dropped**: M1 (Grassmannian, anti-discriminative), M6 (curvature,
anti-discriminative). These are not merely "uninformative" — they actively
confound the composite by giving null embeddings higher scores than real ones.

## 4. Validation plan (confirmatory — must pass before any v6b claim)

### 4a. Held-out tissue confound gate

The module selection was performed on LUNG data. v6b must pass the confound
gate on tissues that did not inform the selection.

**Tissues**: liver (UBERON:0002107), kidney (UBERON:0002113).
**Design**: Identical to Exp 9 — real (Geneformer, scVI) vs null (random
projection 512d, untrained encoder 512d), cross-assay shifted + same-assay
control (donor-matched where possible).
**Seed**: 20260712.
**Bootstrap**: 10,000 resamples.

**H-v6b.1** [CONFIRMATORY]: On liver AND kidney separately, the v6b composite
tier gap (control - shifted) for real models exceeds the null gap by >= 1.0
tier, 95% bootstrap CI excluding 0.

Pass = H-v6b.1 holds on BOTH held-out tissues. Failure on either = v6b is
still confounded and requires further revision.

### 4b. Donor-matched control (addresses control-worse-than-shifted)

The Exp 9 lung result showed controls scoring WORSE than shifted (negative
tier gaps for all embeddings). This suggests the "control" (same-assay,
different donors) introduces more geometric divergence than the assay shift.
To isolate the assay effect:

**Design**: Where the same donor_id appears in both 10x 3' v3 AND 10x 5' v2
data for a tissue, construct donor-matched pairs. Score both the matched
cross-assay pair (same donor, different assay) and a within-assay pair
(same donor, same assay, random split).

**H-v6b.2** [EXPLORATORY]: Donor-matched cross-assay pairs show lower v6b
composite than donor-matched within-assay pairs (expected direction: shifted
< control when donor variation is controlled).

If donor-matched pairs are unavailable (< 5 donors with both assays), this
test is DROPPED and flagged as a limitation.

## 5. What this means for the paper

The paper reports three things:

1. **v6 failed its own preregistered confound gate.** The prereg had teeth.
   This is the primary result of Exp 9.
2. **The failure mode is Johnson-Lindenstrauss preservation** — subspace
   alignment and graph curvature are invariant under random linear maps,
   so they measure raw distributional geometry, not learned structure.
   This is a field-relevant warning for geometric evaluation methods.
3. **v6b (post-hoc revision, validated on held-out tissues) produces a
   confound-free instrument** — OR it doesn't, and the limitation is that
   no purely geometric diagnostic can distinguish learned from random without
   a task-specific probe.

The JL finding connects to the broader theme: naive geometric metrics on
embeddings are necessary but not sufficient for transportability claims.
Learned structure shows up in direction stability and spectral structure,
not in subspace alignment or graph curvature.

## 6. Preregistration mechanics

Freeze: SHA-256 over this document + the v6b scorer variant source.
Commit before pulling liver/kidney data.
The v6b scorer is a configuration change (weights only), not new code —
implemented by passing custom weights to the existing `run()` function.
