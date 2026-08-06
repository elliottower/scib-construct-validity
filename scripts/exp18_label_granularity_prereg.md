# Exp18: Label Granularity Robustness — Registered Analysis Plan

**Status: FROZEN before any granularity-varied metric value was computed.**
Nothing beyond the exp10 result already published in Paper D has been inspected.

---

## Context

Every scIB bio-conservation metric scores an embedding against a cell-type
labelling. ARI and NMI compare a Leiden partition to it, silhouette and cLISI
measure its geometric separation, isolated-label scores its rare classes. The
labelling is an input, and in every published evaluation it is a single fixed
choice inherited from the atlas.

Cell types are nested. The same cells that carry a leaf annotation also carry
every coarser grouping of it, and each is a defensible answer to "what cell type
is this." No scIB evaluation declares which granularity it used or shows the
result survives a different one.

Paper D already establishes that these metrics invert 46–58% of pairwise model
rankings against cross-assay transfer F1, and that the failure is specific to the
cross-assay setting. That is a failure driven by protocol artifacts. This
experiment asks whether a second, structural failure sits underneath it: whether
the construct is underdetermined by the analyst's granularity choice before any
assay effect enters.

## The question

**Do bio-conservation metrics rank embeddings the same way when the cell-type
labelling is coarsened?**

This is the same test as T2 (cross-dimensionality robustness) one axis over. T2
varies the embedding dimension and requires rankings to survive. T4 varies the
label granularity and asks the same.

---

## Disclosure: observed vs unobserved

### Observed (design parameters and prior published results)
- The exp10 audit design: 4 tissues (lung, liver, kidney, brain), 3 real
  embeddings (geneformer, scvi, scgpt), 3 nulls (random projection, untrained
  encoder, bag-of-genes PCA), 10x 3' v3 → Smart-seq2, 2000 cells per side.
- The five bio metrics: `nmi_leiden`, `ari_leiden`, `silhouette_label`, `clisi`,
  `isolated_label_asw`.
- Paper D's published cross-assay inversion rate, 46–58%.
- T2's published rank-correlation threshold, 0.70.
- That leaf cell-type counts differ by tissue. The counts themselves have not
  been tabulated for this experiment.

### Genuinely unobserved (confirmatory)
- Every metric value at any granularity other than leaf.
- Every ranking, inversion count, and rank correlation across granularity.
- The stability floor computed under this design.
- All quantities in the predictions below.

---

## Design

### The granularity ladder

Coarsening must not depend on any embedding under evaluation, or the ladder
would favour whichever model it was derived from.

For each tissue, with $L$ leaf cell types present after the exp10 filters:

1. Compute a centroid per leaf type in **raw log-normalised gene space**
   (log1p of CP10K), using source-domain cells only.
2. Hierarchically cluster the $L$ centroids, average linkage, correlation
   distance.
3. Cut the tree at $L$, $\lceil 0.75L \rceil$, $\lceil 0.50L \rceil$,
   $\lceil 0.25L \rceil$ clusters, giving four nested labellings
   $G_1 \supset G_{0.75} \supset G_{0.5} \supset G_{0.25}$.

Levels are dropped where the cut yields fewer than 2 classes. Every cell keeps
its identity; only the label it carries changes.

Cell types present in the target domain but absent from the source carry no
centroid and therefore no position in the tree. They remain singleton groups at
every level. Specified here so the handling is not chosen after seeing how many
such types there are.

**Primary comparison: $G_1$ vs $G_{0.25}$.** The intermediate levels are reported
and are not used for the headline.

### What is recomputed

The five bio metrics, for all six embeddings, in all four tissues, at each
granularity. Embeddings, cells, kNN graphs and every other input are held fixed —
only `label_key` changes. Batch metrics are not label-dependent in the relevant
sense and are excluded.

### Primary statistic — pairwise ranking inversions

Matching Paper D's house method rather than introducing a new one.

For each (tissue, metric) and each unordered pair of **real** embeddings
$(A,B)$, the pair is **inverted** between two granularities if
$\mathrm{sign}(m_A - m_B)$ differs between them. With 3 real embeddings this is
3 pairs × 4 tissues × 5 metrics = **60 pairs** per granularity comparison.

Inversion rate is reported per metric (12 pairs each) and pooled.

**Real embeddings only for the primary statistic.** Including the three nulls
would let the trained-versus-random gap dominate and mask inversions among the
models anyone would actually choose between — the same masking that lets scIB
pass null discrimination 8/9 while inverting half its cross-assay rankings.
All-six inversion rates are reported as a secondary.

### The floor — this is what makes the number interpretable

An inversion rate is meaningless without knowing how many inversions arise from
the pipeline's own noise. At fixed granularity $G_1$, recompute all metrics on
3 independent 90% subsamples of the cells, and take the mean pairwise inversion
rate across subsample pairs. That is the **stability floor**.

Resampling rather than seed-setting, because it needs no assumption about which
stochastic arguments the installed `scib-metrics` exposes, and it is executable
against any version. It is a **conservative** floor: it absorbs cell-sampling
noise on top of algorithmic stochasticity, so it is larger than a pure seed
floor would be. Granularity therefore has to clear a higher bar, which is the
direction that favours the null.

Granularity is only shown to matter if its inversion rate exceeds this floor.
Bootstrap 95% CIs on (granularity inversion rate − stability floor) over the 60
pairs, 1000 resamples.

### Secondary — rank correlation, for comparability with T2

Spearman correlation between the six-embedding ranking at $G_1$ and at each
coarser level, per (tissue, metric). Reported against T2's existing 0.70
threshold so the two robustness checks can be read on one scale.

---

## Predictions

Registered before computation, and recorded because they are the outcomes worth
reporting whichever way they fall.

**P1 (primary).** The $G_1$ vs $G_{0.25}$ inversion rate among real embeddings
exceeds the stability floor, with a bootstrap 95% CI on the difference excluding zero.

**P2.** At least 2 of the 5 bio metrics show an inversion rate above 25%.

**P3.** `isolated_label_asw` has the highest inversion rate of the five.
Coarsening merges rare types, and that metric is defined on exactly the rare-type
structure the merge destroys.

**P4.** `silhouette_label` is the most stable. It measures geometric separation
that a relabelling perturbs less sharply than partition-comparison metrics.

**Prediction on the headline: P1 confirms.** If it does not, the construct
survives a test nobody has run on it, and that is a genuine strengthening of
Paper D's protocol rather than a null worth burying.

### What each outcome means

| outcome | reading |
|---|---|
| inversion rate ≫ stability floor | bio-conservation is conservation of a granularity the analyst chose and never declared; the repair is to declare and report across the ladder |
| inversion rate ≈ stability floor | the construct is robust to granularity; T4 becomes a passed check and the protocol gains a dimension |
| inversion rate below stability floor | something is wrong with the floor; investigate before reporting either |

---

## Abort conditions

Stated in advance so they cannot be rationalised afterwards.

1. **Data mismatch.** If the Census pull does not reproduce exp10's per-tissue
   cell counts and leaf type counts, stop. A different subset invalidates
   comparison to the published audit.
2. **Ladder too short.** A tissue with $L < 6$ leaf types cannot support a
   four-level ladder; that tissue is dropped and the drop is reported.
3. **Degenerate metrics.** If a metric returns NaN or a constant across
   embeddings at any level, that (metric, level) cell is excluded and reported,
   never silently imputed.
4. **Floor at ceiling.** If the stability floor alone exceeds 40%, the metrics
   are too unstable at fixed granularity for this test to say anything, and the
   result is reported as uninformative rather than as a null.

---

## Deviations from Paper D's existing protocol

None intended. The metrics, tissues, embeddings, assay pair, cell budget and
seed are exp10's. The single manipulated variable is `label_key`.

If the CL ontology rollup is added later as a robustness check, it is a separate
amendment recorded here with its date, and the centroid ladder remains primary —
the ontology route needs a dependency this repo does not currently pin.

---

## Files

- Script: `scripts/exp18_label_granularity.py`
- Results: `results/exp18_label_granularity/`
- Freeze: `scripts/FREEZE_exp18.txt`
