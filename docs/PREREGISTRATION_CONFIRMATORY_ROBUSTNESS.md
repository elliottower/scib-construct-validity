# Pre-registration: Confirmatory Robustness Tests (Blind)

**Date**: 2026-07-13
**Status**: Pre-registered predictions BEFORE running any analysis
**Author**: Elliot Tower
**Parent**: Extends PREREGISTRATION_SCIB_AUDIT.md (T1/T3) and
PREREGISTRATION_SCIB_EXTENDED_VALIDITY.md (Tests 1-3)

**Blind attestation**: No code for Tests 4-6 has been executed. No metric
values under noise perturbation, hyperparameter variation, or
delta-vs-rho comparison have been observed. All predictions below
derive from first-principles reasoning about metric definitions and
embedding geometry.

---

## Test 4: Noise Dose-Response Monotonicity

**Protocol**: Corrupt each of the 6 embeddings (Geneformer, scVI, scGPT,
random_projection, untrained_encoder, bag_of_genes_pca_512) with
additive Gaussian noise at sigma in {0.01, 0.05, 0.1, 0.25, 0.5, 1.0,
2.0} x embedding_std, across 4 tissues. For each metric, check whether
the score changes monotonically across noise levels. A metric FAILS if
non-monotonic in >1 of the 24 conditions (6 embeddings x 4 tissues).

### Per-metric predictions

#### nmi_leiden: FAIL

**Mechanism**: Leiden community detection is a discrete optimization
problem on the kNN graph. At a fixed resolution parameter, the number
and boundaries of detected communities depend discontinuously on the
graph topology. Small noise acts as a regularizer: if the clean
embedding produces over-fragmented communities (resolution too high for
the data), moderate noise blurs fine-grained substructure, causing
Leiden to merge fragments. This can temporarily INCREASE NMI if the
merged communities better match ground-truth cell-type labels, before
further noise degrades the clustering.

**Where non-monotonicity appears**: sigma in {0.01, 0.05, 0.1}. At
these levels, noise is large enough to change kNN graph edges but small
enough that cell-type-level structure is intact. The Leiden algorithm
finds a different partition whose agreement with ground truth fluctuates
non-monotonically.

**Frequency**: I predict non-monotonicity in 8-16 of 24 conditions.
Embeddings with well-separated types at the default resolution (trained
models in "easy" tissues like brain) may be robust, but most
embedding-tissue combinations should show at least one reversal across
the 7 noise levels.

#### ari_leiden: FAIL

**Mechanism**: Identical to NMI. Both depend on Leiden clustering at a
fixed resolution. ARI is more sensitive than NMI to the exact number of
clusters (it penalizes both over- and under-clustering more harshly),
making it arguably even more prone to non-monotonic jumps when noise
causes community merges or splits.

**Where**: sigma in {0.01, 0.05, 0.1}. Same noise regime as NMI.

**Frequency**: 10-18 of 24 conditions. ARI's sensitivity to cluster
count means even small partition changes produce noticeable score
fluctuations.

#### silhouette_label: PASS

**Mechanism**: Silhouette width is a purely geometric quantity:
s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is the mean
intra-cluster distance and b(i) is the mean nearest-cluster distance.
Adding isotropic Gaussian noise increases all pairwise distances, but
proportionally more for small distances (intra-cluster) than for large
distances (inter-cluster). Specifically, for points x, y with noise
eps_x, eps_y ~ N(0, sigma^2 I):

  E[||x + eps_x - y - eps_y||^2] = ||x - y||^2 + 2 * d * sigma^2

The additive 2*d*sigma^2 is constant regardless of ||x-y||, so it
constitutes a larger proportional increase for small intra-cluster
distances than for large inter-cluster distances. This systematically
degrades the silhouette ratio. No discrete operations are involved;
the metric is a continuous function of pairwise distances, and the
expected value decreases monotonically with sigma.

**Confidence**: High. The only failure mode would be if a specific noise
realization happened to DECREASE some intra-cluster distances more than
inter-cluster distances, but averaging over thousands of cells makes
this negligible.

#### clisi: PASS

**Mechanism**: cLISI (cell-type local inverse Simpson index) measures
the effective number of cell types in each cell's kNN neighborhood.
Higher values indicate less pure neighborhoods (worse bio conservation).
scIB rescales so higher = better (purer neighborhoods).

Adding noise changes kNN neighborhoods. The question is whether the
AVERAGE purity across all cells can increase with noise. For isotropic
noise added to all cells simultaneously, there is no systematic
mechanism by which noise preferentially moves cells toward their own
type. Individual cells may gain or lose same-type neighbors stochastically,
but the average over 2,000-4,000 cells should smooth this to a monotonic
decline in purity.

cLISI values tend to concentrate near ceiling (most cells have
type-pure neighborhoods in both trained and structured-null embeddings).
This ceiling compression makes the metric's decline very shallow at low
noise (plateau near maximum) followed by steeper decline at high noise.
The trajectory is monotonically non-increasing throughout.

**Caveat**: Individual kNN boundary flips are discrete events, creating
micro-non-monotonicity at the individual-cell level. But averaging over
thousands of cells smooths this. With 24 conditions and 7 noise levels,
I predict 0-1 conditions show non-monotonicity (below the >1 failure
threshold).

**Confidence**: Medium-high.

#### isolated_label_asw: PASS

**Mechanism**: Same as silhouette_label (average silhouette width) but
restricted to cell types that appear in only one batch. The computation
is identical; only the cell subset differs. The subset may be small
(tens to hundreds of cells), increasing variance, but the silhouette
calculation itself remains a continuous function of pairwise distances
with the same monotonic degradation property.

**Confidence**: Medium-high. The smaller sample size could in principle
create more variance, but the systematic force (noise degrades cluster
separation) dominates.

#### silhouette_batch: PASS

**Mechanism**: Silhouette width computed on batch labels, rescaled so
higher = better batch mixing. scIB uses the transform:
score = (1 - ASW_batch) / 2, or a similar rescaling that maps ASW in
[-1, 1] to [0, 1] with 0.5 = no batch structure.

Adding noise erases batch structure monotonically. Batch effects are
systematic shifts between groups of cells. Isotropic noise with
variance proportional to embedding_std dilutes these shifts: at sigma =
0, batch structure is intact; at sigma >> 1, noise dominates and
ASW_batch approaches 0 (no batch discrimination). The rescaled score
monotonically approaches 0.5 from whatever starting point. For
embeddings where batches are separated (ASW_batch > 0), the score
increases monotonically. For embeddings where batches are already mixed
(ASW_batch near 0), the score stays near 0.5.

No non-monotonicity mechanism exists because the same geometric argument
applies as for silhouette_label: noise increases all distances by a
constant additive factor in expectation, systematically degrading any
labeling-based structure.

**Confidence**: High.

#### ilisi: PASS

**Mechanism**: iLISI (integration LISI) measures the effective number of
batches in each cell's kNN neighborhood. Higher = better batch mixing.
scIB normalizes as: score = (median_LISI - 1) / (n_batches - 1).

Adding noise to all cells makes pairwise distances more uniform (noise
adds a constant variance to all squared distances). This homogenizes
kNN neighborhoods: cells that were in batch-homogeneous neighborhoods
start acquiring cross-batch neighbors. The average LISI increases
monotonically toward the theoretical maximum (perfect random mixing).

iLISI values tend toward floor (near 0, meaning neighborhoods are
batch-homogeneous). Adding noise can only maintain or improve batch
mixing, never systematically reduce it. The trajectory is monotonically
non-decreasing.

**Confidence**: Medium-high. The discrete kNN graph could cause micro-
fluctuations, but averaging over thousands of cells smooths these.

#### kbet: FAIL

**Mechanism**: kBET applies a chi-squared goodness-of-fit test per cell,
testing whether the batch composition of each cell's kNN neighborhood
matches the global batch proportions. The aggregate score is the
fraction of cells that pass the test (p > alpha, typically 0.05).

Two properties make kBET prone to non-monotonicity:

1. **Binary thresholding**: Each cell either passes or fails the
   chi-squared test. The aggregate is a fraction, not a smooth average.
   When noise changes a cell's neighborhood composition, the chi-squared
   statistic can cross the rejection threshold in either direction. For
   cells whose chi-squared statistic is near the critical value, small
   noise perturbations flip them between pass and fail stochastically.

2. **Count sensitivity**: The chi-squared test operates on integer
   counts of each batch in the neighborhood. Changing even one neighbor
   changes the counts discretely, which can cause a disproportionate
   change in the test statistic. At moderate noise levels (sigma = 0.05
   to 0.25), enough neighborhoods change to create threshold effects,
   but not enough to overwhelm the systematic direction.

kBET is known within the single-cell community as the most variable
metric in the scIB suite. Its sensitivity to local neighborhood
composition and the hard pass/fail threshold create the conditions for
non-monotonic aggregate behavior.

**Where**: sigma in {0.05, 0.1, 0.25}. At very low noise (0.01), too
few neighborhoods change. At high noise (1.0, 2.0), all neighborhoods
are effectively random and pass rates are high. The intermediate regime
is where threshold effects dominate.

**Frequency**: 3-8 of 24 conditions. The binary thresholding creates
more non-monotonicity than smooth metrics but less than Leiden-based
metrics.

**Confidence**: Medium. kBET's known instability supports this
prediction, but the averaging over thousands of cells provides some
smoothing.

#### graph_connectivity: PASS

**Mechanism**: Graph connectivity measures the fraction of cells in the
largest connected component per cell type in the kNN graph. Adding noise
to embeddings tends to homogenize pairwise distances, which makes the
kNN graph MORE connected (previously distant cells of the same type
become reachable through shorter chains of neighbors).

For trained models, connectivity is likely already near 1.0 (cell types
form single connected components). Noise cannot improve this further,
and moderate noise should not disconnect well-connected clusters because
noise moves all cells simultaneously (relative positions change smoothly
rather than individual cells being removed).

For null models, connectivity might be < 1.0 if cell types form
multiple disconnected components. Noise should improve connectivity by
creating new connections across gaps.

In both cases, the metric is non-decreasing. The only failure mode
would be if noise disconnected a previously connected cell-type cluster,
which requires noise to move "bridge" cells away from both subclusters
they connect. This is systematically unlikely with isotropic noise.

**Confidence**: Medium-high. Graph connectivity's tendency toward ceiling
provides a monotonic floor.

#### pcr_comparison: PASS

**Mechanism**: PCR comparison regresses principal components on the batch
variable and compares R-squared before vs after integration. Adding
noise to the embedding adds variance uncorrelated with batch. This
monotonically dilutes the batch-explained variance in the PCs:

- R^2_after = Var(batch-projected PC) / Var(PC)
- Adding noise increases Var(PC) (denominator) without increasing
  Var(batch-projected PC) (numerator), so R^2 decreases.
- pcr_comparison = 1 - R^2_after/R^2_before increases monotonically.

PCA recomputes the top components from the noisy embedding. In high
dimensions, Gaussian noise adds a spherical bulk to the spectrum.
The batch-correlated variance occupies specific directions; as noise
grows, these directions are increasingly dominated by noise variance.
The regression R^2 on those PCs decreases monotonically.

**Confidence**: High. No discrete operations; all quantities are
continuous functions of the embedding.

### Test 4 summary

| Metric | Prediction | Non-monotonicity regime | Mechanism | Confidence |
|--------|-----------|-------------------------|-----------|------------|
| nmi_leiden | **FAIL** | sigma 0.01-0.1 | Discrete Leiden partition jumps | High |
| ari_leiden | **FAIL** | sigma 0.01-0.1 | Discrete Leiden partition jumps | High |
| silhouette_label | PASS | n/a | Continuous geometric degradation | High |
| clisi | PASS | n/a | kNN average smooths boundary effects | Medium-High |
| isolated_label_asw | PASS | n/a | Same as silhouette, smaller subset | Medium-High |
| silhouette_batch | PASS | n/a | Continuous geometric degradation | High |
| ilisi | PASS | n/a | Noise monotonically improves batch mixing | Medium-High |
| kbet | **FAIL** | sigma 0.05-0.25 | Chi-squared threshold effects | Medium |
| graph_connectivity | PASS | n/a | Noise increases graph density | Medium-High |
| pcr_comparison | PASS | n/a | Noise dilutes batch variance in PCs | High |

**Aggregate prediction**: 3 metrics FAIL (nmi_leiden, ari_leiden, kbet),
7 metrics PASS. The failures divide into two mechanisms: discrete
clustering (NMI, ARI) and statistical thresholding (kBET). All
continuous-valued, cell-averaged metrics pass.

---

## Test 5: Hyperparameter Sensitivity of Rankings

**Protocol**: Re-run clustering-based metrics (NMI, ARI) at Leiden
resolutions {0.5, 1.0, 2.0}. Re-run kNN-based metrics (cLISI, iLISI,
kBET, graph_connectivity) at k in {15, 30, 90}. For each metric, compute
Kendall's tau between the ranking of 6 embeddings at different
hyperparameter values.

### Structural constraint on tau

With 6 embeddings (3 trained, 3 null), there are 15 pairwise comparisons.
If the trained-vs-null partition is preserved, 9 cross-group pairs are
concordant regardless of within-group ordering. Even with completely
reversed within-group order, tau >= (9-6)/15 = 0.2. With random
within-group order, expected tau = (9+3)/15 - 3/15 = 0.6. This means
tau < 0.6 implies systematic within-group or cross-group disruption.

### Per-metric predictions

#### nmi_leiden (resolutions 0.5 vs 1.0 vs 2.0)

**Predicted tau**: 0.5-0.7 (between resolution 0.5 and 2.0; higher
between adjacent resolutions).

**Reasoning**: Leiden resolution controls clustering granularity. At
resolution 0.5, Leiden finds fewer, larger communities. This favors
embeddings where cell types form large well-separated groups. At
resolution 2.0, Leiden finds many small communities. This favors
embeddings where cell types have tight, internally homogeneous clusters
(over-clustering is punished more when the subclusters don't match
ground truth, but the penalty depends on the specific embedding
geometry).

The trained-vs-null split should hold at all resolutions: trained models
always cluster better than random projections. The within-trained-model
ranking is what changes. Each model creates a different embedding
geometry, so a resolution that works well for scVI's latent space might
be suboptimal for Geneformer's.

**Most likely ranking flip**: Among trained models. scVI (explicitly
designed for integration, lower-dimensional latent space d=50) may
produce tighter clusters that hold up at resolution 2.0, while
Geneformer (high-dimensional, d=512) may produce more diffuse clusters
that fragment. At resolution 0.5, the situation could reverse if
Geneformer's broad cluster structure captures coarse cell-type
distinctions that scVI's narrow manifold misses.

**Winner change**: Possible. The top-ranked trained model at resolution
0.5 may differ from the top at resolution 2.0. I predict this happens
in at least 2 of 4 tissues.

#### ari_leiden (resolutions 0.5 vs 1.0 vs 2.0)

**Predicted tau**: 0.4-0.6 (slightly lower than NMI).

**Reasoning**: ARI penalizes cluster-count mismatch more sharply than
NMI. At resolution 2.0, the over-clustering penalty in ARI depends
strongly on the specific embedding: an embedding that produces many
small but type-pure subclusters gets a less harsh ARI penalty than one
that produces many impure subclusters. This sharper resolution
dependence creates more instability in rankings.

**Most likely ranking flip**: Same as NMI but more frequent. ARI's
sensitivity to the number of clusters means the relative ARI of two
embeddings can swap when resolution changes the cluster count
differently for each.

**Winner change**: Yes, more likely than for NMI. I predict the winner
changes in 2-3 of 4 tissues between resolution 0.5 and 2.0.

#### clisi (k in {15, 30, 90})

**Predicted tau**: 0.3-0.6.

**Reasoning**: cLISI operates in a regime of ceiling saturation: most
cells have type-pure kNN neighborhoods, producing cLISI scores clustered
near the maximum (1.0 on the scIB-rescaled scale). This ceiling
compression means the ranking of 6 embeddings is determined by tiny
score differences rather than meaningful signal.

At k=15 (small neighborhoods), local purity is high for all embeddings
that preserve any cell-type structure. Even structured null models
(bag_of_genes_pca_512 via PCA on gene expression) should have pure
neighborhoods at k=15, because cell types differ substantially in
expression profiles and the Johnson-Lindenstrauss lemma guarantees
distance preservation. At k=90 (large neighborhoods), purity decreases
for all embeddings, but the decrease rate varies: embeddings with
well-separated type clusters (large inter-type gaps) maintain purity
longer than embeddings with overlapping types.

The crucial effect: at k=15, all embeddings are near ceiling and
rankings reflect noise. At k=90, more spread appears, and rankings
reflect genuine structure differences. These two ranking bases may be
weakly correlated.

**Most likely ranking flip**: Between null models and trained models.
If bag_of_genes_pca_512 has high cLISI at k=15 (PCA preserves local
structure), it might rank among the trained models, then fall below them
at k=90 (PCA doesn't capture the nonlinear manifold structure needed
for large-neighborhood purity).

**Winner change**: Possible but unpredictable (ceiling noise dominates
the "winner" choice at small k).

#### ilisi (k in {15, 30, 90})

**Predicted tau**: 0.2-0.5.

**Reasoning**: iLISI operates in a regime of floor saturation: most
cells have batch-homogeneous kNN neighborhoods, producing iLISI scores
near 0. This floor compression creates the mirror problem of cLISI's
ceiling: rankings are determined by which embeddings have ANY non-zero
batch mixing, not by the magnitude of mixing.

At k=15, neighborhoods are small and batch-homogeneous for all
embeddings. At k=90, neighborhoods are larger and may capture some
cross-batch cells. The identity of which embeddings first achieve
nonzero iLISI as k increases depends on the specific embedding geometry.

For trained models that explicitly model batch effects (scVI), batch
mixing should appear at smaller k (the model actively integrates
batches). For null models, random_projection should have high iLISI
(no batch structure to preserve) while bag_of_genes_pca_512 may have
low iLISI (PCA preserves batch-correlated variance in top components).

**Most likely ranking flip**: random_projection may outrank trained
models for iLISI (random projections have no batch effects by
construction). This ranking could hold at all k, or could change if
scVI's explicit batch correction becomes effective at larger k.

**Winner change**: Yes. If random_projection has the best iLISI at k=15
and scVI has the best at k=90, the winner changes. This is a plausible
outcome because random projections provide uniform mixing at all scales,
while scVI's integration acts at specific neighborhood sizes determined
by its latent-space geometry.

#### kbet (k in {15, 30, 90})

**Predicted tau**: 0.1-0.4 (lowest of all metrics).

**Reasoning**: kBET is the metric most sensitive to k. The chi-squared
test's behavior changes qualitatively with neighborhood size:

- At k=15: The chi-squared test has few degrees of freedom (only 15
  cells in each neighborhood). The test has low power (fails to detect
  real batch imbalance) and high variance (random fluctuations in the
  15-cell sample dominate). This produces noisy, unreliable rankings.

- At k=90: The chi-squared test has much higher power and lower
  variance. The test reliably detects batch imbalance. Rankings
  reflect genuine batch-mixing differences between embeddings.

Because the k=15 rankings are essentially noise-driven while the k=90
rankings are signal-driven, the correlation between them should be
weak. The tau reflects the correlation between noise and signal, which
is near zero.

Additionally, the fraction of cells passing the chi-squared test
(the aggregate kBET score) can flip dramatically with k. An embedding
where most cells just barely pass at k=15 (low-power test with large
acceptance region) might have most cells fail at k=90 (high-power test
detects real imbalance). This creates embedding-specific, k-dependent
behavior that disrupts rankings.

**Most likely ranking flip**: Complete reordering among both trained
and null models. The trained-vs-null split itself might not hold:
random_projection should pass kBET at all k (no batch structure), while
trained models could fail kBET at high k if residual batch effects are
detectable by the more powerful test.

**Winner change**: Yes, with high probability. The winner at k=15 is
likely different from the winner at k=90. I predict the winner changes
in 3-4 of 4 tissues.

#### graph_connectivity (k in {15, 30, 90})

**Predicted tau**: 0.3-0.6.

**Reasoning**: Graph connectivity depends on whether cells of the same
type form a single connected component in the kNN graph. This has a
ceiling effect at high k: as k increases, the graph becomes denser, and
connectivity approaches 1.0 for all embeddings.

At k=15, the graph is sparse and connectivity can vary across
embeddings. Trained models should have higher connectivity (cell types
form coherent clusters with dense internal kNN connections). Null models
may have lower connectivity (cells of the same type are scattered).

At k=90, most embeddings approach connectivity = 1.0 (the denser graph
connects everything). Rankings become meaningless (all tied at 1.0).

The tau between k=15 and k=90 rankings is depressed by the ceiling
effect: the k=90 ranking is degenerate. At k=15 vs k=30, tau should be
higher (~0.6-0.8) because both have meaningful spread.

**Most likely ranking flip**: Between the null models. Among null
models, untrained_encoder and bag_of_genes_pca_512 preserve more
structure than random_projection, so they should have higher
connectivity. But this advantage disappears at high k (everything
connects).

**Winner change**: Unlikely between k=15 and k=30 (same general
ordering). At k=90, the concept of "winner" is meaningless (all tied
at or near 1.0).

### Test 5 summary

| Metric | Hyperparameter | Predicted tau (extreme settings) | Winner changes? |
|--------|---------------|----------------------------------|-----------------|
| nmi_leiden | Leiden resolution | 0.5-0.7 | Yes (2/4 tissues) |
| ari_leiden | Leiden resolution | 0.4-0.6 | Yes (2-3/4 tissues) |
| clisi | k (kNN) | 0.3-0.6 | Possibly (ceiling noise) |
| ilisi | k (kNN) | 0.2-0.5 | Yes (floor saturation) |
| kbet | k (kNN) | 0.1-0.4 | Yes (3-4/4 tissues) |
| graph_connectivity | k (kNN) | 0.3-0.6 | No (ceiling at high k) |

**Aggregate prediction**: kBET shows the most ranking instability
(tau as low as 0.1). ARI is more resolution-sensitive than NMI.
cLISI and iLISI rankings are unstable for different reasons (ceiling
and floor saturation respectively). graph_connectivity saturates at
high k, making rankings degenerate rather than unstable.

**Qualitative takeaway**: A practitioner who runs scIB at k=15 and
another who runs at k=90 could reach opposite conclusions about which
model is best integrated, driven primarily by kBET flips. This
constitutes a validity threat to any benchmark that reports scIB scores
at a single hyperparameter setting without sensitivity analysis.

---

## Test 6: Discriminative-vs-Predictive Dissociation

**Protocol**: For each of the 5 bio metrics, compute T1-delta (how much
better trained models score than null models) and T3-rho (Spearman
correlation between the metric and ground-truth F1 across conditions).
Then compute the Spearman rank correlation between T1-delta and T3-rho
across the 5 bio metrics (n=5 data points).

### First-principles predictions for T1-delta and T3-rho

#### Reasoning about T1-delta (discriminative power)

T1-delta = mean(trained_scores) - mean(null_scores). This depends on
how much each metric separates structured embeddings from random ones.

**silhouette_label**: Large delta. Random projections and untrained
encoders produce embeddings with no systematic cell-type cluster
structure (silhouette near 0). Trained models produce separated
clusters (silhouette > 0). bag_of_genes_pca_512 is intermediate:
PCA on gene expression captures major axes of variation that correlate
with cell type, but less effectively than trained models. Delta is
pulled toward moderate-large by the PCA null model's intermediate
performance.

**nmi_leiden**: Moderate-to-large delta. Leiden clustering on random
embeddings finds communities, but they do not match cell-type labels
well (NMI near 0 for random_projection, somewhat higher for
bag_of_genes_pca_512). Trained models achieve NMI in the 0.5-0.8
range. NMI is bounded [0,1] and tends to be "generous" (even partial
agreement yields moderate NMI), which compresses the delta relative
to ARI.

**ari_leiden**: Large delta. ARI is adjusted for chance: random
partitions produce ARI near 0 by construction. Trained models produce
positive ARI (0.3-0.6 typically). The chance adjustment makes ARI's
null baseline lower than NMI's, producing a LARGER delta. However,
bag_of_genes_pca_512 may have nontrivial ARI (PCA-based clustering can
match cell types), pulling the null mean up.

**clisi**: Small delta. cLISI tends toward ceiling saturation because
cell types in single-cell data differ substantially in expression
profiles. Even distance-preserving null models (random projections via
JL lemma, PCA) maintain reasonably pure kNN neighborhoods. The trained
model mean and null model mean are both high (near ceiling), so the
delta is compressed. This is the defining property of a ceiling-
saturated metric: both groups score near the maximum, leaving little
room for discrimination.

**isolated_label_asw**: Moderate delta. Same mechanism as silhouette_
label but restricted to cell types present in only one batch. The
smaller cell count per type creates noisier scores. The delta should
track silhouette_label but with more variance and potentially a smaller
gap (isolated types may be harder to separate for trained and null
models alike).

**Predicted T1-delta ranking (largest to smallest)**:
1. ari_leiden (chance-adjusted, low null baseline)
2. silhouette_label (geometric, clear trained-null gap)
3. nmi_leiden (generous scale compresses delta)
4. isolated_label_asw (noisier version of silhouette)
5. clisi (ceiling saturation, both groups near maximum)

#### Reasoning about T3-rho (predictive validity)

T3-rho = Spearman correlation between metric scores and ground-truth
cell_type_recovery_f1 across conditions (6 embeddings x 4 tissues = 24
data points). This measures whether the metric tracks the same signal
that determines downstream classification performance.

**silhouette_label**: Moderate-to-high T3-rho. Silhouette measures the
geometric separation of cell-type clusters. kNN classification (which
determines F1) depends on local decision boundaries, which are closely
related to cluster separation. Higher silhouette implies larger margins
between types, which implies better kNN classification. The connection
is direct but imperfect: silhouette is a global average while F1
depends on per-type boundary quality. Predicted rho: 0.4-0.6.

**ari_leiden**: Moderate T3-rho. Leiden clustering quality relates to
classification performance, but through an indirect path: good clusters
imply good structure, which implies good classification. The Leiden
resolution parameter introduces a confounder: the resolution that
maximizes ARI might not correspond to the resolution that best predicts
classification. At a fixed resolution, ARI captures cluster-matching
quality that correlates with F1 but with noise from the resolution
mismatch. Predicted rho: 0.2-0.5.

**nmi_leiden**: Moderate T3-rho. NMI and ARI are highly correlated
(both derive from Leiden clustering), so T3-rho should be similar.
NMI is less sensitive to cluster count, which might make it a slightly
worse or slightly better predictor of F1 depending on whether the
resolution is too high or too low. Predicted rho: 0.2-0.5.

**clisi**: Low or wrong-sign T3-rho. Ceiling saturation means cLISI
has minimal variance across conditions. With minimal variance, the
Spearman correlation with F1 is dominated by noise. The sign could be
positive, zero, or negative essentially at random. If anything, there
is a subtle mechanism for a negative correlation: embeddings that
create very tight, pure neighborhoods (high cLISI) may do so at the
expense of cross-batch alignment (cells of the same type from different
batches are in separate tight clusters rather than one shared cluster).
This would create high cLISI but low F1 (cross-assay classification
fails because the source and target clusters don't overlap). Predicted
rho: -0.2 to 0.1 (centered near 0, possibly negative).

**isolated_label_asw**: Low T3-rho. Isolated labels are cell types
present in only one batch. In the cross-assay setting (source = 10x 3'
v3, target = Smart-seq2), isolated labels are types found in only one
assay. These types by definition CANNOT be classified cross-assay (they
appear only on one side of the train/test split). So isolated_label_asw
measures embedding quality for types that do not contribute to the F1
ground truth. The correlation between isolated_label_asw and F1 is
therefore driven entirely by shared structure (types that are well-
separated overall tend to have well-separated isolated types too), not
by a direct causal link. Predicted rho: 0.1-0.3.

**Predicted T3-rho ranking (highest to lowest)**:
1. silhouette_label (geometric, directly related to kNN boundaries)
2. ari_leiden (cluster quality correlates with classification)
3. nmi_leiden (similar to ARI, slightly different sensitivity)
4. isolated_label_asw (measures irrelevant subset for cross-assay F1)
5. clisi (ceiling saturation, near-zero or wrong-sign correlation)

#### Delta-vs-rho rank correlation

Using the predicted rankings:

| Metric | T1-delta rank | T3-rho rank |
|--------|--------------|-------------|
| ari_leiden | 1 | 2 |
| silhouette_label | 2 | 1 |
| nmi_leiden | 3 | 3 |
| isolated_label_asw | 4 | 4 |
| clisi | 5 | 5 |

Rank differences d: (1-2)=-1, (2-1)=1, (3-3)=0, (4-4)=0, (5-5)=0
Sum d^2 = 1 + 1 + 0 + 0 + 0 = 2
Spearman r = 1 - 6*2 / (5*24) = 1 - 12/120 = 1 - 0.1 = **0.9**

### Prediction

**Sign**: Positive.

**Magnitude**: Spearman rho approximately 0.7-1.0 (point estimate 0.9).

**Driving pattern**: The correlation is strongly positive because the
same property -- ceiling saturation -- makes cLISI both non-discriminative
(low T1-delta) AND non-predictive (low T3-rho). cLISI anchors the
bottom-right of the delta-vs-rho space. At the other end, silhouette_
label and ARI have both meaningful variance (enabling discrimination)
and geometric relevance to classification (enabling prediction).

The correlation is driven by the extremes (cLISI at bottom, silhouette/
ARI at top) rather than by fine-grained ordering in the middle. If my
rankings of NMI and isolated_label_asw are wrong (they swap positions),
the correlation stays positive but drops somewhat.

**Sensitivity analysis**: The correlation is robust to within-group
swaps (NMI vs ARI, NMI vs isolated_label_asw) but depends critically
on cLISI being last in both rankings. If cLISI turns out to have high
T1-delta (contrary to the ceiling saturation hypothesis), the
correlation could drop substantially. If cLISI turns out to have
high T3-rho (ceiling saturation is less severe than expected), the
correlation would also drop. My confidence in the positive sign is
moderate-high; my confidence in the specific magnitude (0.9) is
moderate.

**Alternative scenario (weakly positive)**: If ARI has the largest
delta but only moderate rho (Leiden resolution mismatch confounds
predictive validity), while silhouette has moderate delta but the
highest rho, then the top two swap between delta and rho rankings.
In this scenario, the correlation drops to ~0.6-0.7. Still positive,
still driven by cLISI at the bottom.

### Implications for the scIB overall score

The scIB overall score combines bio and batch metrics with weighting
(typically 0.6 bio + 0.4 batch). The positive delta-rho correlation
for bio metrics means the bio sub-score is moderately self-consistent:
metrics that separate trained from null also tend to predict downstream
performance. This is reassuring at first glance.

However, three concerns emerge:

1. **Dead-weight metrics dilute signal.** cLISI contributes equally to
   the bio sub-score despite having minimal discriminative AND predictive
   power. Its inclusion dilutes the bio sub-score with noise. The
   effective information in the bio sub-score is carried by 2-3 metrics
   (silhouette_label, ARI, possibly NMI), not all 5.

2. **Batch metrics are structurally uncoupled from bio prediction.**
   Batch metrics measure batch mixing, which is orthogonal to
   cell-type-recovery F1. Including batch metrics in the overall score
   adds a dimension that is discriminative (trained integrators mix
   batches; random projections also mix batches, for different reasons)
   but not predictive of biological performance. The overall score's
   correlation with ground-truth F1 is therefore LOWER than the best
   individual bio metric's correlation.

3. **Equal weighting is unjustified.** The implicit assumption of the
   scIB overall score is that each metric contributes independent
   information. If NMI and ARI are near-redundant (both Leiden-based,
   correlation > 0.9), and if cLISI is noise, then the "10 metrics"
   reduce to ~3-4 independent signals. Equal weighting of 10
   mathematically-distinct but informationally-redundant metrics creates
   a false sense of multi-dimensional validation.

**Bottom line**: The positive delta-rho correlation (predicted ~0.9) is
not a vindication of the suite. It reflects the fact that ceiling-
saturated metrics are bad at EVERYTHING (both discrimination and
prediction). The informationally useful portion of the scIB bio score is
a proper subset of its metrics, and the overall score conflates this
signal with batch-metric noise and dead-weight bio metrics.

---

## Summary of all predictions

### Test 4: Noise Monotonicity

| Metric | Verdict | Confidence | Key mechanism |
|--------|---------|------------|---------------|
| nmi_leiden | FAIL | High | Discrete Leiden partitions |
| ari_leiden | FAIL | High | Discrete Leiden partitions |
| silhouette_label | PASS | High | Continuous geometric degradation |
| clisi | PASS | Med-High | Cell average smooths kNN discreteness |
| isolated_label_asw | PASS | Med-High | Same as silhouette on subset |
| silhouette_batch | PASS | High | Continuous geometric degradation |
| ilisi | PASS | Med-High | Noise monotonically improves mixing |
| kbet | FAIL | Medium | Chi-squared threshold effects |
| graph_connectivity | PASS | Med-High | Noise increases graph density |
| pcr_comparison | PASS | High | Noise dilutes batch variance |

**Prediction: 3 FAIL, 7 PASS.**

### Test 5: Hyperparameter Ranking Sensitivity

| Metric | Predicted tau | Winner changes? | Confidence |
|--------|--------------|-----------------|------------|
| nmi_leiden | 0.5-0.7 | Yes (2/4 tissues) | Medium |
| ari_leiden | 0.4-0.6 | Yes (2-3/4 tissues) | Medium |
| clisi | 0.3-0.6 | Possibly | Low-Medium |
| ilisi | 0.2-0.5 | Yes | Medium |
| kbet | 0.1-0.4 | Yes (3-4/4 tissues) | Medium-High |
| graph_connectivity | 0.3-0.6 | No | Medium |

**Prediction: kBET is least stable (tau ~ 0.1-0.4); graph_connectivity
is most stable (but ceiling-degenerate at high k). At least 3 of 6
metrics show winner changes across hyperparameter settings.**

### Test 6: Delta-vs-Rho Dissociation (bio metrics only, n=5)

| Quantity | Prediction |
|----------|-----------|
| Spearman rho (T1-delta vs T3-rho) | +0.7 to +1.0 (point estimate: +0.9) |
| Sign | Positive |
| Driver | cLISI anchors bottom of both rankings (ceiling saturation) |
| Implication | Suite is self-consistent but only because broken metrics are broken at everything |

### Meta-predictions (cross-test consistency checks)

1. **Metrics that FAIL Test 4 should have LOWER tau in Test 5.** The
   same property (discrete/thresholded computation) that causes
   non-monotonicity under noise should cause ranking instability under
   hyperparameter changes. Predicted ordering of instability:
   kBET > ARI > NMI > iLISI > cLISI > graph_connectivity. This ordering
   should be consistent between Test 4 failure rates and Test 5 tau
   values.

2. **Metrics with low T1-delta in Test 6 should have high ceiling
   saturation detectable in Test 4.** Specifically, cLISI's noise
   response curve should show a long plateau near maximum followed by a
   sharp drop. This "hockey stick" shape is the noise-domain signature
   of ceiling saturation that drives low T1-delta.

3. **If kBET fails Test 4 AND has the lowest tau in Test 5, this
   constitutes independent evidence of construct-validity failure.** A
   metric that is both noise-non-monotonic and hyperparameter-unstable
   cannot be trusted as a component of an aggregate score.
