# Pre-registration: scGraph as a dimensionality-independent alternative to CKA

**Version**: 1

## Scientific question

Cell-type centroid CKA suffers a dimensionality-dependent null floor: E[CKA] = 1 - 1.06*(k-1)/(d+k), where k is the number of shared cell types and d is the embedding dimension. At d=512, k=10, the null floor is 0.98 -- real embeddings cannot clear it regardless of quality. scGraph metrics (Wang et al. 2026, Nat Biotech) evaluate relational structure between cell types via k-by-k relationship matrices (centroid distance, kNN overlap, weighted affinity), then correlate the upper triangular entries across assays. Because these metrics operate on k-by-k matrices rather than d-dimensional embeddings directly, they should be immune to the CKA null floor. This experiment tests that claim.

## Design

### Data

- **Tissues**: All tissues discoverable from Census 2023-12-15 with >= 200 cells per assay and >= 8 shared cell types (~22 tissues)
- **Models**: geneformer (d=512), scgpt (d=512), scvi (d=128)
- **Assay pairs**: 10x 3' v3 (source) vs Smart-seq2 (target), same as all prior experiments

### Metrics computed at each (tissue, model) condition

**scGraph metrics** (three Spearman correlations between source and target k-by-k matrices):
- Centroid distance correlation: Spearman rho between upper-triangular entries of Euclidean centroid distance matrices
- kNN overlap correlation: Spearman rho between fraction-of-cross-type-kNN-edges matrices (k=15 neighbors)
- Weighted affinity correlation: Spearman rho between mean-inverse-distance affinity matrices

**Comparison metrics**:
- Cell-type centroid CKA (linear CKA on k centroids)
- Transfer F1 (logistic regression, macro-averaged, shared cell types only)

All metrics computed at native d and at PCA-reduced d=50 (for models with native d > 50).

### What scGraph captures

Each scGraph metric measures whether the relational structure between cell types is preserved across assays. If cell types A and B are close in the source embedding, are they also close in the target? This is computed entirely in the k-by-k space of cell-type pairs, not in the d-dimensional embedding space. The intermediate computation (centroids, kNN) touches d-dimensional vectors, but the final correlation is over k*(k-1)/2 scalar pairs.

## Predictions

### P-1 (primary): scGraph is dimensionality-stable

scGraph metrics will NOT change substantially between native d and PCA d=50. For each (tissue, model) condition with native d > 50, define:

    delta = |scGraph(native_d) - scGraph(d=50)|

**Criterion**: The median delta across all conditions is < 0.10 for each of the three scGraph metrics.

**Rationale**: scGraph operates on k-by-k matrices. PCA changes the embedding coordinates but preserves the dominant neighborhood structure (it retains 85-95% of variance at d=50). The k-by-k matrices should be approximately invariant. CKA, by contrast, changes dramatically (the null floor shifts from 0.98 to 0.84 at k=10).

### P-2 (primary): scGraph discriminates embedding quality

scGraph centroid distance correlation will be higher for geneformer than for scgpt in >= 70% of tissues (paired comparison, same tissue).

**Rationale**: Geneformer consistently outperforms scGPT on transfer F1 across tissues. If scGraph captures real embedding quality (relational structure preserved across assays), it should track this ordering. 70% allows for tissue-specific reversals where scGPT happens to capture a particular tissue's cell-type geometry better.

### P-3 (secondary): scGraph correlates with functional quality

Across all (tissue, model) conditions, the Spearman correlation between scGraph centroid distance correlation and transfer F1 will be >= 0.30.

**Rationale**: CKA's correlation with F1 is confounded by the dimensionality-dependent floor (high-d embeddings get compressed to near 1.0 regardless of quality). scGraph should correlate with F1 at least modestly because both measure aspects of embedding quality. We set the bar at 0.30 (moderate) rather than requiring scGraph to match or beat CKA, since CKA's correlation with F1 across all conditions is inflated by between-model variance in d.

### P-4 (secondary): CKA at native d is below its null floor while scGraph is not

For geneformer (d=512) and scgpt (d=512), CKA at native d will fall below the analytic null floor in >= 90% of tissues. scGraph centroid distance correlation at native d will be >= 0.20 (above trivial correlation) in >= 80% of tissues.

**Rationale**: This is the core dissociation. CKA is uninformative at d=512 because the null floor compresses all values to near 1.0. scGraph should remain informative because it does not have this floor.

### P-5 (secondary): scvi is the natural control

scvi (d=128) will have a higher CKA clearance rate than geneformer (d=512) or scgpt (d=512) -- not because scvi embeddings are better, but because d=128 produces a lower null floor. scGraph will rank scvi relative to geneformer based on embedding quality (F1), not dimensionality.

**Criterion**: If scvi F1 > geneformer F1 in a given tissue, scGraph_centroid should also favor scvi; and vice versa. Agreement rate >= 60%.

## Analysis plan

1. For each (tissue, model), compute all five metrics at native d and at PCA d=50.
2. **P-1**: For each scGraph metric, compute delta = |native - d50| per condition. Report median and IQR. Test against 0.10 threshold.
3. **P-2**: Paired comparison of scGraph_centroid(geneformer) vs scGraph_centroid(scgpt) per tissue. Report fraction of tissues where geneformer > scgpt.
4. **P-3**: Spearman correlation of scGraph_centroid with F1 across all conditions. Report rho and p-value.
5. **P-4**: For d=512 models, report fraction of tissues where CKA < null_cka(k, 512). For scGraph_centroid, report fraction >= 0.20.
6. **P-5**: For tissues where both scvi and geneformer are available, compare scGraph ordering to F1 ordering. Report agreement rate.

## Decision rules

- **If P-1 and P-2 both hold**: scGraph is a dimensionality-independent alternative to CKA for cross-assay evaluation. Report as a confirmatory result alongside the analytic null floor and random-projection control.
- **If P-1 holds but P-2 fails**: scGraph is stable across d but does not track embedding quality. Report as a negative result: dimensionality-independence alone does not make a metric useful.
- **If P-1 fails**: scGraph is also affected by dimensionality, just via a different mechanism (kNN structure changes with d). Report and investigate whether the centroid-distance variant (which does not use kNN) is more stable than the kNN-based variants.
- **If P-1 and P-2 hold and P-3 also holds**: strongest result. scGraph is dimensionality-independent, discriminates quality, and predicts functional performance. Recommend scGraph as a complement to dimensionality-corrected CKA.

## Code

Script: `scripts/exp12_scgraph_eval.py`
Modal wrapper: `scripts/modal_scgraph_eval.py`
Output: `results/scgraph_eval/`

## Prior results motivating this experiment

- Analytic null: E[CKA] = 1 - 1.06*(k-1)/(d+k). At d=512, k=10: null = 0.98. At d=50, k=10: null = 0.84.
- PCA reduction (d=512 -> d=50): Geneformer 0/22 -> 18/22 tissues clear CKA null; scGPT 0/22 -> 8/22.
- Wang et al. 2026 (Nat Biotech): Islander games demonstrate scIB failures; scGraph proposed as relational-structure metric.
- Random-projection control (Experiment B): tests whether d-reduction itself (not denoising) clears the CKA floor.
- scGraph operates on k-by-k cell-type relationship matrices, which should be immune to the d-dependent CKA floor by construction.
