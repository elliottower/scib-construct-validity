# Deviation Log: Biological-Structure Metrics (V3b)

Pre-registration: `PREREGISTRATION_BIOLOGICAL_STRUCTURE_V2.md`
SHA-256: `b018262f2f4d056eaca638058b38af0c14ef3f5521741a482c26c24376cd40b9`

All deviations from the frozen pre-registration are logged here.

---

## D1: HB3 (kNN purity) reporting

**Pre-registered as:** Primary hypothesis HB3 — "kNN purity has positive
partial Spearman correlation with transfer F1 (expected significant;
informative only if it FAILS, which would indicate a data quality problem)."

**Result:** Partial rho = +0.518, permutation p < 1e-4. Behaves as expected
(near-tautological with F1).

**Deviation:** Not reported as a primary result in the main text. Demoted
to a declared positive control reported in the supplementary, consistent
with the pre-registration's own language: "near-tautological with the F1
outcome... included as a positive control, not as independent evidence."

**Justification:** Reporting kNN purity alongside CKA and Procrustes as a
"primary" result would inflate the appearance of three independent
confirmations when only two are structurally independent. The
pre-registration itself anticipated this by labeling HB3 as a positive
control and stating it is "informative only if it FAILS."

**Status:** Result computed and saved in `summary.json` under
`positive_control.knn_purity`. Should be reported in the paper as a
declared positive control (supplementary or inline), not omitted entirely.

---

## D2: MIN_SHARED_TYPES changed from 3 to 8

**Pre-registered (V1):** MIN_SHARED_TYPES = 3

**Pre-registered (V2, operative):** MIN_SHARED_TYPES = 8, with documented
rationale: "raised from 3 to avoid unreliable small-matrix alignment in
CKA/Procrustes."

**Status:** Not a deviation from the operative pre-registration (V2), which
explicitly documents the change. V1 is superseded. No action needed, but
the paper's Methods should state MIN_SHARED_TYPES = 8.

---

## D3: Procrustes PCA reduction step

**Pre-registered:** "Procrustes similarity: 1 minus Procrustes disparity on
PCA-reduced cell-type centroids. Centroids projected to min(50, n_types-1, d)
dimensions via PCA before alignment."

**Implementation:** The script (`exp_biological_structure_v3.py`, lines
94-107) implements the PCA step exactly as specified:
```python
k = min(n_components, len(shared) - 1, src_centroids.shape[1])
pca = PCA(n_components=k, random_state=0)
src_reduced = pca.fit_transform(src_centroids)
tgt_reduced = PCA(n_components=k, random_state=0).fit_transform(tgt_centroids)
```

**Deviation:** The paper's Methods subsection describes Procrustes as
"orthogonal Procrustes on column-centered centroid matrices" without
mentioning the PCA reduction step. The implementation matches the
pre-registration; the paper description is incomplete.

**Fix:** Add to Methods: "Centroids are PCA-reduced to min(50, k-1, d)
dimensions before Procrustes alignment, where k is the number of shared
cell types."

---

## D4: Assay-pair discrepancy between sections

**scIB audit (sec:scib):** Source = 10x 3' v3 (EFO:0009922), Target =
Smart-seq2. 4 tissues, 6 embeddings, 24 conditions. Ground truth =
macro-averaged kNN F1.

**Biological-structure panel (sec:structure):** Source = 10x 3' v3
(EFO:0009922), Target = 10x 5' (EFO:0008931). 25 tissues, 8 models,
104 contenders. Ground truth = logistic-regression macro F1.

**Status:** Both assay pairs are correctly implemented per their respective
pre-registrations. The paper currently implies the CKA success and scIB
failure are on the same evaluation — they are not. The design differences
(assay pair, ground truth, panel size) must be stated explicitly.

**Fix:** Add a paragraph or table acknowledging the design asymmetry.
The strongest fix: recompute scIB metrics on the 25-tissue 10x 3' v3 →
10x 5' panel with logreg-F1 ground truth, to verify inversions persist
under identical conditions.

---

## D5: Ground-truth classifier mismatch

**scIB audit:** kNN classifier for ground-truth F1.

**Biological-structure panel + external validation:** Logistic regression
for ground-truth F1.

**Status:** The two sections use different ground-truth probes. The paper's
headline comparison ("scIB fails, CKA succeeds") conflates metric failure
with design differences. Should be disclosed; ideally resolved by running
both probes on both panels.
