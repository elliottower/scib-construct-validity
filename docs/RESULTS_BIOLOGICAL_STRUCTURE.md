# Results: Biological-Structure Metrics Predict Cross-Assay Transfer

Pre-registration: `PREREGISTRATION_BIOLOGICAL_STRUCTURE_V2.md`
SHA-256: `b018262f2f4d056eaca638058b38af0c14ef3f5521741a482c26c24376cd40b9`

## Setup

**Outcome:** Cross-assay transfer F1 (macro-averaged). Logistic regression
trained on 10x 3' v3 embeddings, evaluated on 10x 5' embeddings, same tissue.
CellxGene Census v2023-12-15.

**Scale:** 25 tissues × up to 8 models = 154 total conditions, 104 contenders
(excluding random_projection and untrained_encoder baselines).

**Tissues (auto-discovered, >= 200 cells/assay, >= 8 shared cell types):**
anterior tongue, aorta, bladder, blood, bone marrow, decidua, endocrine pancreas,
inguinal lymph node, large intestine, left frontal lobe, liver, lung, lymph node,
mammary gland, pelvic diaphragm muscle, muscle tissue, rectus abdominis, right
frontal lobe, right temporal lobe, skin of abdomen, skin of chest, small intestine,
spleen, thymus, trachea.

**Models:** geneformer, scvi, scgpt, geneformer_v2_104m, geneformer_v2_316m,
bog_pca_512, random_projection, untrained_encoder.

## Result 1: Raw geometry anti-predicts transfer (V2, 4 tissues)

Ten geometric/transportability metrics tested against cross-assay transfer F1
across 32 conditions (8 models × 4 tissues).

| Metric | Module | rho | p | Sign |
|---|---|---|---|---|
| Chordal distance | Grassmannian | +0.418 | 0.017 | **WRONG** (expected -) |
| Subspace overlap | Grassmannian | -0.390 | 0.028 | **WRONG** (expected +) |
| Geodesic distance | Grassmannian | +0.380 | 0.032 | **WRONG** (expected -) |
| Domain classifier AUC | Domain shift | +0.365 | 0.043 | **WRONG** (expected -) |
| Sliced Wasserstein | Domain shift | -0.087 | 0.634 | null |
| MMD | Domain shift | +0.044 | 0.814 | null |
| Direction stability | Stability | +0.043 | 0.814 | null |
| Ollivier-Ricci curvature | Curvature | +0.039 | 0.832 | null |
| Centroid distance | Domain shift | -0.030 | 0.871 | null |
| Proxy A-distance | Domain shift | +0.028 | 0.882 | null |

Four metrics reach significance — all predict in the wrong direction. Six are null.
Zero out of ten predict cross-assay transfer correctly.

**Interpretation:** Geometric metrics confound model expressiveness with
transferability. Good models learn assay-specific representations (large subspace
divergence) while preserving cell-type identity (good transfer). The correlation
between geometric divergence and F1 is positive because both are driven by model
quality, not because divergence causes transfer.

## Result 2: scIB bio-conservation metrics fail noise monotonicity

All five bio-conservation metrics fail to decrease monotonically when Gaussian noise
is added to embeddings (7 noise levels, σ = 0.01 to 2.0, 24 conditions).

| Metric | Monotonic conditions | Verdict |
|---|---|---|
| ARI (Leiden) | 2 / 24 | **FAIL** |
| Graph connectivity | 2 / 24 | **FAIL** |
| NMI (Leiden) | 6 / 24 | **FAIL** |
| cLISI | 13 / 24 | **FAIL** |
| Isolated label ASW | 15 / 24 | **FAIL** |

Batch-correction metrics pass (iLISI 23/24, PCR 23/24, silhouette batch 23/24).

## Result 3: Cell-type structural similarity predicts transfer (V3b, 25 tissues)

Primary analysis: partial Spearman (controlling embedding dimensionality d),
contender models only, tissue-stratified permutation test.

| Metric | Role | partial rho | p (tissue-strat.) | 95% CI (tissue-block) |
|---|---|---|---|---|
| **Procrustes similarity** | Primary | **+0.657** | **< 1e-4** | [0.54, 0.72] |
| **Cell-type CKA** | Primary | **+0.536** | **< 1e-4** | [0.44, 0.66] |
| kNN purity | Pos. control | +0.518 | < 1e-4 | (near-tautological with F1) |

Both primary hypotheses confirmed (HB1, HB2). Concordance (HB4): both CKA and
Procrustes show positive partial-rho. Results survive all pre-registered controls:
dimensionality correction, baseline exclusion, tissue-stratified permutation,
tissue-block bootstrap.

## The dissociation

The same models that ANTI-predict transfer on raw geometry (Grassmannian distance
rho = +0.38, wrong sign) PREDICT transfer on cell-type structure (Procrustes
partial-rho = +0.66, correct sign). This dissociation shows the evaluation gap is
not about statistical power — the signal is strong in both directions. The metrics
are measuring different things: raw subspace geometry captures model expressiveness,
while cell-type structural alignment captures the biological invariant that
determines generalization.

## Implications for the paper

1. scIB bio-conservation metrics lack construct validity for cross-assay evaluation
   (noise monotonicity failure).
2. Geometric transportability metrics anti-predict cross-assay transfer (confound
   with model expressiveness).
3. Cell-type structural metrics (Procrustes, CKA on centroids) predict correctly and
   strongly, pointing toward what a valid cross-assay evaluation metric should measure.
4. The paper moves from pure negative ("metrics are broken") to constructive
   ("here's what works and why").

## Result 4: External validation (in progress)

Pre-registration: `PREREGISTRATION_EXTERNAL_VALIDATION.md`
SHA-256: `46aff40fdd13bbe40f102505857866c2099f7167e977419bf9ffa2819d0d461a`

Testing CKA and Procrustes on 3 completely independent datasets:

| Dataset | Technologies | Tech pairs | Source |
|---|---|---|---|
| Pancreas (Luecken et al.) | CEL-seq, CEL-seq2, Smart-seq2, inDrop, Fluidigm C1, SMARTER-seq | 15 | FigShare |
| Tabula Sapiens | 10x vs Smart-seq2 per organ | ~15 organs | CZ CELLxGENE Census |
| PBMC (Ding et al.) | 10x v2, 10x v3, Drop-seq, inDrops, Seq-Well, CEL-Seq2, Smart-seq2 | 21 | GEO GSE132044 |

Embedding models: PCA (all genes / HVG, d=50 / d=200), random projection, untrained
encoder. Field baselines compared: silhouette, MMD, domain classifier AUC.

**Status:** Launched, awaiting results.
