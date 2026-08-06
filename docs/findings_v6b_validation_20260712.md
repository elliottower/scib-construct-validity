# Findings: M4 PR Diagnostic + v6b Validation (2026-07-12)

**Prereg**: `docs/preregistration_v6b_confound_free_V2.md` (frozen at commit 05b42c4)
**Scripts**: `scripts/exp9b_m4_raw_pr_diagnostic.py`, `scripts/exp_v6b_validation.py`
**Data source**: CellxGene Census 2023-12-15, cross-assay pairs (10x 3' v3 vs Smart-seq2), zero donor overlap

---

## 1. M4 PR Diagnostic: Dimensional Artifact Confirmed

**Question**: Why does UCE (d=1280) score M4=0 on liver, kidney, and brain while passing on lung?

**Pre-committed falsification criterion**: If UCE's min(PR_source, PR_target) normalized > 0.015 on any tissue where M4=0, the "PR/d normalization floor" hypothesis is wrong.

### Results

| Tissue | Model | d | PR_norm (source) | PR_norm (target) | Floor fires? | M4 score |
|--------|-------|------|-----------------|-----------------|--------------|----------|
| Lung | UCE | 1280 | 0.0182 | 0.0156 | Neither | 0.857 |
| Liver | UCE | 1280 | 0.0055 | 0.0075 | Both | 0.0 |
| Kidney | UCE | 1280 | 0.0103 | 0.0062 | Target | 0.0 |
| Brain | UCE | 1280 | 0.0118 | 0.0085 | Target | 0.0 |

For comparison, 512d models never trigger the floor:

| Tissue | Model | d | min(PR_norm) | Floor? | M4 |
|--------|-------|-----|--------------|--------|------|
| Lung | Geneformer | 512 | 0.037 | No | 0.821 |
| Lung | scGPT | 512 | 0.019 | No | 0.600 |
| Kidney | Geneformer | 512 | 0.020 | No | 0.776 |
| Brain | scGPT | 512 | 0.016 | No | 0.935 |

And 50d models are far above:

| Tissue | Model | d | min(PR_norm) | M4 |
|--------|-------|----|--------------|----|
| Lung | scVI | 50 | 0.171 | 0.852 |
| Kidney | scVI | 50 | 0.147 | 0.905 |

### Verdict

**Falsification criterion NOT met.** Every tissue where UCE has M4=0 has min(PR_norm) well below 0.015. The mechanism is clear: PR_normalized = PR_raw / d, and at d=1280 the effective dimensionality (~7-20 raw) divides to ~0.005-0.015, landing near or below the 0.01 threshold. At d=512 the same raw PR values stay above the threshold. At d=50 they're far above.

**Conclusion**: M4's floor is a normalization artifact that penalizes high-dimensional embeddings regardless of their actual spectral structure. The decision to exclude M4 from v6b was correct.

---

## 2. v6b Validation: H-v6b.2 FAILS on Liver

**Composite**: M2 (domain shift, weight=1.0) + M3 (direction stability, weight=2.0)
**Hypotheses**:
- H-v6b.1 (floor): mean(real) > mean(random), bootstrap 95% CI excludes 0
- H-v6b.2 (primary): mean(real) > BoG, bootstrap 95% CI excludes 0

### Liver Results

| Model | Type | v6b score | Tier |
|-------|------|-----------|------|
| random_projection_512d | random null | 0.050 | T1 |
| untrained_encoder_512d | random null | 0.061 | T1 |
| bag_of_genes_pca512 | structured null | 0.303 | T3 |
| scgpt (d=512) | real | 0.279 | T3 |
| geneformer (d=512) | real | 0.323 | T3 |
| scvi (d=50) | real | 0.489 | T4 |

**H-v6b.1** (floor): delta = 0.308, CI = [0.224, 0.434] --- **PASS**
**H-v6b.2** (primary): delta = 0.061, CI = [-0.024, 0.186] --- **FAIL**

### Kidney Results

| Model | Type | v6b score | Tier |
|-------|------|-----------|------|
| random_projection_512d | random null | 0.076 | T1 |
| untrained_encoder_512d | random null | 0.060 | T1 |
| bag_of_genes_pca512 | structured null | 0.322 | T3 |
| geneformer (d=512) | real | 0.230 | T2 |
| scgpt (d=512) | real | 0.267 | T3 |
| scvi (d=50) | real | 0.427 | T4 |

**H-v6b.1** (floor): delta = 0.240, CI = [0.163, 0.359] --- **PASS**
**H-v6b.2** (primary): delta = **-0.014**, CI = [-0.092, 0.105] --- **FAIL** (point estimate negative)

### Interpretation

The composite easily beats random projections (the floor test passes with room to spare). But it cannot reliably distinguish real foundation model embeddings from bag-of-genes PCA. scGPT actually scores *below* BoG (0.279 vs 0.303), Geneformer barely edges above (0.323 vs 0.303).

scVI (0.489 liver, 0.427 kidney) is the only model that clearly beats BoG on both tissues, but scVI operates at d=50 — and Result 1 just demonstrated that this composite's behavior is dimension-sensitive. scVI beating BoG may be the same d-effect (favorable normalization at low d), not superior biology. **scVI cannot be cited as evidence that real embeddings beat BoG without reintroducing the dimensionality confound diagnosed in Result 1.**

Kidney is actually worse than liver: the mean real score (0.308) is *below* BoG (0.322), giving a negative point estimate. Both 512d real models (Geneformer 0.230, scGPT 0.267) score below BoG (0.322) on kidney.

The honest reading: M2+M3 at weights (1.0, 2.0) measures something that log1p + PCA already provides at matched dimensionality. The structured null eats the signal on both held-out tissues.

### Kill condition status

The prereg states: "Pass = all 4 tests (2 tissues x 2 hypotheses). Kill = any CI includes 0."

Both tissues fail H-v6b.2. Kidney is worse than liver (negative point estimate). **Kill is clean and unambiguous across both held-out tissues.**

| Tissue | H-v6b.1 (floor) | H-v6b.2 (primary) | Verdict |
|--------|-----------------|-------------------|---------|
| Liver | PASS (CI [0.224, 0.434]) | FAIL (CI [-0.024, 0.186]) | KILL |
| Kidney | PASS (CI [0.163, 0.359]) | FAIL (CI [-0.092, 0.105]) | KILL |

---

## 3. Consolidated Finding

Results 1 and 2 tell one story: **geometric preflight scores fail against a strong structured baseline, for reasons that are mechanical, not biological.**

M4 fails by PR/d normalization — the 0.01 threshold penalizes high-dimensional embeddings regardless of spectral structure. M2+M3 fails because bag-of-genes PCA already encodes what these modules measure (domain overlap and direction stability). Where a model appears to beat BoG (scVI at d=50), the dimensionality confound diagnosed in Result 1 cannot be ruled out.

The honest headline: *learned-embedding geometry does not beat bag-of-genes PCA for cross-assay transportability at matched dimensionality, and where it appears to, dimensionality confounds the comparison.*

This is the same pattern as the JL confound finding (documented in `SHARED_FINDING_geometric_confounds.md`): structured baselines eat the signal that geometric metrics claim to detect. BoG-PCA here plays exactly the role that the random-projection null plays for M1/M6 — it's the boring control that matches the fancy model.

### What survives

- The floor test (H-v6b.1) passes easily: real embeddings have structure beyond random projections. This is necessary but not sufficient.
- M3 (direction stability) was identified in the confound analysis as the one metric random projections cannot reproduce. Whether M3 *alone* can beat BoG is an open question — diagnosable within the current prereg as a module-level breakdown, but any composite built around M3 alone requires fresh preregistration before confirmatory testing.

### For the paper

Lock v6b as a reported negative. The construct-validity story for preflight-bio is: M1 anti-discriminates (random scores higher), M6 is JL-confounded (random matches real), M4 is a normalization artifact, M2+M3 cannot beat BoG. M3 alone is the only candidate survivor, and it's underpowered as a standalone diagnostic pending further work.

### What is NOT licensed by these results

- **Rebalancing weights or dropping modules to find a passing composite is a post-hoc endpoint switch on a pre-registered kill.** The prereg discipline this program is built on requires that any revised composite (v6c, M3-only, etc.) be pre-registered fresh before testing. Running multiple composites until one beats BoG and reporting the winner is exactly the practice we're testing against.
- **Citing scVI as the counterexample** that "real can beat BoG" reintroduces the d-confound from Result 1. It cannot be used as positive evidence without first ruling out the dimensional mechanism.

### Within-prereg diagnostic (does not require new prereg)

- **Break out M2 vs M3 separately against BoG** to identify which module BoG matches. This names the mechanism without changing the endpoint. If M2 alone fails but M3 alone passes, that's a within-composite diagnostic, not a new claim.

### Recommended path

**Accept the negative.** v6b as defined lacks discriminative validity against BoG on held-out tissue. This is itself an informative result: geometric transportability metrics (domain shift + direction stability) measure structure that log1p + PCA already provides.

---

## 4. Raw data locations

- M4 PR diagnostic: `results/m4_pr_diagnostic` (JSONL, one line per model x tissue)
- v6b validation (complete): `results/v6b_validation` (full JSON with both tissues, all scores, hypothesis tests, overall verdict)
