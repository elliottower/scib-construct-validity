# Shared Finding: Geometric Transportability Metrics Are Confounded by Structure-Preserving Maps

**Status**: Cross-domain convergent finding. Referenced by both preflight-bio and causal-rna/factorization-unified.
**Date crystallized**: 2026-07-12

---

## The finding (one sentence)

Geometric metrics computed on learned embeddings are confounded by properties that any random linear map preserves — subspace alignment, pairwise distances, and graph curvature are intrinsic to the data's covariance/kNN structure and survive projection through a zero-parameter random matrix.

## Evidence from two independent domains

### Domain 1: Single-cell foundation models (preflight-bio)

**Experiment**: Preregistered confound gate (v8, Exp 9). Compared Geneformer/scVI embeddings against two null baselines (random Gaussian projection 512d, untrained 2-layer MLP) on lung cross-assay pairs.

**Result**: v6 composite FAILED. Null baselines reproduced the same tier pattern as real models.

**Which metrics die**:
- **M1 (Grassmannian subspace alignment)**: Random projection preserves covariance → subspace alignment is high for any linear map. Null scored HIGHER than real (anti-discriminative).
- **M6 (Ollivier-Ricci curvature)**: JL preserves pairwise distances to (1±ε) at d=512, preserving kNN graph structure → curvature is invariant. Null scored HIGHER than real.

**Which metrics survive**:
- **M3 (direction stability)**: Learned embeddings have stable meaningful principal directions across domains. Random projections have unstable, arbitrary directions. Real=4.0 tiers, Null=2.5 tiers (+1.5 gap).
- **M4 (participation ratio)**: Learned embeddings have comparable compression across domains (similar PR). Random projections show more heterogeneous PR. Real=6.5, Null=5.0 (+1.5 gap). Caveat: dimension-sensitive at d>1000 due to normalization bug (collapse floor at PR<0.01).

### Domain 2: RNA foundation models (causal-rna / factorization-unified)

**Experiment**: Tested whether Grassmannian subspace distance d_g distinguishes learned RNA embeddings from null baselines.

**Result**: A zero-parameter 3-mer frequency baseline reproduces d_g — the metric captures sequence composition, not learned biology.

**Which metrics die**:
- **d_g (Grassmannian distance)**: Same mechanism as M1. Any linear map that preserves covariance structure shows low d_g between domains. The metric cannot distinguish "alignment because the model learned transferable biology" from "alignment because linear maps preserve covariance."

**Which metrics survive**:
- Direction stability (not yet formally tested in RNA domain but predicted by the same JL argument)

## The mechanism (why this happens)

**Johnson-Lindenstrauss lemma**: A random projection R ∈ ℝ^{d×D} (d ≥ O(log n / ε²)) preserves all pairwise distances to (1±ε). This guarantees:
1. Covariance/subspace structure is preserved (PCA directions are approximately invariant)
2. kNN graphs are approximately invariant (distance preservation → neighbor preservation)
3. Any metric that is a function of (covariance, pairwise distances, or graph topology) is confounded

**What random projections do NOT preserve**:
1. Direction stability across independent samples (random directions are unstable)
2. Structured spectral compression (random projections have ~Marchenko-Pastur spectra)
3. Classifier separability beyond what the raw data geometry provides

## Implications for the field

Most geometric evaluation methods for embeddings (alignment scores, manifold curvature, Procrustes distances, CKA on representations) measure properties of the data's intrinsic geometry that any dimensionality-preserving map retains. They are necessary conditions for a good embedding (a bad embedding could fail them) but not sufficient conditions (a random projection also passes them).

**What distinguishes learned from random**: Learned structure shows up in *direction semantics* (stable, interpretable principal components) and *spectral structure* (task-relevant compression, non-uniform eigenvalue distribution relative to data complexity). Not in subspace overlap or graph curvature.

## Paper framing options

### Option A: Two papers citing shared confound
- **Preflight paper** (Genome Biology): geometric diagnostic for scFMs, reports v6 failure honestly, validates v6b
- **RNA paper** (TRDNT submission): d_g falsification, mutation-sensitivity alternative
- Both cite: "Geometric metrics are confounded by JL preservation (Tower et al., 2026a/b)"

### Option B: One cross-domain methods paper
- **Title**: "Geometric transportability metrics are confounded by structure-preserving maps: evidence from RNA and single-cell foundation models"
- **Venue**: NeurIPS/ICML workshop on evaluation, or Bioinformatics methods
- **Strength**: Cross-domain evidence is stronger than either alone; the JL diagnosis is the contribution
- **Risk**: Neither domain gets full depth; need supplementary for both experiments

### Option C: Hybrid
- One methods contribution (the confound + what survives) as a short paper or workshop contribution
- Full papers for each domain that cite the methods contribution for the confound diagnosis

## Pre-commitments (written 2026-07-12, before results land)

### M4 PR diagnostic: what falsifies the artifact hypothesis

The "PR/d normalization floor" hypothesis predicts that UCE's raw
PR_normalized will be < 0.01 on liver, kidney, and brain (the tissues where
M4 = 0.0). Specifically: min(PR_source_normalized, PR_target_normalized)
should be below the 0.01 threshold.

**Falsification**: If UCE's min(PR_source, PR_target) normalized values are
> 0.015 on ANY tissue where M4 = 0.0 in Exp 8, then the floor is NOT firing
on PR/d normalization and something else causes the zero. In that case the
"dimensional artifact" explanation is wrong, and M4's floor has an unknown
cause that may be a deeper bug.

### v6b validation: what claim it supports (and does not)

The v6b validation tests **discriminative validity**: can v6b distinguish
learned embeddings from random projections (H-v6b.1) and from a structured
biological baseline (H-v6b.2, bag-of-genes)?

It does NOT test **predictive validity**: does a higher v6b score predict
better downstream task performance (cell-type classification, integration
benchmarks)? That is a separate experiment requiring correlation between
v6b scores and task-probe outcomes across many model×tissue cells.

A PASS on v6b validation licenses: "v6b scores foundation model embeddings
higher than both random and structured-biological baselines on held-out
tissues." It does NOT license: "v6b predicts transportability" or "higher
v6b means better transfer." The predictive claim requires Exp 8-style
correlation analysis on the held-out tissues, which is in the exploratory
section (4d) but is not confirmatory.

## Living references

- preflight-bio: `docs/preregistration_v8_a_plus.md` (Exp 9), `results/exp9_confound_controls/summary_20260712_133650.json`
- preflight-bio: `docs/preregistration_v6b_confound_free.md` (v6b revision)
- causal-rna: d_g confound analysis (3-mer baseline result)
- This doc: `preflight-bio/docs/SHARED_FINDING_geometric_confounds.md`
