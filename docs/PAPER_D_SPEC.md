# Paper D Spec: Geometric Transportability Metrics Fail Against Structured Baselines

**Status**: Writing from scratch (not editing whitepaper_v4.tex)
**Voice target**: bracket-norm paper (prose-heavy, few headers, claim-first, detective-story flow)
**File**: `docs/paper_d_v1.tex`

## Thesis (inverted from whitepaper_v4)

Five geometric metrics proposed for evaluating single-cell foundation model embeddings each fail a distinct construct-validity check. No composite score predicts downstream transfer performance. The contribution is the diagnosis: identifying *why* each metric fails, yielding a construct-validity checklist for future geometric evaluation proposals.

## Source material

- `whitepaper_v4.tex`: experiments, tables, module definitions (OBSOLETE conclusion)
- `SHARED_FINDING_geometric_confounds_v5.md`: authoritative framing, full scorecard
- `findings_v6b_validation_20260712.md`: M4 confirmed artifact, v6b killed
- `results/v6b_validation` and `results/m4_pr_diagnostic`: raw numbers
- `bracket-norm/paper/paper_v20.tex`: voice model

## Structure

1. **Abstract** (one dense paragraph): Built five metrics, preregistered, each fails for an identifiable reason, probes are the only signal, contribution is the diagnosis.

2. **Introduction** (2-3 flowing paragraphs): scFM deployment gap, geometric diagnostics seem appealing, we built and preregistered one, the result is negative but informative.

3. **Results** (subsections with claim-as-title, bracket-norm style):
   - The composite separates obvious shift from controls (positive start)
   - Structured baselines eat the geometric signal (the turn)
   - Five metrics, five distinct failure modes (the contribution):
     - M1: JL confound + anti-prediction (rho=-0.76 within 512d)
     - M6: JL confound (random matches real)
     - M4: normalization artifact (PR/d floor at high d)
     - M2: anti-predicts transfer (rho=-0.58 vs kNN F1)
     - M3: sole survivor but underpowered (+0.30 [-0.34, +0.80] at n=12)
   - Transportability tiers invert representation quality (dissociation table)
   - Downstream probes are the only reliable predictor

4. **Methods** (concise):
   - Data: CellxGene Census, cross-assay pairs, 5 models
   - Module definitions (formulas, one paragraph each)
   - Construct-validity tests (null-model discrimination, BoG gate, sign-correctness)
   - Preregistration: SHA-256 chain, frozen scorer

5. **Discussion** (one continuous section, no subsections):
   - Connects to JL literature
   - Cross-domain convergence (RNA d_g, bracket-norm neuron count)
   - What geometric metrics actually measure (raw-data covariance structure)
   - Implications: construct validity before deployment
   - Limitations: Census only, 5 models, M3 underpowered
   - The BoG problem: embarrassingly simple baselines are the hardest test

## Key constraints

- NEVER frame as "preflight works" (that's the obsolete conclusion)
- Tables embedded in narrative flow (not in separate Tables section)
- No bullet lists in the paper body
- Numbers woven into prose, not announced
- Each result flows into the next (detective story, not enumeration)
- Honest negative: "we built X, it failed, here's why, that's the contribution"
- Cross-domain evidence strengthens the generality claim
