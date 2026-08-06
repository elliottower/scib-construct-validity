# Genome Biology Submission TODO

This is a comprehensive map of what could be done, not a commitment to do all of it.
Phases 0-2 are probably necessary. Phases 3-3.5 would maximize acceptance odds.
Phases 5-6 are packaging/logistics. Pick and choose.

Target: Genome Biology (where scIB was published)
Submission type: Benchmark article -> Benchmarks v2.0 collection (or Methods if not maintaining)
Paper: "A Construct-Validity Protocol for Geometric Evaluation of Single-Cell Foundation Model Embeddings"
Current version: paper_d_v12.tex (includes cross-tissue replication)
APC: ~$5,690 | Acceptance rate: ~15% | Desk filter: ~14 days | Median to acceptance: ~9 months
Fallback: PLOS Computational Biology (~$2,500 APC, higher acceptance, less targeted audience)

Existing assets:
- submission/zenodo/ — v1.0.0 archive (v9 paper, exp10 results)
- submission/paper_d/ — v1.0.0 archive with latex source, preregistrations, scripts, results
- submission/genome-biology/ — empty scaffold (supplementary/ created)

---

## Phase 0: What's done

- [x] 3-check construct-validity protocol (null discrimination, cross-d robustness, sign-correctness)
- [x] 5 preregistered geometric modules tested (M1-M6, M5 dropped)
- [x] scIB audit: 10 metrics, 6 embeddings, 4 tissues (exp10)
- [x] Confirmatory test 1: hyperparameter sensitivity (all 6 predictions wrong)
- [x] Confirmatory test 2: noise dose-response (8/10 metrics non-monotonic)
- [x] Cross-tissue replication (exp11): 4 contenders x 6 tissue pairs = 24 conditions
  - Both H11.1 and H11.2 OVERTURNED — metrics work cross-tissue, fail cross-assay
  - Seed stability confirmed across 3 seeds
- [x] Paper v12 updated with cross-tissue results
- [x] Existing Zenodo archive (v1.0.0) with v9 paper + exp10 results

---

## Phase 0.5: Ship current state to Zenodo (can do NOW)

Publish what exists as v2.0.0 (v1.0.0 had v9 paper; this adds cross-tissue).

- [ ] Update submission/zenodo/.zenodo.json version to 2.0.0
- [ ] Copy paper_d_v12.tex into zenodo archive
- [ ] Compile paper_d_v12.pdf
- [ ] Copy exp11 results + seed stability results into archive
- [ ] Copy preregistration_v11_cross_tissue_ground_truth.md into archive
- [ ] Copy exp11 scripts (exp11_cross_tissue_validity.py, exp11_seed_stability.py)
- [ ] Build preflight-bio-v2.0.0.zip
- [ ] Build preflight-bio-v2.0.0.pdf
- [ ] Upload to Zenodo, get DOI
- [ ] Record DOI in paper

This gives you a citable, timestamped archive of everything done so far,
independent of whether you pursue the journal submission.

---

## Phase 1: Expand model panel (engineering, ~1 week)

Goal: Go from 4 contenders to 8-10. Unified panel across ALL experiments.

GB benchmark papers compare 10-16+ methods on 5-6+ datasets. The recently accepted
scIB-E paper had 16 methods on 6 datasets. With 8-10 models we get ~28-45 pairs per
tissue instead of 6, which powers the M3 question (paper says n ~ 85 needed).

### 1a. Fix embedding scripts (2-3h each)

Priority models (4 contenders minimum to add):

- [ ] **Nicheformer** — find/fix embedding extraction script
  - Check if Census has Nicheformer embeddings or if we need to run inference
  - If inference needed: find pretrained weights, write embedding script, test on toy data
  - Output: function that takes AnnData, returns (n_cells, d_model) embeddings

- [ ] **scPrint** — find/fix embedding extraction script
  - Same as above
  - scPrint may need specific preprocessing (gene tokenization)

- [ ] **CellPLM** — find/fix embedding extraction script
  - Same as above

- [ ] **scFoundation** — explicitly named in limitations as a gap
  - Large model, may need GPU for inference
  - Check if feasible within Modal compute budget

Stretch models (if time):

- [ ] **scBERT** — smaller model, should be straightforward
- [ ] **SATURN** — multi-species, interesting comparison

Also consider:

- [ ] **scVI at d=512** — retrain to match dimensionality of other models
  - Removes the d=50 vs d=512 mismatch that reviewers will flag
  - Straightforward, a few hours of compute

### 1b. Validate embeddings locally

- [ ] For each new model: run on a small Census slice (1 tissue, 500 cells)
- [ ] Verify embeddings are non-degenerate (not all zeros, reasonable variance)
- [ ] Verify dimensions match expectations
- [ ] Compute basic sanity: kNN probe F1 on within-tissue cell-type classification

---

## Phase 2: Blind preregistration for expanded panel

Goal: Pre-register the model expansion BEFORE seeing any results for the new models.

### 2a. Write the prereg

- [ ] Launch a FRESH agent (not a fork) with NO knowledge of existing results
- [ ] Agent knows only: model names, existing test framework, existing preregistrations
- [ ] Agent writes prereg specifying:
  - Expanded contender set (original 4 + new models)
  - Same tests applied to all: cross-assay inversions, cross-tissue inversions,
    noise dose-response, hyperparameter sensitivity
  - Hypotheses: do the existing findings (46-58% cross-assay inversions, 22% cross-tissue)
    hold with the expanded panel?
  - Pre-specified analysis: per-model inversion contributions, model-dropped sensitivity
  - Multiple testing correction for the expanded comparisons
- [ ] Freeze with SHA-256 hash BEFORE running any new model

### 2b. Hash and commit

- [ ] SHA-256 hash the prereg document
- [ ] SHA-256 hash all embedding scripts (new models)
- [ ] SHA-256 hash the experiment runner scripts
- [ ] Record hashes in the prereg

---

## Phase 3: Run unified experiments (~2-3 days compute)

Goal: Every test uses the SAME model panel. No more "these 4 here, those 5 there."

### 3a. Cross-assay (exp10 extension)

- [ ] Run scIB audit on new models: 4 tissues, 10x -> Smart-seq2
- [ ] Compute inversions with expanded contender set
- [ ] Compare: do inversion rates change with more models?

### 3b. Cross-tissue (exp11 extension)

- [ ] Run cross-tissue on new models: 6 tissue pairs, matched assay
- [ ] Compute inversions with expanded contender set
- [ ] Compare: does rho hold with more models?

### 3c. Noise dose-response (extension)

- [ ] Run noise dose-response on new models: 7 sigma levels x 4 tissues
- [ ] Check: do new models show same non-monotonicity pattern?

### 3d. Hyperparameter sensitivity (extension)

- [ ] Run sensitivity on new models: Leiden r x kNN k grid
- [ ] Check: ranking stability for new models

### 3e. Aggregate and compare

- [ ] Side-by-side: original panel results vs expanded panel results
- [ ] Per the prereg: report ALL results regardless of outcome
- [ ] If findings change with expanded panel: report honestly

---

## Phase 3.5: Second downstream ground truth (2-4 weeks)

GB reviewers will ask: "What if scIB metrics predict perturbation response even
though they fail at cross-assay cell-type transfer?" The paper acknowledges this
limitation but doesn't address it.

Options (pick at least 1):

- [ ] **Perturbation prediction** (most field-relevant)
  - Geneformer was trained partly for this
  - Use GEARS or scGEN perturbation benchmark data
  - Ground truth: perturbation response correlation
  - Apply same 3-check protocol to scIB vs perturbation prediction

- [ ] **Gene program recovery** (complementary)
  - Hotspot or topic modeling
  - Ground truth: known gene programs / pathways
  - Shows protocol applies beyond classification

- [ ] **Trajectory inference** (stretch)
  - More complex setup, less standardized ground truth
  - Lower priority

Even showing the protocol applied to ONE additional ground truth would satisfy
the "single ground truth" reviewer objection.

---

## Phase 4: Update paper (v13)

- [ ] Update all tables with expanded model panel
- [ ] Update all figures
- [ ] Update abstract numbers if they change
- [ ] Update limitations (model panel no longer a stated gap)
- [ ] Add model descriptions for new embeddings to Methods
- [ ] Re-run figure generation script

---

## Phase 5: Package for Genome Biology submission

### 5a. Zenodo archive (submission/zenodo/)

Structure:
```
submission/zenodo/
  preflight-bio.zip          # Full reproducibility archive
    scripts/                 # All experiment scripts
    results/                 # All result JSONs
    preregistration/         # All preregs with SHA-256 hashes
    src/                     # Source code
    pyproject.toml
    README.md
    LICENSE
    CITATION.cff
    .zenodo.json
  preflight-bio.pdf          # Compiled paper
```

- [ ] Update .zenodo.json with v2.0.0 metadata
- [ ] Include ALL results (exp10 + exp11 + expanded panel)
- [ ] Include ALL preregistrations (v6b, v8a+, v11, expanded panel)
- [ ] Include ALL scripts
- [ ] Compile PDF
- [ ] Create zip
- [ ] Upload to Zenodo, get DOI

### 5b. Genome Biology submission (submission/genome-biology/)

Structure:
```
submission/genome-biology/
  paper_d_v13.pdf            # Main manuscript PDF
  latex_source.zip           # LaTeX source (tex + bib + figures)
  cover_letter.tex           # Cover letter
  supplementary/
    supplementary_tables.pdf # Extended data tables
    supplementary_figures.pdf # Extended figures
```

- [ ] Write cover letter highlighting:
  - Evaluates Genome Biology's own published scIB suite
  - Preregistered protocol with SHA-256 frozen code
  - Cross-tissue falsification test that fired (rare in this field)
  - Practical recommendations for practitioners
  - All code and data public
- [ ] Compile main PDF
- [ ] Package LaTeX source zip
- [ ] Compile supplementary materials
- [ ] Check Genome Biology formatting requirements:
  - Abstract word limit
  - Figure count limits
  - Reference style
  - Data availability statement format
  - ORCID requirement
  - Software availability section

---

## Phase 6: Pre-submission checklist

- [ ] All preregistrations referenced and hashed
- [ ] All exploratory analyses labeled as such
- [ ] No existing claims changed by expanded panel (if they are, report honestly)
- [ ] Zenodo DOI in manuscript
- [ ] GitHub repo public (or ready to make public on acceptance)
- [ ] ORCID linked
- [ ] All co-author approvals (if any)
- [ ] Final read-through for AI-writing tells
- [ ] Final negation-contrast audit (grep for "not.*but")

---

## Stretch goals (if time)

- [ ] Second organism (mouse Census data) — addresses "is this human-specific?"
- [ ] Perturbation prediction as third ground truth — mentioned in limitations
- [ ] pip-installable protocol package — Genome Biology loves reusable software
- [ ] Interactive companion (Marimo notebook) for practitioners to run the 3 checks

---

## Timeline estimate

- Phase 1 (embedding scripts): 1 week
- Phase 2 (blind prereg): 1 day
- Phase 3 (experiments): 2-3 days compute, 1 day analysis
- Phase 4 (paper update): 2-3 days writing
- Phase 5 (packaging): 1 day
- Phase 6 (checklist): 1 day

Total: ~2.5 weeks from start to submission-ready
