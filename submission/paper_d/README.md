# A Construct-Validity Protocol for Geometric Evaluation of Single-Cell Foundation Model Embeddings

**Author:** Elliot Tower (elliot@elliottower.ai)  
**Date:** July 2026  
**License:** MIT

## Overview

This archive accompanies the manuscript "A Construct-Validity Protocol for
Geometric Evaluation of Single-Cell Foundation Model Embeddings." The paper
introduces four construct-validity tests that expose systematic failures in
scIB metrics when applied to single-cell foundation model evaluation:

1. **Null-model discrimination** — random projections and untrained encoders
   score comparably to trained models on geometric metrics
2. **Cross-dimensionality robustness** — model rankings permute under
   Johnson-Lindenstrauss projection, violating assumed stability
3. **Sign-correctness** — 46--67% pairwise inversion rates against
   ground-truth cell-type classification F1
4. **Noise dose-response** — bio-conservation metrics behave non-monotonically
   under controlled embedding degradation

## Archive Contents

```
paper/
  paper_d_v9.pdf               # Compiled manuscript
  paper_d_v9.tex               # LaTeX source
  paper_d_v9.bib               # Bibliography
  figures/                     # Publication figures (PDF + PNG)
    composite_distributions.*  # Test 1: null vs trained score distributions
    inversion_rates.*          # Test 3: pairwise inversion rates vs F1
    noise_dose_response.*      # Test 4: 2x3 panel, 6 metrics under noise
    noise_all_tissues.*        # Test 4: ARI across 4 tissues
    monotonicity_summary.*     # Test 4: non-monotonic fraction per metric
    hypothesis_outcomes.*      # Summary of all hypothesis outcomes
    module_heatmap.*           # Diagnostic module scores

results/
  summary.json                 # Experiment 10: scIB audit summary (raw scores, F1s)
  noise_dose_response.json     # Noise dose-response data (7 sigma levels x 24 conditions)
  extended_validity.json       # Extended validity checks
  hyperparameter_sensitivity.json  # Hyperparameter sensitivity analysis

scripts/
  exp10_scib_audit.py          # Main scIB audit experiment
  exp10_noise_dose_response.py # Noise dose-response experiment
  exp10_extended_validity.py   # Extended validity checks
  exp10_hyperparameter_sensitivity.py  # Sensitivity analysis
  modal_noise_parallel.py      # 24-way parallel Modal wrapper
  modal_noise_retry_and_aggregate.py   # Retry + aggregation
  generate_paper_d_figures.py  # Figure generation from result JSONs

preregistration/
  preregistration_v8_a_plus.md             # Latest preregistration
  preregistration_v6b_confound_free_V2.md  # Confound-free protocol
  frozen_prereg_scib_confirmatory/         # SHA-256 frozen specs

LICENSE                        # MIT License
CITATION.cff                   # Citation metadata
.zenodo.json                   # Zenodo deposit metadata
```

## Reproducing the Results

```bash
# Install dependencies
pip install preflight-bio

# Run the scIB audit (requires CELLxGENE Census access)
python scripts/exp10_scib_audit.py

# Run noise dose-response (24 conditions, ~45 min on Modal with 24 containers)
python scripts/modal_noise_parallel.py

# Generate figures from result JSONs
pip install matplotlib numpy
python scripts/generate_paper_d_figures.py
```

## Data Availability

All experiments use the CELLxGENE Census (version 2023-12-15), a publicly
available single-cell atlas. No private or restricted data were used.

## Related Work

This paper is part of the Preflight Bio project. The main platform paper
(geometric transportability diagnostics) is archived separately. The bracket
norm confound (geometric metrics confounded by neuron count) is available at
DOI 10.5281/zenodo.21093518.
