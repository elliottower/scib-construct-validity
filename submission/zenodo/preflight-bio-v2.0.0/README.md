# Construct-validity testing for geometric evaluation of single-cell foundation model embeddings

**Author:** Elliot Tower (elliot@elliottower.ai)
**Version:** 2.0.0 (July 2026)
**License:** MIT

## Overview

A three-check construct-validity protocol for geometric evaluation metrics,
applied first to five preregistered geometric modules (four eliminated) and then
to the field-standard scIB bio-conservation suite. The scIB audit finds 46-58%
pairwise inversion rates against cross-assay transfer F1. A preregistered
cross-tissue replication overturns this finding: scIB metrics positively predict
cross-tissue transfer (rho = +0.63, p = 0.0014), localizing the failure to the
cross-assay setting.

## What's new in v2.0.0

- Paper restructured (v15): scIB audit leads Results, module self-audit
  compressed with scorecard table, detail tables moved to appendix
- Cross-tissue replication (exp11): 4 contenders x 6 tissue pairs, both
  preregistered hypotheses overturned
- Seed stability check: headline results robust across 3 seeds
- Preregistration v11: cross-tissue ground truth hypotheses
- Prose fixes: abstract precision, superlative reduction, conclusion
  deduplicated from abstract

## Archive Contents

```
paper/
  paper_d_v15.tex              # Restructured manuscript (LaTeX source)
  paper_d_v11.bib              # Bibliography

preregistration/
  preregistration_v7.md        # Original module preregistration
  preregistration_v8_a_plus.md # scIB audit preregistration
  preregistration_v11_cross_tissue_ground_truth.md  # Cross-tissue prereg

results/
  exp10_scib_audit/            # scIB audit: 10 metrics, 6 embeddings, 4 tissues
    summary.json               # Main scIB audit results
    extended_validity.json     # Extended validity analysis
    hyperparameter_sensitivity.json  # Confirmatory test 1
    noise_dose_response.json   # Confirmatory test 2
  exp11_cross_tissue_validity/ # Cross-tissue replication
    exp11_cross_tissue_validity.json
  exp11_seed_stability/        # Seed robustness check
    seed_20260802/
    seed_20260803/
  *.json                       # v1.0.0 results (experiments 0-7)

scripts/                       # All experiment scripts (exp0-exp11)

src/preflight/                 # Source code for the scorer
  core/                        # Core runner, preregistration, modules
  extractors/                  # Data loaders (CELLxGENE Census, AnnData)
  modules/                     # Module implementations
  transfer/                    # Transfer evaluation harness

tests/                         # Test suite
```

## Reproducing the Results

```bash
pip install -e .

# Run scIB audit (requires CELLxGENE Census access + scib-metrics)
python scripts/exp10_scib_audit.py

# Run cross-tissue replication
python scripts/exp11_cross_tissue_validity.py

# Run confirmatory tests
python scripts/exp10_hyperparameter_sensitivity.py
python scripts/exp10_noise_dose_response.py
```

## Preregistration

All scorer code and hypotheses are frozen via SHA-256 hashing before data access:
- v7: Original 5-module preregistration (26 hypotheses)
- v8a+: scIB audit preregistration (metric routing, Check 1 predictions)
- v11: Cross-tissue replication (H11.1 inversion boundary, H11.2 composite sign)

## Citation

```bibtex
@article{tower2026construct,
  author = {Tower, Elliot},
  title = {Construct-validity testing reveals that {scIB} bio-conservation
           metrics predict cross-tissue transfer but invert cross-assay
           rankings at chance},
  year = {2026},
  url = {https://github.com/elliottower/preflight-bio},
  note = {Preprint, v2.0.0}
}
```

## Data Availability

All experiments use the CELLxGENE Census (version 2023-12-15), a publicly
available single-cell atlas. No private or restricted data were used.
