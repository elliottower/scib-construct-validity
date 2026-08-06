# Preflight Bio: Geometric Transportability Diagnostics for Single-Cell Foundation Models

**Author:** Elliot Tower (elliot@elliottower.ai)  
**Date:** July 2026  
**License:** MIT

## Overview

Preflight Bio is a geometric diagnostics platform that evaluates embedding
transportability before any downstream task is attempted. Given arbitrary source
and target embedding matrices, it computes five diagnostic modules—subspace
alignment on the Grassmannian, domain separability, direction stability,
participation ratio, and network curvature—and produces a composite
transportability tier. A sixth module (ecological bias) is computed when
site-level metadata is available.

All scorer code and hyperparameters are frozen via SHA-256 hashing before data
access, providing a verifiable audit trail for every analysis.

## Archive Contents

```
paper/
  whitepaper_v4.pdf          # Compiled manuscript (23 pages)
  whitepaper_v4.tex          # LaTeX source
  whitepaper_v4.bib          # Bibliography
  figures/                   # Publication figures (PDF + PNG)

src/preflight/               # Source code for the Preflight scorer
  core/                      # Core runner, preregistration, modules
  extractors/                # Data loaders (CELLxGENE Census, AnnData)
  modules/                   # Module implementations
  transfer/                  # Transfer evaluation harness
  cli.py                     # Command-line interface
  probes.py                  # Cell-type classification probes

scripts/                     # Experiment scripts (Experiments 0-7)

results/                     # Raw JSON results from all experiments
  composite_validation/      # Experiment 0: composite calibration
  bag_of_genes_baseline/     # Experiment 1: BoG baseline comparison
  sweep_v7/                  # Experiment 2: multi-model sweep
  sensitivity/               # Experiment 3: sample-size sensitivity
  metric_comparison/         # Experiment 4: RankMe/MMD/C2ST comparison
  cross_disease/             # Experiment 6: cross-disease (exploratory)
  cross_tissue/              # Experiment 5: cross-tissue transfer
  sweep/                     # Experiment 7: scGPT evaluation

preregistration/             # Frozen preregistration documents
  preregistration_v7.md      # Final preregistration (v7)
  frozen_prereg_v7/          # SHA-256 frozen experiment specifications
    FROZEN_SUMMARY.json      # Git commit, scorer SHAs, hyperparameters

tests/                       # Test suite

pyproject.toml               # Project metadata and dependencies
LICENSE                      # MIT License
CITATION.cff                 # Citation metadata
```

## Reproducing the Results

```bash
# Install dependencies
pip install -e .

# Run the scorer on a pair of embedding matrices
preflight score --source source.h5ad --target target.h5ad

# Run experiment scripts (requires CELLxGENE Census access)
python scripts/exp0_composite_validation.py
```

## Preregistration

All scorer code, hyperparameters, and dataset specifications were frozen via
SHA-256 hashing before data access. The frozen hashes are recorded in
`preregistration/frozen_prereg_v7/FROZEN_SUMMARY.json`. The preregistration
document specifying all 26 confirmatory hypotheses is at
`preregistration/preregistration_v7.md`.

## Citation

```bibtex
@software{tower2026preflight,
  author = {Tower, Elliot},
  title = {Preflight Bio: Geometric Transportability Diagnostics for Single-Cell Foundation Models},
  year = {2026},
  url = {https://github.com/elliottower/preflight-bio},
  license = {MIT}
}
```

## Data Availability

All experiments use the CELLxGENE Census (version 2023-12-15), a publicly
available single-cell atlas. No private or restricted data were used.
