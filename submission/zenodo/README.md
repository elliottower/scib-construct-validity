# Construct Validity Failure in Single-Cell Embedding Evaluation: Null Saturation Bounds and Source Classifier Confidence

**Author:** Elliot Tower (elliot@elliottower.ai)  
**License:** MIT

## Overview

This archive accompanies the manuscript evaluating construct validity of
single-cell embedding evaluation metrics. CKA on cell-type centroids exhibits
null saturation at foundation-model dimensionality. Source classifier confidence
(SCC) is proposed as an alternative, achieving Spearman rho = 0.67 against
cross-technology cell-type transfer F1 across 23 human tissues and 92 embedding
conditions in CELLxGENE Census.

## Archive Contents

```
docs/
  paper_d_v37.tex              # LaTeX source (current version)
  paper_d_v11.bib              # Bibliography
  preregistration_*.md         # All preregistration versions (v1-v11)
  frozen_prereg_*/             # SHA-256 frozen experiment specifications

src/scib_validity/             # pip-installable SCC and CKA null metrics
  metrics/scc.py               # Source classifier confidence
  metrics/cka_null.py          # CKA null saturation bounds

tests/                         # Test suite

scripts/                       # Experiment scripts

results/                       # Raw JSON results from all experiments

.github/workflows/publish.yml  # PyPI publish workflow

pyproject.toml                 # Package metadata (pip install scib-validity)
LICENSE                        # MIT License
README.md                     # Repository README
CITATION.cff                  # Citation metadata
```

## Reproducing the Results

```bash
pip install scib-validity

# Or from source
pip install -e .
python -m pytest tests/
```

## Preregistration

All experiment specifications were frozen via SHA-256 hashing before data
access. Frozen hashes are recorded in `docs/frozen_prereg_*/FROZEN_SUMMARY.json`.

## Citation

```bibtex
@software{tower2026scib_validity,
  author = {Tower, Elliot},
  title = {Construct Validity Failure in Single-Cell Embedding Evaluation},
  year = {2026},
  url = {https://github.com/elliottower/scib-construct-validity},
  license = {MIT}
}
```

## Data Availability

All experiments use the CELLxGENE Census (version 2023-12-15), a publicly
available single-cell atlas. No private or restricted data were used.
