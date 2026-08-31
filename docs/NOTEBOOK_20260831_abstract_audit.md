# Notebook — 2026-08-31 — the transfer-metrics table, and five defects in the abstract

Two unrelated things, both found while checking every number in `paper_d_v38.tex`
against its result files before writing a conference abstract.

## 1. Table 4 does reproduce — recorded independently

Found twice on the same day from different directions, and already recorded as a
correction banner on `NOTEBOOK_20260825_table_reproduction.md` and its review packet.
Restated here only so this entry stands alone:
`scripts/compute_exp15_table.py` reads exp15 (69 rows, three models, no UCE), while
the manuscript's table reports exp16 (92 rows, four models). Recomputed from
`exp16_summary.json`, all twelve rows match `paper_d_v38.tex` to within 0.0005,
including the three RCS rows the 08-25 entry flagged as crossing 0.05.

The `json.dump` recommendation from that entry still stands and is still unapplied:
`compute_exp15_table.py` ends in `print` and writes nothing.

## 2. Five defects in v38, fixed in v39

Found by checking each abstract number against its primary source. None changes a
result; three change what the paper claims.

| # | location | was | is |
|---|---|---|---|
| 1 | abstract, panel | "23 human tissues and 92 embedding conditions" | 25 tissues, 104 conditions |
| 2 | abstract, RCS bound | "$\rho \leq 0.25$" | $\rho \leq 0.32$ |
| 3 | abstract, PAD | "pairwise discrimination accuracy" | proxy A-distance, and it fails rather than predicting marginally |
| 4 | Sec. null saturation | "$d = 50$, $k = 16$ ... 0.772" | 0.759 |
| 5 | Discussion | "sharpens that observation" | "extends that observation to" |

**1.** The abstract set up the 92-condition transfer-aware panel and then reported
CKA's partial $\rho = 0.53$, which comes from the other panel — Table 1 is $n = 104$,
25 tissues. The SCC sentence now names its own panel so the two do not merge.

**2.** RCS (combined) is $+0.315$ and RCS (PCA-10) is $+0.284$, both above 0.25. The
Discussion already said $\rho \in [0.18, 0.32]$; only the abstract was wrong.

**3.** $\rho = 0.003$ is proxy A-distance (Table 5, line 583). "Pairwise" is a column
header in that table for pairwise model discrimination accuracy, a different
quantity, on which PAD scores 46.5%. No metric named "pairwise discrimination
accuracy" exists in either results table. The old wording also grouped PAD with the
RCS variants as predicting "only marginally", where it does not predict at all.

**4.** $1 - 1.06 \times 15 / 66 = 0.759$. The value 0.772 is what the formula returns
at $k = 15$. Confirmed against Monte Carlo (2,000 draws, $k = 16$, $d = 50$: 0.7591).
Table 2 in the same section is correct throughout.

**5.** House style: the paper does not grade its own moves.

## 3. Carried into the NECB 2026 poster abstract, submitted 2026-08-31

The submitted abstract closes:

> at d >= 512, CKA-based rankings carry no more information than ranking against
> random embeddings.

This contradicts its own second paragraph ("CKA retains ranking power, partial
rho = 0.53") and the paper's recommendation, which reads "the metric can rank but
cannot certify." The ranking/certification dissociation is the paper's contribution
and that sentence collapses it in the wrong direction. It also carries defect 3 above,
naming PAD as pairwise discrimination accuracy.

The abstract is submitted and stands. The poster must not repeat either.

## 4. Two smaller things, unfixed

`.zenodo.json` and `CITATION.cff` both describe **"Preflight Bio"** — metadata copied
from another repository. `.zenodo.json` feeds the DOI record.

On novelty, checked because a report described two papers as publishing closed forms
for the CKA null: **Chun et al., arXiv 2502.15104** is *Estimating Neural
Representation Alignment from Sparsely Sampled Inputs and Features*, a corrected CKA
estimator for finite stimulus and neuron sampling. **Gröger et al., arXiv 2602.14486**
is *Revisiting the Platonic Representation Hypothesis: An Aristotelian View*. Both
identifiers resolve and both are worth citing as related work on CKA under limited
sampling. Neither publishes $\mathbb{E}[\mathrm{CKA}] \approx 1 - 1.06(k-1)/(d+k)$ or
an equivalent for the $(k, d)$ centroid regime.
