# Notebook — 2026-08-25 — does Table 4 reproduce from the repository?

A single day's entry, recording one check and what it found. Nothing here changes the
manuscript; it records what was measured so a decision can be made from evidence rather than
from memory.

## What was run

`scripts/compute_exp15_table.py`, unmodified, from a clean working tree at `d8a2cd4`, against
the result files as committed. The script reads
`results/exp15_scc_stress_test/exp15_scc_stress_test/summary/{exp15_summary,exp15_all_rows}.json`,
pools across tissues and models, and prints the LaTeX rows for the transfer-metrics table.

It takes under a second. It reads JSON already on disk and computes Spearman correlations;
no embeddings, no Census download, no network.

## What it found

Every one of the twelve rows differs from `docs/paper_d_v38.tex`.

| metric | paper ρ | paper p | rerun ρ | rerun p | conclusion |
|---|---:|---:|---:|---:|---|
| SCC (LR) | +0.673 | <0.001 | +0.768 | <0.001 | unchanged |
| SCC (kNN) | +0.558 | <0.001 | +0.507 | <0.001 | unchanged |
| SCC (RF) | +0.569 | <0.001 | +0.587 | <0.001 | unchanged |
| SCC (SVM) | +0.548 | <0.001 | +0.517 | <0.001 | unchanged |
| MMD | +0.525 | <0.001 | +0.595 | <0.001 | unchanged |
| CCAL | +0.472 | <0.001 | +0.598 | <0.001 | unchanged |
| RCS (baseline) | +0.218 | 0.050 | +0.298 | 0.019 | **crosses 0.05** |
| RCS (PCA-10) | +0.284 | 0.009 | +0.269 | 0.028 | unchanged |
| RCS (trimmed) | +0.212 | 0.051 | +0.293 | 0.019 | **crosses 0.05** |
| RCS (normalized) | +0.179 | 0.096 | +0.286 | 0.021 | **crosses 0.05** |
| RCS (combined) | +0.315 | 0.004 | +0.316 | 0.014 | unchanged |
| PAD | +0.003 | 0.975 | +0.112 | 0.359 | unchanged (null both ways) |

Nine of twelve reach the same conclusion. Three cross the 0.05 threshold, all in the RCS
family and all toward significance.

## What did not explain it

Checked, each ruled out:

- **The script did not change.** `git log -- scripts/compute_exp15_table.py` returns one
  commit, `23235d4`, which added it.
- **The inputs did not change.** `git status results/` is clean; the files the script reads
  are exactly what is committed.
- **There is no randomness in the correlations.** No seed, no RNG in the script. Spearman ρ
  is determined by its input, so identical input gives identical output. The bootstrap
  intervals vary; the point estimates cannot.
- **The tissue count matches.** The script reports 23 tissues and the paper says 23. An
  earlier check against `compute_full_table.py` found 21, but that is a different script
  producing a different table, and was the wrong comparison.

What remains is that the manuscript's numbers were computed from inputs that are not the
ones in the repository. Everything was added in a single commit, so git holds no earlier
state to compare against.

## Where the numbers do survive

Session transcripts under `~/.claude/projects` hold the printed output of several runs. Both
sets appear: `SCC (LR) +0.673` (the manuscript) and `+0.768` (today), and the same for other
rows. The transcripts are the only record of the earlier run, because the script writes no
file — it ends in `print("LaTeX table rows:")`.

## What follows

The script prints and never writes. A value that exists only in stdout has no address: it
cannot be checked, and which run produced it cannot be established afterward. One line fixes
it for every future run:

```python
json.dump(all_stats, open(OUT / "exp15_table.json", "w"), indent=2)
```

Open questions are in `docs/REVIEW_PACKET_20260825_table_reproduction.md`.
