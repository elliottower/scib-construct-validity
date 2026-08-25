# Review packet — Table 4 does not reproduce from the repository

For whoever picks this up next. The measurement is in
`docs/NOTEBOOK_20260825_table_reproduction.md`; this file is the decisions it needs.

## The situation in one paragraph

`scripts/compute_exp15_table.py`, run unmodified against the committed result files, produces
twelve values and all twelve differ from the table in `docs/paper_d_v38.tex`. The script has
one commit and has not changed. The inputs are tracked and clean. The correlations are
deterministic. So the manuscript's numbers came from inputs that were never committed, and
because everything landed in a single commit there is no earlier state in git to recover.

Nine of the twelve rows reach the same conclusion. Three cross 0.05, all in the RCS family
and all toward significance.

## What is not at risk

Worth stating first, because the headline reads worse than the situation.

- **The main result holds.** SCC, MMD and CCAL are strongly significant under both sets, and
  the ordering among them is unchanged.
- **PAD stays null.** `p = 0.975` in the manuscript, `p = 0.359` today. The argument that PAD
  fails to predict transfer survives, which is the claim the discussion leans on.
- **The tissue count is right.** 23 in both. An earlier concern about 21 came from checking
  the wrong script.

## What is at risk

**Section "Relational consistency scores: dimensionality-stable but weakly predictive."** On
the committed data, RCS baseline (`p = 0.019`), trimmed (`p = 0.019`) and normalized
(`p = 0.021`) all clear 0.05, where the manuscript reports `0.050`, `0.051` and `0.096`. Two
of those are borderline in the manuscript and one is not. "Weakly predictive" is harder to
write if three of five variants are significant.

## The decisions

**1. Which run is authoritative?**

Neither can be assumed. The manuscript's run cannot be reproduced and its inputs are not in
the repository; today's run is reproducible but may be computed from a subset or a superseded
version of the results. Answering this needs someone who remembers what was run, or an
examination of what the summary files contain against what the experiment was meant to cover.

**2. If today's run is authoritative:** regenerate the table, revise the RCS subsection to
match, and check whether anything in the abstract or discussion depends on the three rows
that move.

**3. If the manuscript's run is authoritative:** its inputs need to be found and committed,
or the paper needs a note that the deposited artifact yields different values. A submission
with a data-availability statement should not leave a reviewer to discover this.

**4. Independent of either:** add the `json.dump`. This recurred silently because the script
prints and writes nothing.

## Open questions

- Do `exp15_summary.json` and `exp15_all_rows.json` cover all 23 tissues and all model
  conditions the manuscript describes, or a subset? The row counts should be checked against
  what the experiment was specified to produce.
- Were the manuscript's numbers produced before or after the results were committed in
  `23235d4`? The session transcripts hold both value sets with timestamps and may settle it.
- Does any other table in the manuscript come from a script that only prints? A pass over
  `scripts/` for `json.dump` against `print` would show which tables are addressable and
  which are not.
- Is the same true of the figures?

## How to reproduce this check

```bash
uv run --with numpy --with scipy python scripts/compute_exp15_table.py
```

Compare against `docs/paper_d_v38.tex` lines 565–585. Under a second, reads only committed
files, changes nothing.
