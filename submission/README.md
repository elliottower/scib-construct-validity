# Submission Directory

All contents except this README are gitignored. Compiled PDFs, zips, and
cover letters live here but are not tracked.

## Structure

```
submission/
  README.md                         # this file (tracked)
  genome-biology/                   # primary venue
    cover_letter.tex                # cover letter source
    cover_letter.pdf                # compiled
    manuscript.pdf                  # compiled paper_d_v32
    latex_source.zip                # tex + bib + figures
    supplementary/                  # extended tables/figures
  plos-comp-bio/                    # fallback venue
    cover_letter.tex
    cover_letter.pdf
    manuscript.pdf
    latex_source.zip
  zenodo/                           # reproducibility archive
    scib-construct-validity.zip     # full repo sans submission/
    manuscript.pdf                  # compiled paper
```

## Venue notes

**Genome Biology** (primary)
- Benchmark article
- APC ~$5,690
- Where scIB was published; this paper evaluates scIB metrics
- Format: structured abstract, line numbers, natbib numeric

**PLOS Computational Biology** (fallback)
- Research Article
- APC ~$2,500
- Format: structured abstract, Vancouver-style references

## Checklist before submission

- [ ] Paper compiles clean (zero undefined refs, zero errors)
- [ ] All figures present and referenced
- [ ] Zenodo DOI in manuscript data availability statement
- [ ] GitHub repo is public
- [ ] pip package published (`pip install scib-validity`)
- [ ] Cover letter compiled
- [ ] LaTeX source zipped (tex + bib + figures, no build artifacts)
- [ ] Supplementary materials compiled
- [ ] ORCID linked (Genome Biology requires it)
- [ ] Final negation-contrast audit: `grep -n "not.*but" docs/paper_d_v32.tex`
- [ ] Final AI-tell audit: `grep -niE "delve|leverage|landscape|tapestry|multifaceted" docs/paper_d_v32.tex`
