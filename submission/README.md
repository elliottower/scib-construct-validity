# Submission

All contents except this README are gitignored.

## Structure

```
submission/
  README.md                         # this file (tracked)
  genome-biology/                   # primary venue
    cover_letter.tex
    cover_letter.pdf
  plos-comp-bio/                    # fallback venue
    cover_letter.tex
    cover_letter.pdf
  zenodo/                           # reproducibility archive
    README.md
    manuscript.pdf                  # compiled paper
    scib-construct-validity.zip     # repo archive (docs, src, tests, scripts, results)
```

## Venue notes

**Genome Biology** (primary)
- Benchmark article
- APC ~$5,690
- Where scIB was published; this paper evaluates scIB metrics
- Format: structured abstract (Background/Results/Conclusions), line numbers, natbib numeric
- Requires ORCID

**PLOS Computational Biology** (fallback)
- Research Article
- APC ~$2,500
- Format: structured abstract, Vancouver-style references

## Zenodo

DOI: [10.5281/zenodo.21351298](https://doi.org/10.5281/zenodo.21351298)

## Checklist

- [ ] Paper compiles clean (zero undefined refs, zero errors)
- [ ] All figures present and referenced
- [ ] Zenodo DOI in manuscript data availability statement
- [ ] GitHub repo is public
- [ ] pip package published (`pip install scib-validity`)
- [ ] Cover letter compiled
- [ ] ORCID linked (Genome Biology requires it)
- [ ] Final negation-contrast audit: `grep -n "not.*but" docs/paper_d_v33.tex`
- [ ] Final AI-tell audit: `grep -niE "delve|leverage|landscape|tapestry|multifaceted" docs/paper_d_v33.tex`
