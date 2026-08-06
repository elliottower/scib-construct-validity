"""
Frozen classification rule for etiologic evidence discordance.

Two-criterion rule:
  OBS: "non-trivial" if d >= threshold (exposure-comparable scale)
  MR:  "causal" if CI excludes null AND d >= threshold (after per-SD rescaling)

Classification:
  OBS non-trivial + MR null   -> qualitative discordance -> predict failure
  OBS non-trivial + MR causal -> concordance -> predict success
  OBS trivial + MR null       -> null concordance -> ambiguous
  OBS trivial + MR causal     -> genetic-only signal

OBS instrument types:
  "epidemiological_OR" — population-level exposure-disease OR (neuro/cardio)
  "case_control_SMD"   — case-control biomarker SMD (autoimmune cytokines)
  Both produce Cohen's d: OR via Chinn (2000), SMD directly.
"""

import csv
import json
import math
import sys
from pathlib import Path


def chinn_d(OR: float) -> float:
    """Convert OR to Cohen's d via Chinn (2000): d = |ln(OR)| * sqrt(3) / pi."""
    if OR <= 0:
        raise ValueError(f"OR must be positive, got {OR}")
    return abs(math.log(OR)) * math.sqrt(3) / math.pi


def rescale_per_allele_to_per_sd(or_per_allele: float, sd_per_allele: float) -> float:
    """Rescale per-allele OR to per-SD OR.

    or_per_allele: MR OR per copy of the variant
    sd_per_allele: change in exposure (in SD units) per allele
    Returns: OR per 1 SD change in the exposure
    """
    if sd_per_allele <= 0:
        raise ValueError(f"sd_per_allele must be positive, got {sd_per_allele}")
    return or_per_allele ** (1.0 / sd_per_allele)


def ci_excludes_null(ci_lower: float, ci_upper: float) -> bool:
    """Check whether a confidence interval for an OR excludes 1.0."""
    return not (ci_lower <= 1.0 <= ci_upper)


def classify_mr(ci_lower: float, ci_upper: float, d_value: float,
                threshold: float = 0.10) -> str:
    """Classify MR evidence as 'causal' or 'null' using the two-criterion rule."""
    stat_sig = ci_excludes_null(ci_lower, ci_upper)
    above_floor = d_value >= threshold
    if stat_sig and above_floor:
        return "causal"
    return "null"


def classify_obs(d_value: float, threshold: float = 0.10) -> str:
    """Classify observational evidence as 'non-trivial' or 'trivial'."""
    if d_value >= threshold:
        return "non-trivial"
    return "trivial"


def classify_family(obs_class: str, mr_class: str) -> tuple[str, str]:
    """Classify a mechanism family. Returns (classification, prediction)."""
    if obs_class == "non-trivial" and mr_class == "null":
        return "qualitative discordance", "failure"
    elif obs_class == "non-trivial" and mr_class == "causal":
        return "concordance", "success"
    elif obs_class == "trivial" and mr_class == "null":
        return "null concordance", "ambiguous"
    elif obs_class == "trivial" and mr_class == "causal":
        return "genetic-only signal", "success"
    raise ValueError(f"Unexpected combination: obs={obs_class}, mr={mr_class}")


def run_classification(families: list[dict], threshold: float = 0.10) -> list[dict]:
    """Run the full classification pipeline on a list of families."""
    results = []
    for f in families:
        obs_type = f.get("obs_type", "epidemiological_OR")
        if "obs_d_direct" in f:
            obs_d = f["obs_d_direct"]
        else:
            obs_d = chinn_d(f["obs_OR"])
        obs_class = classify_obs(obs_d, threshold)

        mr_or = f["gen_OR"]
        if f.get("per_allele") and f.get("sd_per_allele"):
            mr_or = rescale_per_allele_to_per_sd(mr_or, f["sd_per_allele"])
        mr_d = chinn_d(mr_or)

        ci_lo = f.get("gen_CI_lower")
        ci_hi = f.get("gen_CI_upper")
        if ci_lo is not None and ci_hi is not None:
            mr_class = classify_mr(ci_lo, ci_hi, mr_d, threshold)
        else:
            mr_class = "null" if mr_d < threshold else "causal"

        classification, prediction = classify_family(obs_class, mr_class)

        actual = f.get("drug_outcome", "unknown")
        if actual in ("Approved", "approved"):
            correct = prediction == "success"
        elif actual in ("Failed", "failed", "No benefit"):
            correct = prediction == "failure"
        elif actual in ("Pending", "pending"):
            correct = None
        else:
            correct = None

        results.append({
            "family": f["family"],
            "domain": f.get("domain", "unknown"),
            "obs_OR": f.get("obs_OR"),
            "obs_d": round(obs_d, 3),
            "obs_type": obs_type,
            "obs_class": obs_class,
            "gen_OR_raw": f["gen_OR"],
            "gen_OR_rescaled": round(mr_or, 3) if f.get("per_allele") else None,
            "gen_d": round(mr_d, 3),
            "gen_CI": f"({ci_lo}-{ci_hi})" if ci_lo else None,
            "mr_class": mr_class,
            "classification": classification,
            "prediction": prediction,
            "drug_outcome": actual,
            "correct": correct,
        })
    return results


NEURO_FAMILIES = [
    {"family": "Metabolic-AD", "domain": "neuro", "obs_OR": 1.53, "gen_OR": 1.01,
     "drug_outcome": "Failed"},
    {"family": "ModRisk-AD", "domain": "neuro", "obs_OR": 1.46, "gen_OR": 1.10,
     "drug_outcome": "Failed"},
    {"family": "Anti-CD20-MS", "domain": "neuro", "obs_OR": 2.23, "gen_OR": 0.83,
     "gen_CI_lower": 0.79, "gen_CI_upper": 0.89,
     "drug_outcome": "Approved"},
    {"family": "Smoking-MS/AD", "domain": "neuro", "obs_OR": 1.46, "gen_OR": 1.03,
     "drug_outcome": "Failed"},
    {"family": "HRT-AD", "domain": "neuro", "obs_OR": 0.67, "gen_OR": 1.00,
     "gen_CI_lower": 0.85, "gen_CI_upper": 1.18,
     "drug_outcome": "Failed"},
    {"family": "BMI-MS", "domain": "neuro", "obs_OR": 1.92, "gen_OR": 1.41,
     "gen_CI_lower": 1.20, "gen_CI_upper": 1.66,
     "drug_outcome": "Pending"},
    {"family": "BMI-AD", "domain": "neuro", "obs_OR": 2.04, "gen_OR": 1.03,
     "gen_CI_lower": 1.01, "gen_CI_upper": 1.05,
     "drug_outcome": "Failed"},
    {"family": "VitaminD-MS", "domain": "neuro", "obs_OR": 1.40, "gen_OR": 2.00,
     "gen_CI_lower": 1.70, "gen_CI_upper": 2.50,
     "drug_outcome": "Failed"},
    {"family": "EBV-MS", "domain": "neuro", "obs_OR": 1.70, "gen_OR": 5.00,
     "gen_CI_lower": 2.00, "gen_CI_upper": 20.0,
     "drug_outcome": "Approved"},
]

CARDIO_FAMILIES = [
    {"family": "HDL/CETP", "domain": "cardio", "obs_OR": 0.62, "gen_OR": 0.93,
     "gen_CI_lower": 0.68, "gen_CI_upper": 1.26, "drug_outcome": "Failed"},
    {"family": "Niacin/HDL", "domain": "cardio", "obs_OR": 0.62, "gen_OR": 0.93,
     "gen_CI_lower": 0.68, "gen_CI_upper": 1.26, "drug_outcome": "Failed"},
    {"family": "Homocysteine", "domain": "cardio", "obs_OR": 1.32, "gen_OR": 1.02,
     "gen_CI_lower": 0.98, "gen_CI_upper": 1.07, "drug_outcome": "Failed"},
    {"family": "CRP", "domain": "cardio", "obs_OR": 1.37, "gen_OR": 1.00,
     "gen_CI_lower": 0.97, "gen_CI_upper": 1.02, "drug_outcome": "No benefit"},
    {"family": "Uric acid", "domain": "cardio", "obs_OR": 1.07, "gen_OR": 1.05,
     "gen_CI_lower": 0.92, "gen_CI_upper": 1.20, "drug_outcome": "Failed"},
    {"family": "LDL/PCSK9", "domain": "cardio", "obs_OR": 1.52, "gen_OR": 1.78,
     "gen_CI_lower": 1.58, "gen_CI_upper": 2.01, "drug_outcome": "Approved"},
    {"family": "Blood pressure", "domain": "cardio", "obs_OR": 1.41, "gen_OR": 1.44,
     "gen_CI_lower": 1.35, "gen_CI_upper": 1.55, "drug_outcome": "Approved"},
    {"family": "Triglycerides", "domain": "cardio", "obs_OR": 1.72, "gen_OR": 1.62,
     "gen_CI_lower": 1.24, "gen_CI_upper": 2.11, "drug_outcome": "Approved"},
    {"family": "Lp(a)", "domain": "cardio", "obs_OR": 1.13, "gen_OR": 0.94,
     "gen_CI_lower": 0.93, "gen_CI_upper": 0.95, "drug_outcome": "Pending"},
    {"family": "IL-6R", "domain": "cardio", "obs_OR": 1.25, "gen_OR": 0.95,
     "gen_CI_lower": 0.93, "gen_CI_upper": 0.97,
     "per_allele": True, "sd_per_allele": 0.34,
     "drug_outcome": "Pending"},
]


AUTOIMMUNE_FAMILIES = [
    # obs_sourcing: "meta_analysis" = pooled SMD from systematic review
    #               "author_estimated" = estimated from reported significance
    #               "construct_limited" = construct-definition problem
    # obs_type: "case_control_SMD" vs "epidemiological_OR" vs "tissue_expression"

    # IL-23: serum SMD=0.66 (Dowlatshahi 2013, CI -0.25 to 1.58, NON-SIGNIFICANT);
    #   but tissue p19 22x elevated (Lee 2004 PMID 14707118).
    #   NOTE: OBS rests on tissue expression, not serum. Serum-vs-tissue switch
    #   must be stated explicitly in the paper.
    {"family": "IL-23-psoriasis", "domain": "autoimmune",
     "obs_d_direct": 0.66, "obs_type": "tissue_expression",
     "obs_sourcing": "meta_analysis",
     "gen_OR": 0.616, "gen_CI_lower": 0.563, "gen_CI_upper": 0.674,
     "drug_outcome": "Approved"},

    # CTLA-4: sCTLA-4 elevated in RA p=0.005 (Liu 2012 PMID 22917707)
    #   6.8 ng/mL vs controls; exact SMD not reported
    {"family": "CTLA-4-RA", "domain": "autoimmune",
     "obs_d_direct": 0.50, "obs_type": "case_control_SMD",
     "obs_sourcing": "author_estimated",
     "gen_OR": 0.86, "gen_CI_lower": 0.78, "gen_CI_upper": 0.95,
     "drug_outcome": "Approved"},

    # TNF-a: meta-analysis of 14 studies, SMD=1.93 (CI 1.23-2.64)
    #   Wang 2015 PMC4694713; 890 RA vs 441 controls
    {"family": "TNF-a-RA", "domain": "autoimmune",
     "obs_d_direct": 1.93, "obs_type": "case_control_SMD",
     "obs_sourcing": "meta_analysis",
     "gen_OR": 1.00,
     "drug_outcome": "Approved"},

    # IL-17: meta-analysis of 8 studies, SMD=0.47 (CI 0.07-0.86)
    #   Zhou 2017 PMID 27925680
    {"family": "IL-17-psoriasis", "domain": "autoimmune",
     "obs_d_direct": 0.47, "obs_type": "case_control_SMD",
     "obs_sourcing": "meta_analysis",
     "gen_OR": 0.998,
     "drug_outcome": "Approved"},

    # STAT4/JAK: serum STAT4 elevated in RA p=0.01 (Osman 2025 PMC12520596);
    #   STAT1/STAT4/Jak3 elevated in synovium (Walker 2006)
    {"family": "JAK-STAT-RA", "domain": "autoimmune",
     "obs_d_direct": 0.50, "obs_type": "case_control_SMD",
     "obs_sourcing": "author_estimated",
     "gen_OR": 1.27, "gen_CI_lower": 1.20, "gen_CI_upper": 1.34,
     "drug_outcome": "Approved"},

    # CD20/B-cell: RF positivity 60-80% in RA; B-cell infiltration in synovium
    #   FCRL3 GWAS OR 2.15 (Kochi 2005, Nature Genetics)
    {"family": "CD20-RA", "domain": "autoimmune",
     "obs_d_direct": 0.50, "obs_type": "case_control_SMD",
     "obs_sourcing": "author_estimated",
     "gen_OR": 1.291, "gen_CI_lower": 1.190, "gen_CI_upper": 1.391,
     "drug_outcome": "Approved"},

    # IL-4Ra/AD: CONSTRUCT-LIMITED — IL-4 mRNA NOT dominant in AD lesions;
    #   IL-13 is 7x elevated (Hamid 1996 PMID 9158100; Jeong 2003 PMID 15014952).
    #   Dupilumab blocks IL-4Ra (shared receptor), works via IL-13.
    #   EXCLUDED FROM SCORING: construct-definition problem, not a clean miss.
    #   Drug outcome "Construct-limited" removes it from denominator.
    {"family": "IL-4Ra-AD", "domain": "autoimmune",
     "obs_d_direct": 0.15, "obs_type": "case_control_SMD",
     "obs_sourcing": "construct_limited",
     "gen_OR": 1.02,
     "drug_outcome": "Construct-limited"},

    # IL-1b/CVD: CRP/IL-1 inflammatory axis observational OR 1.37 (ERFC 2010)
    #   Same construct as cardio CRP family
    {"family": "IL-1b-CVD", "domain": "autoimmune",
     "obs_OR": 1.37, "obs_type": "epidemiological_OR",
     "obs_sourcing": "meta_analysis",
     "gen_OR": 1.00,
     "drug_outcome": "Failed"},
]


def print_results(results: list[dict]) -> None:
    header = (f"{'Family':<20} {'OBS d':>7} {'Src':>4} {'OBS':>12} "
              f"{'GEN d':>7} {'MR':>8} {'Class':<25} "
              f"{'Pred':>8} {'Actual':>16} {'OK?':>5}")
    print(header)
    print("=" * len(header))
    for r in results:
        ok = "✓" if r["correct"] is True else ("✗" if r["correct"] is False else "—")
        ot = r.get("obs_type", "epidemiological_OR")
        if ot == "case_control_SMD":
            src = "SMD"
        elif ot == "tissue_expression":
            src = "tis"
        else:
            src = " OR"
        print(f"{r['family']:<20} {r['obs_d']:>7.3f} {src:>4} {r['obs_class']:>12} "
              f"{r['gen_d']:>7.3f} {r['mr_class']:>8} {r['classification']:<25} "
              f"{r['prediction']:>8} {r['drug_outcome']:>16} {ok:>5}")


def main():
    thresholds = [0.08, 0.10, 0.12, 0.15] if "--sensitivity" in sys.argv else [0.10]
    all_families = NEURO_FAMILIES + CARDIO_FAMILIES + AUTOIMMUNE_FAMILIES

    for t in thresholds:
        print(f"\n{'='*80}")
        print(f"THRESHOLD d = {t:.2f}")
        print(f"{'='*80}")
        results = run_classification(all_families, threshold=t)

        print(f"\n--- Neuro ---")
        print_results([r for r in results if r["domain"] == "neuro"])
        print(f"\n--- Cardio ---")
        print_results([r for r in results if r["domain"] == "cardio"])
        print(f"\n--- Autoimmune ---")
        print_results([r for r in results if r["domain"] == "autoimmune"])

        print(f"\n--- Domain tallies (NOT pooled — different OBS constructs) ---")
        for domain in ["neuro", "cardio", "autoimmune"]:
            dom_results = [r for r in results if r["domain"] == domain]
            dom_known = [r for r in dom_results if r["correct"] is not None]
            dom_unambig = [r for r in dom_known if r["prediction"] != "ambiguous"]
            dom_correct = sum(1 for r in dom_unambig if r["correct"])
            dom_pending = sum(1 for r in dom_results if r["drug_outcome"] == "Pending")
            dom_ambig = sum(1 for r in dom_known if r["prediction"] == "ambiguous")
            dom_construct = sum(1 for r in dom_results
                                if r["drug_outcome"] == "Construct-limited")
            obs_types = set(r.get("obs_type", "epidemiological_OR") for r in dom_results)
            extra = ""
            if dom_construct:
                extra += f", {dom_construct} construct-limited"
            if dom_ambig:
                extra += f", {dom_ambig} ambig"
            if dom_pending:
                extra += f", {dom_pending} pending"
            print(f"  {domain:>10}: {dom_correct}/{len(dom_unambig)} scored"
                  f"{extra} [OBS: {', '.join(obs_types)}]")

        ai_fams = [f for f in all_families if f.get("domain") == "autoimmune"]
        n_estimated = sum(1 for f in ai_fams
                         if f.get("obs_sourcing") == "author_estimated")
        if n_estimated:
            print(f"\n  NOTE: {n_estimated} autoimmune families have "
                  f"author-estimated OBS d (not meta-analysis sourced).")
        print(f"  Do NOT pool autoimmune with neuro/cardio into one accuracy figure.")

    if "--json" in sys.argv:
        results = run_classification(all_families, threshold=0.10)
        out_path = Path(__file__).parent / "classification_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")

    if "--ablation" in sys.argv:
        run_ablation(all_families)


def ablation_rules(result: dict) -> dict[str, str]:
    """Return predictions from four rules for one family.

    always-fail: predict failure for everything
    MR-only: causal -> success, null -> failure
    OBS-only: non-trivial -> success, trivial -> failure
    cross-type: our concordance rule (already computed)
    """
    return {
        "always-fail": "failure",
        "MR-only": "success" if result["mr_class"] == "causal" else "failure",
        "OBS-only": "success" if result["obs_class"] == "non-trivial" else "failure",
        "cross-type": result["prediction"],
    }


def _score_prediction(prediction: str, actual: str) -> bool | None:
    if actual in ("Approved", "approved"):
        return prediction == "success"
    elif actual in ("Failed", "failed", "No benefit"):
        return prediction == "failure"
    return None


def run_ablation(families: list[dict], threshold: float = 0.10) -> None:
    results = run_classification(families, threshold=threshold)
    scorable = [r for r in results
                if r["drug_outcome"] not in ("Pending", "Construct-limited")
                and r["prediction"] != "ambiguous"]

    rule_names = ["always-fail", "MR-only", "OBS-only", "cross-type"]
    scored: dict[str, list[tuple[str, bool]]] = {rn: [] for rn in rule_names}

    for r in scorable:
        preds = ablation_rules(r)
        for rn in rule_names:
            correct = _score_prediction(preds[rn], r["drug_outcome"])
            if correct is not None:
                scored[rn].append((r["family"], correct))

    n_failed = sum(1 for r in scorable
                   if r["drug_outcome"] in ("Failed", "failed", "No benefit"))
    n_approved = sum(1 for r in scorable
                     if r["drug_outcome"] in ("Approved", "approved"))

    print(f"\n{'='*80}")
    print(f"ABLATION ANALYSIS (threshold d={threshold:.2f})")
    print(f"{'='*80}")
    print(f"Base rates: {n_failed} failed, {n_approved} approved, "
          f"{len(scorable)} total scorable")

    header = (f"{'Rule':<15} {'Total':>7} {'Correct':>9} {'Acc':>7} "
              f"{'Fail✓':>7} {'Fail✗':>7} {'Appr✓':>7} {'Appr✗':>7}")
    print(f"\n{header}")
    print("-" * len(header))

    rule_correct_sets: dict[str, set[str]] = {}
    for rn in rule_names:
        pairs = scored[rn]
        total = len(pairs)
        correct = sum(1 for _, c in pairs if c)
        acc = correct / total if total else 0

        fail_correct = sum(1 for f, c in pairs if c
                          and any(r["family"] == f and r["drug_outcome"]
                                  in ("Failed", "failed", "No benefit")
                                  for r in scorable))
        fail_wrong = n_failed - fail_correct
        appr_correct = sum(1 for f, c in pairs if c
                          and any(r["family"] == f and r["drug_outcome"]
                                  in ("Approved", "approved")
                                  for r in scorable))
        appr_wrong = n_approved - appr_correct

        print(f"{rn:<15} {total:>7} {correct:>9} {acc:>7.1%} "
              f"{fail_correct:>7} {fail_wrong:>7} {appr_correct:>7} {appr_wrong:>7}")

        rule_correct_sets[rn] = {f for f, c in pairs if c}

    ct_set = rule_correct_sets["cross-type"]
    mr_set = rule_correct_sets["MR-only"]
    ct_only = ct_set - mr_set
    mr_only = mr_set - ct_set

    print(f"\n--- McNemar disagreements: cross-type vs MR-only ---")
    print(f"  Cross-type correct, MR-only wrong ({len(ct_only)}): "
          f"{', '.join(sorted(ct_only)) if ct_only else 'none'}")
    print(f"  MR-only correct, cross-type wrong ({len(mr_only)}): "
          f"{', '.join(sorted(mr_only)) if mr_only else 'none'}")

    both_right = ct_set & mr_set
    both_wrong = {f for f, _ in scored["cross-type"]} - ct_set - mr_set
    # families wrong under both (actually: families in neither correct set)
    neither = set()
    all_families_scored = {f for f, _ in scored["cross-type"]}
    for f in all_families_scored:
        if f not in ct_set and f not in mr_set:
            neither.add(f)

    print(f"  Both correct ({len(both_right)}): "
          f"{', '.join(sorted(both_right)) if both_right else 'none'}")
    print(f"  Both wrong ({len(neither)}): "
          f"{', '.join(sorted(neither)) if neither else 'none'}")

    if ct_only or mr_only:
        print(f"\n  McNemar 2x2: a={len(ct_only)}, b={len(mr_only)}, "
              f"n_discordant={len(ct_only) + len(mr_only)}")
        if len(ct_only) + len(mr_only) > 0:
            print(f"  (With n_discordant={len(ct_only) + len(mr_only)}, "
                  f"exact binomial is appropriate over chi-squared)")


if __name__ == "__main__":
    main()
