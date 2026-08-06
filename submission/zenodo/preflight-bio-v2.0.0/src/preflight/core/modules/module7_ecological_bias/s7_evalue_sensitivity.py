"""
Supplementary Analysis S7: E-Value Sensitivity for 4CE Moderators

Computes E-values for the elderly-proportion moderator in the 4CE
meta-regression using the VanderWeele & Ding (2017) formula.
"""

import json
import math
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

BETA_MODERATOR = 14.2
SE_MODERATOR = 0.83


def compute_evalue_rr(rr):
    if rr < 1:
        rr = 1 / rr
    return rr + math.sqrt(rr * (rr - 1))


def compute_evalue_from_beta(beta, se):
    rr_point = math.exp(beta)
    rr_lower = math.exp(beta - 1.96 * se)

    evalue_point = compute_evalue_rr(rr_point)
    evalue_ci = compute_evalue_rr(rr_lower)

    return {
        "rr_point": rr_point,
        "rr_ci_lower": rr_lower,
        "evalue_point": evalue_point,
        "evalue_ci_lower": evalue_ci,
    }


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 70)
    print(f"S7: E-Value Sensitivity for 4CE Moderators  [{ts}]")
    print(f"  Moderator: elderly proportion")
    print(f"  Beta = {BETA_MODERATOR}, SE = {SE_MODERATOR}")
    print("=" * 70)

    result = compute_evalue_from_beta(BETA_MODERATOR, SE_MODERATOR)

    print(f"\n  RR (point estimate):  {result['rr_point']:.2e}")
    print(f"  RR (95% CI lower):   {result['rr_ci_lower']:.2e}")
    print(f"  E-value (point):     {result['evalue_point']:.2e}")
    print(f"  E-value (CI lower):  {result['evalue_ci_lower']:.2e}")

    if result["evalue_ci_lower"] > 3:
        interpretation = ("Substantial unmeasured confounding needed to "
                          "explain the moderator effect. The E-value for "
                          "the CI bound alone exceeds 3.")
    elif result["evalue_ci_lower"] > 1.5:
        interpretation = ("Moderate unmeasured confounding needed. E-value "
                          "for CI bound is between 1.5 and 3.")
    else:
        interpretation = ("Modest confounding suffices to explain the "
                          "moderator. E-value for CI bound < 1.5, "
                          "weakening the 4CE claims.")

    print(f"\n  Interpretation: {interpretation}")

    output = {
        "timestamp": ts,
        "moderator": "elderly_proportion",
        "beta": BETA_MODERATOR,
        "se": SE_MODERATOR,
        **result,
        "interpretation": interpretation,
        "method": "VanderWeele & Ding (2017)",
    }

    outpath = OUTPUT_DIR / "s7_evalue_sensitivity.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {outpath}")


if __name__ == "__main__":
    main()
