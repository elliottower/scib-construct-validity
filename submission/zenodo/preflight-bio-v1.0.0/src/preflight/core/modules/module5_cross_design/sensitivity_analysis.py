"""
Pre-registration power analysis and false positive rate verification.

Computes:
1. Power of binomial test at various true accuracy rates and sample sizes
2. False positive rate under null (50% accuracy) via simulation
3. Permutation null: shuffle drug outcomes, recompute classification accuracy
4. Threshold sensitivity: which families flip at each d threshold
"""

import math
import random
from collections import Counter
from scipy import stats


def chinn_d(OR: float) -> float:
    return abs(math.log(OR)) * math.sqrt(3) / math.pi


# === POWER ANALYSIS ===

def binomial_power(n: int, true_p: float, alpha: float = 0.05,
                   null_p: float = 0.5, n_sim: int = 50_000) -> float:
    """Simulate power of one-sided binomial test.

    H0: accuracy = null_p (chance)
    H1: accuracy = true_p
    Reject if binomial p-value < alpha.
    """
    rejects = 0
    for _ in range(n_sim):
        k = sum(1 for _ in range(n) if random.random() < true_p)
        p_val = stats.binomtest(k, n, null_p, alternative="greater").pvalue
        if p_val < alpha:
            rejects += 1
    return rejects / n_sim


def fisher_exact_power(n_disc: int, n_conc: int, true_disc_fail: float,
                       true_conc_success: float, alpha: float = 0.05,
                       n_sim: int = 50_000) -> float:
    """Simulate power of Fisher's exact test on the 2x2 table.

    Discordant families fail with prob true_disc_fail.
    Concordant families succeed with prob true_conc_success.
    """
    rejects = 0
    for _ in range(n_sim):
        disc_fail = sum(1 for _ in range(n_disc) if random.random() < true_disc_fail)
        disc_succ = n_disc - disc_fail
        conc_succ = sum(1 for _ in range(n_conc) if random.random() < true_conc_success)
        conc_fail = n_conc - conc_succ
        table = [[conc_succ, conc_fail], [disc_succ, disc_fail]]
        _, p_val = stats.fisher_exact(table, alternative="greater")
        if p_val < alpha:
            rejects += 1
    return rejects / n_sim


# === FALSE POSITIVE RATE VERIFICATION ===

def fpr_binomial(n: int, null_p: float = 0.5, alpha: float = 0.05,
                 n_sim: int = 50_000) -> float:
    """Verify FPR of binomial test under true null."""
    rejects = 0
    for _ in range(n_sim):
        k = sum(1 for _ in range(n) if random.random() < null_p)
        p_val = stats.binomtest(k, n, null_p, alternative="greater").pvalue
        if p_val < alpha:
            rejects += 1
    return rejects / n_sim


def fpr_fisher(n_total: int, frac_disc: float = 0.5, alpha: float = 0.05,
               n_sim: int = 50_000) -> float:
    """Verify FPR of Fisher's exact test under true null (no association)."""
    n_disc = int(n_total * frac_disc)
    n_conc = n_total - n_disc
    rejects = 0
    for _ in range(n_sim):
        disc_fail = sum(1 for _ in range(n_disc) if random.random() < 0.5)
        disc_succ = n_disc - disc_fail
        conc_succ = sum(1 for _ in range(n_conc) if random.random() < 0.5)
        conc_fail = n_conc - conc_succ
        table = [[conc_succ, conc_fail], [disc_succ, disc_fail]]
        _, p_val = stats.fisher_exact(table, alternative="greater")
        if p_val < alpha:
            rejects += 1
    return rejects / n_sim


# === PERMUTATION NULL ===

def permutation_null(classifications: list[str], outcomes: list[str],
                     n_perm: int = 10_000) -> dict:
    """Shuffle outcomes, recompute accuracy. Returns null distribution stats."""
    observed_correct = sum(
        1 for c, o in zip(classifications, outcomes)
        if (c == "failure" and o in ("Failed", "No benefit")) or
           (c == "success" and o in ("Approved",))
    )
    observed_acc = observed_correct / len(classifications)

    null_accs = []
    for _ in range(n_perm):
        shuffled = outcomes[:]
        random.shuffle(shuffled)
        correct = sum(
            1 for c, o in zip(classifications, shuffled)
            if (c == "failure" and o in ("Failed", "No benefit")) or
               (c == "success" and o in ("Approved",))
        )
        null_accs.append(correct / len(classifications))

    p_value = sum(1 for a in null_accs if a >= observed_acc) / n_perm
    return {
        "observed_accuracy": observed_acc,
        "observed_correct": observed_correct,
        "n_evaluated": len(classifications),
        "null_mean": sum(null_accs) / len(null_accs),
        "null_sd": (sum((a - sum(null_accs)/len(null_accs))**2 for a in null_accs) / len(null_accs)) ** 0.5,
        "null_95th": sorted(null_accs)[int(0.95 * len(null_accs))],
        "null_99th": sorted(null_accs)[int(0.99 * len(null_accs))],
        "permutation_p": p_value,
    }


# === THRESHOLD SENSITIVITY ===

def threshold_sensitivity(families: list[dict], thresholds: list[float]) -> dict:
    """For each threshold, report which families change classification."""
    from classify_families import run_classification
    baseline = {r["family"]: r["classification"]
                for r in run_classification(families, threshold=0.10)}
    changes = {}
    for t in thresholds:
        results = run_classification(families, threshold=t)
        flipped = []
        for r in results:
            if r["classification"] != baseline.get(r["family"]):
                flipped.append({
                    "family": r["family"],
                    "from": baseline[r["family"]],
                    "to": r["classification"],
                })
        changes[t] = flipped
    return changes


def main():
    random.seed(42)

    print("=" * 70)
    print("POWER ANALYSIS: Binomial test (H0: accuracy = 0.50)")
    print("=" * 70)
    print()
    print(f"{'n':>5} {'True acc':>10} {'Power':>8}")
    print("-" * 25)
    for n in [15, 23, 30]:
        for true_p in [0.70, 0.80, 0.85, 0.90, 0.95]:
            pwr = binomial_power(n, true_p, alpha=0.05, n_sim=50_000)
            print(f"{n:>5} {true_p:>10.2f} {pwr:>8.3f}")
        print()

    print("=" * 70)
    print("POWER ANALYSIS: Fisher's exact test (2x2 table)")
    print("=" * 70)
    print()
    print("Scenario: n_disc discordant families, n_conc concordant families")
    print(f"{'n_disc':>7} {'n_conc':>7} {'disc→fail':>10} {'conc→succ':>10} {'Power':>8}")
    print("-" * 45)
    for n_disc, n_conc in [(9, 6), (12, 8), (15, 10)]:
        for df, cs in [(0.90, 0.80), (0.95, 0.85), (1.00, 0.90)]:
            pwr = fisher_exact_power(n_disc, n_conc, df, cs, n_sim=50_000)
            print(f"{n_disc:>7} {n_conc:>7} {df:>10.2f} {cs:>10.2f} {pwr:>8.3f}")
        print()

    print("=" * 70)
    print("FALSE POSITIVE RATE VERIFICATION (under true null)")
    print("=" * 70)
    print()
    for n in [15, 23, 30]:
        fpr = fpr_binomial(n, null_p=0.5, alpha=0.05, n_sim=50_000)
        print(f"Binomial test, n={n}: FPR = {fpr:.4f} (expected <= 0.05)")
    print()
    for n in [15, 23, 30]:
        fpr = fpr_fisher(n, frac_disc=0.55, alpha=0.05, n_sim=50_000)
        print(f"Fisher's exact, n={n}: FPR = {fpr:.4f} (expected <= 0.05)")

    print()
    print("=" * 70)
    print("PERMUTATION NULL: Neuro + Cardio (current data)")
    print("=" * 70)
    print()

    classifications_neuro_cardio = [
        "failure", "failure", "success", "failure", "failure",
        "failure", "success", "success",
        "failure", "failure", "failure", "failure",
        "success", "success", "success",
    ]
    outcomes_neuro_cardio = [
        "Failed", "Failed", "Approved", "Failed", "Failed",
        "Failed", "Failed", "Approved",
        "Failed", "Failed", "Failed", "No benefit",
        "Approved", "Approved", "Approved",
    ]
    family_names = [
        "Metabolic-AD", "ModRisk-AD", "Anti-CD20-MS", "Smoking", "HRT-AD",
        "BMI-AD", "VitaminD-MS", "EBV-MS",
        "HDL/CETP", "Niacin/HDL", "Homocysteine", "CRP",
        "LDL/PCSK9", "Blood pressure", "Triglycerides",
    ]

    print("Families entering permutation test (excluding ambiguous + pending):")
    for name, cls, out in zip(family_names, classifications_neuro_cardio,
                               outcomes_neuro_cardio):
        match = (cls == "failure" and out in ("Failed", "No benefit")) or \
                (cls == "success" and out in ("Approved",))
        print(f"  {name:<20} pred={cls:<10} actual={out:<12} {'✓' if match else '✗'}")

    result = permutation_null(classifications_neuro_cardio,
                               outcomes_neuro_cardio, n_perm=10_000)
    print(f"\nObserved: {result['observed_correct']}/{result['n_evaluated']} "
          f"= {result['observed_accuracy']:.3f}")
    print(f"Null mean: {result['null_mean']:.3f} (SD {result['null_sd']:.3f})")
    print(f"Null 95th percentile: {result['null_95th']:.3f}")
    print(f"Null 99th percentile: {result['null_99th']:.3f}")
    print(f"Permutation p-value: {result['permutation_p']:.4f}")


if __name__ == "__main__":
    main()
