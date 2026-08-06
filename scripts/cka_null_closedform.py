"""
Closed-form expected CKA and Procrustes for random Gaussian centroid matrices.

Goal: derive E[CKA(X, Y)] where X, Y ~ iid N(0,1) in R^{k x d}, centered,
and validate against the simulated values in robustness_controls.json.

Linear CKA (centered, unbiased HSIC):
  CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
where K = H @ X @ X.T @ H, L = H @ Y @ Y.T @ H, H = I - 11^T/k

For independent X, Y:
  E[HSIC(K, L)] = E[tr(K @ L)] / (k-1)^2
  = E[tr(H X X^T H H Y Y^T H)] / (k-1)^2

Since X, Y independent and H is idempotent:
  = tr(H @ E[X X^T] @ H @ E[Y Y^T] @ H) / (k-1)^2  ... NO: E[tr(AB)] != tr(E[A]E[B])

Need E[tr(A B)] where A = H X X^T H, B = H Y Y^T H are independent Wishart-like.
For independent A, B: E[tr(A B)] = tr(E[A] @ E[B]) + Cov terms.

Actually, since X and Y are independent:
  E[tr(A B)] = sum_{i,j} E[A_{ij}] E[B_{ji}] = tr(E[A] @ E[B])

This IS correct because A and B are functions of independent X and Y.

So: E[HSIC_num] = tr(E[H X X^T H]^2) ... wait no, the numerator is tr(K L),
not tr(K^2).

E[tr(K L)] = tr(E[K] E[L]) since K, L independent.

E[K] = E[H X X^T H] = H E[X X^T] H = H (d I_k) H = d H
(since X entries iid N(0,1), E[X X^T] = d I_k, and H^2 = H)

So E[tr(K L)] = tr((d H)^2) = d^2 tr(H) = d^2 (k-1)

For the denominator: E[HSIC(K,K)] = E[tr(K^2)] / (k-1)^2
E[tr(K^2)] = E[tr((H X X^T H)^2)] -- this requires 4th moment of Wishart

Let W = X X^T ~ Wishart_k(I_d, d). Then K = H W H.
E[tr((HWH)^2)] = E[sum_{ijkl} H_{ij} W_{jk} H_{kl} H_{li'} W_{i'j'} H_{j'i}]
... messy. Let's just compute numerically.

Strategy: for each (k, d), generate many random matrix pairs, compute CKA and
Procrustes empirically, and fit a formula. Then compare to the simulation data.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime


def linear_cka_centered(X, Y):
    """Linear CKA on centered matrices X, Y (k x d)."""
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ X @ X.T @ H
    L = H @ Y @ Y.T @ H
    hsic_kl = np.trace(K @ L)
    hsic_kk = np.trace(K @ K)
    hsic_ll = np.trace(L @ L)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return hsic_kl / denom


def procrustes_sim(X, Y):
    """Procrustes similarity on centered matrices."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Xc /= np.linalg.norm(Xc, 'fro') + 1e-12
    Yc /= np.linalg.norm(Yc, 'fro') + 1e-12
    U, _, Vt = np.linalg.svd(Xc.T @ Yc, full_matrices=False)
    nuclear = np.sum(np.linalg.svd(Xc.T @ Yc, compute_uv=False))
    return nuclear ** 2  # Procrustes similarity = 1 - disparity^2, which equals nuclear^2 for unit-norm


def compute_null_stats(k, d, n_trials=5000, seed=42):
    """Compute mean CKA and Procrustes for random Gaussian k x d matrices."""
    rng = np.random.default_rng(seed)
    cka_vals = []
    proc_vals = []
    for _ in range(n_trials):
        X = rng.standard_normal((k, d))
        Y = rng.standard_normal((k, d))
        cka_vals.append(linear_cka_centered(X, Y))
        proc_vals.append(procrustes_sim(X, Y))
    return {
        'cka_mean': np.mean(cka_vals),
        'cka_std': np.std(cka_vals),
        'cka_95ci': [np.percentile(cka_vals, 2.5), np.percentile(cka_vals, 97.5)],
        'proc_mean': np.mean(proc_vals),
        'proc_std': np.std(proc_vals),
        'proc_95ci': [np.percentile(proc_vals, 2.5), np.percentile(proc_vals, 97.5)],
    }


def theoretical_cka_numerator(k, d):
    """
    E[tr(K L)] where K = H X X^T H, L = H Y Y^T H, X,Y independent N(0,1).

    Since X, Y independent:
      E[tr(K L)] = tr(E[K] E[L]) = tr((d H)^2) = d^2 tr(H) = d^2 (k-1)

    Actually this is the numerator of CKA before normalization.
    """
    return d**2 * (k - 1)


def theoretical_cka_denominator_term(k, d):
    """
    E[tr(K^2)] where K = H X X^T H, X ~ N(0,1)^{k x d}.

    K = H W H where W = X X^T ~ Wishart_k(I_d, d).

    For W ~ Wishart_p(I, n) (p=k, n=d):
      E[tr(W^2)] = n*p + n^2*p = n*p*(1+n) ... no.

    Actually E[W_{ij}] = n * delta_{ij}
    E[W_{ij} W_{kl}] = n^2 delta_{ij} delta_{kl} + n (delta_{ik} delta_{jl} + delta_{il} delta_{jk})

    So E[tr(W^2)] = sum_{ij} E[W_{ij}^2] = sum_{ij} [n^2 delta_{ij}^2 + n(1 + delta_{ij})]
    Wait, let me be more careful.

    E[W_{ij} W_{ji}] = E[W_{ij}^2] for symmetric W.
    E[W_{ij}^2] = n^2 delta_{ij} + n(1 + delta_{ij})  ... for Wishart_p(I, n)

    Hmm, let me just compute this from the Wishart moment formula:
    E[W_{ab} W_{cd}] = n(Sigma_{ac} Sigma_{bd} + Sigma_{ad} Sigma_{bc}) + n^2 Sigma_{ab} Sigma_{cd}

    With Sigma = I:
    E[W_{ab} W_{cd}] = n(delta_{ac} delta_{bd} + delta_{ad} delta_{bc}) + n^2 delta_{ab} delta_{cd}

    tr(W^2) = sum_{ij} W_{ij} W_{ji}
    E[tr(W^2)] = sum_{ij} E[W_{ij} W_{ji}]
              = sum_{ij} [n(delta_{ii} delta_{jj} + delta_{ij} delta_{ji}) + n^2 delta_{ij} delta_{ji}]
              = sum_{ij} [n(1 * 1 + delta_{ij}^2) + n^2 delta_{ij}^2]    ... wait

    a=i, b=j, c=j, d=i:
    E[W_{ij} W_{ji}] = n(delta_{ij} delta_{ji} + delta_{ii} delta_{jj}) + n^2 delta_{ij} delta_{ji}
                     = n(delta_{ij}^2 + 1) + n^2 delta_{ij}^2

    sum over i,j:
    = sum_{ij} [n delta_{ij}^2 + n + n^2 delta_{ij}^2]
    = n * k + n * k^2 + n^2 * k    (since sum delta_{ij}^2 = k, sum 1 = k^2)
    = n*k*(1 + k + n)              ... let me double check
    = k*n + k^2*n + k*n^2 = k*n*(1 + k + n)

    Hmm, that doesn't look standard. Let me verify: for p=k, n=d (Wishart_p(I, n)):
    E[tr(W^2)] = p*n*(1 + p + n)?

    For p=2, n=3: E[tr(W^2)] = 2*3*(1+2+3) = 36.
    """
    # Just use the formula and verify numerically
    p, n = k, d
    return p * n * (1 + p + n)


def main():
    print(f"CKA/Procrustes null-floor analysis — {datetime.now().isoformat()}")
    print("=" * 70)

    # Load simulated values
    results_path = Path(__file__).parent.parent / "results/biological_structure_v3b/biological_structure_v3b/robustness_controls/robustness_controls.json"
    with open(results_path) as f:
        sim_data = json.load(f)
    sim_null = sim_data['random_centroid_null']

    d = 512
    k_values = sorted([int(k) for k in sim_null.keys()])

    # First: verify Wishart moment formula against direct simulation
    print("\n--- Verifying E[tr(W^2)] formula ---")
    rng = np.random.default_rng(0)
    for k in [8, 15, 27, 44]:
        empirical = []
        for _ in range(2000):
            X = rng.standard_normal((k, d))
            W = X @ X.T
            empirical.append(np.trace(W @ W))
        emp_mean = np.mean(empirical)
        theory = k * d * (1 + k + d)
        print(f"  k={k:3d}: theory={theory:.0f}  empirical={emp_mean:.0f}  ratio={emp_mean/theory:.4f}")

    # Now compute E[tr((HWH)^2)] via simulation and try to find formula
    print("\n--- Computing E[tr((HWH)^2)] ---")
    for k in [8, 15, 27, 44]:
        H = np.eye(k) - np.ones((k, k)) / k
        empirical_hwh2 = []
        empirical_cka = []
        for _ in range(3000):
            X = rng.standard_normal((k, d))
            Y = rng.standard_normal((k, d))
            Kx = H @ X @ X.T @ H
            Ky = H @ Y @ Y.T @ H
            empirical_hwh2.append(np.trace(Kx @ Kx))
            empirical_cka.append(np.trace(Kx @ Ky) / np.sqrt(np.trace(Kx @ Kx) * np.trace(Ky @ Ky)))

        mean_hwh2 = np.mean(empirical_hwh2)
        mean_cka = np.mean(empirical_cka)

        # E[tr(K L)] / E[sqrt(tr(K^2) * tr(L^2))]
        # Since K, L independent: E[tr(KL)] = tr(E[K]^2) = d^2 * tr(H) = d^2 * (k-1)
        # But CKA = tr(KL) / sqrt(tr(K^2) * tr(L^2)), and E[A/B] != E[A]/E[B]
        # So we need the full distribution, not just moments.

        # Compare to simulated data
        sim_cka = sim_null[str(k)]['cka_mean'] if str(k) in sim_null else None
        sim_str = f"{sim_cka:.6f}" if sim_cka is not None else "N/A"
        print(f"  k={k:3d}: E[CKA]={mean_cka:.6f}  sim_CKA={sim_str:>10s}  E[tr(HWH)^2]={mean_hwh2:.0f}")

    # Full comparison table
    print("\n--- Full comparison: computed vs simulated null ---")
    print(f"{'k':>4s}  {'Computed CKA':>12s}  {'Sim CKA':>10s}  {'Delta':>8s}  {'Computed Proc':>13s}  {'Sim Proc':>10s}  {'Delta':>8s}")

    results = {}
    for k in k_values:
        stats = compute_null_stats(k, d, n_trials=5000)
        sim_cka = sim_null[str(k)]['cka_mean']
        sim_proc = sim_null[str(k)]['procrustes_mean']
        delta_cka = stats['cka_mean'] - sim_cka
        delta_proc = stats['proc_mean'] - sim_proc
        print(f"{k:4d}  {stats['cka_mean']:12.6f}  {sim_cka:10.6f}  {delta_cka:+8.5f}  {stats['proc_mean']:13.6f}  {sim_proc:10.6f}  {delta_proc:+8.5f}")
        results[k] = stats

    # Try to fit CKA as a function of k and d
    print("\n--- Fitting CKA(k, d) ---")
    k_arr = np.array(k_values, dtype=float)
    cka_arr = np.array([results[k]['cka_mean'] for k in k_values])

    # Try: CKA ≈ 1 - a * k / d
    ratios = k_arr / d
    slopes = (1 - cka_arr) / ratios
    print(f"  1 - CKA ≈ slope * k/d")
    print(f"  slopes: {slopes}")
    print(f"  mean slope: {np.mean(slopes):.4f}, std: {np.std(slopes):.4f}")

    # Try: CKA ≈ 1 - a * (k-1) / (d+k) or similar
    # From random matrix theory, for Wishart k x d:
    # The expected squared cosine between random subspaces goes as (k-1)/(d-1)
    test1 = (k_arr - 1) / (d - 1)
    test2 = (k_arr - 1) / (d + k_arr)
    test3 = (k_arr - 1) / d

    print(f"\n  Testing: 1 - CKA ≈ c * f(k,d)")
    for name, test in [("(k-1)/(d-1)", test1), ("(k-1)/(d+k)", test2), ("(k-1)/d", test3)]:
        coeffs = (1 - cka_arr) / test
        print(f"    {name:15s}: coeffs range [{coeffs.min():.4f}, {coeffs.max():.4f}], mean={coeffs.mean():.4f}, cv={coeffs.std()/coeffs.mean():.4f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'd': d,
        'computed_null': {str(k): results[k] for k in k_values},
        'note': 'Computed with 5000 trials per k, seed=42',
    }
    out_path = Path(__file__).parent.parent / "results/biological_structure_v3b/biological_structure_v3b/robustness_controls/closedform_null.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
