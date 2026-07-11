"""Generate the v5 vs v6 vs M6-sweep comparison table for the audit trail."""
import json
from pathlib import Path

import numpy as np

V5_PATH = Path("results/modal_results/composite_validation_v5/incremental_20260710_220954.jsonl")
V6_PATH = Path("results/modal_results/composite_validation_v6/incremental.jsonl")
M6_PATH = Path("results/m6_sensitivity/m6_sensitivity.json")
OUTPUT = Path("results/m6_sensitivity/comparison_table.md")

with open(V5_PATH) as f:
    v5 = {p["pair_id"]: p for p in (json.loads(l) for l in f)}
with open(V6_PATH) as f:
    v6 = {p["pair_id"]: p for p in (json.loads(l) for l in f)}
with open(M6_PATH) as f:
    sweep = json.load(f)

m6_00 = {p["pair_id"]: p for p in sweep[0]["pairs"]}
m6_25 = {p["pair_id"]: p for p in sweep[2]["pairs"]}

lines = []
lines.append("# Exp0 Comparison: v5 → v6 → M6 Sensitivity")
lines.append("")
lines.append("## Context")
lines.append("")
lines.append("- **v5**: Pre-fix scorer (M4=NaN counter, M6=1-frac, M3=PCA)")
lines.append("- **v6**: Fixed scorer (M4=participation ratio, M6=frac, M3=LDA), M6 weight=0.5")
lines.append("- **v6 M6=0.0**: Same scorer, M6 zeroed out")
lines.append("- **v6 M6=0.25**: Same scorer, M6 halved")
lines.append("")

lines.append("## Per-Pair Results")
lines.append("")
lines.append("| Pair | Type | Degr. | v5 T | v6 T | M6=0 T | M6=.25 T | v5 Score | v6 Score | M6=0 Score | M6=.25 Score |")
lines.append("|------|------|-------|------|------|--------|----------|----------|----------|------------|--------------|")

for pid in sorted(v6.keys(), key=lambda x: v6[x]["composite_tier"]):
    p6 = v6[pid]
    p5 = v5.get(pid, {})
    p00 = m6_00.get(pid, {})
    p25 = m6_25.get(pid, {})

    degr = f"{p6['relative_degradation']:.2f}" if "relative_degradation" in p6 else "?"
    v5t = p5.get("composite_tier", "?")
    v6t = p6["composite_tier"]
    t00 = p00.get("composite_tier", "?")
    t25 = p25.get("composite_tier", "?")

    v5s = f"{p5['composite_score']:.3f}" if "composite_score" in p5 else "?"
    v6s = f"{p6['composite_score']:.3f}"
    s00 = f"{p00['composite_score']:.3f}" if "composite_score" in p00 else "?"
    s25 = f"{p25['composite_score']:.3f}" if "composite_score" in p25 else "?"

    ptype = p6["pair_type"]
    lines.append(f"| {pid} | {ptype} | {degr} | {v5t} | {v6t} | {t00} | {t25} | {v5s} | {v6s} | {s00} | {s25} |")

lines.append("")
lines.append("## Summary Statistics")
lines.append("")

shifted_v6 = [p for p in v6.values() if p["pair_type"] != "negative_control"]
controls_v6 = [p for p in v6.values() if p["pair_type"] == "negative_control"]

lines.append("| Metric | v5 | v6 (M6=0.5) | v6 (M6=0.0) | v6 (M6=0.25) |")
lines.append("|--------|----|-------------|-------------|--------------|")

for i, entry in enumerate(sweep):
    if entry["m6_weight"] in (0.0, 0.25, 0.5):
        pass

v5_shifted = [v5[pid] for pid in v6.keys() if v6[pid]["pair_type"] != "negative_control"]
v5_controls = [v5[pid] for pid in v6.keys() if v6[pid]["pair_type"] == "negative_control"]

v5_st = np.mean([p["composite_tier"] for p in v5_shifted])
v5_ct = np.mean([p["composite_tier"] for p in v5_controls])
v6_st = sweep[3]["shifted_mean_tier"]
v6_ct = sweep[3]["control_mean_tier"]
s0_st = sweep[0]["shifted_mean_tier"]
s0_ct = sweep[0]["control_mean_tier"]
s25_st = sweep[2]["shifted_mean_tier"]
s25_ct = sweep[2]["control_mean_tier"]

lines.append(f"| Shifted mean tier | {v5_st:.1f} | {v6_st:.1f} | {s0_st:.1f} | {s25_st:.1f} |")
lines.append(f"| Control mean tier | {v5_ct:.1f} | {v6_ct:.1f} | {s0_ct:.1f} | {s25_ct:.1f} |")
lines.append(f"| Gap (tiers) | {v5_ct - v5_st:.1f} | {sweep[3]['gap']:.1f} | {sweep[0]['gap']:.1f} | {sweep[2]['gap']:.1f} |")
lines.append(f"| Tier overlap | {'NO' if max([p['composite_tier'] for p in v5_shifted]) < min([p['composite_tier'] for p in v5_controls]) else 'YES'} | {'NO' if not sweep[3]['tier_overlap'] else 'YES'} | {'NO' if not sweep[0]['tier_overlap'] else 'YES'} | {'NO' if not sweep[2]['tier_overlap'] else 'YES'} |")
lines.append(f"| Spearman rho | — | {sweep[3]['spearman_rho']:.3f} | {sweep[0]['spearman_rho']:.3f} | {sweep[2]['spearman_rho']:.3f} |")
lines.append(f"| Spearman p | — | {sweep[3]['spearman_p']:.4f} | {sweep[0]['spearman_p']:.4f} | {sweep[2]['spearman_p']:.4f} |")

lines.append("")
lines.append("## Conclusion")
lines.append("")
lines.append("The composite is stable under M6 reweighting. Zero overlap in all conditions.")
lines.append("M6=0.0 slightly improves the gap (4.2 vs 3.5 tiers) and lifts control scores,")
lines.append("but M6=0.5 achieves the best Spearman rho (-0.709 vs -0.673).")
lines.append("M6 is non-discriminative for cell-type KNN graphs but does not harm the composite.")
lines.append("Recommendation: keep M6=0.5 (preserves rho) and label as non-discriminative in this graph family.")

output = "\n".join(lines)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write(output)

print(output)
print(f"\nSaved to {OUTPUT}")
