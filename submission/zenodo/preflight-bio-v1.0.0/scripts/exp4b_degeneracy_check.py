"""Experiment 4b: Degenerate Embedding Characterization.

Part A — Confirmatory re-analysis: verify M4 values from Exp 1/2.
Part B — Prospective counterfactual: simulate worst-module gate.

No Census access needed. Pure analysis of existing results.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_DIR = Path("results/degeneracy_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_PREREG_PATH = Path("docs/frozen_prereg_v7/exp4b_degeneracy_check.json")

EXP1_PATH = Path("results/modal_results/bag_of_genes_v6/bag_of_genes_baseline/summary_20260711_122910.json")
EXP2_PATH = Path("results/modal_results/sweep_v6/sweep/summary_20260711_221653.json")


def load_exp1():
    with open(EXP1_PATH) as f:
        return json.load(f)


def load_exp2():
    with open(EXP2_PATH) as f:
        return json.load(f)


def extract_m4_from_exp1(data):
    """Extract M4 scores per embedding from Exp 1."""
    results = {}
    for emb_name, emb_data in data["embedding_results"].items():
        results[emb_name] = {
            "m4_score": emb_data["module_scores"]["m4_domain_validity"]["score"],
            "m4_tier": emb_data["module_scores"]["m4_domain_validity"]["tier"],
            "overall_tier": emb_data["overall_tier"],
        }
    return {"lung": results}


def extract_m4_from_exp2(data):
    """Extract M4 scores per tissue x embedding from Exp 2."""
    results = {}
    for tissue_name, tissue_data in data["per_condition"].items():
        tissue_results = {}
        for emb_name, emb_data in tissue_data.items():
            tissue_results[emb_name] = {
                "m4_score": emb_data["module_scores"]["m4_domain_validity"]["score"],
                "m4_tier": emb_data["module_scores"]["m4_domain_validity"]["tier"],
                "overall_tier": emb_data["overall_tier"],
            }
        results[tissue_name] = tissue_results
    return results


def simulate_worst_module_gate(module_scores, current_tier):
    """Apply worst-module gate: any module <= Tier 1 caps overall to <= Tier 3."""
    min_tier = min(m["tier"] for m in module_scores.values())
    if min_tier <= 1:
        return min(current_tier, 3)
    return current_tier


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"Experiment 4b: Degenerate Embedding Characterization")
    print(f"=" * 60)

    exp1 = load_exp1()
    exp2 = load_exp2()

    m4_exp1 = extract_m4_from_exp1(exp1)
    m4_exp2 = extract_m4_from_exp2(exp2)

    all_m4 = {**m4_exp1, **m4_exp2}

    print(f"\n--- Part A: Confirmatory Re-analysis ---\n")

    # H4b.1: BoG-512 has M4 < 0.01 in all tissues
    print("H4b.1: BoG-512 M4 participation ratio < 0.01 in all conditions")
    h4b1_pass = True
    bog_m4_values = []
    for tissue, embeddings in all_m4.items():
        for bog_key in ["bag_of_genes_512", "bag_of_genes_pca512"]:
            if bog_key in embeddings:
                m4 = embeddings[bog_key]["m4_score"]
                bog_m4_values.append({"tissue": tissue, "m4": m4})
                status = "PASS" if m4 < 0.01 else "FAIL"
                if m4 >= 0.01:
                    h4b1_pass = False
                print(f"  {tissue}: M4 = {m4:.4f} [{status}]")
                break
    print(f"  >>> H4b.1: {'SUPPORTED' if h4b1_pass else 'REJECTED'}")

    # H4b.3: No non-degenerate embedding has M4 < 0.05
    print(f"\nH4b.3: No non-degenerate embedding (Geneformer, scVI) has M4 < 0.05")
    h4b3_pass = True
    non_deg_m4_values = []
    for tissue, embeddings in all_m4.items():
        for emb_name, emb_data in embeddings.items():
            if "bag_of_genes" in emb_name:
                continue
            m4 = emb_data["m4_score"]
            non_deg_m4_values.append({"tissue": tissue, "embedding": emb_name, "m4": m4})
            if m4 < 0.05:
                h4b3_pass = False
                print(f"  FAIL: {tissue}/{emb_name}: M4 = {m4:.4f} < 0.05")
    if h4b3_pass:
        print(f"  All non-degenerate embeddings have M4 >= 0.05")
        min_non_deg = min(v["m4"] for v in non_deg_m4_values)
        print(f"  Minimum non-degenerate M4: {min_non_deg:.4f}")
    print(f"  >>> H4b.3: {'SUPPORTED' if h4b3_pass else 'REJECTED'}")

    print(f"\n--- Part B: Prospective Counterfactual ---\n")

    # H4b.2: Worst-module gate reclassifies BoG-512 to <= Tier 3 in all conditions
    print("H4b.2: Worst-module gate (any module <= Tier 1 caps overall to <= Tier 3)")
    h4b2_pass = True
    gate_results = []

    # Check BoG-512 reclassification
    print("  BoG-512 reclassification:")
    for tissue, embeddings in all_m4.items():
        for emb_name in ["bag_of_genes_512", "bag_of_genes_pca512"]:
            if emb_name not in embeddings:
                continue
            emb_data = embeddings[emb_name]
            original_tier = emb_data["overall_tier"]

            # Get full module scores for gate simulation
            if tissue == "lung" and emb_name in exp1["embedding_results"]:
                modules = exp1["embedding_results"][emb_name]["module_scores"]
            elif tissue in exp2.get("per_condition", {}):
                if emb_name in exp2["per_condition"][tissue]:
                    modules = exp2["per_condition"][tissue][emb_name]["module_scores"]
                else:
                    continue
            else:
                continue

            gated_tier = simulate_worst_module_gate(modules, original_tier)
            reclassified = gated_tier <= 3
            if not reclassified and original_tier > 3:
                h4b2_pass = False
            gate_results.append({
                "tissue": tissue,
                "embedding": emb_name,
                "original_tier": original_tier,
                "gated_tier": gated_tier,
                "reclassified": original_tier > 3 and gated_tier <= 3,
            })
            print(f"    {tissue}: Tier {original_tier} -> Tier {gated_tier} "
                  f"{'[RECLASSIFIED]' if original_tier > 3 and gated_tier <= 3 else ''}")
            break

    # Check no non-degenerate embedding is reclassified
    print("  Non-degenerate embedding stability:")
    non_deg_reclassified = False
    for tissue, embeddings in all_m4.items():
        for emb_name, emb_data in embeddings.items():
            if "bag_of_genes" in emb_name:
                continue
            original_tier = emb_data["overall_tier"]

            if tissue == "lung" and emb_name in exp1["embedding_results"]:
                modules = exp1["embedding_results"][emb_name]["module_scores"]
            elif tissue in exp2.get("per_condition", {}):
                if emb_name in exp2["per_condition"][tissue]:
                    modules = exp2["per_condition"][tissue][emb_name]["module_scores"]
                else:
                    continue
            else:
                continue

            gated_tier = simulate_worst_module_gate(modules, original_tier)
            if gated_tier < original_tier:
                non_deg_reclassified = True
                h4b2_pass = False
                print(f"    FAIL: {tissue}/{emb_name}: Tier {original_tier} -> {gated_tier}")

    if not non_deg_reclassified:
        print(f"    No non-degenerate embedding reclassified by gate")
    print(f"  >>> H4b.2: {'SUPPORTED' if h4b2_pass else 'REJECTED'}")

    # Save results
    output = {
        "experiment": "exp4b_degeneracy_check",
        "timestamp": timestamp,
        "prereg_sha": json.load(open(FROZEN_PREREG_PATH))["sha256"] if FROZEN_PREREG_PATH.exists() else "N/A",
        "data_sources": [str(EXP1_PATH), str(EXP2_PATH)],
        "part_a": {
            "H4b.1": {
                "description": "BoG-512 M4 < 0.01 in all conditions",
                "supported": h4b1_pass,
                "values": bog_m4_values,
            },
            "H4b.3": {
                "description": "No non-degenerate embedding has M4 < 0.05",
                "supported": h4b3_pass,
                "values": non_deg_m4_values,
            },
        },
        "part_b": {
            "H4b.2": {
                "description": "Worst-module gate reclassifies BoG-512 to <= Tier 3, no non-degenerate reclassified",
                "supported": h4b2_pass,
                "gate_rule": "any_module_tier_le_1_caps_overall_to_le_3",
                "results": gate_results,
                "non_degenerate_reclassified": non_deg_reclassified,
            },
        },
    }

    output_path = OUTPUT_DIR / "degeneracy_v7.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY:")
    print(f"  H4b.1 (Part A): {'SUPPORTED' if h4b1_pass else 'REJECTED'}")
    print(f"  H4b.3 (Part A): {'SUPPORTED' if h4b3_pass else 'REJECTED'}")
    print(f"  H4b.2 (Part B): {'SUPPORTED' if h4b2_pass else 'REJECTED'}")


if __name__ == "__main__":
    main()
