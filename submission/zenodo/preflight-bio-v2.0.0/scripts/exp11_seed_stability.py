"""Seed stability check for Experiment 11.

Runs the full exp11 pipeline with an alternate seed to verify
headline numbers (rho, inversion rate) are stable across subsamples.

Exploratory robustness — not confirmatory.

Usage:
    python scripts/exp11_seed_stability.py <seed>
"""
import sys
import importlib.util
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/exp11_seed_stability.py <seed>")
        sys.exit(1)

    seed = int(sys.argv[1])
    print(f"Running exp11 with alternate seed: {seed}")

    script_dir = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "exp11", script_dir / "exp11_cross_tissue_validity.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.SEED = seed
    mod.OUTPUT_DIR = Path(f"results/exp11_seed_stability/seed_{seed}")
    mod.main()


if __name__ == "__main__":
    main()
