"""Exp 5: Gene-level probes on Geneformer token embeddings.

Preregistered experiment: extract gene embeddings from Geneformer's
token embedding layer, pull regulatory edges from OmniPath (DoRothEA +
CollecTRI), and run three gene-level probes (tf_target, hub_tf, rsa).

Usage:
    .venv-census/bin/python scripts/exp5_gene_level_probes.py
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

OUTPUT_DIR = Path("results/gene_probes")
PREREG_DIR = Path("docs/frozen_prereg_v4")

GENEFORMER_MODEL_ID = "ctheodoris/Geneformer"

OMNIPATH_URL = (
    "https://omnipathdb.org/interactions?"
    "datasets=dorothea,collectri&organisms=9606"
    "&dorothea_levels=A,B,C"
    "&genesymbols=yes&format=json"
)


def _load_geneformer_gene_embeddings():
    """Load gene token embeddings from Geneformer's HuggingFace model.

    Returns:
        gene_embeddings: (n_genes, d) float64 array
        gene_ids: list of gene identifier strings (Ensembl IDs)
    """
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Loading Geneformer from HuggingFace...")

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed.")
        print("Run: pip install transformers torch")
        return None, None

    try:
        model = AutoModel.from_pretrained(GENEFORMER_MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load Geneformer model: {e}")
        print(f"Model ID: {GENEFORMER_MODEL_ID}")
        print("This may require HuggingFace access approval.")
        return None, None

    # Extract token embedding matrix
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Extracting token embeddings...")
    embedding_layer = model.embeddings.word_embeddings
    weight = np.array(embedding_layer.weight.detach().cpu().tolist())
    print(f"  Raw embedding matrix: {weight.shape}")

    # Load token dictionary — Geneformer uses a pickle dict, not a standard tokenizer
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Loading token dictionary...")
    try:
        import pickle
        from huggingface_hub import hf_hub_download
        token_dict_path = hf_hub_download(
            repo_id=GENEFORMER_MODEL_ID,
            filename="geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl",
        )
        with open(token_dict_path, "rb") as f:
            vocab = pickle.load(f)
        print(f"  Token dictionary: {len(vocab)} entries")
    except Exception as e2:
        print(f"ERROR: Failed to load token dictionary: {e2}")
        return None, None

    # Map: keep only Ensembl gene IDs (ENSG...), skip special tokens
    gene_ids = []
    gene_indices = []
    for token, idx in tqdm(sorted(vocab.items()), desc="Filtering gene tokens"):
        if token.startswith("ENSG") and idx < weight.shape[0]:
            gene_ids.append(token)
            gene_indices.append(idx)

    if not gene_ids:
        print("ERROR: No Ensembl gene IDs found in vocabulary")
        return None, None

    gene_embeddings = weight[gene_indices].astype(np.float64)
    print(f"  Gene embeddings: {gene_embeddings.shape} ({len(gene_ids)} genes)")

    return gene_embeddings, gene_ids


def _load_omnipath_edges():
    """Pull TF-target regulatory edges from OmniPath REST API.

    Returns:
        edges: list of (source_symbol, target_symbol) tuples
        edge_metadata: dict with counts and sources
    """
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Querying OmniPath API...")

    try:
        req = urllib.request.Request(OMNIPATH_URL)
        req.add_header("User-Agent", "preflight-bio/1.0")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"ERROR: OmniPath query failed: {e}")
        return None, None

    edges = []
    sources_seen = set()
    targets_seen = set()

    for row in tqdm(data, desc="Processing OmniPath edges"):
        src = row.get("source_genesymbol", "")
        tgt = row.get("target_genesymbol", "")
        if src and tgt:
            edges.append((src, tgt))
            sources_seen.add(src)
            targets_seen.add(tgt)

    metadata = {
        "n_edges": len(edges),
        "n_unique_tfs": len(sources_seen),
        "n_unique_targets": len(targets_seen),
        "url": OMNIPATH_URL,
    }

    print(f"  OmniPath edges: {len(edges)} ({len(sources_seen)} TFs, {len(targets_seen)} targets)")

    return edges, metadata


def _map_symbols_to_ensembl(gene_ids, edges):
    """Build a mapping from gene symbols (in OmniPath) to Ensembl IDs (in Geneformer).

    OmniPath uses gene symbols (e.g., TP53) while Geneformer uses Ensembl IDs
    (e.g., ENSG00000141510). We use the MyGene.info REST API for mapping.

    Returns the edges remapped to Ensembl IDs, plus the mapping dict.
    """
    print(f"  [{datetime.now(timezone.utc).isoformat()}] Mapping gene symbols to Ensembl IDs...")

    # Collect all unique symbols from edges
    all_symbols = set()
    for src, tgt in edges:
        all_symbols.add(src)
        all_symbols.add(tgt)

    print(f"  Unique symbols to map: {len(all_symbols)}")

    gene_id_set = set(gene_ids)

    # Query MyGene.info in batches
    symbol_to_ensembl = {}
    symbols_list = sorted(all_symbols)
    batch_size = 1000

    for i in tqdm(range(0, len(symbols_list), batch_size), desc="MyGene.info batches"):
        batch = symbols_list[i:i + batch_size]
        query = ",".join(batch)
        url = f"https://mygene.info/v3/query?q={query}&scopes=symbol&fields=ensembl.gene&species=human&size={len(batch)}"

        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "preflight-bio/1.0")

            # POST for large batches
            post_data = json.dumps({
                "q": batch,
                "scopes": ["symbol"],
                "fields": ["ensembl.gene"],
                "species": "human",
            }).encode()
            req = urllib.request.Request(
                "https://mygene.info/v3/query",
                data=post_data,
                headers={"Content-Type": "application/json", "User-Agent": "preflight-bio/1.0"},
            )

            with urllib.request.urlopen(req, timeout=60) as resp:
                results = json.loads(resp.read().decode())

            for item in results:
                symbol = item.get("query", "")
                ensembl = item.get("ensembl", {})
                if isinstance(ensembl, list):
                    ensembl = ensembl[0]
                if isinstance(ensembl, dict):
                    ens_id = ensembl.get("gene", "")
                    if ens_id in gene_id_set:
                        symbol_to_ensembl[symbol] = ens_id
        except Exception as e:
            print(f"  WARNING: MyGene.info batch failed: {e}")
            continue

    print(f"  Mapped {len(symbol_to_ensembl)} / {len(all_symbols)} symbols to Geneformer Ensembl IDs")

    # Remap edges
    remapped_edges = []
    for src, tgt in edges:
        src_ens = symbol_to_ensembl.get(src)
        tgt_ens = symbol_to_ensembl.get(tgt)
        if src_ens and tgt_ens:
            remapped_edges.append((src_ens, tgt_ens))

    print(f"  Remapped edges: {len(remapped_edges)} / {len(edges)}")

    return remapped_edges, symbol_to_ensembl


def main():
    from preflight.core.preregister import (
        DatasetSpec,
        load_preregistration,
        verify_preregistration,
    )
    from preflight.core.runner import ALL_MODULE_SOURCES, resolve_hyperparameters
    from preflight.probes import tf_target_probe, hub_tf_probe, rsa_probe

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{datetime.now(timezone.utc).isoformat()}] Exp 5: Gene-level probes")

    # ==================================================================
    # STEP 1: Load and verify frozen preregistration
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [1/7] Loading frozen preregistration...")

    prereg_path = PREREG_DIR / "exp5_gene_level_probes.json"
    if not prereg_path.exists():
        print(f"ERROR: Frozen prereg not found at {prereg_path}")
        print("Run: python scripts/freeze_preregistration_v4.py")
        return 1

    prereg_record = load_preregistration(prereg_path)
    frozen_sha = prereg_record.sha256
    print(f"  Frozen SHA: {frozen_sha[:32]}...")

    hp, weights = resolve_hyperparameters({"k": 5})
    canonical_hp = {**hp, "weights": weights}

    dataset_spec = DatasetSpec(
        name="exp5_gene_level_probes",
        source_description="Geneformer gene token embeddings (from model weights)",
        target_description="N/A (gene-level, not cell-level)",
        n_source=None,
        n_target=None,
        extra={
            "model": "geneformer",
            "regulatory_source": "OmniPath (DoRothEA + CollecTRI)",
            "probes": ["tf_target", "hub_tf", "rsa"],
        },
    )

    match, msg = verify_preregistration(prereg_record, ALL_MODULE_SOURCES, canonical_hp, dataset_spec)
    if not match:
        print(f"  VERIFICATION FAILED: {msg}")
        return 1
    print(f"  VERIFIED: {msg}")

    # ==================================================================
    # STEP 2: Load Geneformer gene embeddings
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [2/7] Loading Geneformer gene embeddings...")

    gene_embeddings, gene_ids = _load_geneformer_gene_embeddings()
    if gene_embeddings is None:
        print("\nFATAL: Could not load Geneformer gene embeddings.")
        print("This experiment requires HuggingFace model access.")
        return 1

    # ==================================================================
    # STEP 3: Pull regulatory edges from OmniPath
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [3/7] Pulling regulatory edges from OmniPath...")

    raw_edges, edge_metadata = _load_omnipath_edges()
    if raw_edges is None:
        print("\nFATAL: Could not load OmniPath regulatory edges.")
        return 1

    # ==================================================================
    # STEP 4: Map gene symbols to Ensembl IDs
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [4/7] Mapping symbols to Ensembl IDs...")

    edges, symbol_map = _map_symbols_to_ensembl(gene_ids, raw_edges)
    if len(edges) < 100:
        print(f"\nWARNING: Only {len(edges)} edges mapped. Results may be unreliable.")

    gene_ids_arr = np.array(gene_ids, dtype=str)

    # ==================================================================
    # STEP 5: Run gene-level probes
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [5/7] Running gene-level probes...")

    # TF-target probe
    print(f"\n  [{datetime.now(timezone.utc).isoformat()}] Running tf_target probe...")
    tf_result = tf_target_probe(
        gene_embeddings=gene_embeddings,
        gene_ids=gene_ids_arr,
        regulatory_edges=edges,
        model_name="geneformer",
        min_targets=30,
        seed=0,
    )
    print(f"  TF-target: mean AUC = {tf_result.mean:.4f} "
          f"[{tf_result.ci_lo:.4f}, {tf_result.ci_hi:.4f}], "
          f"n_TFs = {tf_result.n_items}")

    # Hub-TF probe
    print(f"\n  [{datetime.now(timezone.utc).isoformat()}] Running hub_tf probe...")
    hub_result = hub_tf_probe(
        gene_embeddings=gene_embeddings,
        gene_ids=gene_ids_arr,
        regulatory_edges=edges,
        model_name="geneformer",
        hub_quantile=0.90,
        k=5,
        seed=0,
    )
    print(f"  Hub-TF: mean AUC = {hub_result.mean:.4f} "
          f"[{hub_result.ci_lo:.4f}, {hub_result.ci_hi:.4f}], "
          f"n_TFs = {hub_result.n_items}")

    # RSA probe
    print(f"\n  [{datetime.now(timezone.utc).isoformat()}] Running rsa probe...")
    rsa_result = rsa_probe(
        gene_embeddings=gene_embeddings,
        gene_ids=gene_ids_arr,
        regulatory_edges=edges,
        model_name="geneformer",
    )
    rsa_rho = rsa_result.details.get("rho", float("nan"))
    rsa_p = rsa_result.details.get("p_value", float("nan"))
    print(f"  RSA: Spearman rho = {rsa_rho:.4f}, p = {rsa_p:.4e}, "
          f"n_pairs = {rsa_result.n_items}")

    # ==================================================================
    # STEP 6: Test hypotheses
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [6/7] Testing hypotheses...")

    # H5.1: TF-target mean AUC > 0.6
    h5_1_pass = tf_result.mean > 0.6
    print(f"\n  H5.1: TF-target mean AUC > 0.6")
    print(f"    Observed: {tf_result.mean:.4f}")
    print(f"    H5.1 {'PASS' if h5_1_pass else 'FAIL'}")

    # H5.2: Hub-TF AUC > 0.6
    h5_2_pass = hub_result.mean > 0.6
    print(f"\n  H5.2: Hub-TF AUC > 0.6")
    print(f"    Observed: {hub_result.mean:.4f}")
    print(f"    H5.2 {'PASS' if h5_2_pass else 'FAIL'}")

    # H5.3: RSA Spearman rho > 0 (p < 0.05)
    h5_3_pass = rsa_rho > 0 and rsa_p < 0.05
    print(f"\n  H5.3: RSA Spearman rho > 0 (p < 0.05)")
    print(f"    Observed: rho = {rsa_rho:.4f}, p = {rsa_p:.4e}")
    print(f"    H5.3 {'PASS' if h5_3_pass else 'FAIL'}")

    # ==================================================================
    # STEP 7: Save results
    # ==================================================================
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] [7/7] Saving results...")

    def _probe_result_to_dict(pr):
        return {
            "model_name": pr.model_name,
            "probe_name": pr.probe_name,
            "mean": pr.mean,
            "ci_lo": pr.ci_lo,
            "ci_hi": pr.ci_hi,
            "n_items": pr.n_items,
            "scores": pr.scores[:50],  # truncate for readability
            "details": {k: v for k, v in pr.details.items()
                        if k != "probed_tfs"},  # skip large lists
        }

    summary = {
        "timestamp": timestamp,
        "frozen_prereg_sha": frozen_sha,
        "geneformer_model": GENEFORMER_MODEL_ID,
        "gene_embeddings_shape": list(gene_embeddings.shape),
        "n_genes": len(gene_ids),
        "regulatory_edges": {
            "source": "OmniPath (DoRothEA + CollecTRI)",
            "n_raw_edges": edge_metadata["n_edges"],
            "n_mapped_edges": len(edges),
            "n_mapped_symbols": len(symbol_map),
        },
        "probes": {
            "tf_target": _probe_result_to_dict(tf_result),
            "hub_tf": _probe_result_to_dict(hub_result),
            "rsa": _probe_result_to_dict(rsa_result),
        },
        "hypotheses": {
            "H5.1_tf_target_auc": {
                "pass": h5_1_pass,
                "observed": tf_result.mean,
                "threshold": 0.6,
            },
            "H5.2_hub_tf_auc": {
                "pass": h5_2_pass,
                "observed": hub_result.mean,
                "threshold": 0.6,
            },
            "H5.3_rsa_rho": {
                "pass": h5_3_pass,
                "rho": rsa_rho,
                "p_value": rsa_p,
                "rho_threshold": 0.0,
                "p_threshold": 0.05,
            },
        },
    }

    summary_path = OUTPUT_DIR / f"summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")

    # Save full probe results with all TF names
    full_results = {
        "tf_target": {
            "scores": tf_result.scores,
            "details": tf_result.details,
        },
        "hub_tf": {
            "scores": hub_result.scores,
            "details": hub_result.details,
        },
        "rsa": {
            "scores": rsa_result.scores,
            "details": rsa_result.details,
        },
    }
    full_path = OUTPUT_DIR / f"full_probe_results_{timestamp}.json"
    with open(full_path, "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"  Full results: {full_path}")

    # ==================================================================
    # Final report
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("EXP 5 RESULTS: Gene-level probes")
    print(f"{'=' * 60}")
    print(f"  Prereg SHA: {frozen_sha[:32]}...")
    print(f"  Gene embeddings: {gene_embeddings.shape}")
    print(f"  Regulatory edges: {len(edges)} (mapped from {edge_metadata['n_edges']})")
    print(f"  TF-target AUC:  {tf_result.mean:.4f}  {'PASS' if h5_1_pass else 'FAIL'} (> 0.6)")
    print(f"  Hub-TF AUC:     {hub_result.mean:.4f}  {'PASS' if h5_2_pass else 'FAIL'} (> 0.6)")
    print(f"  RSA rho:        {rsa_rho:.4f}  {'PASS' if h5_3_pass else 'FAIL'} (> 0, p < 0.05)")
    n_pass = sum([h5_1_pass, h5_2_pass, h5_3_pass])
    print(f"  Total: {n_pass}/3 hypotheses confirmed")
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Done. Artifacts in {OUTPUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
