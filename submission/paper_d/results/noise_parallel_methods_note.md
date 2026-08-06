# Noise Dose-Response: Parallelization and Infrastructure Notes

Date: 2026-07-13

## Runtime comparison

- **Sequential (single container)**: >28 hours, zero output, appeared stuck/timed out
- **Parallel (24 containers)**: ~45 minutes wall-clock (all conditions complete)
- Infrastructure: Modal, 24 containers (one per tissue x embedding), 16GB memory each
- Each container: pulls Census data independently, generates embedding, runs 7 sigma levels

## S3 timeout on brain_bog_pca_512

23/24 workers completed on first launch. `brain_bog_pca_512` hit an S3 timeout
pulling Census expression matrix data (the X matrix for brain tissue):

```
SOMAError: S3: Failed to read S3 object [...]/a0.tdb
curlCode: 28, Timeout was reached;
Details: Operation too slow. Less than 1 bytes/sec transferred the last 3 seconds
```

Cause: 24 simultaneous Census pulls from the same S3 bucket caused throttling.
Brain tissue has the largest expression matrix, making it the most timeout-prone.

Retried successfully as a single container (no competing traffic). All 24 condition
JSONs saved to volume, aggregated into noise_dose_response.json.

## Modal app IDs

- `ap-TBkzzMSuGsM9UsAsHvhktM`: parallel launch (23/24 succeeded, brain_bog_pca_512 failed)
- `ap-3cT9qGJ9yw5mnuEdSETrLj`: retry brain_bog_pca_512 + aggregation (succeeded)
- `ap-k9OVAqOgAdNpTLZRLTs4xE`: old sequential run (stuck, zero output after 28h)
- `ap-ffk2NRmHlUuWvIp4VFRe3T`: second sequential run (eventually completed but slow)

## Scripts

- `scripts/modal_noise_parallel.py`: 24-way parallel wrapper
- `scripts/modal_noise_retry_and_aggregate.py`: retry + aggregation
- `scripts/exp10_noise_dose_response.py`: original sequential logic (hash bug fixed)

## Seed derivation

Deterministic: `SEED + tissue_idx * 10000 + emb_idx * 1000 + sigma_idx`
where SEED = 20260713, tissue_idx in {0..3}, emb_idx in {0..5}, sigma_idx in {0..6}.
