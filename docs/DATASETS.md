# Datasets (M1)

For M1, datasets are stored as small `.npz` files containing a single time series.
The companion Python package provides a minimal standard container,
`TimeSeriesDataset`, and helpers to save/load it.

## On-disk format: `.npz`

A dataset file is a NumPy compressed archive with keys:

- `t` (float array, shape `(T,)`): strictly increasing time points
- `y` (array, shape `(T,)`): observations (e.g. reported cases); expected non-negative
- `name` (string): optional human-readable label

The intended reader-facing API is:

- `diff_epi_inference.save_timeseries_npz(ds, path)`
- `diff_epi_inference.load_timeseries_npz(path)`

## Synthetic paired SEIR example

A tiny paired deterministic + stochastic SEIR example can be written to disk via:

```bash
python scripts/make_synthetic_dataset.py --outdir data/synthetic/paired-seir
```

This produces:

- `data/synthetic/paired-seir/reported_deterministic.npz`
- `data/synthetic/paired-seir/reported_stochastic.npz`

Both are generated from the same conceptual model and share the same observation
pipeline (reporting + delay + NegBin noise). They differ only in the simulator:
Euler discretisation (deterministic) vs tau-leaping (stochastic).
