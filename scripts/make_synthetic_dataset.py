#!/usr/bin/env python3
"""Write tiny synthetic datasets used by the handbook.

Currently produces a paired deterministic + stochastic SEIR example, both passed
through the same reporting observation model, and stores them as .npz files.

Usage
-----
python scripts/make_synthetic_dataset.py --outdir data/synthetic/paired-seir

This script is intentionally lightweight (stdlib + NumPy + the local package).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from diff_epi_inference.examples import make_paired_seir_reported_datasets
from diff_epi_inference.io import save_timeseries_npz


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/synthetic/paired-seir"),
        help="Output directory to write .npz datasets into.",
    )
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--reporting-rate", type=float, default=0.3)
    p.add_argument("--seed-det", type=int, default=0)
    p.add_argument("--seed-sto", type=int, default=1)

    args = p.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ds_det, ds_sto = make_paired_seir_reported_datasets(
        seed_det=args.seed_det,
        seed_sto=args.seed_sto,
        steps=args.steps,
        dt=args.dt,
        reporting_rate=args.reporting_rate,
    )

    save_timeseries_npz(ds_det, outdir / "reported_deterministic.npz")
    save_timeseries_npz(ds_sto, outdir / "reported_stochastic.npz")

    print(f"Wrote: {outdir / 'reported_deterministic.npz'}")
    print(f"Wrote: {outdir / 'reported_stochastic.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
