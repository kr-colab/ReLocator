#!/usr/bin/env python
"""Benchmark the genotype data loader through repeated holdout training.

Drives the public ``Locator.train_holdout`` API on a synthetic genotype matrix
so it runs unchanged against both the old and new loader. To compare before and
after the GPU-resident loader refactor, run it once with the ``locator/`` changes
stashed and once with them applied:

    git stash
    pixi run python scripts/benchmark_dataloader.py
    git stash pop
    pixi run python scripts/benchmark_dataloader.py

The synthetic matrix is fixed-shape and seeded, so timings are comparable. One
Locator instance is reused across folds, mirroring how a Ray actor reuses a
worker; with the new loader the genotype table is then built once and reused,
so fold 0 pays the build and later folds do not.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-snps", type=int, default=1_000_000)
    p.add_argument("--n-samples", type=int, default=236)
    p.add_argument("--epochs", type=int, default=30, help="fixed epochs per fold")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--holdout", type=int, default=10, help="samples held out per fold")
    p.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help="enable PCA-init projection (the realistic n_SNPs >> n_samples "
        "regime); keeps the trainable model small so the data path is visible",
    )
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def gpu_snapshot():
    """Return a one-line nvidia-smi utilisation/memory snapshot, or a note."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "(no GPU reported)"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "(nvidia-smi unavailable)"


def make_synthetic_data(n_snps, n_samples, seed):
    """Build a pre-filtered int8 genotype matrix and matching sample metadata."""
    rng = np.random.default_rng(seed)
    # (n_snps, n_samples) is locator's filtered-genotype layout.
    filtered = rng.integers(0, 3, size=(n_snps, n_samples), dtype=np.int8)
    samples = np.array([f"s{i}" for i in range(n_samples)])
    sample_df = pd.DataFrame(
        {
            "sampleID": samples,
            "x": rng.uniform(-120, -115, size=n_samples),
            "y": rng.uniform(35, 45, size=n_samples),
        }
    )
    return filtered, samples, sample_df


def main():
    args = parse_args()

    from locator.core import Locator

    print("GPU before run :", gpu_snapshot())
    print(
        f"Synthetic data : {args.n_snps:,} SNPs x {args.n_samples} samples "
        f"(int8, {args.n_snps * args.n_samples / 1e6:.0f} MB resident)"
    )

    t0 = time.perf_counter()
    filtered, samples, sample_df = make_synthetic_data(
        args.n_snps, args.n_samples, args.seed
    )
    print(f"Data generation: {time.perf_counter() - t0:.1f} s\n")

    config = {
        "out": "/tmp/locator_benchmark",
        "sample_data": sample_df,
        "max_epochs": args.epochs,
        "patience": args.epochs + 1000,  # disable early stopping
        "batch_size": args.batch_size,
        "width": args.width,
        "nlayers": args.n_layers,
        "keras_verbose": 0,
        "holdout_no_intermediate_saves": True,
        "save_fold_models": False,
    }
    if args.pca_components:
        config["pca_components"] = args.pca_components
        # Keep the projection frozen so per-epoch time reflects the data path
        # rather than a second fine-tuning phase.
        config["pca_finetune"] = False

    # One Locator reused across folds, as a Ray worker would be.
    loc = Locator(config)
    rng = np.random.default_rng(args.seed)

    fold_times = []
    for fold in range(args.folds):
        holdout_idx = rng.choice(args.n_samples, args.holdout, replace=False)
        t_fold = time.perf_counter()
        loc.train_holdout(
            genotypes=None,
            samples=samples,
            holdout_indices=list(holdout_idx),
            filtered_genotypes=filtered,
        )
        elapsed = time.perf_counter() - t_fold
        fold_times.append(elapsed)
        print(
            f"fold {fold}: {elapsed:7.2f} s total  "
            f"{elapsed / args.epochs * 1000:7.1f} ms/epoch"
        )

    print()
    print("GPU during/after:", gpu_snapshot())
    print()
    steady = fold_times[1:] or fold_times
    print(f"fold 0 (incl. table build): {fold_times[0]:.2f} s")
    print(f"steady-state fold mean    : {np.mean(steady):.2f} s")
    print(f"steady-state ms/epoch     : {np.mean(steady) / args.epochs * 1000:.1f} ms")
    print(
        f"build-once saving (fold0 - steady mean): "
        f"{fold_times[0] - np.mean(steady):+.2f} s"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
