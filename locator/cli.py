"""Command line interface for locator"""

import argparse
import json
import os
import sys
import time

from .core import Locator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", help="VCF with SNPs for all samples.")
    parser.add_argument("--zarr", help="zarr file of SNPs for all samples.")
    parser.add_argument(
        "--matrix",
        help="tab-delimited matrix of genotype dosage with first column "
        "named 'sampleID'. Accepts both hard-call dosage (integers 0/1/2) "
        "and continuous expected dosage from genotype-likelihood pipelines "
        "(floats in [0, 2]). For the GL preprocessing workflow, see "
        "`scripts/gl_to_locator.py --help`. E.g., "
        "sampleID\\tsite1\\tsite2\\t... "
        "msp1\\t0\\t1\\t... "
        "msp2\\t2\\t0\\t...",
    )
    parser.add_argument(
        "--microsat",
        help="tab-delimited microsatellite genotype table with 'sampleID' "
        "as the first column and one column per locus (pair format, "
        "e.g. '12,14'), or two consecutive columns per locus (two-column "
        "format). Loaded natively into Locator as a multi-allelic dosage "
        "matrix; missing genotypes are imputed to per-allele site mean. "
        "No preprocessing script is required.",
    )
    parser.add_argument(
        "--microsat_maf",
        default=0.01,
        type=float,
        help="Drop microsat alleles below this per-locus frequency. default: 0.01",
    )
    parser.add_argument(
        "--sample_data",
        help="tab-delimited text file with columns\
                         'sampleID \t x \t y'.\
                          SampleIDs must exactly match those in the \
                          VCF. X and Y values for \
                          samples without known locations should \
                          be NA.",
    )
    parser.add_argument(
        "--train_split",
        default=0.9,
        type=float,
        help="0-1, proportion of samples to use for training. \
                          default: 0.9 ",
    )
    parser.add_argument(
        "--bootstrap",
        default=False,
        action="store_true",
        help="Run bootstrap replicates by retraining on bootstrapped data.",
    )
    parser.add_argument(
        "--jacknife",
        default=False,
        action="store_true",
        help="Run jacknife uncertainty estimate on a trained network. \
                    NOTE: we recommend this only as a fast heuristic \
                    -- use the bootstrap option or run windowed analyses \
                    for final results.",
    )
    parser.add_argument(
        "--jacknife_prop",
        default=0.05,
        type=float,
        help="proportion of SNPs to remove for jacknife resampling.\
                    default: 0.05",
    )
    parser.add_argument(
        "--nboots",
        default=50,
        type=int,
        help="number of bootstrap replicates to run.\
                    default: 50",
    )
    parser.add_argument("--batch_size", default=32, type=int, help="default: 32")
    parser.add_argument("--max_epochs", default=5000, type=int, help="default: 5000")
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="n epochs to run the optimizer after last \
                          improvement in validation loss. \
                          default: 100",
    )
    parser.add_argument(
        "--min_mac",
        default=2,
        type=int,
        help="minimum minor allele count.\
                          default: 2",
    )
    parser.add_argument(
        "--max_SNPs",
        default=None,
        type=int,
        help="randomly select max_SNPs variants to use in the analysis",
    )
    parser.add_argument(
        "--impute_missing",
        default=False,
        action="store_true",
        help="impute missing genotypes using mean allele frequency",
    )
    parser.add_argument(
        "--dropout_prop",
        default=0.25,
        type=float,
        help="proportion of weights to zero at the dropout layer. \
                          default: 0.25",
    )
    parser.add_argument(
        "--nlayers",
        default=10,
        type=int,
        help="number of layers in the network. \
                          default: 10",
    )
    parser.add_argument(
        "--width",
        default=256,
        type=int,
        help="number of units in each layer. \
                          default: 256",
    )
    parser.add_argument(
        "--pca_components",
        default=None,
        type=int,
        help="If set, prepend a PCA-initialized linear projection of this "
        "width as the first layer and fine-tune it. Recommended when the "
        "number of SNPs greatly exceeds the number of samples. "
        "default: None (disabled)",
    )
    parser.add_argument(
        "--no_pca_finetune",
        dest="pca_finetune",
        default=True,
        action="store_false",
        help="Keep the PCA projection frozen at its PCA initialization "
        "instead of running the low-learning-rate fine-tuning phase.",
    )
    parser.add_argument(
        "--pca_finetune_lr",
        default=1e-4,
        type=float,
        help="Learning rate for the PCA fine-tuning phase. default: 1e-4",
    )
    parser.add_argument(
        "--out",
        help="file name stem for output",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="random seed. default: None"
    )
    parser.add_argument(
        "--gpu_number",
        default=None,
        type=str,
        help="Specify which GPU to use (0-based index). For example, use '1' to use the second GPU. "
        "If not specified, uses the first available GPU. "
        "Use --disable_gpu to force CPU usage. default: None",
    )
    parser.add_argument(
        "--plot_history",
        default=True,
        type=bool,
        help="plot training history? default: True",
    )
    parser.add_argument(
        "--keras_verbose",
        default=1,
        type=int,
        help="verbose argument passed to keras in model training. \
                    0 = silent. 1 = progress bars for minibatches. 2 = show epochs. \
                    Yes, 1 is more verbose than 2. Blame keras. \
                    default: 1. ",
    )
    parser.add_argument(
        "--windows",
        default=False,
        action="store_true",
        help="Run windowed analysis over a single chromosome (requires zarr input).",
    )
    parser.add_argument("--window_start", default=0, help="default: 0")
    parser.add_argument("--window_stop", default=None, help="default: max snp position")
    parser.add_argument("--window_size", default=5e5, help="default: 500000")
    parser.add_argument("--load_params", help="Load parameters from previous run")
    parser.add_argument(
        "--predict_from_weights",
        help="Load saved weights",
    )
    parser.add_argument("--keep_weights", default=False, action="store_true")
    parser.add_argument(
        "--no_verbose",
        dest="verbose",
        default=True,
        action="store_false",
        help="Suppress prediction metrics",
    )
    parser.add_argument(
        "--disable_gpu",
        action="store_true",
        help="Disable GPU usage even if available. Useful when running multiple jobs "
        "or when GPU memory is needed for other tasks. default: False",
    )

    return parser.parse_args()


def main():  # noqa: C901
    """Main entry point for CLI"""
    args = parse_args()

    # Set GPU and seed
    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)
    if args.gpu_number is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number

    # Load old parameters if specified
    if args.load_params is not None:
        with open(args.load_params) as f:
            args.__dict__ = json.load(f)

    # Initialize locator
    loc = Locator(vars(args))

    # Store run parameters
    if args.out is not None:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out + "_params.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Load and sort data
    genotypes, samples = loc.load_genotypes(
        vcf=args.vcf,
        zarr=args.zarr,
        matrix=args.matrix,
        microsat=args.microsat,
        microsat_min_allele_freq=args.microsat_maf,
    )
    sample_data, locs = loc.sort_samples(samples, args.sample_data)

    # Track runtime
    start = time.time()

    # Run analysis based on mode
    if args.predict_from_weights:
        # Load genotypes and predict using saved weights
        loc.predict_from_weights(
            weights_path=args.predict_from_weights,
            genotypes=genotypes,
            samples=samples,
            sample_data_file=args.sample_data,
            save_preds_to_disk=True,
            return_df=True,
        )
    elif args.windows:
        if args.zarr is None:
            raise ValueError("Windows mode requires zarr input")

        window_start = int(args.window_start)
        window_size = int(args.window_size)
        window_stop = int(args.window_stop) if args.window_stop else None

        loc.run_windows(
            genotypes,
            samples,
            window_start=window_start,
            window_size=window_size,
            window_stop=window_stop,
        )

    elif args.jacknife:
        # Run jacknife analysis
        loc.train(
            genotypes=genotypes, samples=samples, sample_data_file=args.sample_data
        )
        loc.run_jacknife(genotypes, samples, prop=args.jacknife_prop)

    elif args.bootstrap:
        # Run bootstrap replicates
        for boot in range(args.nboots):
            print(f"\nBootstrap {boot + 1}/{args.nboots}")
            loc.train(
                genotypes=genotypes,
                samples=samples,
                sample_data_file=args.sample_data,
                boot=boot,
            )
            loc.predict(boot=boot, verbose=args.verbose)

    else:
        # Standard run
        loc.train(
            genotypes=genotypes, samples=samples, sample_data_file=args.sample_data
        )
        loc.predict(verbose=args.verbose)

    # Clean up weights if not keeping them
    if not args.keep_weights:
        if args.bootstrap:
            for boot in range(args.nboots):
                try:
                    os.remove(f"{args.out}_boot{boot}_weights.h5")
                except FileNotFoundError:
                    pass
        else:
            try:
                os.remove(f"{args.out}_weights.h5")
            except FileNotFoundError:
                pass

    # Report runtime
    end = time.time()
    print(f"Run time: {(end - start) / 60:.2f} minutes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
