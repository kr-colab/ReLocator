import os

# Suppress all TensorFlow and CUDA messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF logging
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU completely (CPU only)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress oneDNN messages

# Suppress XLA and CUDA messages
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

# Suppress CUDA/cuDNN messages
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Also suppress absl logging
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from locator import Locator
from locator.plotting import plot_error_summary
from locator.utils import weight_samples

vcf_path = "/sietch_colab/data_share/turtles_Actinemys/58-Actinemys/QC/58-Actinemys.pruned.vcf.gz"
coords_path = "/sietch_colab/data_share/turtles_Actinemys/actinemys_locator_metadata.tsv"
output_dir = "/sietch_colab/data_share/turtles_Actinemys/locator_output"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Configuration for Locator - FIXED parameter names
config = {
    "out": os.path.join(output_dir, "actinemys_basic"),
    "sample_data": coords_path,
    "vcf": vcf_path,
    "batch_size": 32,
    "width": 256,  # Number of units in hidden layers
    "nlayers": 8,  # Number of hidden layers
    "dropout_prop": 0.25,
    "max_epochs": 6,  # FIXED: was "epochs"
    "train_split": 0.8,  # FIXED: was "test_split" (0.2), now correct proportion for training
    "patience": 100,  # Early stopping patience
    "keras_verbose": 0,  # Suppress keras output since verbose=False in k-fold
    "weight_samples": {
        "enabled": True,  # Enable sample weighting
        "method": "KD",  # Use holdout method for training
        "xbins": 30,
        "ybins": 30,
    },
    "disable_gpu": False,  # Force CPU-only execution
}

# Create Locator instance
locator = Locator(config)

# Load genotype data
print("\nLoading genotype data from VCF...")
genotypes, samples = locator.load_genotypes(vcf=vcf_path)
print(f"Loaded genotypes shape: {genotypes.shape}")
print(f"Number of samples: {len(samples)}")
print(f"Number of SNPs: {genotypes.shape[0]}")

# Check data quality (new feature)
print("\nChecking data quality...")
status = locator.check_data(genotypes, samples, verbose=True)


# Train the model with k-fold holdouts
k = 2
print(f"\nRunning {k}-fold cross-validation...")
ho_preds = locator.run_k_fold_holdouts(
    genotypes, samples, k=k, verbose=True, return_df=True  # Progress bar will still show
)

# Plot the error summary
print("\nGenerating error summary plots...")
plot_error_summary(
    predictions=ho_preds,
    sample_data=coords_path,
    plot_map=True,
    include_training_locs=True,
    out_prefix=os.path.join(output_dir, "actinemys_holdout_summary"),
)

# Find the six samples with the biggest prediction errors
print("\nFinding samples with largest prediction errors...")

# First, we need to merge predictions with true locations
# Load the sample data to get true locations
sample_data_df = pd.read_csv(coords_path, sep="\t")
ho_preds_merged = ho_preds.merge(
    sample_data_df[["sampleID", "x", "y"]], on="sampleID", suffixes=("_pred", "_true")
)

# Rename columns to match expected format
ho_preds_merged = ho_preds_merged.rename(columns={"x": "x_true", "y": "y_true"})

# Calculate prediction errors
ho_preds_merged["error_km"] = (
    np.sqrt(
        (ho_preds_merged["x_true"] - ho_preds_merged["x_pred"]) ** 2
        + (ho_preds_merged["y_true"] - ho_preds_merged["y_pred"]) ** 2
    )
    * 111.32
)  # Convert degrees to km (approximate)

# Sort by error and get top 6
worst_predictions = ho_preds_merged.nlargest(6, "error_km")
print(f"\nTop 6 prediction errors (km):")
print(
    worst_predictions[["sampleID", "error_km", "x_true", "y_true", "x_pred", "y_pred"]]
)

# Get indices of these samples
worst_sample_ids = worst_predictions["sampleID"].values
print(f"\nWorst predicted samples: {worst_sample_ids}")

# Find the indices of these samples in the original data
sample_list = list(samples)
worst_indices = [
    sample_list.index(sid) for sid in worst_sample_ids if sid in sample_list
]
print(f"Sample indices: {worst_indices}")

# Run window analysis on these specific samples as holdouts
print(f"\nRunning window analysis for {len(worst_indices)} worst-predicted samples...")

# Need to check if we have position information (for VCF data we should)
if hasattr(locator, "positions") or locator.config.get("vcf"):
    window_results = locator.run_windows_holdouts(
        genotypes=genotypes,
        samples=samples,
        holdout_indices=worst_indices,
        window_size=500_000,  # 500kb windows
        window_start=0,
        return_df=True,
        save_full_pred_matrix=True,
    )

    # Plot window analysis results
    if window_results is not None:
        print("\nPlotting window analysis results...")

        # First, rename the window columns to match what plot_predictions expects
        # The function expects columns like x_0, x_1, etc., not x_pos0, x_pos400000
        window_cols = [
            col for col in window_results.columns if col.startswith(("x_pos", "y_pos"))
        ]
        x_window_cols = sorted([col for col in window_cols if col.startswith("x_pos")])
        y_window_cols = sorted([col for col in window_cols if col.startswith("y_pos")])

        # Create a renamed version for plotting
        plot_df = window_results.copy()
        for i, (x_col, y_col) in enumerate(zip(x_window_cols, y_window_cols)):
            plot_df[f"x_{i}"] = plot_df[x_col]
            plot_df[f"y_{i}"] = plot_df[y_col]

        # Use the built-in plot_predictions function to visualize window predictions
        # This will create KDE plots showing the distribution of predictions across windows
        from locator.plotting import plot_predictions

        plot_predictions(
            predictions=plot_df,
            locator=locator,
            out_prefix=os.path.join(output_dir, "worst_samples_windows"),
            samples=worst_sample_ids,  # Plot only the worst samples
            n_cols=3,
            plot_map=False,  # Set to True if you want map background
            width=5,
            height=4,
            show=False,  # Don't display, just save
        )
        print(f"  Saved window KDE plots for worst samples")

        # Also create a custom summary plot showing prediction variance
        print("\nCreating window variance summary plot...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Variance across windows for each sample
        for i, sample_id in enumerate(worst_sample_ids[:6]):
            if sample_id in window_results["sampleID"].values:
                sample_window_data = window_results[
                    window_results["sampleID"] == sample_id
                ]

                # Get predictions across windows
                x_preds = [sample_window_data[col].values[0] for col in x_window_cols]
                y_preds = [sample_window_data[col].values[0] for col in y_window_cols]

                # Calculate distance from mean prediction for each window
                mean_x = np.mean(x_preds)
                mean_y = np.mean(y_preds)

                distances = [
                    np.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2) * 111.32
                    for x, y in zip(x_preds, y_preds)
                ]

                window_positions = [
                    int(col.split("x_pos")[1]) / 1e6 for col in x_window_cols
                ]  # Convert to Mb

                # Get error for coloring
                error_km = worst_predictions[worst_predictions["sampleID"] == sample_id][
                    "error_km"
                ].values[0]

                ax1.plot(
                    window_positions,
                    distances,
                    "o-",
                    label=f"{sample_id} ({error_km:.0f} km error)",
                    alpha=0.7,
                    linewidth=1.5,
                )

        ax1.set_xlabel("Window start position (Mb)")
        ax1.set_ylabel("Distance from mean prediction (km)")
        ax1.set_title("Prediction Variance Across Genomic Windows")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Right panel: Heatmap of prediction errors by window
        error_matrix = []
        sample_labels = []

        for sample_id in worst_sample_ids[:6]:
            if sample_id in window_results["sampleID"].values:
                sample_window_data = window_results[
                    window_results["sampleID"] == sample_id
                ]
                true_sample = worst_predictions[
                    worst_predictions["sampleID"] == sample_id
                ].iloc[0]

                window_errors = []
                for x_col, y_col in zip(x_window_cols, y_window_cols):
                    x_pred = sample_window_data[x_col].values[0]
                    y_pred = sample_window_data[y_col].values[0]
                    error_km = (
                        np.sqrt(
                            (x_pred - true_sample["x_true"]) ** 2
                            + (y_pred - true_sample["y_true"]) ** 2
                        )
                        * 111.32
                    )
                    window_errors.append(error_km)

                error_matrix.append(window_errors)
                sample_labels.append(f"{sample_id} ({true_sample['error_km']:.0f} km)")

        if error_matrix:
            error_matrix = np.array(error_matrix)
            im = ax2.imshow(
                error_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest"
            )

            # Set ticks
            ax2.set_xticks(range(len(x_window_cols)))
            ax2.set_xticklabels(
                [f'{int(col.split("x_pos")[1])/1e6:.1f}' for col in x_window_cols],
                rotation=45,
            )
            ax2.set_yticks(range(len(sample_labels)))
            ax2.set_yticklabels(sample_labels)

            ax2.set_xlabel("Window start position (Mb)")
            ax2.set_ylabel("Sample (overall error)")
            ax2.set_title("Prediction Error by Window (km)")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("Prediction error (km)")

            # Mark best window for each sample
            for i in range(len(error_matrix)):
                best_window_idx = np.argmin(error_matrix[i])
                ax2.text(
                    best_window_idx,
                    i,
                    "★",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=12,
                    weight="bold",
                )

        plt.suptitle("Window Analysis Summary for Worst-Predicted Samples")
        plt.tight_layout()
        variance_plot_filename = os.path.join(output_dir, "window_analysis_summary.png")
        plt.savefig(variance_plot_filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved window analysis summary plot")

else:
    print("  Warning: No position information available for window analysis")
    print("  Window analysis requires VCF input or position-labeled genotype data")

print(f"\nAnalysis complete! Results saved to {output_dir}")
