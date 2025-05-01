"""Plotting functionality for locator predictions"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.stats import gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from geopy.distance import geodesic
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.axes as maxes

__all__ = ["kde_predict", "plot_predictions", "plot_error_summary", "plot_sample_weights"]


def kde_predict(x_coords, y_coords, xlim=(0, 50), ylim=(0, 50), n_points=100):
    """Calculate kernel density estimate of predictions

    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        xlim: Tuple of (min, max) x values for grid
        ylim: Tuple of (min, max) y values for grid
        n_points: Number of points for density estimation grid

    Returns:
        Tuple of (x_grid, y_grid, density)
    """
    try:
        # Calculate kernel density
        positions = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(positions)

        # Create grid of points using full plot range
        x_grid = np.linspace(xlim[0], xlim[1], n_points)
        y_grid = np.linspace(ylim[0], ylim[1], n_points)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Evaluate kernel on grid
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = np.reshape(kernel(positions).T, xx.shape)

        return xx, yy, density

    except Exception as e:
        print(f"KDE failed: {e}")
        return None, None, None


def plot_predictions(
    predictions,
    locator,
    out_prefix,
    samples=None,
    n_samples=9,
    n_cols=3,
    plot_map=False,
    width=5,
    height=4,
    dpi=300,
    n_levels=3,
):
    """Plot locator predictions from jacknife, bootstrap, or windows analyses.

    This function visualizes predictions from any of locator's prediction methods:
    - run_jacknife()
    - run_bootstraps()
    - run_windows()

    The function expects prediction data with:
    - A 'sampleID' column
    - Multiple prediction columns ('x_0', 'x_1'... and 'y_0', 'y_1'...)

    For each sample, the plot shows:
    - KDE contours of predictions (blue lines)
    - True location if known (red star)
    - All training sample locations (gray circles)

    Args:
        predictions: DataFrame or path to predictions file. Output from any of:
            - locator.run_jacknife(return_df=True)
            - locator.run_bootstraps(return_df=True)
            - locator.run_windows(return_df=True)
        locator: Locator instance containing training data configuration
        out_prefix: Prefix for output files
        samples: List of sample IDs to plot. If None, randomly selects n_samples
        n_samples: Number of samples to plot if samples not specified
        n_cols: Number of columns in plot grid
        plot_map: Whether to plot on a map (requires cartopy)
        width: Width of each subplot
        height: Height of each subplot
        dpi: DPI for output figure
        n_levels: Number of KDE contour levels to plot

    Returns:
        matplotlib figure object

    Example:
        >>> # For jacknife analysis
        >>> predictions = locator.run_jacknife(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "jacknife_example")

        >>> # For bootstrap analysis
        >>> predictions = locator.run_bootstraps(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "bootstrap_example")

        >>> # For windows analysis
        >>> predictions = locator.run_windows(genotypes, samples, return_df=True)
        >>> plot_predictions(predictions, locator, "windows_example")
    """
    # Load predictions
    if isinstance(predictions, (str, Path)):
        pred_path = Path(predictions)
        if pred_path.is_file():
            preds = pd.read_csv(pred_path)
        else:
            pred_files = list(pred_path.glob("*predlocs.txt"))
            preds = pd.concat([pd.read_csv(f) for f in pred_files])
    else:
        preds = predictions

    # Get sample data from locator
    if isinstance(locator.config["sample_data"], pd.DataFrame):
        samples_df = locator.config["sample_data"].copy()
    else:
        samples_df = pd.read_csv(
            locator.config["sample_data"], sep="\t", na_values="NA", quotechar='"'
        )
    samples_df.columns = samples_df.columns.str.strip('"')
    if "sampleID" in samples_df.columns:
        samples_df["sampleID"] = samples_df["sampleID"].str.strip('"')

    # Select samples to plot if not provided
    if samples is None:
        available_samples = preds["sampleID"].unique()
        samples = np.random.choice(
            available_samples,
            size=min(n_samples, len(available_samples)),
            replace=False,
        )

    # Create figure
    n_rows = int(np.ceil(len(samples) / n_cols))
    fig = plt.figure(figsize=(width * n_cols, height * n_rows), dpi=dpi)

    # Get x and y columns and calculate limits
    x_cols = [col for col in preds.columns if col.startswith("x_")]
    y_cols = [col for col in preds.columns if col.startswith("y_")]

    # Calculate global min/max for x and y coordinates
    x_all = preds[x_cols].values.ravel()
    y_all = preds[y_cols].values.ravel()

    # Add some padding (10%) to the limits
    padding = 0.1
    x_range = x_all.max() - x_all.min()
    y_range = y_all.max() - y_all.min()

    xlim = (x_all.min() - x_range * padding, x_all.max() + x_range * padding)
    ylim = (y_all.min() - y_range * padding, y_all.max() + y_range * padding)

    # Plot each sample
    for i, sample in enumerate(samples, 1):
        ax = fig.add_subplot(
            n_rows, n_cols, i, projection=ccrs.PlateCarree() if plot_map else None
        )

        sample_preds = preds[preds["sampleID"] == sample]
        sample_true = samples_df[samples_df["sampleID"] == sample]

        if plot_map:
            ax.add_feature(cfeature.LAND, facecolor="lightgray")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # Plot all training sample locations as background
        # Only plot samples that have true locations (not NA)
        training_locs = samples_df[
            pd.notna(samples_df["x"]) & pd.notna(samples_df["y"])
        ]
        if not training_locs.empty:
            ax.scatter(
                training_locs["x"],
                training_locs["y"],
                c="gray",
                marker="o",
                s=20,
                facecolors="none",
                alpha=0.5,
                linewidth=0.5,
                label="Training samples",
            )

        # Plot predictions using KDE
        if x_cols:  # Changed from checking columns again to using existing x_cols
            # Multiple predictions per sample (e.g., jacknife)
            # Collect all predictions
            x_preds = sample_preds[x_cols].values.ravel()
            y_preds = sample_preds[y_cols].values.ravel()

            # Calculate KDE using plot limits
            xx, yy, density = kde_predict(x_preds, y_preds, xlim=xlim, ylim=ylim)
            if density is not None:
                # Calculate percentile-based contour levels
                density_flat = density.ravel()
                levels = np.percentile(density_flat[density_flat > 0], [85, 90, 95, 99])

                # Plot contour lines
                ax.contour(
                    xx,
                    yy,
                    density,
                    levels=levels,
                    colors="blue",
                    alpha=0.8,
                    linewidths=0.5,
                )

        # Plot true location if it exists and is not NA
        if len(sample_true) > 0 and pd.notna(sample_true.iloc[0]["x"]):
            ax.scatter(
                sample_true.iloc[0]["x"],
                sample_true.iloc[0]["y"],
                c="red",
                marker="*",
                s=100,
                label="True",
            )

        ax.set_title(f"Sample {sample}")

    plt.tight_layout()
    if out_prefix:
        plt.savefig(f"{out_prefix}_predictions.pdf")
    plt.show()
    return None


def plot_error_summary(
    predictions,
    sample_data,
    out_prefix=None,
    plot_map=True,
    width=20,
    height=10,
    dpi=300,
    use_geodesic=True,
    include_training_locs=True,
):
    """Plot summary of prediction errors from holdout analysis

    Args:
        predictions: DataFrame of predictions
        sample_data: DataFrame or path to sample locations
        out_prefix: Prefix for output files
        plot_map: Whether to plot on a map
        width: Figure width
        height: Figure height
        dpi: Figure resolution
        use_geodesic: Use geodesic distances (km) if True, else Euclidean distances
        include_training_locs: Whether to plot training locations and use their extent
    """
    # Validate predictions input
    if predictions.empty:
        raise ValueError("Predictions DataFrame cannot be empty")

    # Consolidate loading and validation of sample_data
    if isinstance(sample_data, pd.DataFrame):
        samples = sample_data.copy()
    elif isinstance(sample_data, (str, Path)):
        sample_path = Path(sample_data)
        if not sample_path.is_file():
            raise ValueError(f"sample_data file {sample_data} does not exist")
        samples = pd.read_csv(sample_path, sep="\t")
    else:
        raise ValueError("sample_data must be a DataFrame or a valid file path")

    if samples.empty:
        raise ValueError("Sample data cannot be empty")

    # Validate required columns in predictions and samples
    required_pred_cols = ["sampleID", "x_pred", "y_pred"]
    required_sample_cols = ["sampleID", "x", "y"]
    missing_pred_cols = [
        col for col in required_pred_cols if col not in predictions.columns
    ]
    missing_sample_cols = [
        col for col in required_sample_cols if col not in samples.columns
    ]
    if missing_pred_cols:
        raise ValueError(
            f"Missing required columns in predictions: {missing_pred_cols}"
        )
    if missing_sample_cols:
        raise ValueError(
            f"Missing required columns in sample data: {missing_sample_cols}"
        )

    samples = samples.rename(columns={"x": "x_true", "y": "y_true"})

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Merge predictions with true locations
    merged = predictions.merge(samples[["sampleID", "x_true", "y_true"]], on="sampleID")
    if merged.empty:
        raise ValueError(
            "No matching samples found between predictions and sample data"
        )

    # Calculate errors
    if use_geodesic:
        merged["error"] = merged.apply(
            lambda row: geodesic(
                (row["y_true"], row["x_true"]), (row["y_pred"], row["x_pred"])
            ).kilometers,
            axis=1,
        )
        error_units = "km"
    else:
        merged["error"] = np.sqrt(
            (merged["x_pred"] - merged["x_true"]) ** 2
            + (merged["y_pred"] - merged["y_true"]) ** 2
        )
        error_units = "coordinate units"

    # Set up figure and primary axis based on plot_map flag
    if plot_map:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = fig.add_gridspec(1, 3)
        map_ax = fig.add_subplot(gs[0:2], projection=ccrs.PlateCarree())
    else:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = fig.add_gridspec(1, 2)
        map_ax = fig.add_subplot(gs[0])

    # Common axis setup
    map_ax.set_xticks([])
    map_ax.set_yticks([])
    if plot_map:
        map_ax.add_feature(cfeature.LAND, facecolor="lightgray")
        map_ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

    # Determine bounds and optionally plot training locations
    if include_training_locs:
        x_min, x_max = samples["x_true"].min(), samples["x_true"].max()
        y_min, y_max = samples["y_true"].min(), samples["y_true"].max()
        training_mask = ~samples["sampleID"].isin(predictions["sampleID"])
        training_locs = samples[training_mask]
        if not training_locs.empty:
            map_ax.scatter(
                training_locs["x_true"],
                training_locs["y_true"],
                c="gray",
                marker="o",
                s=20,
                alpha=0.5,
                label="Training locations",
            )
    else:
        x_min, x_max = merged["x_true"].min(), merged["x_true"].max()
        y_min, y_max = merged["y_true"].min(), merged["y_true"].max()

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    if plot_map:
        # Use set_extent only for map projections.
        map_ax.set_extent(
            [
                x_min - x_range * padding,
                x_max + x_range * padding,
                y_min - y_range * padding,
                y_max + y_range * padding,
            ]
        )
    else:
        # For regular axes, set x and y limits.
        map_ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)
        map_ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)

    # Plot scatter, colorbar, and error connections
    scatter = map_ax.scatter(
        merged["x_true"],
        merged["y_true"],
        c=merged["error"],
        cmap="RdYlBu_r",
        s=20,
        **({"label": "Test locations"} if plot_map else {}),
    )
    cbar = plt.colorbar(scatter, ax=map_ax, label=f"Error ({error_units})")
    cbar.outline.set_visible(False)
    for _, row in merged.iterrows():
        map_ax.plot(
            [row["x_true"], row["x_pred"]],
            [row["y_true"], row["y_pred"]],
            "k-",
            linewidth=0.5,
            alpha=0.5,
        )
    if plot_map and include_training_locs:
        map_ax.legend(loc="upper right")

    # Set up histogram panel (common to both layouts)
    hist_ax = fig.add_subplot(gs[2] if plot_map else gs[1])
    sns.histplot(data=merged, x="error", ax=hist_ax)
    hist_ax.set_xlabel(f"Error ({error_units})", fontsize=14)
    hist_ax.set_ylabel("Count", fontsize=14)
    stats_text = (
        f"Mean error: {merged['error'].mean():.2f} {error_units}\n"
        f"Median error: {merged['error'].median():.2f} {error_units}\n"
        f"Max error: {merged['error'].max():.2f} {error_units}\n"
        f"R² (x): {np.corrcoef(merged['x_pred'], merged['x_true'])[0,1]**2:.3f}\n"
        f"R² (y): {np.corrcoef(merged['x_pred'], merged['x_true'])[0,1]**2:.3f}"
    )
    hist_ax.text(
        0.95,
        0.95,
        stats_text,
        transform=hist_ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=12,
    )

    plt.tight_layout()
    if out_prefix:
        plt.savefig(f"{out_prefix}_error_summary.png")
    plt.show()
    plt.close()
    return None

def plot_sample_weights(
    locator,
    out_prefix=None,
    plot_map=True,
    width=5,
    height=3,
    dpi=300,
):
    """Plot sample weights assined to training locations

    Args:
        sample_data: DataFrame or path to sample locations
        sample_weights: DataFrame or path to sample weights
        out_prefix: Prefix for output files
        plot_map: Whether to plot on a map
        width: Figure width
        height: Figure height
        dpi: Figure resolution
    """
    sample_data = locator._sample_data_df
    sample_weights = locator.sample_weights['sample_weights_df']
    # Validate inputs
    if sample_data.empty or sample_weights.empty:
        raise ValueError("Sample data and weights cannot be empty DataFrames")
    # Check for required columns
    required_weight_cols = ["sampleID", "sample_weight"]
    required_sample_cols = ["sampleID", "x", "y"]

    missing_weight_cols = [
        col for col in required_weight_cols if col not in sample_weights.columns
    ]
    missing_sample_cols = [
        col for col in required_sample_cols if col not in sample_data.columns
    ]

    if missing_weight_cols:
        raise ValueError(
            f"Missing required columns in predictions: {missing_weight_cols}"
        )
    if missing_sample_cols:
        raise ValueError(
            f"Missing required columns in sample data: {missing_sample_cols}"
        )

    # Set larger font sizes globally
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    # Load sample data if path provided
    if isinstance(sample_data, pd.DataFrame):
        samples = sample_data.copy()
    else:
        samples = pd.read_csv(sample_data, sep="\t")
        # Load sample data if path provided
    if isinstance(sample_weights, pd.DataFrame):
        weights = sample_weights.copy()
    else:
        weights = pd.read_csv(sample_weights, sep="\t")

    # Merge predictions with true locations
    merged = sample_weights.merge(samples, on="sampleID")
    # Check if merge was successful
    if merged.empty:
        raise ValueError(
            "No matching samples found between sample data and sample weights"
        )

    # Create figure
    if plot_map:
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = fig.add_gridspec(1, 2)

        ax1 = fig.add_subplot(gs[0:1], projection=ccrs.PlateCarree())
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax1.add_feature(cfeature.LAND, facecolor="lightgray")
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)

        x_min, x_max = merged["x"].min(), merged["x"].max()
        y_min, y_max = merged["y"].min(), merged["y"].max()

        # Add padding to bounds
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Set map extent
        ax1.set_extent(
            [
                x_min - x_range * padding,
                x_max + x_range * padding,
                y_min - y_range * padding,
                y_max + y_range * padding,
            ]
        )

        # Plot predictions scatter with error colors
        scatter = ax1.scatter(
            merged["x"],
            merged["y"],
            c=merged["sample_weight"],
            cmap="viridis",
            s=10,
            label="Training locations",
            norm=matplotlib.colors.LogNorm(),
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1, label="Sample Weights")
        cbar.outline.set_visible(False)
        #plt.gca().set_aspect('equal')

        #

        #plt.tight_layout()

        if out_prefix:
            plt.savefig(f"{out_prefix}_sample_weights.png")

        plt.show()
        plt.close()
    else:
        # Create figure
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        gs = fig.add_gridspec(1, 2)

        # Create left panel (map + colorbar) without frame
        ax1 = fig.add_subplot(gs[0])

        # Calculate bounds with some padding
        x_min, x_max = merged["x"].min(), merged["x"].max()
        y_min, y_max = merged["y"].min(), merged["y"].max()

        # Add padding to bounds
        padding = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Set map extent
        ax1.set(
            xlim = (
                x_min - x_range * padding,
                x_max + x_range * padding),
            ylim = (
                y_min - y_range * padding,
                y_max + y_range * padding)
        )

        # Plot predictions scatter with error colors
        scatter = ax1.scatter(
            merged["x"],
            merged["y"],
            c=merged["sample_weight"],
            cmap="viridis",
            s=10,
            label="Training locations",
            norm=matplotlib.colors.LogNorm(),
        )

        cbar = plt.colorbar(scatter, ax=ax1, label="Sample Weights")
        cbar.outline.set_visible(False)
        plt.gca().set_aspect('equal')

        #

        #plt.tight_layout()

        if out_prefix:
            plt.savefig(f"{out_prefix}_sample_weights.png")

        plt.show()
        plt.close()
    return None