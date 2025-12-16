"""Plotting functionality for locator predictions"""

import base64
import io
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from geopy.distance import geodesic
from scipy.stats import gaussian_kde

__all__ = [
    "kde_predict",
    "plot_predictions",
    "plot_error_summary",
    "plot_interactive_error_map",
    "plot_sample_weights",
    "PlottingMixin",
]


def _handle_plot_display(show=None):
    """Handle whether to display a plot based on environment.

    Args:
        show: None (auto-detect), True (always show), or False (never show)
    """
    if show is None:
        # Auto-detect: show only if in interactive environment (Jupyter/IPython)
        try:
            get_ipython()  # This is defined in IPython/Jupyter
            plt.show()
        except NameError:
            # Not in interactive environment, don't show
            pass
    elif show:
        plt.show()


def kde_predict(x_coords, y_coords, xlim=(0, 50), ylim=(0, 50), n_points=100):
    """Calculate kernel density estimate of predictions.

    This is a helper function used internally by plot_predictions() to compute
    kernel density estimates for visualizing prediction uncertainty.

    Args:
        x_coords (array-like): Array of x coordinates (longitude values)
        y_coords (array-like): Array of y coordinates (latitude values)
        xlim (tuple): Tuple of (min, max) x values for grid. Default: (0, 50)
        ylim (tuple): Tuple of (min, max) y values for grid. Default: (0, 50)
        n_points (int): Number of points for density estimation grid. Default: 100

    Returns
    -------
        tuple: A 3-tuple containing:

            - **x_grid** (*numpy.ndarray*): X coordinates of the mesh grid
            - **y_grid** (*numpy.ndarray*): Y coordinates of the mesh grid
            - **density** (*numpy.ndarray*): Density values at each grid point

            Returns (None, None, None) if KDE calculation fails.

    Note:
        The function uses scipy.stats.gaussian_kde for density estimation.
        Grid limits should match the geographic extent of your predictions.
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


def plot_predictions(  # noqa: C901
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
    show=None,
):
    """Plot locator predictions from jacknife, bootstrap, or windows analyses.

    This function visualizes predictions from any of locator's prediction methods
    that generate multiple predictions per sample. It creates a grid of subplots,
    one per sample, showing the distribution of predictions as KDE contours.

    The function expects prediction data with:

    - A 'sampleID' column
    - Multiple prediction columns ('x_0', 'x_1'... and 'y_0', 'y_1'...)

    For each sample, the plot shows:

    - KDE contours of predictions (blue lines)
    - True location if known (red star)
    - All training sample locations (gray circles)

    Args:
        predictions (pandas.DataFrame or str): DataFrame or path to predictions file.
            Output from any of:

            - ``locator.run_jacknife(return_df=True)``
            - ``locator.run_bootstraps(return_df=True)``
            - ``locator.run_windows(return_df=True)``

        locator (Locator): Locator instance containing training data configuration
        out_prefix (str): Prefix for output files. Plot saved as {out_prefix}_predictions.pdf
        samples (list, optional): List of sample IDs to plot. If None, randomly selects n_samples
        n_samples (int): Number of samples to plot if samples not specified. Default: 9
        n_cols (int): Number of columns in plot grid. Default: 3
        plot_map (bool): Whether to plot on a geographic map (requires cartopy). Default: False
        width (float): Width of each subplot in inches. Default: 5
        height (float): Height of each subplot in inches. Default: 4
        dpi (int): DPI resolution for output figure. Default: 300
        n_levels (int): Number of KDE contour levels to plot. Default: 3
        show (bool or None): Whether to display plot. None=auto-detect environment. Default: None

    Returns
    -------
        None: Saves plot to file and optionally displays it

    Examples
    --------
        For jacknife analysis::

            predictions = locator.run_jacknife(genotypes, samples, return_df=True)
            plot_predictions(predictions, locator, "jacknife_example")

        For bootstrap analysis::

            predictions = locator.run_bootstraps(genotypes, samples, return_df=True)
            plot_predictions(predictions, locator, "bootstrap_example")

        For windows analysis::

            predictions = locator.run_windows(genotypes, samples, return_df=True)
            plot_predictions(predictions, locator, "windows_example")

        Plot specific samples::

            plot_predictions(predictions, locator, "selected",
                           samples=['HG001', 'HG002', 'HG003'])

    Note:
        - Requires matplotlib and scipy for KDE calculation
        - If plot_map=True, requires cartopy for geographic projections
        - Automatically adjusts plot limits based on prediction ranges
        - KDE may fail for samples with very few predictions
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
        training_locs = samples_df[pd.notna(samples_df["x"]) & pd.notna(samples_df["y"])]
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
    _handle_plot_display(show)
    plt.close()
    return None


def plot_error_summary(  # noqa: C901
    predictions,
    sample_data,
    out_prefix=None,
    plot_map=True,
    width=20,
    height=10,
    dpi=300,
    use_geodesic=True,
    include_training_locs=True,
    show=None,
    return_merged=False,
):
    """
    Plot summary of prediction errors from holdout analysis.

    Creates a comprehensive error visualization with two panels:

    1. **Map/Scatter panel**: Shows true locations colored by prediction error,
       with lines connecting true and predicted locations
    2. **Histogram panel**: Distribution of errors with summary statistics

    This function is designed for analyzing results from holdout methods like:

    - ``run_holdouts()``
    - ``run_k_fold_holdouts()``
    - ``run_leave_one_out()``

    Args:
        predictions (pandas.DataFrame): DataFrame with columns:
            - ``sampleID``: Sample identifiers
            - ``x_pred``: Predicted longitude
            - ``y_pred``: Predicted latitude

        sample_data (pandas.DataFrame or str): DataFrame or path to TSV file with columns:
            - ``sampleID``: Sample identifiers (must match predictions)
            - ``x``: True longitude
            - ``y``: True latitude

        out_prefix (str, optional): Prefix for output files. If provided, saves as
            {out_prefix}_error_summary.png (or .html for interactive). Default: None
        plot_map (bool): Whether to plot on a geographic map using cartopy projection.
            If False, uses regular scatter plot. Default: True
        width (float): Figure width in inches. Default: 20
        height (float): Figure height in inches. Default: 10
        dpi (int): Figure resolution in dots per inch. Default: 300
        use_geodesic (bool): If True, calculate geodesic distances in kilometers.
            If False, use Euclidean distances in coordinate units. Default: True
        include_training_locs (bool): Whether to plot training locations (gray circles)
            and use their extent for map bounds. Default: True
        show (bool or None): Whether to display plot. None=auto-detect environment,
            True=always show, False=never show. Default: None
        return_merged (bool): If True, return the internal merged DataFrame used for plotting.
            Default: False

    Returns
    -------
        None: Saves plot to file and optionally displays it.
        If return_merged is True, returns the internal merged DataFrame containing prediction errors and true locations.

    Raises
    ------
        ValueError: If predictions or sample_data are empty, have missing columns,
            or have no matching samples

    Examples
    --------
        Basic usage with k-fold results::

            predictions = locator.run_k_fold_holdouts(genotypes, samples, return_df=True)
            plot_error_summary(predictions, "samples.tsv", "kfold_errors")

        With DataFrame input and Euclidean distances::

            plot_error_summary(predictions, sample_df,
                             out_prefix="holdout_errors",
                             use_geodesic=False)

        Without map projection::

            plot_error_summary(predictions, sample_df,
                             plot_map=False,
                             width=10, height=5)

        Return merged DataFrame::

            merged = plot_error_summary(predictions, sample_df, return_merged=True)

    Note:
        - Summary statistics shown: mean, median, max error, R² for x and y
        - Training locations help visualize geographic sampling bias
        - Geodesic distances account for Earth's curvature
        - Map projection requires cartopy to be installed
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
        raise ValueError(f"Missing required columns in predictions: {missing_pred_cols}")
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
        raise ValueError("No matching samples found between predictions and sample data")

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
        f"R² (y): {np.corrcoef(merged['y_pred'], merged['y_true'])[0,1]**2:.3f}"
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

    _handle_plot_display(show)
    plt.close()
    if return_merged:
        return merged
    return None


def plot_interactive_error_map(
    predictions,
    sample_data,
    out_prefix=None,
    width=1200,
    height=600,
    use_geodesic=True,
    include_training_locs=True,
    show_histogram=True,
):
    """Create an interactive map of prediction errors using Plotly.

    This function creates an interactive visualization of prediction errors from
    holdout analyses, with hover tooltips showing detailed information for each
    sample. The plot uses Plotly's geographic features to display results on a
    world map with coastlines and land features.

    Args:
        predictions (pandas.DataFrame): DataFrame with columns:
            - ``sampleID``: Sample identifiers
            - ``x_pred``: Predicted longitude
            - ``y_pred``: Predicted latitude

        sample_data (pandas.DataFrame or str): DataFrame or path to TSV file with columns:
            - ``sampleID``: Sample identifiers (must match predictions)
            - ``x``: True longitude
            - ``y``: True latitude

        out_prefix (str, optional): Prefix for output files. If provided, saves as
            {out_prefix}_interactive_error_map.html. Default: None

        width (int): Width of plot in pixels. Default: 1200
        height (int): Height of plot in pixels. Default: 600

        use_geodesic (bool): If True, calculate geodesic distances in kilometers.
            If False, use Euclidean distances in coordinate units. Default: True

        include_training_locs (bool): Whether to plot training locations (gray circles).
            Default: True

        show_histogram (bool): Whether to include error distribution histogram panel.
            Default: True

    Returns
    -------
        plotly.graph_objects.Figure: The interactive figure object. Can be displayed
        directly in Jupyter notebooks with fig.show() or just 'fig' in a cell.

    Examples
    --------
        Basic usage::

            fig = plot_interactive_error_map(predictions, sample_data)
            fig.show()  # Display in notebook

        Save to file::

            plot_interactive_error_map(predictions, sample_data,
                                     out_prefix="analysis")

        Without histogram panel::

            fig = plot_interactive_error_map(predictions, sample_data,
                                           show_histogram=False)
    """
    # Validate and load data
    if predictions.empty:
        raise ValueError("Predictions DataFrame cannot be empty")

    # Load sample data
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

    # Validate columns
    required_pred_cols = ["sampleID", "x_pred", "y_pred"]
    required_sample_cols = ["sampleID", "x", "y"]

    missing_pred_cols = [
        col for col in required_pred_cols if col not in predictions.columns
    ]
    missing_sample_cols = [
        col for col in required_sample_cols if col not in samples.columns
    ]

    if missing_pred_cols:
        raise ValueError(f"Missing required columns in predictions: {missing_pred_cols}")
    if missing_sample_cols:
        raise ValueError(
            f"Missing required columns in sample data: {missing_sample_cols}"
        )

    # Rename columns for clarity
    samples = samples.rename(columns={"x": "x_true", "y": "y_true"})

    # Merge predictions with true locations
    merged = predictions.merge(samples[["sampleID", "x_true", "y_true"]], on="sampleID")
    if merged.empty:
        raise ValueError("No matching samples found between predictions and sample data")

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

    # Create subplots
    if show_histogram:
        # Two-panel layout with map and histogram
        fig = sp.make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=("", "Error Distribution"),
            horizontal_spacing=0.05,
            specs=[[{"type": "geo"}, {"type": "xy"}]],
        )
    else:
        # Single geo panel
        fig = go.Figure()

    # Determine bounds for the plot
    if include_training_locs:
        x_min, x_max = samples["x_true"].min(), samples["x_true"].max()
        y_min, y_max = samples["y_true"].min(), samples["y_true"].max()
    else:
        x_min, x_max = merged["x_true"].min(), merged["x_true"].max()
        y_min, y_max = merged["y_true"].min(), merged["y_true"].max()

    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min_padded = x_min - x_range * padding
    x_max_padded = x_max + x_range * padding
    y_min_padded = y_min - y_range * padding
    y_max_padded = y_max + y_range * padding

    # We'll configure geographic features in the layout section

    # First, add training locations if requested
    if include_training_locs:
        training_mask = ~samples["sampleID"].isin(predictions["sampleID"])
        training_locs = samples[training_mask]
        if not training_locs.empty:
            trace = go.Scattergeo(
                lon=training_locs["x_true"],
                lat=training_locs["y_true"],
                mode="markers",
                marker=dict(
                    size=4,
                    color="rgba(128, 128, 128, 0.5)",  # Gray with alpha
                    symbol="circle-open",
                    line=dict(width=0.5, color="gray"),
                ),
                name="Training locations",
                hoverinfo="skip",
                showlegend=True,
            )
            if show_histogram:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

    # Add lines connecting true and predicted locations
    for idx, row in merged.iterrows():
        trace = go.Scattergeo(
            lon=[row["x_true"], row["x_pred"], None],
            lat=[row["y_true"], row["y_pred"], None],
            mode="lines",
            line=dict(color="black", width=1),
            opacity=0.3,
            showlegend=False,
            hoverinfo="skip",
        )
        if show_histogram:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    # Add scatter plot of true locations colored by error
    hover_text = merged.apply(
        lambda row: f"Sample: {row['sampleID']}<br>"
        f"Error: {row['error']:.2f} {error_units}<br>"
        f"True: ({row['x_true']:.2f}, {row['y_true']:.2f})<br>"
        f"Predicted: ({row['x_pred']:.2f}, {row['y_pred']:.2f})",
        axis=1,
    )

    trace = go.Scattergeo(
        lon=merged["x_true"],
        lat=merged["y_true"],
        mode="markers",
        marker=dict(
            size=8,
            color=merged["error"],
            colorscale="RdYlBu_r",
            showscale=True,
            colorbar=dict(
                title=f"Error ({error_units})",
                x=1.02,
                xanchor="left",
                len=0.8,
                thickness=15,
                yanchor="middle",
                y=0.5,
            ),
            line=dict(width=0.5, color="rgba(255,255,255,0.8)"),  # White outline
        ),
        text=hover_text,
        hoverinfo="text",
        name="Test locations",
        showlegend=True,
    )
    if show_histogram:
        fig.add_trace(trace, row=1, col=1)
    else:
        fig.add_trace(trace)

    # Add histogram if requested
    if show_histogram:
        fig.add_trace(
            go.Histogram(
                x=merged["error"],
                name="Error distribution",
                showlegend=False,
                marker=dict(color="steelblue", line=dict(color="white", width=1)),
                opacity=0.8,
            ),
            row=1,
            col=2,
        )

    # Calculate statistics
    mean_error = merged["error"].mean()
    median_error = merged["error"].median()
    max_error = merged["error"].max()
    r2_x = np.corrcoef(merged["x_pred"], merged["x_true"])[0, 1] ** 2
    r2_y = np.corrcoef(merged["y_pred"], merged["y_true"])[0, 1] ** 2

    # Add statistics annotation
    stats_text = (
        f"Mean: {mean_error:.2f} {error_units}<br>"
        f"Median: {median_error:.2f} {error_units}<br>"
        f"Max: {max_error:.2f} {error_units}<br>"
        f"R² (x): {r2_x:.3f}<br>"
        f"R² (y): {r2_y:.3f}"
    )

    # Add statistics annotation
    if show_histogram:
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="x2 domain",
            yref="y2 domain",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11, family="monospace"),
            align="left",
            xanchor="right",
            yanchor="top",
        )
    else:
        # Place stats on the map
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11, family="monospace"),
            align="left",
            xanchor="left",
            yanchor="top",
        )

    # Configure geo subplot
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        coastlinecolor="rgb(204, 204, 204)",
        coastlinewidth=0.5,
        showlakes=True,
        lakecolor="white",
        showocean=True,
        oceancolor="rgb(230, 245, 255)",
        lataxis=dict(range=[y_min_padded, y_max_padded]),
        lonaxis=dict(range=[x_min_padded, x_max_padded]),
        bgcolor="white",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)",
        countrywidth=0.5,
        showsubunits=False,
        showframe=False,
        resolution=50,  # 50m resolution
    )

    # Histogram panel axes (if present)
    if show_histogram:
        fig.update_xaxes(title_text=f"Error ({error_units})", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_layout(
        width=width,
        height=height,
        title_text="Prediction Error Map",
        title_font_size=16,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            font=dict(size=11),
        ),
        template="plotly_white",
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
    )

    # Save to HTML file
    if out_prefix:
        output_file = f"{out_prefix}_interactive_error_map.html"
        fig.write_html(output_file)
        print(f"Interactive map saved to: {output_file}")

    # Return the figure so it can be displayed in Jupyter notebooks
    return fig


def plot_sample_weights(
    locator,
    out_prefix=None,
    plot_map=True,
    width=5,
    height=3,
    dpi=300,
    show=None,
):
    """Plot sample weights assigned to training locations.

    Visualizes the geographic distribution of sample weights used during training.
    This is useful for understanding which regions are upweighted or downweighted
    based on sampling density.

    Sample weights are typically computed using:

    - Kernel density (KD) method: Upweights samples in sparse regions
    - Histogram binning method: Based on 2D histogram counts

    The plot uses a log-scale color mapping to better show weight variations.

    Args:
        locator (Locator): Locator instance that has been trained with sample
            weighting enabled. Must have computed sample_weights attribute.
        out_prefix (str, optional): Prefix for output files. If provided, saves as
            {out_prefix}_sample_weights.png. Default: None
        plot_map (bool): Whether to plot on a geographic map using cartopy projection.
            If False, uses regular scatter plot with equal aspect ratio. Default: True
        width (float): Figure width in inches. Default: 5
        height (float): Figure height in inches. Default: 3
        dpi (int): Figure resolution in dots per inch. Default: 300
        show (bool or None): Whether to display plot. None=auto-detect environment,
            True=always show, False=never show. Default: None

    Returns
    -------
        None: Saves plot to file and optionally displays it

    Raises
    ------
        ValueError: If locator doesn't have computed sample weights, or if
            required data is missing

    Examples
    --------
        After training with KDE weighting::

            config = {
                "weight_samples": {
                    "enabled": True,
                    "method": "KD"
                }
            }
            locator = Locator(config)
            locator.train(genotypes, samples)
            plot_sample_weights(locator, "kde_weights")

        With histogram binning weights::

            config = {
                "weight_samples": {
                    "enabled": True,
                    "method": "hist",
                    "xbins": 20,
                    "ybins": 20
                }
            }
            locator = Locator(config)
            locator.train(genotypes, samples)
            plot_sample_weights(locator, "hist_weights", plot_map=False)

    Note:
        - Requires that locator was trained with weight_samples enabled
        - Log scale coloring helps visualize large weight variations
        - Higher weights (yellow) indicate undersampled regions
        - Lower weights (purple) indicate oversampled regions
        - Map projection requires cartopy to be installed
    """
    sample_data = locator._sample_data_df
    sample_weights = locator.sample_weights["sample_weights_df"]
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
        # weights = sample_weights.copy()  # noqa: F841
        pass
    else:
        # weights = pd.read_csv(sample_weights, sep="\t")  # noqa: F841
        pass

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
        # plt.gca().set_aspect('equal')

        #

        # plt.tight_layout()

        if out_prefix:
            plt.savefig(f"{out_prefix}_sample_weights.png")

        _handle_plot_display(show)
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
            xlim=(x_min - x_range * padding, x_max + x_range * padding),
            ylim=(y_min - y_range * padding, y_max + y_range * padding),
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
        plt.gca().set_aspect("equal")

        #

        # plt.tight_layout()

        if out_prefix:
            plt.savefig(f"{out_prefix}_sample_weights.png")

        _handle_plot_display(show)
        plt.close()
    return None


class PlottingMixin:
    """Mixin class providing plotting functionality for Locator.

    This mixin is inherited by the main Locator class to provide visualization
    methods for training history and Jupyter notebook integration.

    Methods
    -------
        plot_history: Plot training and validation loss curves
        _repr_html_: Generate rich HTML representation for Jupyter notebooks
    """

    def plot_history(self, history):
        """Plot training history and prediction error.

        Creates a figure with two subplots showing the validation loss and training loss
        over epochs. This helps visualize model convergence and potential overfitting.

        The plot shows:

        - Left panel: Validation loss over epochs (excluding first 3)
        - Right panel: Training loss over epochs (excluding first 3)

        First 3 epochs are excluded as they often have very high initial losses
        that would compress the scale of the plot.

        Args:
            history (keras.callbacks.History): History object returned by model.fit()
                containing training metrics for each epoch

        Returns
        -------
            None: Saves plot to {config['out']}_fitplot.pdf if config['plot_history'] is True

        Note:
            - Only creates plot if config['plot_history'] is True
            - Uses 'agg' backend to avoid display issues on servers
            - Creates a compact figure suitable for publication

        Example:
            Enable history plotting in config::

                config = {"out": "analysis", "plot_history": True}
                locator = Locator(config)
                history = locator.train(genotypes, samples)
                # Plot is automatically saved as analysis_fitplot.pdf
        """
        if self.config.get("plot_history", False):
            plt.switch_backend("agg")
            fig = plt.figure(figsize=(4, 1.5), dpi=200)
            plt.rcParams.update({"font.size": 7})
            ax1 = fig.add_axes([0, 0, 0.4, 1])
            ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
            ax1.set_xlabel("Validation Loss")
            ax2 = fig.add_axes([0.55, 0, 0.4, 1])
            ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
            ax2.set_xlabel("Training Loss")
            fig.savefig(self.config["out"] + "_fitplot.pdf", bbox_inches="tight")

    def _repr_html_(self):  # noqa: C901
        """Return HTML representation of Locator instance for Jupyter notebooks.

        Generates a rich HTML display showing:

        - Model configuration parameters
        - Current model status (trained/not trained)
        - Training history plot (if available)
        - Data loading status
        - Sample weighting information
        - Holdout sample information

        This method is automatically called by Jupyter/IPython when displaying
        a Locator instance in a notebook cell.

        Returns
        -------
            str: HTML string with styled content including embedded plots

        Note:
            - Training history plot is embedded as base64 PNG
            - Holdout samples shown in collapsible list if > 0
            - Automatically detects which data has been loaded

        Example:
            In a Jupyter notebook::

                locator = Locator(config)
                locator  # Rich HTML display appears automatically
        """
        html = [
            "<div style='font-family: monospace'>",
            "<h3>Locator Model</h3>",
            "<table>",
            "<tr><th style='text-align:left; padding:5px'>Configuration</th><th style='text-align:left; padding:5px'>Value</th></tr>",
        ]

        # Add key configuration parameters
        key_params = [
            "train_split",
            "batch_size",
            "min_mac",
            "max_SNPs",
            "width",
            "nlayers",
            "dropout_prop",
            "max_epochs",
            "optimizer_algo",
            "learning_rate",
            "weight_decay",
            "use_range_penalty",
            "species_range_shapefile",
            "resolution",
            "penalty_weight",
            "species_range_geom",
        ]

        for param in key_params:
            if param in self.config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.config[param]}</td></tr>"
                )
        # add weight samples to end, deal with weird dictionary thing
        if self.config.get("weight_samples", {}).get("enabled", False):
            html.append(
                f"<tr><td style='padding:5px'>{'weight_samples'}</td>"
                f"<td style='padding:5px'>{'True'}</td></tr>"
            )
            for k in ["method", "xbins", "ybins", "lam", "bandwidth"]:
                if k in self.config["weight_samples"].keys():
                    if self.config["weight_samples"][k] is not None:
                        html.append(
                            f"<tr><td style='padding:5px'>{'weight_samples '+k}</td>"
                            f"<td style='padding:5px'>{self.config['weight_samples'][k]}</td></tr>"
                        )

        html.append("</table>")

        # Add model status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        # Model trained status and training history
        if self.model is not None:
            html.append("<li>Model: Trained ✓</li>")
            if hasattr(self, "traingen"):
                html.append(f"<li>Training samples: {self.traingen.shape[0]}</li>")
                html.append(f"<li>Features: {self.traingen.shape[1]}</li>")

            # Add training history if available
            if hasattr(self, "history") and self.history is not None:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 4))
                hist = self.history.history

                # Plot training and validation loss
                ax.plot(hist["loss"], label="Training Loss", color="blue")
                axV = ax.twinx()
                axV.plot(hist["val_loss"], label="Validation Loss", color="orange")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Training Loss")
                axV.set_ylabel("Validation Loss")
                ax.legend()
                axV.legend(loc="upper center")

                # Get final validation loss
                final_val_loss = hist["val_loss"][-1]

                # Convert plot to base64 string
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)

                # Add plot and metrics to HTML
                html.append("</ul>")  # Close the status list
                html.append("<h4>Training History:</h4>")
                html.append(f"<p>Final validation loss: {final_val_loss:.4f}</p>")
                html.append(
                    f'<img src="data:image/png;base64,{plot_data}" style="max-width:100%">'
                )
                html.append("<ul>")  # Reopen list for remaining items
        else:
            html.append("<li>Model: Not trained</li>")

        # Location normalization status
        if all(
            x is not None for x in [self.meanlong, self.sdlong, self.meanlat, self.sdlat]
        ):
            html.append("<li>Location normalization: Computed ✓</li>")
        else:
            html.append("<li>Location normalization: Not computed</li>")

        # Sample data status
        if hasattr(self, "_sample_data_df"):
            html.append(
                f"<li>Sample data loaded: {len(self._sample_data_df)} samples</li>"
            )
        elif "sample_data" in self.config:
            html.append("<li>Sample data: Path provided</li>")
        else:
            html.append("<li>Sample data: Not provided</li>")

        # Genotype data status
        if hasattr(self, "_genotype_df"):
            html.append(
                f"<li>Genotype data loaded: {self._genotype_df.shape[1]} SNPs</li>"
            )
        elif any(x in self.config for x in ["zarr", "vcf", "genotype_data"]):
            html.append("<li>Genotype data: Path provided</li>")
        else:
            html.append("<li>Genotype data: Not provided</li>")

        if hasattr(self, "sample_weights"):
            html.append(
                f"<li>Samples weighted using {self.config['weight_samples'].get('method')}</li>"
            )

        # Add holdout information
        if hasattr(self, "holdout_idx") and self.samples is not None:
            n_holdout = len(self.holdout_idx)
            html.append(f"<li>Holdout samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.holdout_idx]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")
        elif hasattr(self, "pred_indices") and self.samples is not None:
            n_holdout = len(self.pred_indices)
            html.append(f"<li>Prediction samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.pred_indices]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)
