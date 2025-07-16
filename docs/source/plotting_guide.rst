Plotting Guide
==============

Locator provides comprehensive visualization functions for analyzing predictions and model performance. This guide covers the various plotting functions and their usage.

Overview
--------

The plotting module (``locator.plotting``) provides functions for:

* Visualizing prediction distributions from resampling analyses
* Creating error summary plots from holdout analyses
* Displaying geographic distribution of sample weights
* Rich Jupyter notebook integration

All plotting functions support both display and file saving, with automatic environment detection for Jupyter notebooks.

Prediction Visualization
------------------------

plot_predictions()
~~~~~~~~~~~~~~~~~~

The ``plot_predictions()`` function visualizes results from analyses that generate multiple predictions per sample:

.. code-block:: python

    from locator.plotting import plot_predictions

    # After jacknife analysis
    predictions = locator.run_jacknife(genotypes, samples, return_df=True)
    plot_predictions(predictions, locator, "jacknife_viz")

This creates a grid of subplots, one per sample, showing:

* **KDE contours** (blue) representing prediction uncertainty
* **True location** (red star) if known
* **Training locations** (gray circles) as geographic context

Customizing the visualization:

.. code-block:: python

    # Plot specific samples with custom layout
    plot_predictions(
        predictions,
        locator,
        "custom_viz",
        samples=['HG001', 'HG002', 'HG003'],  # Specific samples
        n_cols=1,           # Single column layout
        width=8,            # Wider plots
        height=6,           # Taller plots
        plot_map=True,      # Use geographic map
        dpi=150            # Lower resolution for faster rendering
    )

Works with any multi-prediction analysis:

* ``run_jacknife()`` - Shows effect of SNP subsampling
* ``run_bootstraps()`` - Shows effect of SNP resampling
* ``run_windows()`` - Shows predictions from different genomic regions

Error Analysis
--------------

plot_error_summary()
~~~~~~~~~~~~~~~~~~~~

For holdout-based analyses, ``plot_error_summary()`` provides comprehensive error visualization:

.. code-block:: python

    from locator.plotting import plot_error_summary

    # After k-fold cross-validation
    predictions = locator.run_k_fold_holdouts(genotypes, samples, k=10, return_df=True)

    # Create error summary
    plot_error_summary(
        predictions,
        "samples.tsv",      # Path to true coordinates
        "kfold_errors",     # Output prefix
        use_geodesic=True   # Calculate distances in km
    )

The plot shows:

1. **Map panel**: True locations colored by prediction error, with lines to predictions
2. **Histogram panel**: Error distribution with summary statistics

Options for different use cases:

.. code-block:: python

    # Without map projection (faster, no cartopy required)
    plot_error_summary(
        predictions,
        sample_data_df,     # Can use DataFrame directly
        "errors_scatter",
        plot_map=False,     # Regular scatter plot
        width=12,           # Smaller figure
        height=6
    )

    # Euclidean distances instead of geodesic
    plot_error_summary(
        predictions,
        sample_data_df,
        "errors_euclidean",
        use_geodesic=False  # Use coordinate units
    )

Sample Weight Visualization
---------------------------

plot_sample_weights()
~~~~~~~~~~~~~~~~~~~~~

When using sample weighting, visualize the geographic distribution of weights:

.. code-block:: python

    from locator.plotting import plot_sample_weights

    # Configure and train with sample weighting
    config = {
        "out": "weighted_analysis",
        "weight_samples": {
            "enabled": True,
            "method": "KD",      # Kernel density method
            "bandwidth": None    # Auto-calculate
        }
    }

    locator = Locator(config)
    locator.train(genotypes, samples)

    # Plot the weights
    plot_sample_weights(locator, "kde_weights")

The visualization uses:

* **Log-scale coloring** to show weight variations
* **Yellow/bright colors** for high weights (undersampled regions)
* **Purple/dark colors** for low weights (oversampled regions)

Jupyter Notebook Integration
----------------------------

Rich Display
~~~~~~~~~~~~

In Jupyter notebooks, Locator instances display rich HTML automatically:

.. code-block:: python

    # In a Jupyter cell
    locator = Locator(config)
    locator.train(genotypes, samples)
    locator  # Shows configuration, status, and training plot

The display includes:

* Configuration parameters
* Model training status
* Training history plot (if trained)
* Data loading status
* Sample weighting information
* Holdout sample lists

Training History
~~~~~~~~~~~~~~~~

Plot training and validation loss curves:

.. code-block:: python

    # Enable history plotting
    config = {"out": "analysis", "plot_history": True}
    locator = Locator(config)

    history = locator.train(genotypes, samples)
    # Automatically saves analysis_fitplot.pdf

Common Plotting Patterns
------------------------

Comparing Methods
~~~~~~~~~~~~~~~~~

Compare predictions from different analyses:

.. code-block:: python

    # Run multiple analyses
    jack_preds = locator.run_jacknife(genotypes, samples, return_df=True)
    boot_preds = locator.run_bootstraps(genotypes, samples, return_df=True)

    # Plot same samples from each
    test_samples = ['HG001', 'HG002', 'HG003']

    plot_predictions(jack_preds, locator, "jacknife_comparison",
                    samples=test_samples)
    plot_predictions(boot_preds, locator, "bootstrap_comparison",
                    samples=test_samples)

Publication Figures
~~~~~~~~~~~~~~~~~~~

Create publication-quality figures:

.. code-block:: python

    # High-resolution error summary
    plot_error_summary(
        predictions,
        sample_data,
        "figure_2",
        dpi=600,            # Publication quality
        width=7,            # Single column width
        height=4,           # Appropriate height
        include_training_locs=False  # Cleaner look
    )

    # Convert to other formats
    import matplotlib.pyplot as plt
    plt.savefig("figure_2.eps", format='eps')  # For journals

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple datasets:

.. code-block:: python

    datasets = ['population1', 'population2', 'population3']

    for dataset in datasets:
        # Load data for dataset
        genotypes, samples = load_data(dataset)

        # Run analysis
        predictions = locator.run_k_fold_holdouts(
            genotypes, samples, k=5, return_df=True
        )

        # Plot with dataset-specific prefix
        plot_error_summary(
            predictions,
            f"{dataset}_samples.tsv",
            f"{dataset}_kfold_errors"
        )

Customization Tips
------------------

Environment Control
~~~~~~~~~~~~~~~~~~~

Control plot display behavior:

.. code-block:: python

    # Always show plots (interactive mode)
    plot_predictions(predictions, locator, "output", show=True)

    # Never show plots (batch mode)
    plot_predictions(predictions, locator, "output", show=False)

    # Auto-detect (default) - shows in Jupyter, not in scripts
    plot_predictions(predictions, locator, "output", show=None)

Geographic Projections
~~~~~~~~~~~~~~~~~~~~~~

When ``plot_map=True``, plots use cartopy for geographic projections:

.. code-block:: python

    # Ensure cartopy is installed
    # pip install cartopy

    # Plot with coastlines and land features
    plot_predictions(predictions, locator, "map_viz", plot_map=True)

    # Troubleshooting cartopy issues
    plot_predictions(predictions, locator, "no_map", plot_map=False)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets or many samples:

.. code-block:: python

    # Reduce DPI for faster rendering
    plot_predictions(predictions, locator, "quick_viz", dpi=100)

    # Plot fewer samples
    plot_predictions(predictions, locator, "subset", n_samples=6)

    # Use matplotlib Agg backend for headless systems
    import matplotlib
    matplotlib.use('Agg')

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**KDE fails for some samples**:

* Samples may have too few predictions
* Predictions may be too clustered
* Check prediction variance for affected samples

**Map projection errors**:

* Install cartopy: ``pip install cartopy``
* Use ``plot_map=False`` as workaround
* Check cartopy data downloads

**Memory issues with high DPI**:

* Reduce DPI: ``dpi=150`` instead of 300
* Plot fewer samples per figure
* Close figures: ``plt.close('all')``

**Plots not showing**:

* Check ``show`` parameter
* In scripts, add ``plt.show()`` explicitly
* In Jupyter, check ``%matplotlib inline``

Next Steps
----------

* See :doc:`api` for complete function documentation
* Check :doc:`examples` for real-world usage
* Review :doc:`usage` for analysis workflows
