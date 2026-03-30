Plotting Guide
==============

Locator provides visualization functions for analyzing predictions and model
performance. The plotting module (``locator.plotting``) supports prediction
distributions, error summaries, sample weight maps, and rich Jupyter notebook
integration. All functions can save to files or display inline.

Prediction Visualization
------------------------

plot_predictions()
~~~~~~~~~~~~~~~~~~

The ``plot_predictions()`` function visualizes results from analyses that
generate multiple predictions per sample (jacknife, bootstraps, windowed):

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

    plot_predictions(
        predictions,
        locator,
        "custom_viz",
        samples=['HG001', 'HG002', 'HG003'],  # Specific samples
        n_cols=1,           # Single column layout
        width=8,            # Wider plots
        height=6,           # Taller plots
        plot_map=True,      # Use geographic map
        dpi=150             # Lower resolution for faster rendering
    )

Error Analysis
--------------

plot_error_summary()
~~~~~~~~~~~~~~~~~~~~

For holdout-based analyses, ``plot_error_summary()`` provides error
visualization with a map panel (true locations colored by error, with lines
to predictions) and a histogram panel (error distribution with summary
statistics):

.. code-block:: python

    from locator.plotting import plot_error_summary

    # After k-fold cross-validation
    predictions = locator.run_k_fold_holdouts(
        genotypes, samples, k=10, return_df=True
    )

    # Create error summary
    plot_error_summary(
        predictions,
        "samples.tsv",      # Path to true coordinates (or DataFrame)
        "kfold_errors",     # Output prefix
        use_geodesic=True,  # Calculate distances in km
        plot_map=True,      # Geographic map (requires cartopy)
        width=16,
        height=8
    )

Set ``plot_map=False`` for a plain scatter plot (faster, no cartopy
dependency). Set ``use_geodesic=False`` to report errors in coordinate units
instead of kilometers.

Sample Weight Visualization
---------------------------

plot_sample_weights()
~~~~~~~~~~~~~~~~~~~~~

When using sample weighting, visualize the geographic distribution of weights:

.. code-block:: python

    from locator.plotting import plot_sample_weights

    config = {
        "out": "weighted_analysis",
        "weight_samples": {
            "enabled": True,
            "method": "KD",
            "bandwidth": None   # Auto-calculate
        }
    }

    locator = Locator(config)
    locator.train(genotypes, samples)

    # Plot the weights
    plot_sample_weights(locator, "kde_weights")

The visualization uses log-scale coloring: bright/yellow for high weights
(undersampled regions) and dark/purple for low weights (oversampled regions).

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

See Also
--------

* :doc:`usage` — Python API and analysis workflows
* :doc:`api` — Complete function documentation
