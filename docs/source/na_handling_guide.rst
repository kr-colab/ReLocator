Handling Missing Coordinates Guide
==================================

This guide provides comprehensive information about handling samples without geographic coordinates in Locator.

Overview
--------

Locator now provides consistent and flexible handling of samples that lack geographic coordinates (NA samples). This is controlled through the ``na_action`` parameter, which can be set at the instance level or overridden for individual methods.

Understanding NA Samples
------------------------

Samples are considered to have "NA" (missing) coordinates when:

- The x (longitude) coordinate is NaN or missing
- The y (latitude) coordinate is NaN or missing
- Either coordinate is missing (a sample needs both x and y to have "known" coordinates)

Common scenarios where you might have NA samples:

1. **New samples to predict**: Samples collected from unknown locations that you want to predict
2. **Quality control failures**: Samples that failed location verification
3. **Historical samples**: Museum specimens with uncertain provenance
4. **Blind prediction sets**: Samples intentionally withheld for validation

NA Action Modes
---------------

Locator provides three modes for handling NA samples:

separate (default)
~~~~~~~~~~~~~~~~~~

The default mode that separates samples into training (known locations) and prediction (unknown locations) sets.

.. code-block:: python

   locator = Locator({"na_action": "separate"})

   # Trains on samples with coordinates
   # Can predict on samples without coordinates

**Use when**: You have new samples without known locations that you want to predict.

**Behavior**:
- Training uses only samples with known coordinates
- Prediction includes all samples (both known and unknown)
- Unknown samples get predicted coordinates

exclude
~~~~~~~

Filters out all samples without coordinates before any analysis.

.. code-block:: python

   locator = Locator({"na_action": "exclude"})

   # Only uses samples with known coordinates
   # NA samples are ignored completely

**Use when**: You only want to analyze samples with verified locations.

**Behavior**:
- NA samples are removed from all analyses
- Only known-location samples are used
- Reduces dataset size but ensures all samples have coordinates

fail
~~~~

Raises an error if any samples lack coordinates.

.. code-block:: python

   locator = Locator({"na_action": "fail"})

   # Raises ValueError if any NA samples are found

**Use when**: You want to ensure data completeness before analysis.

**Behavior**:
- Stops execution if NA samples are detected
- Forces you to handle missing data explicitly
- Useful for quality control pipelines

Checking Your Data
------------------

Always check your data before analysis:

.. code-block:: python

   # Load your data
   genotypes, samples = locator.load_genotypes(vcf="data.vcf")

   # Check sample status
   locator.check_data(genotypes, samples, verbose=True)

This will display:

.. code-block:: text

   ===== Data Summary =====
   Total samples: 231
   Samples with coordinates: 211
   Samples without coordinates: 20
   Total SNPs: 1000

   Current NA handling mode: separate
   - Will train on samples with known locations
   - Can predict on samples without locations

   Samples without coordinates (first 10):
   - sample_X123
   - sample_X124
   - ...

Programmatic Status Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For programmatic access to sample status:

.. code-block:: python

   status = locator.get_sample_status(samples)

   print(f"Known samples: {status['n_known']}")
   print(f"NA samples: {status['n_na']}")
   print(f"NA sample IDs: {status['na_samples']}")

Method-Specific Behavior
------------------------

Different analysis methods handle NA samples differently:

Methods Supporting Full 'separate' Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods can train on known samples and predict on unknown samples:

- ``train()`` / ``predict()``
- ``run_bootstraps()``
- ``run_windows()``
- ``run_jacknife()``

.. code-block:: python

   # These work with 'separate' mode to predict NA samples
   locator.train(genotypes, samples)  # Trains on known only
   predictions = locator.predict()     # Predicts for all samples

Holdout Methods
~~~~~~~~~~~~~~~

Holdout methods require known coordinates for evaluation, so 'separate' behaves like 'exclude':

- ``run_holdouts()``
- ``run_k_fold_holdouts()``
- ``run_jacknife_holdouts()``
- ``run_windows_holdouts()``

.. code-block:: python

   # These only use samples with known coordinates
   locator.run_k_fold_holdouts(genotypes, samples)
   # Even with na_action='separate', only known samples are used

Best Practices
--------------

1. **Always check your data first**

   .. code-block:: python

      locator.check_data(genotypes, samples, verbose=True)

2. **Choose the appropriate na_action for your analysis**

   - Predicting new samples? Use ``'separate'`` (default)
   - Only want complete data? Use ``'exclude'``
   - Enforce data quality? Use ``'fail'``

3. **Be aware of method limitations**

   - Holdout methods need coordinates for evaluation
   - Document which samples lack coordinates in your results

4. **Consider creating separate analyses**

   .. code-block:: python

      # Analysis 1: Predict unknown samples
      loc_predict = Locator({"na_action": "separate"})

      # Analysis 2: Evaluate only on known samples
      loc_eval = Locator({"na_action": "exclude"})

Troubleshooting
---------------

**Issue**: "Found X samples without coordinates" error

**Solution**: You're using ``na_action='fail'``. Switch to ``'separate'`` or ``'exclude'``:

.. code-block:: python

   locator.train(genotypes, samples, na_action='separate')

**Issue**: Holdout analysis not using all samples

**Solution**: This is expected. Holdout methods require known coordinates for evaluation.

**Issue**: Predictions DataFrame has fewer rows than expected

**Solution**: Check if you're using ``'exclude'`` mode, which filters out NA samples.

Migration from Older Versions
-----------------------------

Older versions of Locator had inconsistent NA handling. The new system:

- Defaults to ``'separate'`` mode (backward compatible)
- Makes NA handling explicit and consistent
- Provides clear reporting of sample status

If your existing code relies on the old behavior, it should continue to work with the default ``'separate'`` mode.
