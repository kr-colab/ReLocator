Handling Missing Coordinates Guide
==================================

Locator provides flexible handling of samples that lack geographic
coordinates (NA samples), controlled through the ``na_action`` parameter.
This can be set at the instance level or overridden per method call.

Understanding NA Samples
------------------------

Samples are considered to have "NA" (missing) coordinates when:

- The x (longitude) coordinate is NaN or missing
- The y (latitude) coordinate is NaN or missing
- Either coordinate is missing (a sample needs both x and y to be "known")

NA Action Modes
---------------

separate (default)
~~~~~~~~~~~~~~~~~~

Separates samples into training (known locations) and prediction (unknown
locations) sets. Training uses only known-coordinate samples; prediction
includes all samples.

.. code-block:: python

   locator = Locator({"na_action": "separate"})

**Use when**: You have new samples without known locations that you want
to predict.

exclude
~~~~~~~

Filters out all samples without coordinates before any analysis. Only
known-location samples participate; dataset size is reduced accordingly.

.. code-block:: python

   locator = Locator({"na_action": "exclude"})

**Use when**: You only want to analyze samples with verified locations.

fail
~~~~

Raises a ``ValueError`` if any samples lack coordinates. Forces you to
handle missing data explicitly before proceeding.

.. code-block:: python

   locator = Locator({"na_action": "fail"})

**Use when**: You want to enforce data completeness in a QC pipeline.

Checking Your Data
------------------

Always inspect your data before analysis:

.. code-block:: python

   genotypes, samples = locator.load_genotypes(vcf="data.vcf")
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

Method-Specific Behavior
------------------------

Different analysis methods handle NA samples in different ways.

Methods Supporting Full 'separate' Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These methods can train on known samples and predict on unknown samples:

- ``train()`` / ``predict()``
- ``run_bootstraps()``
- ``run_windows()``
- ``run_jacknife()``

Holdout Methods
~~~~~~~~~~~~~~~

Holdout methods require known coordinates for evaluation, so ``'separate'``
behaves like ``'exclude'`` -- only samples with known coordinates are used:

- ``run_holdouts()``
- ``run_k_fold_holdouts()``
- ``run_jacknife_holdouts()``
- ``run_windows_holdouts()``
