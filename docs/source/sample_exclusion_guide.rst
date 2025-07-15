Sample Exclusion Guide
======================

This guide explains how to exclude specific samples from Locator analyses, which is useful for removing outliers, low-quality samples, or samples with known issues.

Overview
--------

Sample exclusion in Locator allows you to remove specific samples from all analyses while maintaining an audit trail of what was excluded and why. This feature is complementary to the NA handling system - while NA handling deals with samples missing coordinates, sample exclusion lets you remove samples for quality control or other reasons.

Key features:

- Exclude samples from file or programmatically
- Track exclusion reasons for reproducibility
- Interactive exclusion based on analysis results
- Works with all Locator methods (train, predict, holdouts, ensemble)

When to Use Sample Exclusion
-----------------------------

Common scenarios for excluding samples:

1. **Quality Control**: Remove samples with low genotype rates or high missingness
2. **Outlier Removal**: Exclude samples with unusually high prediction errors
3. **Geographic Outliers**: Remove samples that are geographically isolated
4. **Technical Issues**: Exclude samples with known sequencing or sampling problems
5. **Population Structure**: Remove admixed individuals or migrants

File-Based Exclusion
--------------------

The simplest way to exclude samples is to provide a list when initializing Locator:

From a File
~~~~~~~~~~~

Create a text file with one sample ID per line:

.. code-block:: text

   # outliers.txt
   # Samples identified as outliers in preliminary analysis
   sample_001
   sample_045

   # Low quality samples
   sample_023
   sample_089

Then use it when initializing Locator:

.. code-block:: python

   locator = Locator({
       "exclude_samples": "outliers.txt",
       "sample_data": "samples.txt"
   })

File format notes:

- One sample ID per line
- Lines starting with ``#`` are treated as comments
- Empty lines are ignored
- Sample IDs must match exactly (case-sensitive)

From a List
~~~~~~~~~~~

You can also provide a list directly:

.. code-block:: python

   locator = Locator({
       "exclude_samples": ["sample_001", "sample_045", "sample_023"],
       "sample_data": "samples.txt"
   })

Interactive Exclusion
---------------------

Exclude samples dynamically based on analysis results:

Basic Exclusion
~~~~~~~~~~~~~~~

.. code-block:: python

   # Exclude specific samples
   locator.exclude_samples("sample_001", reason="outlier")

   # Exclude multiple samples
   locator.exclude_samples(
       ["sample_002", "sample_003", "sample_004"],
       reason="batch_effect"
   )

Conditional Exclusion
~~~~~~~~~~~~~~~~~~~~~

Exclude samples based on criteria:

.. code-block:: python

   # After running holdout analysis
   predictions = pd.read_csv("holdout_predictions.txt", sep="\t")

   # Calculate prediction errors
   errors = calculate_errors(predictions, true_coords)

   # Exclude samples with high error
   locator.exclude_samples_by_condition(
       lambda df: df['error'] > 100,  # 100 km threshold
       sample_df=errors,
       reason="high_prediction_error"
   )

Managing Exclusions
-------------------

View Excluded Samples
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get DataFrame of excluded samples
   excluded_df = locator.get_excluded_samples()
   print(excluded_df)

   # Output:
   #      sampleID                reason
   # 0  sample_001               outlier
   # 1  sample_002          batch_effect
   # 2  sample_003          batch_effect
   # 3  sample_004          batch_effect
   # 4  sample_045  high_prediction_error

Include Samples Back
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Include specific samples back
   n_included = locator.include_samples(["sample_001", "sample_002"])
   print(f"Included {n_included} samples back")

Clear All Exclusions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Remove all exclusions
   locator.clear_exclusions()

Integration with Data Checking
------------------------------

The ``check_data()`` method reports exclusion statistics:

.. code-block:: python

   locator.check_data(genotypes, samples)

   # Output:
   # Data Summary
   # ==================================================
   # Total samples: 500
   # Samples with coordinates: 480
   # Samples without coordinates: 20
   # Excluded samples: 15
   #   - Excluded samples with coordinates: 12
   # Available samples for training: 468
   # Total SNPs: 10000

Complete Workflow Example
-------------------------

Here's a complete workflow showing sample exclusion in practice:

.. code-block:: python

   from locator import Locator
   import pandas as pd
   import numpy as np

   # Step 1: Initial analysis to identify problematic samples
   loc = Locator({
       "out": "initial_analysis",
       "sample_data": "samples.txt"
   })

   genotypes, samples = loc.load_genotypes(vcf="genotypes.vcf.gz")

   # Run holdout analysis
   loc.run_holdouts(genotypes, samples, k=50, replicates=10)

   # Step 2: Identify outliers
   predictions = pd.read_csv("initial_analysis_holdouts.txt", sep="\t")

   # Calculate errors (assuming you have true coordinates)
   from locator.plotting import plot_error_summary
   merged = plot_error_summary(
       predictions,
       "samples.txt",
       return_merged=True
   )

   # Find outliers (e.g., >2 SD from mean error)
   mean_error = merged['error'].mean()
   std_error = merged['error'].std()
   outlier_threshold = mean_error + 2 * std_error

   outliers = merged[merged['error'] > outlier_threshold]['sampleID'].tolist()
   print(f"Found {len(outliers)} outliers with error > {outlier_threshold:.1f} km")

   # Step 3: Create new analysis excluding outliers
   loc_filtered = Locator({
       "out": "filtered_analysis",
       "sample_data": "samples.txt",
       "exclude_samples": outliers
   })

   # Check the data
   loc_filtered.check_data(genotypes, samples)

   # Step 4: Train final model
   loc_filtered.train(genotypes, samples)

   # Step 5: If needed, exclude more samples interactively
   # For example, after examining the training history
   if loc_filtered.history.history['val_loss'][-1] > threshold:
       # Load sample metadata
       sample_meta = pd.read_csv("sample_metadata.csv")

       # Exclude samples from a problematic batch
       loc_filtered.exclude_samples_by_condition(
           lambda df: df['batch'] == 'batch_7',
           sample_df=sample_meta,
           reason="problematic_batch"
       )

       # Retrain
       loc_filtered.train(genotypes, samples)

   # Step 6: Save exclusion list for reproducibility
   excluded = loc_filtered.get_excluded_samples()
   excluded.to_csv("excluded_samples.csv", index=False)

Best Practices
--------------

1. **Document Exclusions**: Always provide meaningful reasons when excluding samples
2. **Save Exclusion Lists**: Export excluded samples for reproducibility
3. **Iterative Refinement**: Start with obvious exclusions, then refine based on results
4. **Validate Impact**: Compare results with and without exclusions
5. **Consider Alternatives**: Sometimes transformation or weighting is better than exclusion

Technical Details
-----------------

How Exclusions Work
~~~~~~~~~~~~~~~~~~~

- Excluded samples are removed during the ``sort_samples()`` step
- They are treated similarly to NA samples in the ``IndexSet``
- Exclusions are applied before any analysis begins
- All downstream methods respect the exclusions

Interaction with NA Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sample exclusion works alongside NA handling:

- NA handling: Deals with samples missing coordinates
- Sample exclusion: Removes samples for other reasons
- Both can be used together
- Excluded samples are removed before NA handling is applied

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Exclusions are applied once during data loading
- No performance penalty during training or prediction
- Memory usage is reduced by not loading excluded samples
- Exclusion checking is O(n) where n is the number of samples

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Sample IDs not found**:

.. code-block:: python

   # Check exact sample IDs in your data
   print(samples[:10])  # First 10 sample IDs

   # Ensure IDs match exactly (case-sensitive)
   # "Sample_001" != "sample_001"

**Exclusions not applied**:

.. code-block:: python

   # Verify exclusions were loaded
   print(f"Excluded samples: {len(locator._excluded_sample_ids)}")
   print(locator.get_excluded_samples())

   # Make sure to reload data after adding exclusions
   genotypes, samples = locator.load_genotypes(...)

**Too many samples excluded**:

.. code-block:: python

   # Review your exclusion criteria
   # Check the distribution before excluding
   errors = merged['error']
   print(f"Error distribution: mean={errors.mean():.1f}, std={errors.std():.1f}")
   print(f"Would exclude {(errors > threshold).sum()} samples with threshold={threshold}")

See Also
--------

- :doc:`na_handling_guide` - For handling samples without coordinates
- :doc:`api` - Complete API reference
- :doc:`examples` - More code examples
