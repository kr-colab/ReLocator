Quality Control Guide
=====================

This guide covers genotype quality control features in Locator, including sample quality assessment
and SNP subsetting for computational efficiency.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Locator provides comprehensive quality control features to help identify and handle problematic
samples and reduce computational requirements through intelligent SNP subsetting. These features
integrate seamlessly with the existing sample exclusion system.

Sample Quality Assessment
-------------------------

The ``check_genotypes()`` method analyzes genotype quality metrics to identify samples with
anomalous patterns, particularly high missingness rates.

Basic Usage
^^^^^^^^^^^

.. code-block:: python

    from locator import Locator

    # Initialize Locator
    locator = Locator({"out": "analysis", "sample_data": "coords.txt"})

    # Load genotypes
    genotypes, samples = locator.load_genotypes(vcf="data.vcf.gz")

    # Run quality control check
    qc_results = locator.check_genotypes(
        genotypes,
        samples,
        method='mad',  # Outlier detection method
        plot=True,     # Generate diagnostic plots
        verbose=True   # Print summary
    )

Output Example
^^^^^^^^^^^^^^

::

    Genotype Quality Control Summary:
    Total samples: 384
    Mean missing rate: 0.032
    Median missing rate: 0.028

    Outlier detection method: mad
    MAD multiplier: 3.0
    Outliers found: 5

    Top outlier samples:
      Sample_101: 0.215 missing
      Sample_247: 0.187 missing
      Sample_089: 0.156 missing
      Sample_331: 0.142 missing
      Sample_022: 0.128 missing

    Suggested action:
    locator.exclude_samples(['Sample_101', 'Sample_247', ...], reason='high_missingness')

Outlier Detection Methods
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``check_genotypes()`` method supports multiple outlier detection approaches:

1. **MAD (Median Absolute Deviation)**

   .. code-block:: python

       # Robust to outliers, recommended default
       qc_results = locator.check_genotypes(
           genotypes, samples,
           method='mad',
           n_mad=3.0  # Number of MADs from median
       )

2. **IQR (Interquartile Range)**

   .. code-block:: python

       # Standard boxplot method
       qc_results = locator.check_genotypes(
           genotypes, samples,
           method='iqr',
           iqr_multiplier=1.5  # Standard boxplot outlier threshold
       )

3. **Z-score**

   .. code-block:: python

       # Based on standard deviations
       qc_results = locator.check_genotypes(
           genotypes, samples,
           method='zscore',
           n_std=3.0  # Number of standard deviations
       )

4. **Threshold**

   .. code-block:: python

       # Simple cutoff value
       qc_results = locator.check_genotypes(
           genotypes, samples,
           method='threshold',
           threshold=0.1  # 10% missing data cutoff
       )

Visualization
^^^^^^^^^^^^^

When ``plot=True``, the function generates a 4-panel diagnostic plot:

- **Top-left**: Histogram of missing rates with outliers highlighted
- **Top-right**: Boxplot comparing normal vs outlier samples
- **Bottom-left**: Scatter plot of heterozygosity vs missing rate
- **Bottom-right**: Missing rate by sample order (useful for batch effects)

Integration with Sample Exclusion
---------------------------------

The quality control results integrate seamlessly with Locator's sample exclusion system:

.. code-block:: python

    # Run QC check
    qc_results = locator.check_genotypes(genotypes, samples)

    # Automatically exclude outliers
    if qc_results['outliers']:
        locator.exclude_samples(
            qc_results['outliers'],
            reason='high_missingness'
        )

    # View excluded samples
    excluded_df = locator.get_excluded_samples()
    print(excluded_df)

    # Continue with analysis - excluded samples are automatically filtered
    locator.train(genotypes, samples)

Advanced QC Workflow
^^^^^^^^^^^^^^^^^^^^

For more control over the exclusion process:

.. code-block:: python

    # Get detailed statistics
    qc_results = locator.check_genotypes(
        genotypes, samples,
        return_stats=True
    )

    # Access the statistics DataFrame
    stats_df = qc_results['stats']

    # Custom filtering based on multiple metrics
    problematic = stats_df[
        (stats_df['missing_rate'] > 0.1) |
        (stats_df['heterozygosity'] < 0.1)
    ]

    # Exclude based on custom criteria
    locator.exclude_samples(
        problematic['sampleID'].tolist(),
        reason='failed_qc_metrics'
    )

SNP Subsetting
--------------

The ``subset_genotypes()`` method provides efficient ways to reduce the number of SNPs for
computational efficiency or testing purposes.

Random Subsetting
^^^^^^^^^^^^^^^^^

Random selection maintains the overall allele frequency distribution:

.. code-block:: python

    # Subset to specific number of SNPs
    genotypes_subset = locator.subset_genotypes(
        genotypes,
        method='random',
        n_snps=100000,
        seed=42  # For reproducibility
    )

    # Or use a fraction
    genotypes_subset = locator.subset_genotypes(
        genotypes,
        method='random',
        fraction=0.1,  # Keep 10% of SNPs
        seed=42
    )

Uniform Subsetting
^^^^^^^^^^^^^^^^^^

Uniform selection provides even genomic coverage:

.. code-block:: python

    # Select every Nth SNP
    genotypes_subset = locator.subset_genotypes(
        genotypes,
        method='uniform',
        n_snps=50000
    )

    # Output shows spacing information
    # "Average spacing: every 20.0 SNPs"

Getting Selected Indices
^^^^^^^^^^^^^^^^^^^^^^^^

For reproducibility and downstream analysis:

.. code-block:: python

    # Return both subsetted genotypes and indices
    genotypes_subset, indices = locator.subset_genotypes(
        genotypes,
        method='random',
        n_snps=100000,
        seed=42,
        return_indices=True
    )

    # Use indices for other data
    if hasattr(locator, 'positions'):
        positions_subset = locator.positions[indices]

Complete QC Workflow Example
----------------------------

Here's a complete example combining quality control and subsetting:

.. code-block:: python

    from locator import Locator
    import numpy as np

    # Initialize Locator
    config = {
        "out": "turtle_analysis",
        "sample_data": "turtle_coords.txt",
        "max_epochs": 1000,
        "patience": 100
    }
    locator = Locator(config)

    # Load full dataset
    print("Loading genotype data...")
    genotypes, samples = locator.load_genotypes(vcf="turtle_genotypes.vcf.gz")
    print(f"Loaded {genotypes.shape[0]:,} SNPs for {genotypes.shape[1]} samples")

    # Step 1: Quality control check
    print("\nRunning quality control...")
    qc_results = locator.check_genotypes(
        genotypes,
        samples,
        method='mad',
        n_mad=3.0,
        plot=True,
        verbose=True
    )

    # Step 2: Exclude problematic samples
    if qc_results['outliers']:
        print(f"\nExcluding {len(qc_results['outliers'])} outlier samples")
        locator.exclude_samples(
            qc_results['outliers'],
            reason='high_missingness'
        )

    # Step 3: Subset SNPs for efficiency
    print("\nSubsetting SNPs for analysis...")
    genotypes_subset = locator.subset_genotypes(
        genotypes,
        method='random',
        n_snps=100000,
        seed=42,
        verbose=True
    )

    # Step 4: Run analysis with clean, subsetted data
    print("\nTraining model with QC-filtered data...")
    locator.train(genotypes_subset, samples)

    # Make predictions
    predictions = locator.predict()

Performance Considerations
--------------------------

SNP Subsetting Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Random subsetting**: Best for maintaining population structure signals
- **Uniform subsetting**: Better for analyses requiring genomic coverage
- **Typical reduction**: 50-90% reduction often maintains prediction accuracy

Memory and Speed Benefits
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Example: 1M SNPs -> 100k SNPs
    # Memory reduction: ~90%
    # Training time reduction: ~85-90%
    # Minimal accuracy loss: typically <5%

Best Practices
--------------

1. **Always run QC before analysis**: Identify problematic samples early
2. **Document exclusions**: The exclusion system tracks reasons automatically
3. **Test subsetting levels**: Start conservative, increase reduction if needed
4. **Use seeds for reproducibility**: Essential for random subsetting
5. **Consider your research question**:

   - Population structure: Random subsetting preferred
   - Local adaptation: Consider uniform or window-based (future feature)
   - Testing/development: Aggressive subsetting acceptable

API Reference
-------------

For detailed parameter descriptions, see:

- :meth:`locator.core.Locator.check_genotypes`
- :meth:`locator.core.Locator.subset_genotypes`
