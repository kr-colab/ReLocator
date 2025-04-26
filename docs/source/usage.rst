Usage Guide
===========

This guide covers how to use Locator for predicting geographic coordinates from genotype matrices.

For complete API documentation, see :doc:`api`.

Basic Usage
-----------
Loading Data
~~~~~~~~~~~~
For detailed API reference of data loading functions, see :doc:`api`.

Locator supports multiple input formats for genotype data:

.. code-block:: python

   from locator import Locator

   # Create a Locator instance with configuration
   config = {
       "out": "my_analysis",
       "batch_size": 32,
       "width": 256,
       "nlayers": 8,
       "dropout_prop": 0.25
   }
   
   locator = Locator(config)
   
   # Load data from various formats:
   #
   # 1. From VCF
   genotypes, samples = locator.load_genotypes(vcf="path/to/genotypes.vcf")
   #
   # 2. From zarr (recommended for large datasets)
   genotypes, samples = locator.load_genotypes(zarr="path/to/genotypes.zarr")
   #
   # 3. From pandas DataFrame
   locator = Locator({
       "out": "my_analysis",
       "genotype_data": genotype_df,  # DataFrame with samples as index, SNPs as columns
       "sample_data": coords_df       # DataFrame with sampleID, x, y columns
   })

Training and Prediction
-----------------------
Train the model and make predictions:

.. code-block:: python

   # Train the model
   history = locator.train(genotypes=genotypes, samples=samples)
   
   # Make predictions
   predictions = locator.predict(return_df=True)  # Returns DataFrame with sampleID, x, y

Advanced Usage
--------------
For complete API documentation of advanced features, see :doc:`api`.

Holdout Analysis
----------------
Evaluate model performance by holding out samples:

.. code-block:: python

   # Hold out k samples during training
   locator.train_holdout(
       genotypes=genotypes,
       samples=samples,
       k=10
   )
   
   # Get predictions for held-out samples
   holdout_preds = locator.predict_holdout(
       return_df=True,
       plot_summary=True
   )

Ensemble Models
---------------
Use multiple models for improved predictions:

.. code-block:: python

   from locator import EnsembleLocator
   
   # Create ensemble with 5 models
   ensemble = EnsembleLocator(
       base_config=config,
       k_folds=5
   )
   
   # Train ensemble
   histories = ensemble.train(
       genotypes=genotypes,
       samples=samples
   )
   
   # Get ensemble predictions
   predictions = ensemble.predict(return_df=True)

Windowed Analysis
-----------------
Analyze predictions across genomic windows:

.. code-block:: python

   # Run windowed analysis
   window_predictions = locator.run_windows(
       genotypes=genotypes,
       samples=samples,
       window_size=5e5,  # 500kb windows
       return_df=True
   )

Jacknife Analysis
-----------------
Assess prediction uncertainty:

.. code-block:: python

   # Run jacknife analysis
   jacknife_predictions = locator.run_jacknife(
       genotypes=genotypes,
       samples=samples,
       prop=0.05,  # Proportion of SNPs to mask
       n_replicates=100,
       return_df=True
   )

Using Range Masks
-----------------
Incorporate species range constraints:

.. code-block:: python

   # Configure model with range penalty
   config = {
       "out": "range_constrained",
       "use_range_penalty": True,
       "species_range_shapefile": "path/to/range.shp",
       "resolution": 0.05,
       "penalty_weight": 1.0
   }
   
   locator = Locator(config)

GPU Configuration
-----------------
Configure GPU usage:

.. code-block:: python

   # Specify GPU device
   config = {
       "out": "gpu_analysis",
       "gpu_number": 0  # Use first GPU
   }
   
   # Or disable GPU
   config = {
       "out": "cpu_analysis",
       "disable_gpu": True
   }

Data Augmentation
-----------------
Enable data augmentation during training:

.. code-block:: python

   config = {
       "out": "augmented",
       "augmentation": {
           "enabled": True,
           "flip_rate": 0.05  # Rate at which to flip genotypes
       }
   }

Next Steps
----------
* Check the :doc:`api` reference for detailed information about all available functions and classes.
* See the :doc:`examples` section for more advanced usage examples.
* Learn how to :doc:`contributing` to the project. 