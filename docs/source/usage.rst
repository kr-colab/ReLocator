Usage Guide
===========

This guide will walk you through the basic usage of Locator for predicting geographic coordinates from genotype matrices.

Basic Usage
----------

Loading Data
~~~~~~~~~~~

Locator expects genotype data in a specific format. Here's how to load your data:

.. code-block:: python

   import locator
   from locator.data import load_genotype_data

   # Load genotype matrix
   genotype_data = load_genotype_data("path/to/genotype_matrix.csv")
   
   # Load known coordinates (for training)
   coordinates = load_genotype_data("path/to/coordinates.csv")

Creating a Model
~~~~~~~~~~~~~~

Locator provides several pre-built models. Here's how to create and configure one:

.. code-block:: python

   from locator.models import LocatorModel

   # Create a model with default settings
   model = LocatorModel(
       input_shape=(n_loci,),  # Number of genetic loci
       hidden_layers=[128, 64],
       dropout_rate=0.2
   )

Training
~~~~~~~~

Train the model using your genotype data and known coordinates:

.. code-block:: python

   # Train the model
   history = model.fit(
       genotype_data,
       coordinates,
       epochs=100,
       batch_size=32,
       validation_split=0.2
   )

Making Predictions
~~~~~~~~~~~~~~~~

Once trained, you can use the model to predict coordinates for new samples:

.. code-block:: python

   # Load new genotype data
   new_samples = load_genotype_data("path/to/new_samples.csv")
   
   # Make predictions
   predicted_coords = model.predict(new_samples)

Advanced Usage
-------------

Using Species Range Masks
~~~~~~~~~~~~~~~~~~~~~~~

Locator can incorporate species range masks to improve prediction accuracy:

.. code-block:: python

   from locator.utils import load_range_mask

   # Load range mask
   range_mask = load_range_mask("path/to/range_mask.tif")
   
   # Create model with range mask
   model = LocatorModel(
       input_shape=(n_loci,),
       range_mask=range_mask
   )

Custom Loss Functions
~~~~~~~~~~~~~~~~~~~

You can define custom loss functions for specific needs:

.. code-block:: python

   from locator.losses import CustomLoss

   # Create custom loss
   custom_loss = CustomLoss(
       mse_weight=1.0,
       range_penalty_weight=0.5
   )
   
   # Use in model
   model = LocatorModel(
       input_shape=(n_loci,),
       loss=custom_loss
   )

Batch Processing
~~~~~~~~~~~~~~

For large datasets, use batch processing:

.. code-block:: python

   from locator.data import DataGenerator

   # Create data generator
   generator = DataGenerator(
       genotype_data,
       coordinates,
       batch_size=32
   )
   
   # Train with generator
   model.fit(
       generator,
       epochs=100
   )

Evaluation
---------

Locator provides various metrics for evaluating prediction accuracy:

.. code-block:: python

   from locator.metrics import evaluate_predictions

   # Evaluate predictions
   metrics = evaluate_predictions(
       true_coords,
       predicted_coords
   )
   
   print(f"Mean Squared Error: {metrics['mse']}")
   print(f"Mean Absolute Error: {metrics['mae']}")
   print(f"R-squared Score: {metrics['r2']}")

Visualization
-----------

Visualize your results using built-in plotting functions:

.. code-block:: python

   from locator.visualization import plot_predictions

   # Plot predictions
   plot_predictions(
       true_coords,
       predicted_coords,
       range_mask=range_mask
   )

Next Steps
---------

* Check out the :doc:`api` reference for detailed information about all available functions and classes
* See the :doc:`examples` section for more advanced usage examples
* Learn how to :doc:`contributing` to the project 