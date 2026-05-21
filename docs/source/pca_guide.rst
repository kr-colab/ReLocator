The PCA Model
=============

Locator's networks struggle when the genotype data has many more SNPs than
samples -- for example, whole-genome data with hundreds of thousands or
millions of SNPs but only a few hundred individuals. The first layer of the
network grows with the number of SNPs, so with millions of SNPs it has
hundreds of millions of values to learn. That is far too many for a few
hundred samples: the network memorizes the training samples instead of
learning a real pattern, and its predictions on new samples get *worse* as
more SNPs are added.

The PCA model fixes this. Before the network sees the genotypes, Locator runs
a PCA and keeps only a handful of components. The network then learns from
those few components instead of from millions of raw SNPs, so it stays small
and accurate no matter how many SNPs you give it.

.. contents:: Table of Contents
   :local:
   :depth: 2

When to use it
--------------

Turn the PCA model on when you have many more SNPs than samples -- roughly,
whole-genome data with 100,000 or more SNPs and a few hundred samples. With
fewer SNPs the plain network works well and PCA is not needed. The PCA model's
job is to keep accuracy steady as the SNP count grows into the millions, where
a plain network gets worse.

It works with normal training, holdouts, k-fold and leave-one-out
cross-validation, and ensembles.

Basic usage
-----------

Let Locator choose the size (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``pca_components`` to ``"auto"`` and Locator decides how many components to
keep, based on the data:

.. code-block:: python

   from locator import Locator

   config = {
       "out": "wgs_analysis",
       "batch_size": 32,
       "width": 256,
       "nlayers": 8,
       "max_epochs": 500,
       "patience": 100,
       "pca_components": "auto",
   }

   locator = Locator(config)
   genotypes, samples = locator.load_genotypes(zarr="genotypes.zarr")
   locator.train(genotypes=genotypes, samples=samples)

Locator prints the number it chose at the start of training and stores it, so
every fold of a run uses the same number.

Set the size yourself
~~~~~~~~~~~~~~~~~~~~~~

Pass a number instead to keep exactly that many components:

.. code-block:: python

   config["pca_components"] = 64

Leaving ``pca_components`` out (or set to ``None``) turns the PCA model off.
That is the default.

How it works
------------

When ``pca_components`` is set, the genotypes pass through an extra PCA step
before the rest of the network::

   genotype data  ->  PCA step (keeps a few components)  ->  rest of the network

A few details:

* **The PCA is run on the training samples only.** Samples held out for
  testing are never used to build it, so cross-validation stays fair.
* **The PCA step starts as an exact PCA.** It is set up to reproduce the PCA
  result exactly, and training is then allowed to adjust it. The rest of the
  network starts from random values, as usual.
* **Training happens in two stages.** In the first stage the PCA step is held
  fixed while the rest of the network learns. In the second stage the PCA step
  is allowed to change too, more slowly, so it can adjust to better predict
  location. Set ``pca_finetune`` to ``False`` to skip the second stage and
  keep the PCA step fixed throughout.

Choosing how many components to keep
------------------------------------

With ``pca_components="auto"``, Locator looks at how much variation each
component captures. The first few components capture a lot; after that, each
one adds little. Locator keeps components up to the point where the curve
levels off -- the natural cut-off in the data. This is usually a small number,
and in practice it predicts just as well as a much larger hand-picked number.

To see this cut-off yourself before choosing a number:

.. code-block:: python

   from locator.pca import scree_elbow

   # training genotypes, shape (samples, SNPs)
   n_components = scree_elbow(train_genotypes)

The number of components cannot be larger than the number of training samples
or the number of SNPs; a larger value raises a clear error.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Setting
     - Default
     - What it does
   * - ``pca_components``
     - ``None``
     - Turns the PCA model on or off: ``None`` is off, a number keeps that
       many components, and ``"auto"`` lets Locator choose.
   * - ``pca_finetune``
     - ``True``
     - Whether the second training stage adjusts the PCA step. ``False`` keeps
       the PCA step fixed the whole time.
   * - ``pca_finetune_lr``
     - ``1e-4``
     - How fast the PCA step is allowed to change in the second stage.

When you cannot use it
----------------------

The PCA model does not work with:

* **Bootstrap or jacknife runs.** These resample or reorder the SNPs on every
  replicate, and the PCA step needs a fixed set of SNPs. Passing ``site_order``
  together with ``pca_components`` raises an error.
* **Windowed analysis.** Each window uses its own set of SNPs, so windowed
  runs reject ``pca_components``.

For these, leave ``pca_components`` off.
