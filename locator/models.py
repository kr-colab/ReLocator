"""Neural network model definitions"""

from typing import Optional

import geopandas as gpd
import numpy as np
import tensorflow as tf
from affine import Affine
from rasterio.features import rasterize
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

PCA_LAYER_NAME = "pca_projection"
PCA_GATE_NAME = "pca_finetune_gate"


def build_optimizer(algo, learning_rate, weight_decay=0.004):
    """Build a Keras optimizer from an algorithm name.

    Args:
        algo: "adam" or "adamw" (case-insensitive).
        learning_rate: Learning rate.
        weight_decay: Weight decay, used only for AdamW.

    Returns
    -------
        A configured keras optimizer.
    """
    algo = algo.lower()
    if algo == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if algo == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {algo}")


def rasterize_species_range(shapefile_path, resolution=0.1):
    gdf = gpd.read_file(shapefile_path)
    geom = gdf.unary_union
    bounds = gdf.total_bounds  # xmin, ymin, xmax, ymax

    xmin, ymin, xmax, ymax = bounds
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    transform = Affine.translation(xmin, ymin) * Affine.scale(resolution, resolution)

    mask = rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )

    return mask.astype(np.float32), transform


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function using Euclidean distance.

    Args:
        y_true: Tensor of true coordinates
        y_pred: Tensor of predicted coordinates

    Returns
    -------
        Euclidean distance between true and predicted coordinates
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def mask_lookup(pred_coords, mask_tensor, transform, resolution):
    """
    pred_coords: (batch_size, 2), float32 [lon, lat]
    mask_tensor: (H, W), float32
    transform: Affine, from rasterize step
    resolution: float, grid resolution used
    """
    # Convert coordinates to mask indices
    lon = pred_coords[:, 0]
    lat = pred_coords[:, 1]
    col = tf.clip_by_value(
        ((lon - transform.c) / resolution), 0, mask_tensor.shape[1] - 1
    )
    row = tf.clip_by_value(
        ((lat - transform.f) / resolution), 0, mask_tensor.shape[0] - 1
    )

    # Integer indices (nearest-neighbor lookup)
    col = tf.cast(tf.round(col), tf.int32)
    row = tf.cast(tf.round(row), tf.int32)

    # Gather values from mask
    idx = tf.stack([row, col], axis=-1)
    valid = tf.gather_nd(mask_tensor, idx)

    return valid


def loss_with_range_penalty(
    y_true, y_pred, mask_tensor, transform, resolution, penalty_weight=1.0
):
    # Euclidean distance
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

    # GPU-friendly range mask
    valid_mask = mask_lookup(y_pred, mask_tensor, transform, resolution)

    # Penalize out-of-range predictions
    penalty = tf.square(1.0 - valid_mask)
    # tf.print("Euclidean:", euclidean, summarize=10)
    # tf.print("Penalty:", penalty, summarize=10)
    # tf.print("Valid mask:", valid_mask, summarize=10)
    return euclidean + penalty_weight * penalty


class GradientGate(keras.layers.Layer):
    """Pass values through unchanged while scaling the gradient by a gate.

    Lets the PCA-initialized projection switch between frozen (phase 1) and
    fine-tuning (phase 2) without changing the training graph. The layer's
    output value is always its input, but the gradient that reaches the input
    -- and therefore the upstream projection's weights -- is multiplied by
    ``gate``: 0 holds the projection at its PCA initialization, 1 lets it
    train. ``gate`` is a non-trainable variable, so flipping it neither
    retraces nor recompiles the graph, which keeps the compiled training
    function reusable across both phases and across folds.
    """

    def build(self, input_shape):
        """Create the non-trainable gate variable (starts closed, at 0)."""
        self.gate = self.add_weight(
            name="gate",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Return the input value with its gradient scaled by the gate."""
        # gate == 0: output value == inputs, gradient to inputs == 0.
        # gate == 1: output == inputs with the full gradient.
        frozen = tf.stop_gradient(inputs)
        return frozen + self.gate * (inputs - frozen)


def create_network(
    input_shape: int,
    width: int = 256,
    n_layers: int = 8,
    dropout_prop: float = 0.25,
    pca_components: Optional[int] = None,
    optimizer_config: Optional[dict] = None,
    loss_fn: Optional[callable] = None,
) -> keras.Model:
    """Create a neural network model for geographic location prediction.

    :param input_shape: Number of input features (SNPs).
    :type input_shape: int
    :param width: Width of the dense layers, defaults to 256.
    :type width: int, optional
    :param n_layers: Total number of dense layers (excluding final layers),
        defaults to 8.
    :type n_layers: int, optional
    :param dropout_prop: Dropout proportion for middle dropout layer,
        defaults to 0.25.
    :type dropout_prop: float, optional
    :param pca_components: If set, prepend a linear projection layer named
        "pca_projection" of this width as the first layer. The caller is
        responsible for initializing its weights with PCA loadings. Defaults
        to None (no projection layer).
    :type pca_components: int, optional
    :param optimizer_config: Configuration for the optimizer. Should be a dict containing keys:
        "algo" (str): "adam" or "adamw";
        "learning_rate" (float);
        "weight_decay" (float, only used for "adamw").
        Defaults to None (uses Adam with default settings).

    :type optimizer_config: dict, optional
    :param loss_fn: Loss function to use. If None, defaults to
        euclidean_distance_loss, defaults to None.
    :type loss_fn: callable, optional
    :return: Compiled Keras model ready for training.
    :rtype: keras.Model

    Example:
        >>> model = create_network(input_shape=1000)
        >>> model.summary()
    """
    # Create input layer explicitly
    inputs = keras.Input(shape=(input_shape,))

    # Optional PCA-initialized linear projection as the first layer. Placed
    # before BatchNormalization so it sees raw genotype counts, which lets a
    # caller set its weights to PCA loadings and reproduce PCA scores exactly.
    # Pinned to float32: it computes raw_counts @ loadings + bias, where the
    # two terms are large and nearly cancel, so float16 loses the result.
    if pca_components is not None:
        x = layers.Dense(
            pca_components,
            activation="linear",
            name=PCA_LAYER_NAME,
            dtype="float32",
        )(inputs)
        # Gradient gate: switches the projection between frozen and fine-tuning
        # by flipping a variable, so the training graph never changes.
        x = GradientGate(name=PCA_GATE_NAME, dtype="float32")(x)
        x = layers.BatchNormalization()(x)
    else:
        x = layers.BatchNormalization()(inputs)

    # First half of layers
    for i in range(int(np.floor(n_layers / 2))):
        x = layers.Dense(width, activation="elu")(x)

    # Middle dropout layer
    x = layers.Dropout(dropout_prop)(x)

    # Second half of layers
    for i in range(int(np.ceil(n_layers / 2))):
        x = layers.Dense(width, activation="elu")(x)

    # Two final coordinate prediction layers
    x = layers.Dense(2)(x)
    outputs = layers.Dense(2)(x)

    # Create model with explicit inputs/outputs
    model = keras.Model(inputs=inputs, outputs=outputs, name="locator_network")

    # Configure optimizer
    if optimizer_config is None:
        optimizer = "Adam"
    else:
        optimizer = build_optimizer(
            optimizer_config["algo"],
            optimizer_config["learning_rate"],
            optimizer_config.get("weight_decay", 0.004),
        )

    # Use provided loss function if available; else default to euclidean_distance_loss
    if loss_fn is None:
        loss_fn = euclidean_distance_loss

    # Compile model with configured optimizer and loss
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model


class IndexedGenotypeModel(keras.Model):
    """Model that gathers genotypes from a GPU-resident table by sample index.

    The genotype matrix for realistic runs is small enough to live on the GPU
    for the entire run (n_snps x n_samples x dtype bytes; sub-GB as int8). This
    wrapper holds the whole matrix as a GPU-resident, sample-major tensor and
    makes the model's input a vector of sample indices instead of a genotype
    batch. Each training step the only host-to-device traffic is the index
    vector; the row gather, dtype cast, and optional augmentation all run on the
    GPU, and ``inner`` -- the ordinary coordinate-prediction network -- sees the
    same ``(batch, n_snps)`` float features it always has.

    The genotype table is held as a plain tensor, never a tracked weight, so it
    stays out of ``get_weights()`` and checkpoints. ``save_weights`` /
    ``load_weights`` / ``get_layer`` delegate to ``inner`` so the on-disk weight
    format and PCA-layer wiring are identical to a bare network.

    Parameters
    ----------
    inner : keras.Model
        Coordinate-prediction network from ``create_network``; consumes
        ``(batch, n_snps)`` genotype features.
    genotype_table : tf.Tensor
        Sample-major genotype matrix, shape ``(n_samples, n_snps)``, native
        dtype (int8 hard calls or float32 dosage).
    site_order : array-like or None
        Optional SNP resampling order (bootstrap/jacknife), applied as a
        per-batch column gather after the row gather.
    augment : dict or None
        Optional augmentation config with ``enabled`` and ``flip_rate`` keys;
        genotype flipping is applied only during training.
    """

    def __init__(self, inner, genotype_table, site_order=None, augment=None):
        super().__init__(name="indexed_genotype_model")
        self.inner = inner
        # Deliberately a plain tensor, not a tf.Variable: a Variable attribute
        # would be tracked by Keras and copied on every get_weights() call
        # (e.g. by EarlyStopping(restore_best_weights=True)). A constant tensor
        # is captured by reference into the traced call and stays GPU-resident.
        self._table = genotype_table
        self._site_order = (
            tf.constant(np.asarray(site_order), dtype=tf.int32)
            if site_order is not None
            else None
        )
        self._augment = augment or {}
        # Imported here rather than at module scope to keep the models module
        # free of any import-time dependency on the data subpackage.
        from .data.tf_dataset import flip_genotypes_tf

        self._flip_fn = flip_genotypes_tf

    @property
    def genotype_table(self):
        """The GPU-resident genotype table this model gathers from.

        The compiled ``call`` captures this tensor by reference, so a model
        may only be reused while its table is unchanged.
        """
        return self._table

    def call(self, idx, training=False):
        """Gather genotypes for a batch of sample indices and predict."""
        idx = tf.cast(idx, tf.int32)
        g = tf.gather(self._table, idx, axis=0)  # (batch, n_snps), on GPU
        if self._site_order is not None:
            g = tf.gather(g, self._site_order, axis=1)  # bootstrap SNP resample
        # float32 keeps the optional pca_projection layer (float32, raw counts)
        # exact; mixed-precision layers inside inner downcast internally.
        g = tf.cast(g, tf.float32)
        if training and self._augment.get("enabled", False):
            g = self._flip_fn(g, self._augment.get("flip_rate", 0.05))
        return self.inner(g, training=training)

    def get_layer(self, *args, **kwargs):
        """Delegate so lookups of inner layers (e.g. PCA_LAYER_NAME) resolve."""
        return self.inner.get_layer(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        """Persist only the inner network -- on-disk weight format is unchanged."""
        return self.inner.save_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        """Load weights into the inner network."""
        return self.inner.load_weights(*args, **kwargs)


__all__ = [
    "PCA_LAYER_NAME",
    "PCA_GATE_NAME",
    "GradientGate",
    "IndexedGenotypeModel",
    "build_optimizer",
    "create_network",
    "euclidean_distance_loss",
    "loss_with_range_penalty",
    "rasterize_species_range",
    "mask_lookup",
]
