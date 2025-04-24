"""Neural network model definitions"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from shapely.geometry import Point
import geopandas as gpd
from rasterio.features import rasterize
from affine import Affine

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
        dtype="uint8"
    )

    return mask.astype(np.float32), transform

def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function using Euclidean distance.

    Args:
        y_true: Tensor of true coordinates
        y_pred: Tensor of predicted coordinates

    Returns:
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
    col = tf.clip_by_value(((lon - transform.c) / resolution), 0, mask_tensor.shape[1] - 1)
    row = tf.clip_by_value(((lat - transform.f) / resolution), 0, mask_tensor.shape[0] - 1)

    # Integer indices (nearest-neighbor lookup)
    col = tf.cast(tf.round(col), tf.int32)
    row = tf.cast(tf.round(row), tf.int32)

    # Gather values from mask
    idx = tf.stack([row, col], axis=-1)
    valid = tf.gather_nd(mask_tensor, idx)

    return valid

def loss_with_range_penalty(y_true, y_pred, mask_tensor, transform, resolution, penalty_weight=1.0):
    # Euclidean distance
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

    # GPU-friendly range mask
    valid_mask = mask_lookup(y_pred, mask_tensor, transform, resolution)

    # Penalize out-of-range predictions
    penalty = tf.square(1.0 - valid_mask)
    #tf.print("Euclidean:", euclidean, summarize=10)
   # tf.print("Penalty:", penalty, summarize=10)
   # tf.print("Valid mask:", valid_mask, summarize=10)
    return euclidean + penalty_weight * penalty

def create_network(input_shape, width=256, n_layers=8, dropout_prop=0.25, optimizer_config=None, loss_fn=None):
    """Create a neural network model for geographic location prediction.

    Args:
        input_shape (int): Number of input features (SNPs)
        width (int, optional): Width of the dense layers. Defaults to 256.
        n_layers (int, optional): Total number of dense layers (excluding final layers).
            Defaults to 8.
        dropout_prop (float, optional): Dropout proportion for middle dropout layer.
            Defaults to 0.25.
        optimizer_config (dict, optional): Configuration for the optimizer.
            Should contain:
                - algo: str, "adam" or "adamw"
                - learning_rate: float
                - weight_decay: float (only for adamw)
            Defaults to None (uses Adam with default settings).
        loss_fn (callable, optional): Loss function to use. If None, defaults to euclidean_distance_loss.

    Returns:
        keras.Model: Compiled Keras model ready for training

    Example:
        >>> model = create_network(input_shape=1000)
        >>> model.summary()
    """
    # Create input layer explicitly
    inputs = keras.Input(shape=(input_shape,))

    # Batch normalization on input
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
        if optimizer_config["algo"].lower() == "adam":
            optimizer = keras.optimizers.Adam(learning_rate=optimizer_config["learning_rate"])
        elif optimizer_config["algo"].lower() == "adamw":
            optimizer = keras.optimizers.AdamW(
                learning_rate=optimizer_config["learning_rate"],
                weight_decay=optimizer_config["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config['algo']}")

    # Use provided loss function if available; else default to euclidean_distance_loss
    if loss_fn is None:
        loss_fn = euclidean_distance_loss

    # Compile model with configured optimizer and loss
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model

__all__ = [
    "create_network",
    "euclidean_distance_loss",
    "loss_with_range_penalty",
    "rasterize_species_range",
    "mask_lookup",
]
