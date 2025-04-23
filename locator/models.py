"""Neural network model definitions"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np


def euclidean_distance_loss(y_true, y_pred):
    """Custom loss function using Euclidean distance.

    Args:
        y_true: Tensor of true coordinates
        y_pred: Tensor of predicted coordinates

    Returns:
        Euclidean distance between true and predicted coordinates
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def create_network(input_shape, width=256, n_layers=8, dropout_prop=0.25, optimizer_config=None):
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

    # Compile model with configured optimizer and loss
    model.compile(optimizer=optimizer, loss=euclidean_distance_loss)

    return model


__all__ = [
    "create_network",
    "euclidean_distance_loss",  # Added to __all__ since it's a public function
]
