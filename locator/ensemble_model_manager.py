"""Model manager for ensemble predictions."""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tensorflow import keras

from .data import NormalizationParams


class EnsembleModelManager:
    """Manages multiple models for ensemble predictions.

    This class handles:
    - Saving and loading ensemble models with metadata
    - Lazy loading of model weights
    - Efficient storage of normalization parameters
    - Model versioning and validation
    """

    def __init__(self, base_path: str):
        """Initialize model manager.

        Args:
            base_path: Base path for saving/loading models
        """
        self.base_path = base_path
        self.models_info = []
        self.loaded_models = {}  # Cache for loaded models

    def save_ensemble(
        self, models_info: List[Dict], ensemble_metadata: Optional[Dict] = None
    ) -> None:
        """Save ensemble models and metadata.

        Args:
            models_info: List of model info dictionaries from training
            ensemble_metadata: Optional metadata about the ensemble
        """
        # Create ensemble directory
        os.makedirs(self.base_path, exist_ok=True)

        # Save ensemble metadata
        metadata = {
            "n_models": len(models_info),
            "ensemble_version": "1.0",
            "models": [],
        }

        if ensemble_metadata:
            metadata.update(ensemble_metadata)

        # Save each model
        for i, model_info in enumerate(models_info):
            model_path = os.path.join(self.base_path, f"model_fold_{i}.weights.h5")

            # Save model weights
            model = model_info["model"]
            model.save_weights(model_path)

            # Prepare model metadata
            model_meta = {
                "fold": model_info["fold"],
                "model_path": f"model_fold_{i}.weights.h5",
                "norm_params": model_info["norm_params"],
                "train_indices_count": len(model_info["train_indices"]),
                "val_indices_count": len(model_info["val_indices"]),
            }

            # Add history if available
            if "history" in model_info and model_info["history"]:
                history = model_info["history"]
                # Check if history has the expected keys
                if hasattr(history, "history") and isinstance(history.history, dict):
                    if "loss" in history.history and len(history.history["loss"]) > 0:
                        model_meta["final_loss"] = float(history.history["loss"][-1])
                    if (
                        "val_loss" in history.history
                        and len(history.history["val_loss"]) > 0
                    ):
                        model_meta["final_val_loss"] = float(
                            history.history["val_loss"][-1]
                        )
                    if "loss" in history.history:
                        model_meta["epochs_trained"] = len(history.history["loss"])

            metadata["models"].append(model_meta)

        # Save metadata
        metadata_path = os.path.join(self.base_path, "ensemble_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved ensemble with {len(models_info)} models to {self.base_path}")

    def load_ensemble(self, model_builder_fn=None) -> List[Dict]:
        """Load ensemble models and metadata.

        Args:
            model_builder_fn: Function to build model architecture

        Returns:
            List of model info dictionaries
        """
        # Load metadata
        metadata_path = os.path.join(self.base_path, "ensemble_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Load model info
        self.models_info = []
        for model_meta in metadata["models"]:
            model_info = {
                "fold": model_meta["fold"],
                "norm_params": model_meta["norm_params"],
                "model_path": os.path.join(self.base_path, model_meta["model_path"]),
                "model": None,  # Lazy loading
                "model_builder_fn": model_builder_fn,
            }
            self.models_info.append(model_info)

        print(f"Loaded ensemble metadata for {len(self.models_info)} models")
        return self.models_info

    def get_model(self, fold: int, n_features: int) -> keras.Model:
        """Get a specific model, loading if necessary.

        Args:
            fold: Fold index
            n_features: Number of features for model construction

        Returns:
            Loaded model
        """
        if fold in self.loaded_models:
            return self.loaded_models[fold]

        # Find model info
        model_info = self.models_info[fold]

        # Build model architecture
        if model_info["model_builder_fn"] is None:
            raise ValueError("Model builder function required for lazy loading")

        model = model_info["model_builder_fn"](n_features)

        # Load weights
        model.load_weights(model_info["model_path"])

        # Cache the model
        self.loaded_models[fold] = model

        return model

    def get_normalization_params(self, fold: int) -> NormalizationParams:
        """Get normalization parameters for a specific fold.

        Args:
            fold: Fold index

        Returns:
            NormalizationParams instance
        """
        params = self.models_info[fold]["norm_params"]
        return NormalizationParams(
            meanlong=params["meanlong"],
            sdlong=params["sdlong"],
            meanlat=params["meanlat"],
            sdlat=params["sdlat"],
        )

    def get_averaged_normalization_params(self) -> NormalizationParams:
        """Get averaged normalization parameters across all folds.

        Returns:
            Averaged NormalizationParams
        """
        all_params = [info["norm_params"] for info in self.models_info]

        avg_params = {
            "meanlong": np.mean([p["meanlong"] for p in all_params]),
            "sdlong": np.mean([p["sdlong"] for p in all_params]),
            "meanlat": np.mean([p["meanlat"] for p in all_params]),
            "sdlat": np.mean([p["sdlat"] for p in all_params]),
        }

        return NormalizationParams(**avg_params)

    def save_predictions(
        self, predictions: pd.DataFrame, prediction_type: str = "ensemble"
    ) -> None:
        """Save predictions to disk.

        Args:
            predictions: DataFrame with predictions
            prediction_type: Type of predictions (e.g., "ensemble", "fold_0")
        """
        pred_path = os.path.join(self.base_path, f"predictions_{prediction_type}.csv")
        predictions.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    def clear_cache(self) -> None:
        """Clear loaded models from memory."""
        self.loaded_models.clear()
        keras.backend.clear_session()
