"""Ensemble functionality for locator - Legacy compatibility wrapper"""

import warnings

import numpy as np
import pandas as pd

from .core import Locator


class EnsembleLocator(Locator):
    """Legacy compatibility wrapper for ensemble functionality.

    This class provides backward compatibility for code using the old EnsembleLocator API.
    New code should use Locator's ensemble methods directly:
    - locator.train_ensemble()
    - locator.predict_ensemble()

    The old API is preserved but now uses the modern implementation internally.
    """

    def __init__(self, base_config=None, k_folds=5, training_set_indices=None):
        """Initialize EnsembleLocator with configuration parameters.

        Args:
            base_config (dict, optional): Base configuration shared by all models.
                Each model will get a copy of this config.
            k_folds (int, optional): Number of folds for cross-validation.
                Defaults to 5.
            training_set_indices (array-like, optional): Indices of samples to use for
                training and validation. If provided, only these samples will be used
                to create k-folds, while others will automatically be assigned to
                prediction set.
        """
        warnings.warn(
            "EnsembleLocator is deprecated. Use Locator with ensemble methods instead:\n"
            "  locator = Locator(config)\n"
            "  locator.train_ensemble(genotypes, samples, k=5)\n"
            "  predictions = locator.predict_ensemble()",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize parent Locator class
        super().__init__(base_config)

        # Store ensemble-specific parameters
        self.k_folds = k_folds
        self.training_set_indices = (
            np.array(training_set_indices) if training_set_indices is not None else None
        )

        # Legacy attributes for compatibility
        self.models = []
        self.fold_indices = {}

    def create_folds(self, genotypes, samples, locations, training_set_indices=None):
        """Create k-fold splits of the data.

        Legacy method - internally uses create_ensemble_folds().
        """
        # Store samples for later use
        self.samples = samples

        # Use the new implementation
        fold_info = self.create_ensemble_folds(
            genotypes=genotypes,
            samples=samples,
            k=self.k_folds,
            training_set_indices=training_set_indices or self.training_set_indices,
            na_action=self.config.get("na_action", "separate"),
        )

        # Extract legacy format fold_indices
        self.fold_indices = fold_info["fold_indices"]
        self._fold_info = fold_info

        return self.fold_indices

    def train(self, genotypes, samples, sample_data_file=None):
        """Train k models on different folds of the data.

        Legacy method - internally uses train_ensemble().
        """
        # Store samples
        self.samples = samples

        # Call the new implementation
        result = self.train_ensemble(
            genotypes=genotypes,
            samples=samples,
            k=self.k_folds,
            training_set_indices=self.training_set_indices,
            na_action=self.config.get("na_action", "separate"),
            augment_data=self.config.get("augmentation", {}).get("enabled", False),
            flip_rate=self.config.get("augmentation", {}).get("flip_rate", 0.05),
            save_fold_models=True,
            verbose=self.config.get("keras_verbose", 1) >= 1,
        )

        # Store results in legacy format
        self.models = result["models"]
        self.fold_indices = result["fold_info"]["fold_indices"]

        # Extract histories for return
        histories = result["histories"]

        # Set averaged normalization parameters
        self.meanlong = result["normalization_params"]["meanlong"]
        self.sdlong = result["normalization_params"]["sdlong"]
        self.meanlat = result["normalization_params"]["meanlat"]
        self.sdlat = result["normalization_params"]["sdlat"]

        return histories

    def predict(
        self, return_df=True, save_preds_to_disk=True, include_val_predictions=True
    ):
        """Make predictions using the ensemble of models.

        Legacy method - internally uses predict_ensemble().
        """
        if not hasattr(self, "models") or not self.models:
            raise ValueError("No trained models in ensemble")

        # For validation predictions, we need to collect them differently
        if include_val_predictions:
            # Get predictions on all samples
            all_predictions = self.predict_ensemble(
                genotypes=self._get_stored_genotypes(),
                samples=self.samples,
                indices=None,  # Predict on all samples
                include_fold_predictions=False,
                return_std=False,
                return_df=True,
                save_predictions=False,
            )

            # Also get validation predictions from each fold
            val_predictions = {}
            for i, model_info in enumerate(self.models):
                val_idx = model_info["val_indices"]
                # Get predictions for validation samples from the all_predictions
                for idx in val_idx:
                    sample_id = self.samples[idx]
                    val_predictions[sample_id] = all_predictions[
                        all_predictions["sampleID"] == sample_id
                    ][["x", "y"]].values[0]

            # Create validation DataFrame
            if val_predictions:
                val_df = pd.DataFrame.from_dict(
                    val_predictions, orient="index", columns=["x", "y"]
                )
                val_df.index.name = "sampleID"
                val_df.reset_index(inplace=True)

                # For legacy compatibility, rename columns in all_predictions
                all_predictions = all_predictions.rename(
                    columns={"x": "x_pred", "y": "y_pred"}
                )

                # Combine (but this might have duplicates, so handle carefully)
                # For now, just return all_predictions as the main output
                result_df = all_predictions
            else:
                result_df = all_predictions

        else:
            # Only prediction set results
            # Get NA samples
            status = self.get_sample_status(self.samples)
            pred_indices = status["na_indices"]

            result_df = self.predict_ensemble(
                genotypes=self._get_stored_genotypes(),
                samples=self.samples,
                indices=pred_indices if len(pred_indices) > 0 else None,
                include_fold_predictions=False,
                return_std=False,
                return_df=True,
                save_predictions=False,
            )

            # Rename columns for legacy compatibility
            result_df = result_df.rename(columns={"x": "x_pred", "y": "y_pred"})

        # Save if requested
        if save_preds_to_disk:
            filename = (
                f"{self.config['out']}_ensemble_predlocs.csv"
                if include_val_predictions
                else f"{self.config['out']}_ensemble_predlocs_pred_only.csv"
            )
            result_df.to_csv(filename, index=False)

        if return_df:
            return result_df

        # Return just coordinates
        return result_df[["x_pred", "y_pred"]].values

    def _get_stored_genotypes(self):
        """Get the stored genotype data."""
        # This is a helper to access genotypes that were used during training
        if hasattr(self, "_ensemble_genotypes"):
            return self._ensemble_genotypes
        else:
            # Try to load from the original source
            # This is a simplified version - in practice would need proper loading
            raise ValueError(
                "Genotype data not available. For predict(), genotypes must be "
                "stored during train() or provided explicitly."
            )

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        html = [
            "<div style='font-family: monospace'>",
            "<h3>EnsembleLocator (Legacy API)</h3>",
            "<p style='color: orange'>⚠️ This class is deprecated. Use Locator.train_ensemble() instead.</p>",
            "<table>",
            "<tr><th style='text-align:left; padding:5px'>Configuration</th><th style='text-align:left; padding:5px'>Value</th></tr>",
        ]

        # Add key configuration parameters
        key_params = [
            "k_folds",
            "train_split",
            "batch_size",
            "min_mac",
            "max_SNPs",
            "width",
            "nlayers",
            "dropout_prop",
            "max_epochs",
            "optimizer_algo",
            "learning_rate",
            "weight_decay",
        ]

        # Add k_folds
        html.append(
            f"<tr><td style='padding:5px'>k_folds</td>"
            f"<td style='padding:5px'>{self.k_folds}</td></tr>"
        )

        for param in key_params[1:]:  # Skip k_folds since we already added it
            if param in self.config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.config[param]}</td></tr>"
                )

        html.append("</table>")

        # Add status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        if hasattr(self, "models") and self.models:
            html.append(f"<li>Models trained: {len(self.models)} models ✓</li>")

            # Add fold information
            if self.fold_indices:
                html.append("<li>Fold splits:")
                html.append("<ul>")
                for fold, indices in self.fold_indices.items():
                    html.append(
                        f"<li>Fold {fold}: {len(indices['train'])} train, "
                        f"{len(indices['val'])} val, {len(indices['pred'])} pred</li>"
                    )
                html.append("</ul></li>")
        else:
            html.append("<li>Models: Not trained</li>")

        # Normalization status
        if all(
            hasattr(self, attr) and getattr(self, attr) is not None
            for attr in ["meanlong", "sdlong", "meanlat", "sdlat"]
        ):
            html.append("<li>Location normalization: Computed (averaged) ✓</li>")
        else:
            html.append("<li>Location normalization: Not computed</li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)
