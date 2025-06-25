"""Visualization functionality for locator"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64


class VisualizationMixin:
    """Mixin class providing visualization functionality for Locator."""
    
    def plot_history(self, history):
        """Plot training history and prediction error.

        Creates a figure with two subplots showing the validation loss and training loss
        over epochs. Saves the plot to a PDF file using the output prefix specified in config.

        Args:
            history: keras.callbacks.History object containing training history
        """
        if self.config.get("plot_history", False):
            plt.switch_backend("agg")
            fig = plt.figure(figsize=(4, 1.5), dpi=200)
            plt.rcParams.update({"font.size": 7})
            ax1 = fig.add_axes([0, 0, 0.4, 1])
            ax1.plot(history.history["val_loss"][3:], "-", color="black", lw=0.5)
            ax1.set_xlabel("Validation Loss")
            ax2 = fig.add_axes([0.55, 0, 0.4, 1])
            ax2.plot(history.history["loss"][3:], "-", color="black", lw=0.5)
            ax2.set_xlabel("Training Loss")
            fig.savefig(self.config["out"] + "_fitplot.pdf", bbox_inches="tight")

    def _repr_html_(self):
        """Return HTML representation of Locator instance for Jupyter notebooks."""
        html = [
            "<div style='font-family: monospace'>",
            "<h3>Locator Model</h3>",
            "<table>",
            "<tr><th style='text-align:left; padding:5px'>Configuration</th><th style='text-align:left; padding:5px'>Value</th></tr>",
        ]

        # Add key configuration parameters
        key_params = [
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
            "use_range_penalty",
            "species_range_shapefile",
            "resolution",
            "penalty_weight",
            "species_range_geom",
        ]

        for param in key_params:
            if param in self.config:
                html.append(
                    f"<tr><td style='padding:5px'>{param}</td>"
                    f"<td style='padding:5px'>{self.config[param]}</td></tr>"
                )
        # add weight samples to end, deal with weird dictionary thing
        if self.config.get("weight_samples", {}).get("enabled", False):
            html.append(
                    f"<tr><td style='padding:5px'>{'weight_samples'}</td>"
                    f"<td style='padding:5px'>{'True'}</td></tr>"
                ) 
            for k in ['method', 'xbins', 'ybins', 'lam', 'bandwidth']:
                if k in self.config['weight_samples'].keys():
                    if self.config['weight_samples'][k] is not None:
                        html.append(
                            f"<tr><td style='padding:5px'>{'weight_samples '+k}</td>"
                            f"<td style='padding:5px'>{self.config['weight_samples'][k]}</td></tr>"
                        )

        html.append("</table>")

        # Add model status
        html.append("<h4>Status:</h4>")
        html.append("<ul>")

        # Model trained status and training history
        if self.model is not None:
            html.append("<li>Model: Trained ✓</li>")
            if hasattr(self, "traingen"):
                html.append(f"<li>Training samples: {self.traingen.shape[0]}</li>")
                html.append(f"<li>Features: {self.traingen.shape[1]}</li>")

            # Add training history if available
            if hasattr(self, "history") and self.history is not None:
                # Create figure
                fig, ax = plt.subplots(figsize=(8, 4))
                hist = self.history.history

                # Plot training and validation loss
                ax.plot(hist["loss"], label="Training Loss", color="blue")
                axV = ax.twinx()
                axV.plot(hist["val_loss"], label="Validation Loss", color="orange")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Training Loss")
                axV.set_ylabel("Validation Loss")
                ax.legend()
                axV.legend(loc='upper center')

                # Get final validation loss
                final_val_loss = hist["val_loss"][-1]

                # Convert plot to base64 string
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight")
                buf.seek(0)
                plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                plt.close(fig)

                # Add plot and metrics to HTML
                html.append("</ul>")  # Close the status list
                html.append("<h4>Training History:</h4>")
                html.append(f"<p>Final validation loss: {final_val_loss:.4f}</p>")
                html.append(
                    f'<img src="data:image/png;base64,{plot_data}" style="max-width:100%">'
                )
                html.append("<ul>")  # Reopen list for remaining items
        else:
            html.append("<li>Model: Not trained</li>")

        # Location normalization status
        if all(
            x is not None
            for x in [self.meanlong, self.sdlong, self.meanlat, self.sdlat]
        ):
            html.append("<li>Location normalization: Computed ✓</li>")
        else:
            html.append("<li>Location normalization: Not computed</li>")

        # Sample data status
        if hasattr(self, "_sample_data_df"):
            html.append(
                f"<li>Sample data loaded: {len(self._sample_data_df)} samples</li>"
            )
        elif "sample_data" in self.config:
            html.append("<li>Sample data: Path provided</li>")
        else:
            html.append("<li>Sample data: Not provided</li>")

        # Genotype data status
        if hasattr(self, "_genotype_df"):
            html.append(
                f"<li>Genotype data loaded: {self._genotype_df.shape[1]} SNPs</li>"
            )
        elif any(x in self.config for x in ["zarr", "vcf", "genotype_data"]):
            html.append("<li>Genotype data: Path provided</li>")
        else:
            html.append("<li>Genotype data: Not provided</li>")

        if hasattr(self, "sample_weights"):
            html.append(
                f"<li>Samples weighted using {self.config['weight_samples'].get('method')}</li>"
            )

        # Add holdout information
        if hasattr(self, "holdout_idx") and self.samples is not None:
            n_holdout = len(self.holdout_idx)
            html.append(f"<li>Holdout samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.holdout_idx]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")
        elif hasattr(self, "pred_indices") and self.samples is not None:
            n_holdout = len(self.pred_indices)
            html.append(f"<li>Prediction samples: {n_holdout} samples held out</li>")
            # Add collapsible list of held out sample IDs
            if n_holdout > 0:
                sample_list = self.samples[self.pred_indices]
                html.append("<li>Held out samples: <details>")
                html.append("<summary>Click to show/hide</summary>")
                html.append("<ul style='max-height:200px;overflow-y:auto'>")
                for sample in sample_list:
                    html.append(f"<li>{sample}</li>")
                html.append("</ul></details></li>")

        html.append("</ul>")
        html.append("</div>")

        return "".join(html)