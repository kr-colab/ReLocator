"""Helper methods for analysis functionality."""

import numpy as np
import zarr


class HelpersMixin:
    """Mixin providing helper methods used by analysis methods."""

    def _ensure_positions(self):
        """Load SNP positions from VCF, zarr, or DataFrame if not already stored.

        Sets self.positions (and self.chromosomes when available from VCF).

        Raises
        ------
            ValueError: If no source of SNP positions is available.
        """
        if hasattr(self, "positions") and self.positions is not None:
            return

        if hasattr(self, "_genotype_df"):
            # Use positions from DataFrame columns
            self.positions = np.array(self._genotype_df.columns, dtype=int)
        elif self.config.get("zarr"):
            # Get positions from zarr file
            callset = zarr.open_group(self.config["zarr"], mode="r")
            self.positions = callset["variants/POS"][:]
        elif self.config.get("vcf"):
            # Re-read VCF to get positions and chromosomes
            print("Loading SNP positions from VCF...")
            import allel

            vcf = allel.read_vcf(self.config["vcf"], fields=["POS", "CHROM"])
            if vcf is not None and "variants/POS" in vcf:
                self.positions = vcf["variants/POS"]
                if "variants/CHROM" in vcf:
                    self.chromosomes = vcf["variants/CHROM"]
                print(f"Loaded {len(self.positions)} SNP positions")
            else:
                raise ValueError(
                    f"Could not load positions from VCF: {self.config['vcf']}"
                )
        else:
            raise ValueError(
                "SNP positions required for windowed analysis. Use VCF, zarr input or "
                "genotype DataFrame with position-labeled columns."
            )

        # Final check
        if not hasattr(self, "positions") or self.positions is None:
            raise ValueError(
                "SNP positions required for windowed analysis. Use zarr input or "
                "genotype DataFrame with position-labeled columns."
            )

    def _validate_na_action(self, samples, na_action, analysis_name):
        """Resolve NA action default, check sample status, print summary, and enforce fail mode.

        Args:
            samples: Array of sample IDs
            na_action: How to handle NA samples, or None to use instance default
            analysis_name: Human-readable name for log messages (e.g. "Window analysis")

        Returns
        -------
            tuple: (na_action, status) where na_action is the resolved string and
                status is the dict returned by get_sample_status.

        Raises
        ------
            ValueError: If na_action is 'fail' and NA samples are present.
        """
        if na_action is None:
            na_action = self.na_action

        status = self.get_sample_status(samples)

        print(
            f"{analysis_name}: {status['n_known']} samples with coordinates, "
            f"{status['n_na']} without"
        )
        if status["n_na"] > 0:
            print(f"NA handling mode: {na_action}")

        if na_action == "fail" and status["n_na"] > 0:
            raise ValueError(
                f"Found {status['n_na']} samples without coordinates. "
                f"Set na_action='separate' or 'exclude' to proceed."
            )

        return na_action, status
