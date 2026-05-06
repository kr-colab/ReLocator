"""Data loading functionality for locator"""

import warnings

import allel
import numpy as np
import pandas as pd
import zarr


def _counts_to_genotype_array(gmat):
    """Convert a genotype count matrix (0/1/2) to an allel.GenotypeArray.

    Uses vectorized numpy operations instead of Python loops.

    Args:
        gmat: numpy array of shape (n_samples, n_snps) with values 0, 1, or 2

    Returns
    -------
        allel.GenotypeArray of shape (n_snps, n_samples, 2)
    """
    h1 = np.minimum(gmat, 1).astype(np.int8)
    h2 = np.clip(gmat - 1, 0, 1).astype(np.int8)
    hmat = np.empty((gmat.shape[0] * 2, gmat.shape[1]), dtype=np.int8)
    hmat[0::2] = h1
    hmat[1::2] = h2
    return allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)


class DataLoaderMixin:
    """Mixin class providing data loading functionality for Locator."""

    def _report_variant_metadata(self):
        """Print summary of loaded positions and chromosomes."""
        if self.positions is not None:
            print(f"Loaded {len(self.positions)} SNP positions for window analysis")
        if self.chromosomes is not None:
            unique_chroms = np.unique(self.chromosomes)
            if len(unique_chroms) > 5:
                print(f"Found {len(unique_chroms)} chromosomes: {unique_chroms[:5]}...")
            else:
                print(f"Found chromosomes: {unique_chroms}")

    def _load_from_zarr(self, zarr_path):
        """Load genotypes from zarr file.

        Supports both scikit-allel format (calldata/GT, samples) and
        VCF Zarr / bio2zarr format (call_genotype, sample_id).

        Args:
            zarr_path: Path to zarr file containing genotype data

        Returns
        -------
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        print("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")

        if "call_genotype" in callset:
            # bio2zarr / VCF Zarr format
            genotypes = allel.GenotypeArray(callset["call_genotype"][:])
            samples = np.array([str(x) for x in callset["sample_id"][:]])
            if "variant_position" in callset:
                self.positions = np.array(callset["variant_position"][:])
            if "variant_contig" in callset and "contig_id" in callset:
                contig_ids = np.array([str(x) for x in callset["contig_id"][:]])
                contig_idx = np.array(callset["variant_contig"][:])
                self.chromosomes = contig_ids[contig_idx]
        elif "calldata/GT" in callset:
            # scikit-allel format
            genotypes = allel.GenotypeArray(callset["calldata/GT"][:])
            samples = callset["samples"][:]
            if "variants/POS" in callset:
                self.positions = callset["variants/POS"][:]
            if "variants/CHROM" in callset:
                self.chromosomes = callset["variants/CHROM"][:]
        else:
            raise ValueError(
                f"Unrecognized zarr format in {zarr_path}. "
                f"Expected 'call_genotype' (bio2zarr) or 'calldata/GT' "
                f"(scikit-allel)."
            )

        self._report_variant_metadata()
        return genotypes, samples

    def _load_from_vcf(self, vcf_path):
        """Load genotypes from VCF file.

        Args:
            vcf_path: Path to VCF file containing genotype data

        Returns
        -------
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs

        Raises
        ------
            ValueError: If VCF file cannot be read
        """
        print("reading VCF")
        vcf = allel.read_vcf(vcf_path, fields=["samples", "GT", "POS", "CHROM"])
        if vcf is None:
            raise ValueError(f"Could not read VCF file: {vcf_path}")
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]

        if "variants/POS" in vcf:
            self.positions = vcf["variants/POS"]
        if "variants/CHROM" in vcf:
            self.chromosomes = vcf["variants/CHROM"]

        self._report_variant_metadata()
        return genotypes, samples

    def _load_from_matrix(self, matrix_path):
        """Load genotypes from matrix file.

        Two input dialects are accepted, distinguished by dtype:

        - **Hard-call dosage** (integer 0/1/2): routed through
          ``_counts_to_genotype_array`` and returned as an ``allel.GenotypeArray``
          of shape ``(n_sites, n_samples, 2)``. Original behavior.
        - **Continuous dosage** (float column with values in [0, 2], e.g.
          expected dosage from GL-based callers): the matrix is returned
          directly as a 2D ``np.ndarray`` of shape ``(n_sites, n_samples)``
          with no allel.GenotypeArray round trip. Downstream
          ``_filter_genotypes`` in training.py recognizes this branch and
          applies MAC/max_snps filters on the continuous values directly,
          skipping biallelic checks (which are not meaningful for continuous
          dosage). NaN values are silently dropped at the MAC filter —
          callers should impute upstream (gl_to_locator.py site-mean fill
          handles this for ANGSD beagle inputs).

        Args:
            matrix_path: Path to tab-delimited matrix file containing genotype data.
                File should have a header row with 'sampleID' as first column,
                followed by variant columns.

        Returns
        -------
            tuple: (genotypes, samples)
        """
        gmat = pd.read_csv(matrix_path, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)

        if np.issubdtype(gmat.values.dtype, np.floating):
            # Continuous dosage path. Shape becomes (n_sites, n_samples) to
            # match the downstream ``ac`` representation produced by
            # ``filter_snps`` for the integer path.
            dosage = np.asarray(gmat.values, dtype=np.float32).T
            return dosage, samples

        if not np.all(np.isin(gmat, [0, 1, 2])):
            raise ValueError("Genotype values must be 0, 1, or 2")
        gmat = np.array(gmat, dtype="int8")

        genotypes = _counts_to_genotype_array(gmat)
        return genotypes, samples

    def load_genotypes(self, vcf=None, zarr=None, matrix=None):  # noqa: C901
        """Load genotype data from various input sources.

        This method can load genotype data from:
        1. A stored DataFrame provided during initialization
        2. A VCF file
        3. A zarr file (scikit-allel or bio2zarr format)
        4. A tab-delimited matrix file

        For windowed analysis, SNP positions must be available either from:
        - Column names in the genotype DataFrame
        - The zarr file's variants/POS array
        - The VCF file's POS field (automatically loaded)

        Args:
            vcf (str, optional): Path to VCF format genotype data
            zarr (str, optional): Path to zarr format genotype data
            matrix (str, optional): Path to tab-delimited matrix file

        Returns
        -------
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray of shape (n_sites, n_samples, 2)
                - samples is a numpy array of sample IDs

        Examples
        --------
            >>> # Using stored DataFrame from initialization
            >>> locator = Locator({
            ...     "genotype_data": geno_df,  # DataFrame with genotypes
            ...     "sample_data": coords_df   # DataFrame with coordinates
            ... })
            >>> genotypes, samples = locator.load_genotypes()

            >>> # Using zarr file (recommended for windowed analysis)
            >>> locator = Locator({"sample_data": coords_df})
            >>> genotypes, samples = locator.load_genotypes(zarr="path/to/geno.zarr")

            >>> # Using VCF file
            >>> genotypes, samples = locator.load_genotypes(vcf="path/to/geno.vcf")

            >>> # Using matrix file
            >>> genotypes, samples = locator.load_genotypes(matrix="path/to/geno.txt")

        Raises
        ------
            ValueError: If no input source is provided or if input format is invalid
        """
        # First load sample data if not already loaded
        if not hasattr(self, "_sample_data_df") and "sample_data" in self.config:
            sample_df = pd.read_csv(self.config["sample_data"], sep="\t")
            required_cols = ["sampleID", "x", "y"]
            if not all(col in sample_df.columns for col in required_cols):
                raise ValueError(f"sample_data must contain columns: {required_cols}")
            self._sample_data_df = sample_df

        # Use stored DataFrame if available
        if hasattr(self, "_genotype_df"):
            print("using stored genotype DataFrame")
            geno_df = self._genotype_df
            samples = np.array([str(x) for x in geno_df.index], dtype=object)
            if self.positions is None:
                try:
                    self.positions = geno_df.columns.astype(float).values
                except ValueError:
                    raise ValueError(
                        "Column names must be convertible to integers (SNP positions)"
                    )

            genotypes = np.zeros((geno_df.shape[1], geno_df.shape[0], 2), dtype=int)

            for i, count in enumerate([0, 1, 2]):
                mask = count == geno_df.values.T
                if count == 0:
                    continue
                elif count == 1:
                    genotypes[mask, 0] = 1
                else:
                    genotypes[mask] = 1

            return allel.GenotypeArray(genotypes), samples

        elif zarr is not None:
            return self._load_from_zarr(zarr)

        elif vcf is not None:
            return self._load_from_vcf(vcf)

        elif matrix is not None:
            return self._load_from_matrix(matrix)

        else:
            raise ValueError(
                "No genotype data provided. Either initialize with genotype_data DataFrame "
                "or provide vcf/zarr/matrix path."
            )

    def sort_samples(self, samples=None, sample_data_file=None, reorder=True):  # noqa: C901
        """Sort samples and match with location data.

        Matches samples with their location data and ensures consistent ordering
        between genotype and location data.

        Args:
            samples (numpy.ndarray): Array of sample IDs from the genotype data
            sample_data_file (str, optional): Override path to tab-delimited file with
                columns 'sampleID', 'x', 'y'. If not provided, uses stored sample data.
            reorder (bool): If True, automatically reorder metadata to match genotype order.
                If False, raise error on order mismatch (default: True)

        Returns
        -------
            tuple: (sample_data DataFrame, locs array of shape (n_samples, 2))
        """
        if samples is None:
            raise ValueError("samples must be provided")

        if hasattr(self, "_sample_data_df"):
            sample_data = self._sample_data_df.copy()
        else:
            sample_data_path = sample_data_file or self.config.get("sample_data")
            if not sample_data_path:
                raise ValueError("sample_data must be provided in config or as argument")
            sample_data = pd.read_csv(sample_data_path, sep="\t")

        if "sampleID" not in sample_data.columns:
            raise ValueError("sample_data must contain 'sampleID' column")

        sample_data["sampleID"] = sample_data["sampleID"].astype(str)
        samples_str = [str(s) for s in samples]

        if len(sample_data) != len(samples):
            if reorder:
                print(
                    f"Sample count mismatch: {len(samples)} in genotypes, "
                    f"{len(sample_data)} in metadata"
                )
            else:
                raise ValueError(
                    f"Sample count mismatch: genotypes has {len(samples)} samples "
                    f"but metadata has {len(sample_data)}. "
                    f"Set reorder=True to handle this automatically."
                )

        min_samples = min(len(sample_data), len(samples))
        order_matches = len(sample_data) == len(samples) and all(
            sample_data["sampleID"].iloc[x] == samples_str[x] for x in range(min_samples)
        )

        if not order_matches:
            if not reorder:
                raise ValueError(
                    "Sample ordering mismatch. Set reorder=True to "
                    "automatically reorder metadata to match genotype order."
                )

            sample_order_df = pd.DataFrame(
                {"sampleID": samples_str, "geno_order": range(len(samples_str))}
            )
            reordered_data = sample_order_df.merge(
                sample_data, on="sampleID", how="left"
            )

            missing_in_meta = reordered_data[["x", "y"]].isna().any(axis=1).sum()
            if missing_in_meta > 0:
                missing_ids = reordered_data[reordered_data["x"].isna()][
                    "sampleID"
                ].tolist()
                warnings.warn(
                    f"{missing_in_meta} samples in genotypes have no metadata. "
                    f"First 10 missing: {missing_ids[:10]}"
                )
                if missing_in_meta == len(reordered_data):
                    raise ValueError(
                        "No samples from genotypes found in metadata! "
                        "Check that sample IDs match between files."
                    )

            samples_set = set(samples_str)
            extra_in_meta = sample_data[~sample_data["sampleID"].isin(samples_set)]
            if len(extra_in_meta) > 0:
                warnings.warn(
                    f"{len(extra_in_meta)} samples in metadata are not in genotypes. "
                    f"First 10 extra: {extra_in_meta['sampleID'].tolist()[:10]}"
                )

            sample_data = reordered_data.sort_values("geno_order").drop(
                "geno_order", axis=1
            )

            print("Reordered metadata to match genotype sample order.")
            print(f"Total samples in genotypes: {len(samples)}")
            print(f"Samples with coordinates: {len(samples) - missing_in_meta}")
            if missing_in_meta > 0:
                print(f"Samples without coordinates (NA): {missing_in_meta}")
                print(
                    f"Note: K-fold CV will only use the "
                    f"{len(samples) - missing_in_meta} samples with known locations"
                )

        locs = np.array(sample_data[["x", "y"]])
        return sample_data, locs
