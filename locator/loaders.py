"""Data loading functionality for locator"""

import sys

import allel
import numpy as np
import pandas as pd
import zarr


class DataLoaderMixin:
    """Mixin class providing data loading functionality for Locator."""

    def _load_from_zarr(self, zarr_path):
        """Load genotypes from zarr file.

        Args:
            zarr_path: Path to zarr file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        print("reading zarr")
        callset = zarr.open_group(zarr_path, mode="r")
        gt = callset["calldata/GT"]
        genotypes = allel.GenotypeArray(gt[:])
        samples = callset["samples"][:]
        return genotypes, samples

    def _load_from_vcf(self, vcf_path):
        """Load genotypes from VCF file.

        Args:
            vcf_path: Path to VCF file containing genotype data

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs

        Raises:
            ValueError: If VCF file cannot be read
        """
        print("reading VCF")
        vcf = allel.read_vcf(vcf_path, fields=["GT", "POS", "CHROM"])
        if vcf is None:
            raise ValueError(f"Could not read VCF file: {vcf_path}")
        genotypes = allel.GenotypeArray(vcf["calldata/GT"])
        samples = vcf["samples"]

        # Store positions and chromosomes for window analysis
        if "variants/POS" in vcf:
            self.positions = vcf["variants/POS"]
            print(f"Loaded {len(self.positions)} SNP positions for window analysis")

        if "variants/CHROM" in vcf:
            self.chromosomes = vcf["variants/CHROM"]
            unique_chroms = np.unique(self.chromosomes)
            print(
                f"Found {len(unique_chroms)} chromosomes: {unique_chroms[:5]}..."
                if len(unique_chroms) > 5
                else f"Found chromosomes: {unique_chroms}"
            )

        return genotypes, samples

    def _load_from_matrix(self, matrix_path):
        """Load genotypes from matrix file.

        Args:
            matrix_path: Path to tab-delimited matrix file containing genotype data.
                File should have a header row with 'sampleID' as first column,
                followed by variant columns. Each row contains genotype counts (0,1,2)
                for one sample.

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray containing genetic data
                - samples is a numpy array of sample IDs
        """
        gmat = pd.read_csv(matrix_path, sep="\t")
        samples = np.array(gmat["sampleID"])
        gmat = gmat.drop(labels="sampleID", axis=1)
        if not np.all(np.isin(gmat, [0, 1, 2])):
            raise ValueError("Genotype values must be 0, 1, or 2")
        gmat = np.array(gmat, dtype="int8")

        # Convert to haplotype format
        hmat = None
        for i in range(gmat.shape[0]):
            h1 = []
            h2 = []
            for j in range(gmat.shape[1]):
                count = gmat[i, j]
                if count == 0:
                    h1.append(0)
                    h2.append(0)
                elif count == 1:
                    h1.append(1)
                    h2.append(0)
                elif count == 2:
                    h1.append(1)
                    h2.append(1)
            if i == 0:
                hmat = h1
                hmat = np.vstack((hmat, h2))
            else:
                hmat = np.vstack((hmat, h1))
                hmat = np.vstack((hmat, h2))

        genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
        return genotypes, samples

    def load_genotypes(self, vcf=None, zarr=None, matrix=None):  # noqa: C901
        """Load genotype data from various input sources.

        This method can load genotype data from:
        1. A stored DataFrame provided during initialization
        2. A VCF file
        3. A zarr file
        4. A tab-delimited matrix file

        For windowed analysis, SNP positions must be available either from:
        - Column names in the genotype DataFrame
        - The zarr file's variants/POS array
        - The VCF file's POS field (automatically loaded)

        Args:
            vcf (str, optional): Path to VCF format genotype data
            zarr (str, optional): Path to zarr format genotype data
            matrix (str, optional): Path to tab-delimited matrix file

        Returns:
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray of shape (n_sites, n_samples, 2)
                - samples is a numpy array of sample IDs

        Examples:
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

        Raises:
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
            # Convert samples to Python's native str type
            samples = np.array([str(x) for x in geno_df.index], dtype=object)
            # Store positions for windowed analysis if not already set
            if self.positions is None:
                try:
                    self.positions = geno_df.columns.astype(float).values
                except ValueError:
                    raise ValueError(
                        "Column names must be convertible to integers (SNP positions)"
                    )

            # Convert DataFrame values to genotype array format
            # Shape needs to be (n_sites, n_samples, 2) for compatibility
            genotypes = np.zeros((geno_df.shape[1], geno_df.shape[0], 2), dtype=int)

            # Convert each genotype count to allele counts
            # e.g., 0 -> [0,0], 1 -> [1,0], 2 -> [1,1]
            for i, count in enumerate([0, 1, 2]):
                mask = geno_df.values.T == count
                if count == 0:
                    continue  # already zeros
                elif count == 1:
                    genotypes[mask, 0] = 1
                else:  # count == 2
                    genotypes[mask] = 1

            return allel.GenotypeArray(genotypes), samples

        # Load from zarr
        elif zarr is not None:
            return self._load_from_zarr(zarr)

        # Load from VCF
        elif vcf is not None:
            print("reading VCF")
            vcf_data = allel.read_vcf(vcf, log=sys.stderr)
            if vcf_data is None:
                raise ValueError(f"Could not read VCF file: {vcf}")
            genotypes = allel.GenotypeArray(vcf_data["calldata/GT"])
            samples = vcf_data["samples"]
            return genotypes, samples

        # Load from matrix
        elif matrix is not None:
            print("reading matrix")
            gmat = pd.read_csv(matrix, sep="\t")
            samples = np.array(gmat["sampleID"])
            gmat = gmat.drop(labels="sampleID", axis=1)
            if not np.all(np.isin(gmat, [0, 1, 2])):
                raise ValueError("Genotype values must be 0, 1, or 2")
            gmat = np.array(gmat, dtype="int8")

            # Convert to haplotype format
            hmat = None
            for i in range(gmat.shape[0]):
                h1 = []
                h2 = []
                for j in range(gmat.shape[1]):
                    count = gmat[i, j]
                    if count == 0:
                        h1.append(0)
                        h2.append(0)
                    elif count == 1:
                        h1.append(1)
                        h2.append(0)
                    elif count == 2:
                        h1.append(1)
                        h2.append(1)
                if i == 0:
                    hmat = h1
                    hmat = np.vstack((hmat, h2))
                else:
                    hmat = np.vstack((hmat, h1))
                    hmat = np.vstack((hmat, h2))

            genotypes = allel.HaplotypeArray(np.transpose(hmat)).to_genotypes(ploidy=2)
            return genotypes, samples

        else:
            raise ValueError(
                "No genotype data provided. Either initialize with genotype_data DataFrame "
                "or provide vcf/zarr/matrix path."
            )
