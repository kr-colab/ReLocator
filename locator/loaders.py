"""Data loading functionality for locator"""

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
        return self._load_from_vcf_allel(vcf_path)

    def _load_from_vcf_cyvcf2(self, vcf_path):
        """Load genotypes using cyvcf2 (htslib-based).

        Alternative to the default scikit-allel loader. Useful when allel
        is unavailable or for streaming access. Requires cyvcf2.
        """
        from cyvcf2 import VCF

        print("reading VCF (cyvcf2)")
        vcf = VCF(vcf_path)
        samples = np.array(vcf.samples)
        n_samples = len(samples)

        chunk_size = 65536
        gt_chunks = []
        pos_chunks = []
        chrom_chunks = []

        gt_buf = np.empty((chunk_size, n_samples, 2), dtype=np.int8)
        pos_buf = np.empty(chunk_size, dtype=np.int32)
        chrom_buf = np.empty(chunk_size, dtype=object)
        idx = 0

        for variant in vcf:
            gt_buf[idx] = variant.genotype.array()[:, :2]
            pos_buf[idx] = variant.POS
            chrom_buf[idx] = variant.CHROM
            idx += 1
            if idx == chunk_size:
                gt_chunks.append(gt_buf.copy())
                pos_chunks.append(pos_buf.copy())
                chrom_chunks.append(chrom_buf.copy())
                idx = 0

        vcf.close()

        if idx > 0:
            gt_chunks.append(gt_buf[:idx].copy())
            pos_chunks.append(pos_buf[:idx].copy())
            chrom_chunks.append(chrom_buf[:idx].copy())

        if not gt_chunks:
            raise ValueError(f"No variants found in VCF: {vcf_path}")

        genotypes = allel.GenotypeArray(np.concatenate(gt_chunks))
        self.positions = np.concatenate(pos_chunks)
        self.chromosomes = np.concatenate(chrom_chunks)

        self._report_variant_metadata()
        return genotypes, samples

    def _load_from_vcf_allel(self, vcf_path):
        """Load genotypes using scikit-allel."""
        print("reading VCF (scikit-allel)")
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

        Args:
            matrix_path: Path to tab-delimited matrix file containing genotype data.
                File should have a header row with 'sampleID' as first column,
                followed by variant columns. Each row contains genotype counts (0,1,2)
                for one sample.

        Returns
        -------
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
