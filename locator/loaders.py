"""Data loading functionality for locator"""

import warnings

import allel
import numpy as np
import pandas as pd
import zarr

from locator import _gl as _gl
from locator import _microsat as _ms


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
            if not ((dosage >= 0.0) & (dosage <= 2.0)).all():
                raise ValueError(
                    "Continuous-dosage matrix has values outside [0, 2]; "
                    "expected expected-dosage encoding (e.g. from "
                    "scripts/gl_to_locator.py)."
                )
            return dosage, samples

        if not np.all(np.isin(gmat, [0, 1, 2])):
            raise ValueError("Genotype values must be 0, 1, or 2")
        gmat = np.array(gmat, dtype="int8")

        genotypes = _counts_to_genotype_array(gmat)
        return genotypes, samples

    def _load_from_microsat(self, microsat_path, min_allele_freq=0.01):
        """Load microsatellite genotypes as a multi-allelic dosage matrix.

        The input is a tab-delimited file with a 'sampleID' column and one
        column per locus (pair format: ``"12,14"``) or two consecutive
        columns per locus (two-column format). Each unique allele at each
        locus becomes its own column with values 0/1/2 (one-hot allele
        counts encoding the diploid genotype). Missing genotypes are
        imputed to the per-allele site mean.

        Args:
            microsat_path: Path to the tab-delimited microsat genotype table.
            min_allele_freq: Drop alleles below this per-locus frequency.
                Default 0.01.

        Returns the (n_sites, n_samples) float dosage representation used
        by the continuous-dosage path in ``_filter_genotypes`` (same shape
        and dtype as the float branch of ``_load_from_matrix``).
        """
        df = pd.read_csv(microsat_path, sep="\t", dtype=str)
        if "sampleID" not in df.columns:
            raise ValueError(
                f"Microsat input {microsat_path} must have a 'sampleID' column."
            )
        if df["sampleID"].duplicated().any():
            dups = df.loc[df["sampleID"].duplicated(), "sampleID"].unique().tolist()
            raise ValueError(
                f"Duplicate sampleIDs in microsat input: {dups}. "
                f"Each sample must appear once."
            )

        fmt = _ms.detect_format(df)
        if fmt == "two_column":
            df = _ms.convert_two_column_to_pair(df)

        df = df.set_index("sampleID")
        loci = list(df.columns)

        catalog = _ms.build_allele_catalog(df, loci, min_allele_freq=min_allele_freq)
        active_loci = [locus for locus in loci if catalog[locus]]
        n_dropped = len(loci) - len(active_loci)
        if n_dropped:
            print(
                f"Dropped {n_dropped} fully-missing locus/loci "
                f"({len(active_loci)}/{len(loci)} retained)."
            )
        if not active_loci:
            raise ValueError(
                f"No loci have any alleles after MAF filtering "
                f"({len(loci)} loci checked); check the input for "
                f"per-locus missingness or lower min_allele_freq."
            )

        matrix, _col_names = _ms.encode_dosage_block(df, active_loci, catalog)
        # encode_dosage_block returns (n_samples, K). The continuous-dosage
        # contract is (n_sites, n_samples), matching _load_from_matrix's
        # float branch.
        dosage = matrix.T.astype(np.float32, copy=False)
        samples = np.array(df.index, dtype=object)
        return dosage, samples

    def _load_from_gl(self, beagle_path, bam_list_path, gl_mode="dosage"):
        """Load ANGSD genotype-likelihood data as a continuous-dosage matrix.

        Reads an ANGSD ``-doGlf 2`` beagle.gz file plus a paired bam_list
        (sample IDs derived from ``Path(bam).stem``). Returns a 2-D float32
        matrix in the same shape ``_load_from_matrix`` produces for its
        continuous-dosage branch, so downstream filtering dispatches through
        the existing ``is_dosage_matrix`` path without further wiring.

        Two modes:

        - ``"dosage"`` (default): returns ``(n_sites_kept, n_samples)``.
          Each value is expected dosage E[geno] = P(AB) + 2*P(BB) under a
          flat prior. Missing samples (max GL < 0.4) are imputed with
          site-mean dosage. Site filter: ``min_maf=0.01``,
          ``max_missing_frac=0.10``.
        - ``"full_gl"``: returns ``(3 * n_sites_kept, n_samples)``. Three
          pseudo-rows per genomic site holding the AA / AB / BB GL
          probabilities. Preserves genotype uncertainty information that
          the dosage scalar collapses. Missing samples are imputed with the
          per-site mean GL triplet.

        Filter thresholds mirror ``scripts/gl_to_locator.py`` defaults and
        are not currently surfaced as CLI flags.
        """
        if gl_mode not in ("dosage", "full_gl"):
            raise ValueError(f"gl_mode must be 'dosage' or 'full_gl', got {gl_mode!r}")

        sample_ids = _gl.sample_ids_from_bam_list(bam_list_path)
        n_samples = len(sample_ids)

        _markers, gl_flat = _gl.load_beagle(beagle_path)
        _gl.validate_dimensions(gl_flat, n_samples, beagle_path, bam_list_path)
        gl = _gl.reshape_gl(gl_flat, n_samples)

        dosage = _gl.expected_dosage(gl)
        missing_mask = _gl.detect_missing(gl, gl_missing_threshold=0.4)
        dosage = _gl.impute_dosage_with_site_mean(dosage, missing_mask)

        keep, _reasons = _gl.filter_sites(
            dosage, missing_mask, min_maf=0.01, max_missing_frac=0.10
        )
        if not keep.any():
            raise ValueError(
                f"No sites passed the MAF/missingness filter "
                f"(min_maf=0.01, max_missing_frac=0.10) on {beagle_path}."
            )

        if gl_mode == "dosage":
            out = dosage[keep, :].astype(np.float32, copy=False)
        else:  # full_gl
            gl_imputed = _gl.impute_gl_with_site_mean(gl, missing_mask)
            kept_gl = gl_imputed[keep, :, :]  # (n_kept, n_samples, 3)
            # Reshape to (3 * n_kept, n_samples) with row order AA, AB, BB
            # per site. transpose to (n_kept, 3, n_samples) then flatten the
            # first two dims.
            out = (
                kept_gl.transpose(0, 2, 1)
                .reshape(-1, n_samples)
                .astype(np.float32, copy=False)
            )

        samples = np.array(sample_ids, dtype=object)
        return out, samples

    def load_genotypes(  # noqa: C901
        self,
        vcf=None,
        zarr=None,
        matrix=None,
        microsat=None,
        microsat_min_allele_freq=0.01,
        gl=None,
        bam_list=None,
        gl_mode="dosage",
    ):
        """Load genotype data from various input sources.

        This method can load genotype data from:
        1. A stored DataFrame provided during initialization
        2. A VCF file
        3. A zarr file (scikit-allel or bio2zarr format)
        4. A tab-delimited matrix file
        5. A tab-delimited microsatellite genotype table
        6. ANGSD beagle GL file paired with a BAM list

        For windowed analysis, SNP positions must be available either from:
        - Column names in the genotype DataFrame
        - The zarr file's variants/POS array
        - The VCF file's POS field (automatically loaded)

        Args:
            vcf (str, optional): Path to VCF format genotype data
            zarr (str, optional): Path to zarr format genotype data
            matrix (str, optional): Path to tab-delimited matrix file
            microsat (str, optional): Path to tab-delimited microsatellite genotype table
            microsat_min_allele_freq (float, optional): Drop microsat alleles below
                this per-locus frequency. Default 0.01.
            gl (str, optional): Path to ANGSD ``-doGlf 2`` beagle.gz file
            bam_list (str, optional): Path to BAM file list used in ANGSD run
                (one path per line). Required when ``gl`` is provided.
                Sample IDs are derived from ``Path(bam).stem``.
            gl_mode (str): GL encoding mode, one of ``"dosage"`` (default) or
                ``"full_gl"``. ``"dosage"`` returns one expected-dosage value
                per site per sample; ``"full_gl"`` returns all three AA/AB/BB
                GL probabilities as separate rows.

        Returns
        -------
            tuple: (genotypes, samples) where:
                - genotypes is an allel.GenotypeArray of shape (n_sites, n_samples, 2)
                  for VCF/zarr/integer-matrix inputs, or a float32 ndarray of shape
                  (n_sites, n_samples) for continuous-dosage inputs (matrix float,
                  microsat, or GL)
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

            >>> # Using microsatellite genotypes
            >>> genotypes, samples = locator.load_genotypes(microsat="path/to/microsats.tsv")

            >>> # Using ANGSD genotype-likelihood file
            >>> genotypes, samples = locator.load_genotypes(
            ...     gl="output.beagle.gz", bam_list="bams.txt", gl_mode="dosage"
            ... )

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

        elif microsat is not None:
            return self._load_from_microsat(
                microsat, min_allele_freq=microsat_min_allele_freq
            )

        elif gl is not None:
            if bam_list is None:
                raise ValueError(
                    "--gl / gl= requires a paired --bam_list / bam_list= "
                    "to derive sample IDs from the ANGSD BAM filelist."
                )
            return self._load_from_gl(gl, bam_list, gl_mode=gl_mode)

        else:
            raise ValueError(
                "No genotype data provided. Either initialize with genotype_data DataFrame "
                "or provide vcf/zarr/matrix/microsat/gl path."
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
