"""Convert an ANGSD beagle GL file to a continuous-dosage zarr store.

Reads a ``-doGlf 2`` beagle.gz plus its paired bam_list, computes expected
dosage with the same impute/filter pipeline as the native ``--gl`` loader, and
writes a zarr store holding the dosage matrix plus ``variants/POS`` and
``variants/CHROM`` parsed from the beagle marker column. The resulting store
loads via ``load_genotypes(zarr=...)`` and carries the per-site positions that
windowed analysis needs.
"""

import argparse

import numpy as np
import zarr

from locator import _gl


def beagle_to_zarr(
    beagle,
    bam_list,
    out_zarr,
    gl_missing_threshold=0.4,
    gl_min_maf=0.01,
    gl_max_missing_frac=0.10,
    chunk_sites=10000,
    overwrite=False,
):
    """Write a continuous-dosage zarr store from a beagle GL file.

    The dosage / imputation / site-filter steps mirror
    ``DataLoaderMixin._load_from_gl`` so the stored matrix is identical to what
    ``load_genotypes(gl=..., gl_mode="dosage")`` would return.
    """
    sample_ids = _gl.sample_ids_from_bam_list(bam_list)
    n_samples = len(sample_ids)

    markers, gl_flat = _gl.load_beagle(beagle)
    _gl.validate_dimensions(gl_flat, n_samples, beagle, bam_list)
    gl = _gl.reshape_gl(gl_flat, n_samples)

    dosage = _gl.expected_dosage(gl)
    missing_mask = _gl.detect_missing(gl, gl_missing_threshold=gl_missing_threshold)
    dosage = _gl.impute_dosage_with_site_mean(dosage, missing_mask)

    keep, _reasons = _gl.filter_sites(
        dosage,
        missing_mask,
        min_maf=gl_min_maf,
        max_missing_frac=gl_max_missing_frac,
    )
    if not keep.any():
        raise ValueError(
            f"No sites passed the MAF/missingness filter "
            f"(gl_min_maf={gl_min_maf}, gl_max_missing_frac={gl_max_missing_frac}) "
            f"on {beagle}."
        )

    dosage = dosage[keep, :].astype(np.float32, copy=False)
    chroms_all, pos_all = _gl.parse_markers(markers)
    chroms = chroms_all[keep]
    positions = pos_all[keep]

    n_sites = dosage.shape[0]
    root = zarr.open_group(out_zarr, mode="w" if overwrite else "w-")
    root.attrs["locator_format"] = "dosage"
    # Chunk along the sites axis only: windowed analysis slices rows (sites),
    # and every read needs all samples for a site.
    root.create_array(
        "dosage",
        data=dosage,
        chunks=(min(chunk_sites, n_sites), n_samples),
    )
    variants = root.create_group("variants")
    variants.create_array("POS", data=positions.astype(np.int64))
    variants.create_array("CHROM", data=np.array([str(c) for c in chroms], dtype=str))
    root.create_array("samples", data=np.array(sample_ids, dtype=str))

    print(
        f"Wrote {out_zarr}: dosage {dosage.shape} (float32), "
        f"{n_sites} sites x {n_samples} samples.",
        flush=True,
    )
    return out_zarr


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert an ANGSD beagle GL file to a dosage zarr store."
    )
    parser.add_argument("--beagle", required=True, help="ANGSD beagle.gz file")
    parser.add_argument(
        "--bam_list",
        required=True,
        help="BAM file list used in the ANGSD run (one path per line)",
    )
    parser.add_argument("--zarr", required=True, help="path for zarr output")
    parser.add_argument("--gl_min_maf", type=float, default=0.01)
    parser.add_argument("--gl_max_missing_frac", type=float, default=0.10)
    parser.add_argument("--gl_missing_threshold", type=float, default=0.4)
    parser.add_argument(
        "--chunk_sites",
        type=int,
        default=10000,
        help="zarr chunk size along the sites axis. default: 10000",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="overwrite existing zarr store"
    )
    args = parser.parse_args(argv)
    beagle_to_zarr(
        args.beagle,
        args.bam_list,
        args.zarr,
        gl_missing_threshold=args.gl_missing_threshold,
        gl_min_maf=args.gl_min_maf,
        gl_max_missing_frac=args.gl_max_missing_frac,
        chunk_sites=args.chunk_sites,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    main()
