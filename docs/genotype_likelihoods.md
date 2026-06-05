# Genotype-likelihood input (ANGSD beagle)

ReLocator loads ANGSD genotype-likelihood data natively. No preprocessing
script, no intermediate TSV — the same shape as VCF/zarr/matrix inputs:

```bash
locator \
    --gl output.beagle.gz \
    --bam_list bam.filelist \
    --sample_data data/sample_data.txt \
    --out out/gl_run/run1
```

Or from the Python API:

```python
from locator.core import Locator

loc = Locator({"sample_data": "data/sample_data.txt"})
genotypes, samples = loc.load_genotypes(
    gl="output.beagle.gz",
    bam_list="bam.filelist",
    gl_mode="dosage",  # or "full_gl"
)
```

## Generating the beagle input

Use ANGSD with `-doGlf 2` to write the beagle file Locator reads:

```bash
angsd \
    -bam bam.filelist \
    -ref reference.fa \
    -GL 2 \
    -doGlf 2 \
    -doMajorMinor 1 \
    -doMaf 1 \
    -SNP_pval 1e-6 \
    -minMapQ 20 \
    -minQ 20 \
    -minInd 10 \
    -minMaf 0.05 \
    -out output
```

Pass the exact same `bam.filelist` to both ANGSD and Locator. Sample IDs
in `sample_data.txt` must match `Path(bam).stem` for each BAM, in the
same column order ANGSD wrote into the beagle.

## Modes

`--gl_mode dosage` (default) emits one row per kept site holding expected
dosage under a flat prior:

```
E[geno] = P(AB) + 2 * P(BB)
```

`--gl_mode full_gl` emits three rows per kept site (AA / AB / BB),
preserving genotype uncertainty that the dosage scalar collapses. Useful
at low coverage where a confidently-homozygous site `(0.9, 0.05, 0.05)`
and a genuinely-uncertain site `(0.4, 0.4, 0.2)` would otherwise map to
similar dosage values.

## Site filtering

Hard-coded defaults applied inside the loader:

| Threshold | Value | Effect |
|---|---|---|
| `gl_missing_threshold` | 0.4 | Sample at site is missing if `max(GL_AA, GL_AB, GL_BB) < 0.4` (near-uniform GL = no information) |
| `max_missing_frac` | 0.10 | Site dropped if missing fraction across samples exceeds this |
| `min_maf` | 0.01 | Site dropped if mean-dosage-derived MAF falls below this |

These are not yet surfaced as CLI flags. If your dataset needs different
thresholds, edit `locator/loaders.py:_load_from_gl` directly or open an
issue.

## Missing data

Sample-level missingness at a site (per `gl_missing_threshold` above) is
imputed inside the loader to per-site mean dosage (dosage mode) or per-site
mean GL triplet (full_gl mode). The output flows through ReLocator's
continuous-dosage filter path (`filter_dosage_matrix`) with no NaNs.

Do **not** pass `--impute_missing` on the CLI for GL inputs; that flag
operates on biallelic SNP `allel.GenotypeArray` via `np.random.binomial`
and is not the right primitive for expected-dosage or GL-triplet inputs.

## Sample ID alignment

Sample IDs are derived from BAM filenames (`Path(bam).stem` for each line
in `--bam_list`). These must match the `sampleID` column in
`sample_data.txt`. The standard ReLocator reordering in `sort_samples`
matches the two and warns on any mismatches.
