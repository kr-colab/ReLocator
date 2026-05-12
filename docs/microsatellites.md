# Microsatellite input

ReLocator loads microsatellite genotypes natively. No preprocessing script,
no intermediate file — the same way VCF/zarr/matrix inputs work:

```bash
locator \
    --microsat data/microsats.tsv \
    --sample_data data/sample_data.txt \
    --out out/microsat_run/run1
```

Or from the Python API:

```python
from locator.core import Locator

loc = Locator({"sample_data": "data/sample_data.txt"})
genotypes, samples = loc.load_genotypes(microsat="data/microsats.tsv")
```

## Input format

Tab-delimited, one row per sample. First column is `sampleID`. Two layouts
are auto-detected:

**Pair format** — one column per locus, alleles separated by `,` `/` or `|`.
Space-separated cells (`"10 11"`) parse correctly once a file is recognised
as pair format, but they are *not* auto-detected as pair format because CSV
whitespace would create ambiguity; if every cell is space-separated, convert
to one of the supported separators or fall into the two-column layout below.

```
sampleID    L1      L2
s1          10,11   20,22
s2          10,10   20,20
s3          11,12   22,24
s4          10,12   NA
```

**Two-column format** — two consecutive columns per locus:

```
sampleID    variant_0   variant_1   variant_2   variant_3
s1          10          11          20          22
s2          10          10          20          20
```

Locus names in two-column inputs are renamed `locus_0`, `locus_1`, ... internally.

## Encoding

Each unique allele at each locus becomes its own column with values 0/1/2 —
a one-hot count encoding the diploid genotype. A locus with three alleles
{10, 11, 12} expands to three columns; a sample with genotype `10,12`
gets dosages `[1, 0, 1]` across those columns.

Rare alleles are dropped below a per-locus frequency threshold (default
1%). Override via `--microsat_maf <float>` on the CLI or
`load_genotypes(microsat=..., microsat_min_allele_freq=<float>)` from the
API. Loci with no parseable genotypes in any sample are dropped before
encoding (a one-line message reports how many).

## Missing data

Missing values (`NA`, `nan`, `.`, empty, `0,0`, `0/0`) are imputed to the
per-allele site mean inside the loader. The output dosage matrix is dense
float in `[0, 2]` with no `NaN`, so it flows directly through ReLocator's
continuous-dosage filter path (`filter_dosage_matrix`) — the same path GL
input uses.

Do **not** pass `--impute_missing` on the CLI to handle microsat missingness;
that flag operates on biallelic SNPs via `np.random.binomial` and is the
wrong primitive for multi-allelic encoding (independent per-allele binomial
draws can violate the per-locus diploid constraint).

## Sample ID alignment

The `sampleID` column of the microsat TSV must match the `sampleID` column
of `sample_data.txt`. ReLocator reorders metadata to match the microsat
input's row order; mismatched IDs produce a warning and missing-coords
rows are excluded from training.
