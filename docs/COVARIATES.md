# Covariate hardening layer

This layer adds reusable, id-keyed covariates for obvious genomic composition
confounds that sequence foundation models can learn directly:

- `repeat_frac`: fraction of the model window overlapped by UCSC hg38
  RepeatMasker intervals.
- `mappability`: fraction of the model window covered by UCSC hg38 Umap k100
  single-read uniquely mappable intervals.
- `gene_density`: fraction of the model window covered by any GENCODE v44 gene
  span, when `data/annotations/gencode_v44_features.pkl` is present.

The sidecars do not overwrite manifests. They are TSVs keyed by `id`.

## Files

- `src/common/region_covariates.py`: importable covariate functions.
- `src/common/annotate_manifest.py`: manifest-to-sidecar CLI.
- `data/annotations/rmsk.txt.gz`: UCSC hg38 RepeatMasker table dump.
- `data/annotations/rmsk_intervals.pkl`: merged interval cache built from
  `rmsk.txt.gz`.
- `data/annotations/umap_k100_unique_mappability.bb`: UCSC hg38 Umap k100
  single-read unique mappability bigBed.
- `data/aim1_sv/covariates_extra.tsv`: 700 annotated Aim 1 rows.
- `data/aim2_popgen/covariates_extra.tsv`: 2,400 annotated Aim 2 rows.

`data/` is gitignored; these track files are reproducible from the commands
below.

## Sources

- RepeatMasker: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz`
- Umap k100 single-read unique mappability:
  `https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Unique.Mappability.bb`
- GENCODE: existing local `data/annotations/gencode.v44.annotation.gtf.gz`,
  previously parsed into `data/annotations/gencode_v44_features.pkl`.

All coordinates are GRCh38 / UCSC `chr` style, 0-based half-open.

## Exact Commands Run

```bash
curl -L --fail --retry 3 \
  --output data/annotations/rmsk.txt.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz

curl -L --fail --retry 3 \
  --output data/annotations/umap_k100_unique_mappability.bb \
  https://hgdownload.soe.ucsc.edu/gbdb/hg38/hoffmanMappability/k100.Unique.Mappability.bb

uv pip install --python .venv/bin/python pyBigWig

.venv/bin/python -m src.common.annotate_manifest \
  --manifest data/aim1_sv/manifest.jsonl \
  --out data/aim1_sv/covariates_extra.tsv

.venv/bin/python -m src.common.annotate_manifest \
  --manifest data/aim2_popgen/manifest.jsonl \
  --out data/aim2_popgen/covariates_extra.tsv
```

The first RepeatMasker lookup builds `data/annotations/rmsk_intervals.pkl`.

## API

```python
from src.common.region_covariates import (
    repeat_fraction,
    mappability,
    gene_density,
    gc_from_fasta,
)

repeat_fraction("chr1", 10000, 11447)  # float in [0, 1]
mappability("chr1", 10000, 11447)      # float in [0, 1], or nan if unavailable
gene_density("chr1", 10000, 11447)     # float in [0, 1]
```

`gc_from_fasta(...)` is optional and returns `nan` unless `pysam` is installed
and an indexed FASTA is provided.

For manifest annotation:

```bash
.venv/bin/python -m src.common.annotate_manifest \
  --manifest data/<aim>/manifest.jsonl \
  --out data/<aim>/covariates_extra.tsv
```

## Manifest Coordinate Handling

Plain region records use their `[start0, end0)` interval.

SV records with `ref`/`alt` and `flank`/`max_allele` use the reference-backed
pieces of the Evo2 reference window:

- left flank: `[start0 - flank, start0)`, clamped at zero;
- reference interior, capped head+tail to `max_allele` when needed;
- right flank: `[end0, end0 + flank)`.

Inserted ALT sequence is not reference-backed, so it has no RepeatMasker or
Umap track coordinate. The sidecar therefore controls reference-window repeat
and mappability composition, not ALT-insert sequence composition.

## Self-Test

Command:

```bash
.venv/bin/python - <<'PY'
from src.common.region_covariates import repeat_fraction, mappability
tests = [
    ("telomeric_repeat_chr1", "chr1", 10000, 11447),
    ("GAPDH_gene_body_chr12", "chr12", 6533923, 6538374),
]
for name, chrom, start0, end0 in tests:
    print(name, repeat_fraction(chrom, start0, end0), mappability(chrom, start0, end0))
PY
```

Observed:

```text
telomeric_repeat_chr1 1.0 0.6247408431237043
GAPDH_gene_body_chr12 0.0 1.0
```

This is the expected direction: the telomeric repeat interval is fully masked,
while the GAPDH body interval has no RepeatMasker overlap and full k100 Umap
coverage.

## Merging Into `run_report`

Join by `id` in the same order as the feature rows or labels table, then append
these columns to the existing covariate matrix passed to
`src.common.analysis.run_report`.

Example:

```python
import numpy as np
import pandas as pd
from src.common.analysis import run_report

labels = pd.read_parquet("data/aim1_sv/labels.parquet")
extra = pd.read_csv("data/aim1_sv/covariates_extra.tsv", sep="\t")
labels = labels.merge(extra[["id", "repeat_frac", "mappability", "gene_density"]],
                      on="id", how="left", validate="one_to_one")

covariate_cols = [
    "log_svlen",
    "gc",
    "repeat_frac",
    "mappability",
    "gene_density",
]
covariates = labels[covariate_cols].to_numpy(float)

run_report(
    X,
    labels["y"].to_numpy(),
    groups=labels["chrom"].to_numpy(),
    covariates=covariates,
    outdir="results/aim1_sv",
    task="classification",
)
```

Use the covariate names that exist in the aim-specific label table (`log_svlen`
and `gc` above are placeholders for that aim's actual columns). The key point is
that `repeat_frac`, `mappability`, and `gene_density` should be included in both
the covariates-only baseline and the residualized-feature analysis.

## Current Sidecar Summaries

Aim 1 (`data/aim1_sv/covariates_extra.tsv`, 700 rows):

- `repeat_frac`: min 0.0000, median 0.4057, max 1.0000.
- `mappability`: min 0.0000, median 1.0000, max 1.0000.
- `gene_density`: min 0.0000, median 0.8403, max 1.0000.

Aim 2 (`data/aim2_popgen/covariates_extra.tsv`, 2,400 rows):

- `repeat_frac`: min 0.0000, median 0.5186, max 1.0000.
- `mappability`: min 0.0000, median 1.0000, max 1.0000.
- `gene_density`: min 0.0000, median 1.0000, max 1.0000.

No values were fabricated. Mappability is present for all current Aim 1 and Aim
2 records because the Umap k100 bigBed was available locally and readable with
`pyBigWig`.

