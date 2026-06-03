# DATA_SV — HPRC release2 structural-variant table (Aim 1)

Curated structural-variant (SV) table for Aim 1: HPRC SVs → Evo2 layer-26 SAE
ref-vs-alt feature delta → does the delta separate SVs by functional consequence?
This doc records the exact sources, commands, filters, label definitions, class
counts, and file schema. All data access is **public, unsigned** S3.

> Status: `<<COUNTS_PENDING>>` placeholders are filled by the run summary at the
> bottom once `label_svs.py` finishes on the complete streamed file.

---

## 1. Source

HPRC pangenome **release2**, minigraph-cactus, **GRCh38**-referenced *wave* VCF
(deconstructed + vcfwave-decomposed graph variants, includes SVs):

```
s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/
  hprc-v2.0-mc-grch38.wave.vcf.gz        (~2.28 GB)
  hprc-v2.0-mc-grch38.wave.vcf.gz.tbi
```

Listed with:
```
aws s3 ls --no-sign-request \
  s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/
```

VCF properties (from header):
- VCFv4.2, GRCh38 contigs (`chr1`..`chrM`, plus `_random`/`chrUn`/`_alt`), CHM13 + 232 samples (241 columns, haplotype phased `GT`).
- Pipeline: `vg deconstruct` → `bcftools norm -m -any` (split multiallelics → **biallelic, sequence-resolved** REF/ALT) → `vcfwave` decomposition → `bcftools norm -m +any`.
- INFO always carries population stats: **`AC`, `AF`, `AN`, `NS`** (computed across the 232 samples), plus graph fields `LV`, `PS`, `CONFLICT`. Decomposed records additionally carry `TYPE` (snp/mnp/ins/del/complex), `LEN` (allele length), `ORIGIN`, and an `INV` flag for detected inversions.

### Why stream instead of download
The local Mac had **< 3 GB free disk**, so the 2.28 GB VCF was **never stored**.
We streamed it once from S3 straight through a filter, discarding the large
per-sample genotype matrix (we only need `AF`/`AC`/`AN` from INFO, which the
pipeline already computed). This is fully reproducible and disk-light.

```bash
# /tmp/sv_filter.awk : keep biallelic, sequence-resolved SVs (|len(ALT)-len(REF)| >= 50),
# emit CHROM POS ID REF ALT INFO  (drop the 232 genotype columns)
aws s3 cp --no-sign-request \
  s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.wave.vcf.gz - \
  | gzip -dc \
  | awk -f /tmp/sv_filter.awk \
  > data/hprc_sv/sv_raw.tsv
```
`sv_filter.awk` (the SV definition / filter):
```awk
/^#/ { next }
{
  ref=$4; alt=$5;
  if (alt ~ /[<>\[\]]/) next;   # drop symbolic alleles (none expected post-vcfwave)
  if (alt ~ /,/)        next;   # drop any residual multiallelic
  rl=length(ref); al=length(alt);
  d = al-rl; if (d<0) d=-d;
  if (d >= 50) print $1,$2,$3,ref,alt,$8;   # SV = indel length delta >= 50 bp
}
```

**SV definition:** `abs(len(ALT) - len(REF)) >= 50` on the sequence-resolved,
biallelic, normalized records. Because the wave VCF is already decomposed, this
captures insertions and deletions (and `INV`-flagged inversions) at single-event
resolution. Symbolic/multiallelic records are excluded (none remain after vcfwave).

---

## 2. Functional-consequence labels (truth proxy)

We use a **reproducible interval-overlap annotation** rather than a heavy VEP/AnnotSV
install. This is a transparent, leakage-controllable truth proxy and is documented
as such.

### Annotation sources
- **GENCODE v44 (GRCh38)** gene models —
  `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz`
  Parsed (`src/common/build_gencode_features.py`) into merged, sorted interval
  arrays per contig for: `cds`, `utr` (5′/3′), `exon` (all biotypes), `gene_any`,
  `gene_coding` (protein_coding gene spans), and `splice` = ±8 bp windows around
  every exon boundary (canonical donor/acceptor region).
- **ENCODE SCREEN cCRE registry, GRCh38** (Registry V4) —
  `https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed` (2.35 M elements).
  Parsed (`src/common/build_ccre_features.py`) into a merged `any` set plus coarse
  classes `promoter` (PLS), `enhancer` (pELS/dELS), `ctcf` (CA-CTCF), `other`.

Overlap is computed against the SV's **reference footprint** on GRCh38:
`[POS-1, POS-1+len(REF))` (0-based half-open; ≥1 bp width enforced for pure
insertions). For deletions this is the deleted reference span (plus the anchor base);
for insertions it is the anchor point.

### Consequence hierarchy (first match wins, most→least severe)
| label         | rule                                                        |
|---------------|-------------------------------------------------------------|
| `cds`         | overlaps a CDS interval                                      |
| `splice`      | overlaps a splice-region window (±8 bp of an exon boundary)  |
| `utr`         | overlaps a 5′/3′ UTR exon                                    |
| `exon_noncod` | overlaps an exon but not CDS/UTR (ncRNA / noncoding-gene exon)|
| `intronic`    | within a gene span but in no exon                           |
| `regulatory`  | not genic, but overlaps an ENCODE cCRE                       |
| `intergenic`  | none of the above                                           |

### Targets emitted
- **`consequence`** — the 7-class label above (multiclass).
- **`consequence_coarse`** — `{coding (cds|splice), noncoding_genic (utr|exon_noncod|intronic), regulatory, intergenic}`.
- **`is_coding_disrupting`** — binary: `1` iff `consequence ∈ {cds, splice}`; the
  cleanest "function-altering" proxy. This is the primary Aim-1 target.

### Leakage control
SV length is **recorded but never used to define the label** (labels come purely
from genomic-position overlap). `svlen_abs`/`svlen_signed` are columns so length
can be regressed out / matched later. The pilot subset's per-class median
`svlen_abs` is reported below so any residual length confound is visible up front.

> Choice / shortcut, stated explicitly: we did **not** run VEP or AnnotSV (heavy
> install, and AnnotSV expects symbolic SV VCFs). The GENCODE+cCRE overlap labels
> are an accepted truth proxy per the task spec. A ClinVar-pathogenic high-confidence
> subset was **not** added in this pass (optional; can be layered on by intersecting
> `svs.parquet` coords with ClinVar SV records later).

---

## 3. Files (`data/hprc_sv/`)

| file | description |
|------|-------------|
| `sv_raw.tsv`         | streamed intermediate: `CHROM POS ID REF ALT INFO`, SVs only |
| `svs.parquet` / `svs.tsv` | **full curated table**, one row per SV (schema below) |
| `svs_pilot.parquet` / `svs_pilot.tsv` | balanced, deduplicated pilot subset (~700/class) |

### Schema (`svs.parquet`)
| column | meaning |
|--------|---------|
| `sv_id` | stable id `hprcv2_<chrom>_<pos>_<svtype>_<sha1[:12]>` |
| `chrom`, `pos` | GRCh38 contig, VCF 1-based position (anchor base) |
| `vcf_id` | original graph variant ID (`>node>node...`) |
| `ref`, `alt` | sequence-resolved alleles (anchor base included, as in VCF) |
| `svlen_signed` | `len(ALT)-len(REF)` (+ = insertion, − = deletion) |
| `svlen_abs` | `abs(svlen_signed)` |
| `svtype` | `INS` / `DEL` / `INV` (from `INV` flag) |
| `af`, `ac`, `an`, `ns` | allele frequency / count / number / #samples (from INFO) |
| `inv_flag` | bool, INFO `INV` present |
| `ref_start0`, `ref_end0` | reference footprint, 0-based half-open (`POS-1 .. POS-1+len(REF)`) |
| `window_start0`, `window_end0` | footprint ± `flank_bp` (Evo2 window bounds) |
| `flank_bp` | flank size used (default 4096) |
| `ov_cds … ov_ccre` | per-class overlap booleans (provenance for the label) |
| `consequence` | 7-class label |
| `consequence_coarse` | 4-class label |
| `is_coding_disrupting` | binary target |

---

## 4. Building ref & alt sequence windows for Evo2

Each row carries everything needed to construct both windows; no re-querying the
VCF. You need a local **GRCh38 FASTA whose contig names match the VCF**
(`chr`-prefixed, includes `_random`/`chrUn`). Recommended (matches exactly):

```bash
# UCSC hg38 (chr-named, incl. random/alt) — ~900 MB; index once.
curl -L -o data/annotations/hg38.fa.gz \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip data/annotations/hg38.fa.gz
samtools faidx data/annotations/hg38.fa
```
(Alternative identical-contig option: NCBI `GCA_000001405.15_GRCh38_no_alt_analysis_set`.)
This FASTA is reusable by the rest of the pipeline. We did **not** download it here
(local disk was full); it is a one-line fetch on the GPU VM / a roomier disk.

Window construction (`src/common/fetch_sv_windows.py`, needs `pysam`):
```
REF window = genome[window_start0 : window_end0]
ALT window = genome[window_start0 : ref_start0] + ALT + genome[ref_end0 : window_end0]
```
The helper also asserts `genome[ref_start0] == REF[0]` to confirm the FASTA matches
the VCF coordinate frame. Run:
```bash
python src/common/fetch_sv_windows.py data/annotations/hg38.fa \
    data/hprc_sv/svs_pilot.parquet data/hprc_sv/svs_pilot_windows.parquet
```

---

## 5. Reproduce end-to-end
```bash
# 1. stream + filter SVs (no full download)
aws s3 cp --no-sign-request s3://human-pangenomics/pangenomes/freeze/release2/minigraph-cactus/hprc-v2.0-mc-grch38.wave.vcf.gz - \
  | gzip -dc | awk -f /tmp/sv_filter.awk > data/hprc_sv/sv_raw.tsv
# 2. annotations
curl -L -o data/annotations/gencode.v44.annotation.gtf.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz
curl -L -o data/annotations/encode_ccre_grch38.bed https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed
python src/common/build_gencode_features.py
python src/common/build_ccre_features.py
# 3. label + curate
python src/common/label_svs.py
```

---

## 6. Run summary

Run on 2026-06-03, full GRCh38 wave VCF streamed end-to-end (chr10→…→chrY, 43
contigs incl. alt/random). No download errors; `sv_filter.err` empty.

**Full table `svs.parquet` / `svs.tsv`: 155,432 SVs** (62 MB / 165 MB). `af` present
for 100% of rows (AF range 0–1; 1,292 with AF=0, 511 with AF=1). Zero duplicates on
`(chrom,pos,ref,alt)`.

svtype: `DEL` 89,880 · `INS` 65,552. (No `INV`-flagged records: the wave VCF encodes
inversions as large balanced indels rather than the `INV` flag, so the `svtype`
column is INS/DEL by construction; `inv_flag` retained for provenance, all False here.)

`svlen_abs` quantiles: median 141 bp, p90 1,501 bp, p99 ≈ 12.6 kb.

### Class counts (full)
| `consequence` (7-class) | n | | `consequence_coarse` | n |
|---|---:|---|---|---:|
| intronic     | 79,678 | | noncoding_genic | 83,113 |
| intergenic   | 52,507 | | intergenic      | 52,507 |
| regulatory   | 14,710 | | regulatory      | 14,710 |
| splice       |  3,103 | | coding          |  5,102 |
| exon_noncod  |  2,291 | | | |
| cds          |  1,999 | | **is_coding_disrupting** | |
| utr          |  1,144 | | 0 / 1 | 150,330 / 5,102 |

### Pilot `svs_pilot.parquet` / `svs_pilot.tsv`: 4,900 SVs
Balanced 700 per `consequence` class (all 7 classes hit 700; deduplicated, shuffled,
seed=42/7). Coarse: noncoding_genic 2,100 · coding 1,400 · regulatory 700 · intergenic
700. Binary `is_coding_disrupting`: 3,500 / 1,400.

### Leakage check (pilot median `svlen_abs` by class, bp)
cds 530 · splice 1,693 · regulatory 228 · intergenic 136 · intronic 138 · exon_noncod
125 · utr 114. **Length is correlated with class** (coding/splice SVs skew larger —
larger SVs are likelier to hit exons). This is expected and is exactly why
`svlen_abs`/`svlen_signed` are stored as columns: control for / regress out length
when evaluating whether the Evo2 SAE feature-delta carries consequence signal beyond
size. Do **not** let a classifier see length as a free feature without a
length-matched control.

### Files written
```
data/hprc_sv/svs.parquet         62 MB   155,432 rows
data/hprc_sv/svs.tsv            165 MB
data/hprc_sv/svs_pilot.parquet  5.1 MB     4,900 rows
data/hprc_sv/svs_pilot.tsv       12 MB
data/hprc_sv/sv_raw.tsv         155 MB   (streamed intermediate; regenerable)
data/annotations/gencode.v44.annotation.gtf.gz, gencode_v44_features.pkl
data/annotations/encode_ccre_grch38.bed,        encode_ccre_features.pkl
src/common/{build_gencode_features,build_ccre_features,label_svs,fetch_sv_windows}.py
```

### Blockers / shortcuts (explicit)
- **Local disk was full (<3 GB free)** at start → the 2.28 GB VCF was **streamed,
  never stored**; uv cache cleared to free space. Fully reproducible.
- **No VEP/AnnotSV** — GENCODE v44 + ENCODE cCRE interval-overlap labels used as the
  truth proxy (per task allowance). Transparent and leakage-controllable.
- **ClinVar pathogenic high-confidence subset not added** (optional). Can be layered
  by intersecting `svs.parquet` coords with ClinVar SV records later.
- **Reference FASTA not downloaded** (disk + "only if quick"): one-line fetch
  documented in §4 (`hg38.fa.gz`, contigs match the VCF exactly).
