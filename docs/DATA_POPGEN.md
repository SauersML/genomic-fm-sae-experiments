# Popgen Region Data For Aim 2

Generated on 2026-06-03 under `data/popgen/`. All final coordinates are GRCh38 / UCSC `chr` style, 0-based half-open BED coordinates.

## Sources

Selective sweeps:
- Source: Prendergast, Maclean, Chue Hong. `hapbin: An efficient program for performing haplotype based scans for positive selection in large genomic datasets`, University of Edinburgh DataShare, DOI `10.7488/ds/214`.
- URL: `https://datashare.ed.ac.uk/handle/10283/714`
- Used files: bedGraph normalized iHS files for five representative 1000G Phase 3 populations: `CEU`, `CHB`, `GIH`, `PEL`, `YRI`, chromosomes 1-22.
- Rule: keep SNP rows with normalized iHS `>= 4.0`, merge same-population/same-chromosome outlier SNPs separated by <= 50 kb, lift the signal interval to GRCh38, and emit a centered 32,768 bp analysis region. The original lifted signal span is retained in `signal_start` / `signal_end`.

Archaic introgression:
- Source: Browning et al. 2018 SPrime results for 1000 Genomes non-African populations and SGDP Papuans, Mendeley Data, DOI `10.17632/y7hyt83vxr.1`.
- URL: `https://data.mendeley.com/datasets/y7hyt83vxr/1`
- Used files: all `*_sprime_results.tar.gz` archives available from the public Mendeley API plus README.
- Rule: group SPrime SNP rows by population, chromosome, and `SEGMENT`; use min/max SNP position as the segment span; keep SPrime score and Neanderthal/Denisovan match rates; lift segment spans to GRCh38.

Liftover / masks:
- UCSC hg19 to hg38 chain: `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz`
- UCSC `liftOver` binary: `https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver`
- GRCh38 chromosome sizes: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes`
- GRCh38 gap track: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz`

Reference FASTA to use downstream:
- Use a UCSC-style GRCh38/hg38 FASTA with `chr1`-style names, e.g. UCSC `hg38.fa.gz`.
- Assembly accession: GRCh38, GenBank `GCA_000001405.15`, RefSeq `GCF_000001405.26`.

## Reproduction Commands

Tool setup:

```bash
brew install bedtools
mkdir -p data/popgen/raw/sweeps_hapbin data/popgen/raw/sprime data/popgen/tools data/popgen/intermediate data/popgen/metadata
curl -L -o data/popgen/tools/liftOver 'https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver'
chmod +x data/popgen/tools/liftOver
curl -L -o data/popgen/tools/hg19ToHg38.over.chain.gz 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz'
```

GRCh38 masks:

```bash
curl -L -s -o data/popgen/metadata/hg38.chrom.sizes.all 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes'
awk '$1 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {print}' data/popgen/metadata/hg38.chrom.sizes.all | sort -V > data/popgen/hg38.autosomes.chrom.sizes
curl -L -s -o data/popgen/metadata/hg38.gap.txt.gz 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/gap.txt.gz'
gzip -dc data/popgen/metadata/hg38.gap.txt.gz | awk 'BEGIN{OFS="\t"} $2 ~ /^chr([1-9]|1[0-9]|2[0-2])$/ {print $2,$3,$4,$8}' | sort -k1,1V -k2,2n > data/popgen/hg38.gaps.autosomes.bed
```

Sweep downloads:

```bash
for pop in CEU CHB GIH PEL YRI; do
  for chr in $(seq 1 22); do
    curl -L -s -o "data/popgen/raw/sweeps_hapbin/${pop}_chr${chr}.bg.gz" \
      "https://datashare.ed.ac.uk/bitstream/handle/10283/714/${pop}_chr${chr}.bg.gz?isAllowed=y"
  done
done
curl -L -s -o data/popgen/raw/sweeps_hapbin/fileDescriptions.txt \
  'https://datashare.ed.ac.uk/bitstream/handle/10283/714/fileDescriptions.txt?isAllowed=y'
```

SPrime downloads:

```bash
curl -L -s -H 'Accept: application/vnd.mendeley-public-dataset.1+json' \
  'https://data.mendeley.com/public-api/datasets/y7hyt83vxr/files?folder_id=root&version=1' \
  > data/popgen/metadata/sprime_mendeley_files.json

python3 - <<'PY' > data/popgen/metadata/sprime_downloads.tsv
import json
files=json.load(open('data/popgen/metadata/sprime_mendeley_files.json'))
print('filename\turl\tsha256\tsize')
for f in files:
    name=f['filename']
    if name.endswith('_sprime_results.tar.gz') or name=='README':
        cd=f['content_details']
        print(f"{name}\t{cd['download_url']}\t{cd['sha256_hash']}\t{f['size']}")
PY

while IFS=$'\t' read -r name url sha size; do
  [ "$name" = "filename" ] && continue
  curl -L -A 'Mozilla/5.0' -s -o "data/popgen/raw/sprime/$name" "$url"
done < data/popgen/metadata/sprime_downloads.tsv
```

Build regions and controls:

```bash
python3 data/popgen/build_popgen_regions.py
python3 data/popgen/make_controls.py
```

## Outputs

Primary region tables:
- `data/popgen/sweeps_regions.grch38.tsv`: sweep positives plus controls.
- `data/popgen/introgression_regions.grch38.tsv`: introgression positives plus controls.
- `data/popgen/popgen_regions.grch38.tsv`: combined table.

Positive BEDs:
- `data/popgen/sweeps_positives.grch38.bed`
- `data/popgen/introgression_positives.grch38.bed`

Control BEDs:
- `data/popgen/sweeps_controls.grch38.bed`
- `data/popgen/introgression_controls.grch38.bed`

Important TSV columns:
- `chrom,start,end`: final region coordinates. Sweep positives and controls are 32,768 bp centered iHS windows. Introgression positives are full lifted SPrime segment spans; introgression controls are full-length where possible.
- `label`: `sweep`, `introgression`, or `control`.
- `source`: source dataset or control-generation method.
- `population`: 1000G/SGDP population label.
- `stat_name,stat_value`: `max_normalized_iHS` for sweeps or `sprime_score` for introgression.
- `signal_start,signal_end`: source-derived lifted signal span.
- `window8_start,window8_end`: centered 8,192 bp analysis window.
- `window32_start,window32_end`: centered 32,768 bp analysis window.
- `split`: held-out chromosome split.

## Counts

Selective sweeps:
- Positive regions: 9,311.
- Controls: 9,311.
- Split: 7,691 train positives, 1,620 test positives; 7,691 train controls, 1,620 test controls.
- Populations: `CEU`, `CHB`, `GIH`, `PEL`, `YRI`.
- LiftOver unmapped iHS clusters: 20.

Archaic introgression:
- Positive regions: 26,005.
- Controls: 26,005.
- Split: 21,994 train positives, 4,011 test positives; 21,994 train controls, 4,011 test controls.
- Populations: `BEB`, `CDX`, `CEU`, `CHB`, `CHS`, `CLM`, `FIN`, `GBR`, `GIH`, `IBS`, `ITU`, `JPT`, `KHV`, `MXL`, `PEL`, `PJL`, `PUR`, `Papuans`, `STU`, `TSI`.
- Best archaic match annotation: 19,011 Neanderthal, 2,123 Denisovan, 4,871 ambiguous.
- LiftOver unmapped SPrime segments: 77.

Combined:
- Total final rows: 70,632.
- Positive rows: 35,316.
- Control rows: 35,316.

## Controls

Controls were generated with `bedtools shuffle`, seed `20260603`, same chromosome as the matched positive row, excluding same-task positive regions and UCSC GRCh38 gaps.

Sweep controls:
- 9,311 / 9,311 are full 32,768 bp matched windows.

Introgression controls:
- 25,934 / 26,005 are full-length same-chromosome controls.
- 71 / 26,005 use the 32,768 bp centered window because the full SPrime segment was longer than any available same-chromosome background interval after excluding positives and gaps.
- These fallback rows have `source=bedtools_shuffle_introgression_window32_impossible_full_length_seed_20260603`.

QA checks:
- Sweep controls overlapping sweep positives or gaps: 0.
- Introgression controls overlapping introgression positives or gaps: 0.
- All final rows have `0 <= start < end`.

## Held-Out Split

Held-out test chromosomes: `chr1`, `chr2`.

All rows on `chr1` or `chr2` are `split=test`; all other autosomes are `split=train`. The same chromosome rule is applied to controls, so no linked regions from held-out chromosomes enter training. Control random seed: `20260603`.

## Notes

The input sources are GRCh37/hg19-style coordinates. SPrime README states positions are build 37. The hapbin DataShare page says the bedGraph tracks are for UCSC `db=hg37`. Both were lifted with the UCSC `hg19ToHg38` chain.

No curated-only sweep loci were added because the hapbin 1000G genome-wide iHS source downloaded cleanly and gives traceable statistic values. No dbPSHP, Johnson-Voight, Vernot/Akey, or hmmix substitution was used.
