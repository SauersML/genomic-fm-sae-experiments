# Ancient-DNA Selection SNP Data

## Source

Primary source used: Akbari and Reich, Harvard Dataverse, DOI `10.7910/DVN/7RVV9N`, dataset title `Ancient DNA reveals pervasive directional selection across West Eurasia`.

Direct files used:

- Selection statistics: `https://dataverse.harvard.edu/api/access/datafile/12126280`
  - Dataverse label: `Selection_Summary_Statistics_01OCT2025.tsv.gz`
  - Header title: `Pervasive findings of directional selection realize the promise of ancient DNA to elucidate human adaptation`
  - Coordinates: GRCh37/hg19
  - Rows: 9,739,624 quality-controlled variants
- Dataverse README: `https://dataverse.harvard.edu/api/access/datafile/13596811`
- UCSC liftOver binary: `https://hgdownload.soe.ucsc.edu/admin/exe/macOSX.arm64/liftOver`
- UCSC chain: `https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz`

The user prompt referred to Akbari et al. 2024. The current Dataverse metadata and README describe the public release as Akbari et al. 2026 / Nature. This build used the primary Reich/Akbari Dataverse release rather than a fallback.

## Outputs

- `data/ancient_selection/snps_hg38.tsv`: full autosomal biallelic SNP table, lifted to chr-prefixed hg38.
- `data/ancient_selection/snps_pilot.tsv`: 5,000-row pilot with 5,001 bp centered windows, labels, covariates, and chromosome split.
- `data/ancient_selection/work/akbari_snps_hg38.unmapped.bed`: UCSC liftOver unmapped log.
- `data/ancient_selection/MANIFEST_SPEC.md` and `summary.json`: machine-readable and human-readable build metadata.
- `data/ancient_selection/READY`: completion sentinel.

The large Akbari source gzip was removed after building to fit the low-free-space checkout. The exact direct URL is above, and `build_ancient_selection.py` will use the same path if the file is re-downloaded.

## Parsing

The source columns used were `CHROM`, `POS`, `REF`, `ALT`, `ANC`, `ID`, `RSID`, `AF`, `S`, `SE`, `X`, `P_X`, `POSTERIOR`, `FDR`, `CHI2_BE`, and `FILTER`.

Rows were filtered to autosomal biallelic SNPs (`len(REF) == len(ALT) == 1`). The output includes `selection_coefficient` from source `S`, plus `selection_se`, `selection_z`, `selection_p`, `posterior`, `fdr`, and `alt_af`. `derived_allele_freq` is computed when `ANC` equals `REF` or `ALT`; otherwise it is `NA`. `matching_af` is `derived_allele_freq` when available and `alt_af` otherwise.

## LiftOver

UCSC `liftOver` was run on 1 bp hg19 BED intervals with `hg19ToHg38.over.chain.gz`.

Counts:

- Source rows: 9,739,624
- Biallelic SNPs submitted to liftOver: 8,074,573
- liftOver mapped: 8,074,523
- liftOver unmapped: 50
- Retained hg38 rows in `snps_hg38.tsv`: 8,074,092

The retained row count is slightly lower than mapped because mapped records outside the 22 chr-prefixed autosomes were dropped.

## Covariates

Window covariates use `[pos_hg38 - 1 - 2500, pos_hg38 + 2500)`, clipped to chromosome bounds.

- `gc`: computed from `data/reference/hg38.fa`.
- `repeat_frac`: RepeatMasker coverage from `data/annotations/rmsk_intervals.pkl`, originally built from UCSC hg38 `rmsk.txt.gz`.
- `gene_density`: GENCODE v44 gene-span coverage from `data/annotations/gencode_v44_features.pkl`.
- `dist_nearest_tss`: nearest strand-aware GENCODE v44 gene TSS parsed from `data/annotations/gencode.v44.annotation.gtf.gz` for protein-coding and lncRNA genes.
- `recomb_rate_cm_per_mb`: value at SNP from UCSC hg38 deCODE average recombination bigWig `https://hgdownload.soe.ucsc.edu/gbdb/hg38/recombRate/recombAvg.bw`.
- `b_statistic`: value at SNP from McVicker background-selection BED `https://raw.githubusercontent.com/gmcvicker/bkgd/master/data/hg38/bkgd_hg38.bed.gz`.

## Pilot Controls And Split

Seed: `20260603`.

Positive set: top 1,500 FDR-significant SNPs (`FDR <= 0.05`) by `abs(selection_coefficient)`.

Neutral controls: 1,500 SNPs with `FDR >= 0.95`, sampled from the lowest 20% of `abs(selection_coefficient)`, then nearest-neighbor matched to selected SNPs on `matching_af`, `b_statistic`, `recomb_rate_cm_per_mb`, `gc`, and `gene_density`.

Continuous sample: 2,000 SNPs sampled across the signed `selection_coefficient` distribution, excluding selected/control SNPs.

Pilot counts:

- selected: 1,500
- matched controls: 1,500
- continuous: 2,000
- total: 5,000
- train: 4,498
- test: 502

Split is chromosome-based: `chr1` and `chr2` are `test`, all other autosomes are `train`.

Selection coefficient range:

- Full hg38 SNP table: `[-0.056680718, 0.063378873]`
- Pilot: `[-0.056680718, 0.060289048]`
