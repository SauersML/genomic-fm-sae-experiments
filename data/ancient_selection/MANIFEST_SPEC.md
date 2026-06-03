# Ancient Selection SNP Manifest

Built by `data/ancient_selection/build_ancient_selection.py` with seed `20260603`.

## Files

- `snps_hg38.tsv`: full lifted autosomal biallelic SNP table with Akbari selection statistics and hg38 covariates.
- `snps_pilot.tsv`: pilot modeling table with 5,001 bp centered windows (`start0`, `end0`), selected/control/continuous labels, and chromosome split.
- `summary.json`: machine-readable build counts.
- `work/akbari_snps_hg38.unmapped.bed`: UCSC liftOver unmapped records retained as the mapping log.
- `READY`: completion sentinel.

## Coordinate Conventions

- Source coordinates are GRCh37/hg19, 1-based.
- Output `chrom`, `start0`, and `end0` are chr-prefixed GRCh38/hg38, 0-based half-open.
- `pos_hg38` is 1-based.
- Windows are `[pos_hg38 - 1 - 2500, pos_hg38 + 2500)`, clipped to chromosome bounds.

## Build Counts

- Source rows: 9739624
- Source biallelic SNP rows: 8074573
- Autosomal SNPs submitted to liftOver: 8074573
- liftOver mapped: 8074523
- liftOver unmapped: 50
- Full output rows: 8074092
- Rows with ancestral allele-derived DAF: 7740391
- Pilot rows: 5000
- Pilot selected/control/continuous: 1500/1500/2000
- Pilot train/test by chromosome split: 4498/502

## Pilot Labels

- `selected`: top FDR-significant (`FDR <= 0.05`) SNPs by `abs(selection_coefficient)`, capped at 1,500.
- `control`: nearest-neighbor matched neutral SNPs from `FDR >= 0.95` and the lowest 20% of `abs(selection_coefficient)`.
- `continuous`: stratified sample across the signed `selection_coefficient` distribution, excluding selected/control SNPs.
- Matching covariates: `matching_af`, `b_statistic`, `recomb_rate_cm_per_mb`, `gc`, and `gene_density`.
- `split`: `test` for `chr1`/`chr2`, otherwise `train`.
