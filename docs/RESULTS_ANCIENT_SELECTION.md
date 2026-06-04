# RESULTS — Ancient-DNA selection coefficients vs Evo2-SAE features

## Question
Do Evo2 layer-26 Goodfire SAE features of a SNP's 5 kb reference window predict that SNP's
ancient-DNA **selection coefficient** (Akbari & Reich, Harvard Dataverse 10.7910/DVN/7RVV9N,
West-Eurasia aDNA time-series; hg19→hg38), beyond B-statistic / recombination / composition?

## Data
5,000-SNP pilot: 1,480 strongly-selected (top |s|, FDR≤0.05) + 1,491 covariate-matched neutral
controls + ~2,000 spanning the signed-s distribution. 5,001 bp windows → Evo2-7B → SAE (32,768)
→ mean-pooled. Held-out **test split = chr1, chr2**. Covariates: gc, repeat_frac, gene_density,
recomb_rate, b_statistic, dist_nearest_tss.

## Result — NULL on held-out chromosomes

Regression (Ridge on SVD-128, predefined chr1/chr2 hold-out, Spearman):

| target | model | test Spearman | train Spearman |
|---|---|---:|---:|
| signed s | SAE features | **−0.06** | +0.51 |
| signed s | covariate-only | −0.05 | +0.33 |
| \|s\| | SAE features | −0.02 | — |
| \|s\| | covariate-only | +0.02 | — |

- SAE features fit the training chromosomes (Spearman 0.51) but **do not generalize** to held-out
  chromosomes (≈0). The same is true of the confound covariates — even B-statistic/recombination
  do not predict signed s across chromosomes.

## Classification is ill-posed under this split
The held-out chromosomes contain **0 selected SNPs** (the top-|s| FDR set is chromosome-clustered),
so selected-vs-control AUROC is undefined on chr1/chr2. An earlier GroupKFold run reported
AUROC≈0.07 — that is an artifact of degenerate per-fold class balance + cross-fold score pooling,
**not** a real (inverse) signal. Do not trust it.

## Verdict
**Evo2 SAE features do not predict ancient-DNA selection coefficients on held-out chromosomes.**
This matches the program-wide pattern (Aims 1–2): apparent in-sample fit, no held-out signal,
and the only structure recoverable is genomic-context composition, not function/selection.

## Caveats / next steps
- Held-out-by-chromosome is the leakage-safe choice but is harsh and makes the positive set absent
  from the test chromosomes. A leakage-controlled **random split with LD pruning** would let the
  selected-vs-control classification be evaluated; given the clean regression null, it is unlikely
  to change the verdict but is the honest follow-up.
- Mean-pooling over 5 kb may wash out a localized signal; a ref/alt **delta** at the SNP (single-base)
  was planned but not run here.
