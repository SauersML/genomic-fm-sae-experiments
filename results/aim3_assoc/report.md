# Aim 3 — do Evo2-SAE haplotype feature profiles predict gene expression?

Per-individual feature = **mean** of the two haplotype SAE vectors. Held out by individual (CV groups = individual; plus an explicit held-out test set).

- genes analysed: **8**
- feature preprocessing: **unsupervised_truncated_svd_128_from_32768**; raw feature dims: **[32768]**
- genes with CV held-out signal (FDR<0.05 perm AND Spearman CI>0): **8/8**
- ... that survive **ancestry** residualization (resid Spearman CI>0): **8/8**
- ... that add signal beyond the **ALT-allele-burden** baseline (resid Spearman>0): **8/8**

## Per-gene (CV)

| gene | n | feat Spearman [CI] | perm q | resid(ancestry) | ancestry-only | resid(altcount) | altcount-only | heldout-test feat [CI] |
|---|---|---|---|---|---|---|---|---|
| C17orf97|ENSG00000187624 | 255 | 0.74 [0.69,0.78] | 0.002 | 0.74 | 0.05 | 0.68 | 0.73 | 0.62 [0.47,0.72] |
| ERAP2|ENSG00000164308 | 255 | 0.60 [0.51,0.68] | 0.002 | 0.57 | -0.14 | 0.46 | 0.58 | 0.34 [0.12,0.53] |
| GSTM1|ENSG00000134184 | 255 | 0.14 [0.01,0.27] | 0.002 | 0.23 | -0.10 | 0.14 | 0.14 | 0.17 [0.01,0.32] |
| PEX6|ENSG00000124587 | 255 | 0.80 [0.73,0.85] | 0.002 | 0.80 | -0.14 | 0.68 | 0.65 | 0.65 [0.46,0.78] |
| POMZP3|ENSG00000146707 | 255 | 0.75 [0.70,0.80] | 0.002 | 0.77 | -0.05 | 0.75 | 0.58 | 0.67 [0.51,0.78] |
| RPS26|ENSG00000197728 | 255 | 0.82 [0.77,0.86] | 0.002 | 0.82 | 0.05 | 0.78 | 0.49 | 0.75 [0.61,0.83] |
| SLFN5|ENSG00000166750 | 255 | 0.73 [0.66,0.78] | 0.002 | 0.71 | -0.06 | 0.60 | 0.76 | 0.78 [0.67,0.85] |
| ZNF266|ENSG00000174652 | 255 | 0.78 [0.70,0.83] | 0.002 | 0.78 | -0.01 | 0.64 | 0.76 | 0.73 [0.58,0.84] |

## EUR-only sensitivity (ancestry confound removed by design)

- genes with CV held-out signal (FDR<0.05 & CI>0): **7/8**

| gene | n | feat Spearman [CI] | perm q | resid(altcount) |
|---|---|---|---|---|
| C17orf97|ENSG00000187624 | 207 | 0.70 [0.63,0.75] | 0.00228 | 0.69 |
| ERAP2|ENSG00000164308 | 207 | 0.64 [0.54,0.72] | 0.00228 | 0.43 |
| GSTM1|ENSG00000134184 | 207 | 0.05 [-0.08,0.17] | 0.00798 | 0.09 |
| PEX6|ENSG00000124587 | 207 | 0.82 [0.74,0.87] | 0.00228 | 0.72 |
| POMZP3|ENSG00000146707 | 207 | 0.76 [0.71,0.81] | 0.00228 | 0.77 |
| RPS26|ENSG00000197728 | 207 | 0.78 [0.73,0.82] | 0.00228 | 0.63 |
| SLFN5|ENSG00000166750 | 207 | 0.73 [0.64,0.79] | 0.00228 | 0.67 |
| ZNF266|ENSG00000174652 | 207 | 0.78 [0.71,0.83] | 0.00228 | 0.54 |

## Interpretation rules

- A gene shows **real, non-confounded** SAE signal only if: CV perm q<0.05, feature Spearman CI>0, the ancestry-residualized Spearman CI stays >0, **and** it holds in the EUR-only run. 
- If features predict only when ancestry varies (collapse EUR-only / on residualization), that's the ancestry confound, not haplotype-specific regulation.
- The ALT-allele-burden baseline is the trivial-eQTL bar: SAE features are interesting insofar as they match/exceed it and add beyond it.
