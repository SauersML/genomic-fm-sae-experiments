# Inversion SAE Analysis

Scope: HPRC release2 `INV`-flagged alleles only, analyzed with Evo2 layer-26 Goodfire SAE deltas. No introgression or expression outcome is used here.

- Inversion alleles embedded: 700
- Reference indel deltas compared from Aim1 pilot: 700
- INV-vs-indel SAE dimensions at BH q<0.05: 5819
- Within-inversion coding-vs-other SAE dimensions at BH q<0.05: 5662
- Inversion-length-associated SAE dimensions at BH q<0.05: 3142
- AF-associated SAE dimensions at BH q<0.05: 1902

## Predictive SAE Checks
- inversion_vs_indel: AUROC=0.598 over 1400 held-out rows by chromosome
- inversion_coding_disrupting: AUROC=0.479 over 700 held-out rows by chromosome
- inversion_log_length: R2=-1.449 over 700 held-out rows by chromosome

## f15532
- INV mean raw delta: 0.00103941; DEL mean: 0.00127257; INS mean: -0.00216679.
- INV-vs-indel f15532 standardized shift: 0.046, q=1.

## Top INV-vs-Indel SAE Features
| feature   |   inv_vs_indel_cohen_d |   inv_vs_indel_mean_diff_raw |           q |
|:----------|-----------------------:|-----------------------------:|------------:|
| f3159     |               0.794038 |                  0.000350047 | 1.27695e-39 |
| f29354    |               0.742382 |                  0.000166131 | 1.75721e-35 |
| f16711    |               0.684904 |                  4.14434e-05 | 3.54313e-30 |
| f5032     |              -0.67625  |                 -0.000129378 | 1.81912e-29 |
| f28973    |              -0.666822 |                 -0.000194791 | 1.14897e-28 |
| f25555    |              -0.653165 |                 -0.000179708 | 1.06508e-27 |
| f13626    |              -0.651271 |                 -0.000154079 | 1.60066e-27 |
| f18914    |               0.646581 |                  0.00126167  | 2.65445e-27 |

## Top Coding-vs-Other Features Within Inversions
| feature   |   inv_coding_vs_other_cohen_d |   inv_coding_vs_other_mean_diff_raw |           q |
|:----------|------------------------------:|------------------------------------:|------------:|
| f3269     |                      0.958407 |                         0.000277051 | 3.2826e-29  |
| f294      |                      0.911048 |                         0.00169011  | 8.97968e-25 |
| f23105    |                      0.888607 |                         0.000106287 | 1.31033e-24 |
| f16232    |                     -0.837381 |                        -8.3607e-05  | 4.07422e-24 |
| f26908    |                     -0.828982 |                        -0.00888224  | 7.16816e-24 |
| f28289    |                     -0.858017 |                        -0.000329352 | 7.16816e-24 |
| f3165     |                      0.937    |                         0.000274177 | 5.94376e-23 |
| f4172     |                     -0.83182  |                        -9.14675e-05 | 5.98166e-23 |

## Top Inversion-Length Features
| feature   |   spearman_log_inv_len |           q |
|:----------|-----------------------:|------------:|
| f29730    |               0.412596 | 9.09515e-26 |
| f28417    |               0.411488 | 9.09515e-26 |
| f11198    |               0.395957 | 1.17564e-23 |
| f6771     |               0.390499 | 5.259e-23   |
| f14060    |               0.385357 | 2.19499e-22 |
| f5056     |              -0.361932 | 2.36191e-19 |
| f5612     |               0.360903 | 2.73727e-19 |
| f13413    |              -0.358429 | 4.92114e-19 |

Plots: `plots/inversion_sae_specificity.png`, `plots/inversion_sae_associations.png`
