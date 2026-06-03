# Aim 2 results — Evo2-SAE feature content of popgen-flagged regions

Does the Evo2 layer-26 SAE feature content of an 8 kb window centered on a popgen-flagged region separate positives from matched controls, **beyond a length/GC/repeat/mappability/gene-density confound baseline** and **beyond a label-shuffle null**, under a by-chromosome held-out split?

## A. Selective sweeps vs controls

- held-out AUROC (logreg, GroupKFold by chrom): **0.624** [0.580, 0.681]
- permutation (within-chromosome label shuffle) p = **0.000999**
- covariates: ['log_length', 'gc', 'repeat_frac', 'mappability', 'gene_density'] (GC available: True)
- confound control: covariate-only AUROC=0.610; residualized AUROC=0.615 (CI lo=0.574)
- beats chance: yes; perm-significant: yes; survives confound: yes
- FDR<0.05 differential features: 3
- n=600, d=128, raw_d=32768, preprocess=unsupervised_truncated_svd_128_from_32768, groups(chrom)=22, missing-feature rows=0
- held-out-chrom class balance: {'chr1:control': 50, 'chr1:sweep': 50, 'chr2:control': 50, 'chr2:sweep': 50}

**Verdict: REAL, non-confounded signal**

Audit caveat: an independent check fit TruncatedSVD inside each chromosome-held-out
fold; the sweeps signal did not weaken (nested-SVD AUROC 0.645; residualized
nested-SVD AUROC 0.638). The reported pipeline still uses full-data
unsupervised SVD for the heavy CV/permutation path, so the preprocessing should
be treated as a transductive dimensionality-reduction step. The significant
`svd_*` components support separability but are not directly interpretable as
individual Evo2-SAE dictionary features.

## B. Archaic introgression vs controls

- held-out AUROC (logreg, GroupKFold by chrom): **0.537** [0.489, 0.592]
- permutation (within-chromosome label shuffle) p = **0.1189**
- covariates: ['log_length', 'gc', 'repeat_frac', 'mappability', 'gene_density'] (GC available: True)
- confound control: covariate-only AUROC=0.605; residualized AUROC=0.494 (CI lo=0.446)
- beats chance: no; perm-significant: no; survives confound: NO
- FDR<0.05 differential features: 1
- n=600, d=128, raw_d=32768, preprocess=unsupervised_truncated_svd_128_from_32768, groups(chrom)=22, missing-feature rows=0
- held-out-chrom class balance: {'chr1:control': 50, 'chr1:introgression': 50, 'chr2:control': 50, 'chr2:introgression': 50}

**Verdict: NOT established beyond confound/chance**

Audit caveat: the stricter fold-local SVD check also kept introgression weak
(nested-SVD AUROC 0.550; residualized nested-SVD AUROC 0.536), and covariates
remained stronger than features.


## How to read this

A feature set carries genuine, non-confounded signal only when all hold: (1) held-out AUROC CI above 0.5, (2) permutation p<0.05, (3) features beat covariate-only AND residualized features stay above chance. If covariate-only is already high and residualized collapses, the apparent signal was length/GC/repeat/mappability/gene-density confound, not function.

Independent audit: `agents/aim2_adversarial_audit.md`.

## Explicit held-out chr1/chr2 test

The fixed test set is chr1+chr2. The model is trained on chr3-22 only; SVD is fit only on the training chromosomes.

- sweeps: SAE AUROC 0.610 [0.531, 0.688], covariates AUROC 0.598, residualized AUROC 0.604; feature permutation p=0.006993, residualized p=0.01199.
- introgression: SAE AUROC 0.536 [0.452, 0.615], covariates AUROC 0.596, residualized AUROC 0.525; feature permutation p=0.1838, residualized p=0.2448.

Summary plot: `plots/aim2_experiment2_summary.png`
UMAP plot: `plots/aim2_experiment2_umap.png`

