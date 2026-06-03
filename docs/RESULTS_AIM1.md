# RESULTS -- Aim 1: HPRC SV ref/alt Evo2-SAE deltas

## Verdict

The pilot does **not** support a robust, non-confounded claim that Evo2 layer-26
Goodfire SAE ref->alt deltas separate HPRC SVs by functional consequence.

There is weak apparent signal in the broad coding-disrupting contrast, but the
obvious covariates are stronger than the SAE deltas, and the critical
length-matched coding-vs-intergenic control is not significant under the
corrected within-chromosome permutation null. After residualizing the deltas
against SV length/type plus repeat, mappability, and gene-density covariates,
the linear residualized signal collapses below chance.

## Data And Features

- SVs: 700 HPRC SVs, balanced across 7 consequence classes.
- Feature matrix: `data/aim1_sv/features.npy`, shape `(700, 32768)`.
- Model: Evo2-7B layer `blocks.26.mlp.l3`.
- SAE: `Goodfire/Evo-2-Layer-26-Mixed`, BatchTopK k=64, 32768 features.
- Feature: mean-pooled `SAE(alt_window) - SAE(ref_window)`.
- Leakage control: chromosome-grouped CV (`24` groups).
- Confound controls: `log_svlen`, `svtype_ins`, `repeat_frac`, `mappability`,
  `gene_density`. `gc_window` was unavailable/all-NaN and was dropped.
- Permutation null: labels shuffled within chromosome for mixed-label chromosome
  groups; this preserves per-chromosome class composition while breaking
  sample-level feature-label association.
- Wide-feature handling: unsupervised SVD to 128 components for CV/permutation;
  univariate differential-feature tests ran on full 32768-d deltas.

## Key Results

| Contrast | n | Feature AUROC | Perm p | Covariate AUROC | Residualized AUROC | FDR<0.05 features | Call |
|---|---:|---:|---:|---:|---:|---:|---|
| coding-disrupting vs not | 700 | 0.586 [0.524, 0.654] | 0.0050 | 0.802 | 0.418 | 362 | apparent but confounded |
| coding/splice vs intergenic, raw | 300 | 0.610 [0.548, 0.675] | 0.011 | 0.990 | 0.263 | 0 | dominated by covariates |
| coding/splice vs intergenic, length-matched | 166 | 0.547 [0.480, 0.636] | 0.296 | 0.991 | 0.049 | 0 | null after adversarial control |
| cds vs intronic, length-matched | 124 | 0.362 [0.255, 0.455] | 0.976 | 0.830 | 0.174 | 0 | null |
| splice vs intergenic, length-matched | 106 | 0.522 [0.424, 0.627] | 0.197 | 0.975 | 0.044 | 0 | not credible beyond covariates |
| cds vs intergenic, length-matched | 132 | 0.432 [0.335, 0.570] | 0.524 | 0.995 | 0.014 | 0 | null |

The histogram-gradient model sometimes recovered higher residualized AUROC than
the linear model, but the pre-registered conservative bar was not met: features
must beat covariates and remain above chance after confound removal in the
length-matched contrast.

## Adversarial Interpretation

The strongest disproof is the covariate-only baseline. In raw coding-vs-
intergenic comparisons, covariates alone reach nearly perfect AUROC (~0.99),
which means the task is mostly encoded by SV length/type and genomic context.
When coding and intergenic SVs are length-matched, the SAE delta AUROC drops to
0.547 with confidence interval crossing 0.5 and corrected permutation p=0.296.

The primary broad contrast has a significant permutation p, but it is still
weaker than covariates-only and does not survive the stricter matched test. The
honest conclusion is that this pilot mostly detects known annotation and SV
composition confounds rather than a validated feature-delta consequence signal.

An independent adversarial audit is recorded in `agents/aim1_adversarial_audit.md`.
It found no local row-alignment or stale-feature problem, but it did identify
that the original group-level permutation null was invalid for mixed-label
chromosome groups. The shared harness was patched and this document reflects the
rerun with the corrected within-chromosome null.

## Reproduce

```bash
.venv/bin/python src/aim1_sv/analyze.py --n-perm 1000 --n-boot 1000 --svd-k 128
```

Outputs:

- `results/aim1_sv/results.json`
- `results/aim1_sv/*/report.md`
- `results/aim1_sv/*/{roc,pca,umap,volcano,top_features}.png`
