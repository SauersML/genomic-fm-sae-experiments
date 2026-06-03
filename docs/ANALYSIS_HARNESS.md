# Analysis & evaluation harness — `src/common/analysis.py`

A shared, **task-generic** statistical harness for all three aims. It takes a
feature matrix `X`, labels `y`, optional `groups` (for leakage-safe splitting)
and optional `covariates` (for confound control), and answers one question
defensibly: **does `X` carry real, non-confounded signal about `y`?**

Built for adversarial scrutiny:
- Every reported number comes with a **bootstrap 95% CI** or a **null**.
- Always compares features against a **shuffled-label** baseline (permutation
  test) and a **covariate-only** baseline + **residualized-feature** model.
- **Group-aware splitting** everywhere — no group leaks across train/test.
- Functions are **pure** (no global state) and importable.

Pure CPU, headless matplotlib (`Agg`). UMAP is optional and skipped gracefully.

## Install

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV="$PWD/.venv" uv pip install -r src/common/requirements.txt
```

## Inputs (shared by every function)

| arg | shape | meaning |
|---|---|---|
| `X` | `(n_samples, n_features)` | SAE feature vectors, ref/alt feature **deltas**, or per-locus haplotype feature profiles. 1-D is reshaped to a single column. |
| `y` | `(n_samples,)` | label: binary (classification) or continuous (regression). |
| `groups` | `(n_samples,)` or `None` | group id (sample / haplotype / locus / contig). No group is allowed to span train and test. If `None`, falls back to stratified (classification) or plain (regression) K-fold. |
| `covariates` | `(n_samples, n_cov)` or `None` | the "obvious" confounders to control for (e.g. SV length, GC content). |
| `task` | `'classification'` \| `'regression'` | — |

## Public API

```python
evaluate_separation(X, y, groups=None, covariates=None,
                    task='classification', seed=0, n_splits=5, n_boot=1000) -> dict
permutation_test(X, y, groups=None, covariates=None, task='classification',
                 metric=None, model=None, n_perm=1000, seed=0, n_splits=5) -> dict
differential_features(X, y, task='classification', feature_names=None) -> pandas.DataFrame
make_plots(X, y, outdir, eval_result=None, diff_table=None,
           task='classification', seed=0) -> list[str]
run_report(X, y, groups=None, covariates=None, outdir='.', title='analysis',
           task='classification', seed=0, feature_names=None,
           n_perm=1000, n_splits=5, n_boot=1000) -> dict
```

### 1. `evaluate_separation`
Group-aware cross-validated evaluation of how well `X` predicts `y`.

- **Split:** `GroupKFold` when `groups` given; else `StratifiedKFold`
  (binary classification) or `KFold` (regression). `n_splits` is auto-capped
  to the number of groups.
- **Models** (held-out, out-of-fold predictions):
  - classification -> **L2 logistic regression** (standardized, class-balanced)
    and **HistGradientBoostingClassifier**.
  - regression -> **Ridge** (standardized) and **HistGradientBoostingRegressor**.
- **Metrics with bootstrap 95% CIs:** AUROC + AUPRC (classification) or
  R2 + Spearman (regression). The bootstrap is a **cluster bootstrap over
  groups** when groups are present (so CIs reflect group-level, not
  sample-level, resampling).
- **Confound control** (when `covariates` given), returns two extra blocks:
  - `covariates_only` — same models trained on the covariates alone. This is
    the bar a feature set must clear.
  - `features_residualized` — features after **out-of-fold linear
    residualization** of the covariates (fit on train folds, subtract on the
    test fold — no leakage). Tells us whether features add signal *beyond* the
    covariates.

Returns a nested dict; each metric is `{value, value_ci95: (lo, hi)}`.

### 2. `permutation_test`
Label-shuffle null for the held-out metric (default AUROC / Spearman).
- The observed statistic is the out-of-fold metric of one baseline model
  (default: logreg / ridge for the task).
- When `groups` are given and each group has one label, labels are permuted
  **at the group level** so the null preserves the group block structure. When
  groups contain mixed labels (e.g. chromosome-held-out SV/popgen analyses),
  labels are shuffled **within each group** instead; this preserves per-group
  class/outcome composition while breaking sample-level feature-label
  association.
- Returns `observed`, `null_mean`, `null_std`, `n_perm_effective`, and a
  one-sided `p_value = (#{null >= observed} + 1) / (n + 1)`.

### 3. `differential_features`
Per-feature univariate association with **Benjamini-Hochberg FDR**.
- binary -> Mann-Whitney U (rank-based, robust) + mean-difference effect size.
- continuous -> Spearman rho.
- Returns a DataFrame ranked by `p_adj_bh`, columns: `feature`, the test
  statistic, the effect, `p_value`, `p_adj_bh`, `direction` (up/down).

### 4. `make_plots`
Saves to `outdir` and returns the file paths:
`pca.png`, `umap.png` (if `umap-learn` is importable, else skipped),
`roc.png` + `calibration.png` (classification), `volcano.png`,
`top_features.png` (from the differential table).

### 5. `run_report`
Runs 1-4 and writes `report.md`, `results.json`, `differential_features.csv`,
and all plots into `outdir`. Returns the results dict.

## How each aim calls it

**Aim 1 — SV functional consequence.** `X` = Evo2 layer-26 SAE **ref-vs-alt
feature deltas** per SV; `y` = functional-consequence label; `groups` = the
HPRC sample / contig the SV came from (so the same locus can't leak across
splits); `covariates` = **SV length** and local **GC content** (the obvious
confounds). The `covariates_only` and `features_residualized` blocks are the
headline result — they show whether the SAE deltas separate SVs *beyond* size
and GC.
```python
run_report(deltas, consequence, groups=sample_id,
           covariates=np.c_[sv_len, gc], outdir='results/aim1_sv',
           title='aim1_sv_consequence', task='classification')
```

**Aim 2 — Selection & introgression.** `X` = SAE feature content of a region;
two separate runs for the two binary labels `y` (selection sweep; archaic
introgression); `groups` = genomic region/contig to keep linked regions
together; `covariates` = region length, GC, mean coverage. Held-out test set =
the held-out group folds. Use the permutation p-value + covariate-only baseline
to argue the signal is real and not a mappability/length artifact.

**Aim 3 — Feature-based association.** `X` = per-locus haplotype feature
profiles; `y` = the outcome (expression / phenotype -> `task='regression'`; or a
selection statistic). `groups` = haplotype or individual id.
`differential_features` surfaces which feature dimensions drive the
association; the bootstrap-CI R2/Spearman plus the permutation null are the
defensible effect estimates.

## Reading the outputs (what counts as a real result)

A feature set carries genuine, non-confounded signal when **all** hold:
1. `separation.features` metric CI sits **above chance** (0.5 AUROC / 0 R2-Spearman),
2. `permutation_test.p_value` is **small** (e.g. < 0.05) — beats the label-shuffle null,
3. with covariates: `features` beats `covariates_only`, **and**
   `features_residualized` **stays above chance** — i.e. the signal survives
   removing the obvious confound.

If `covariates_only` is already near-perfect and `features_residualized`
collapses to chance, the apparent signal was the confound. (The self-test's
Case 3 demonstrates exactly this.)

## Self-test

`python src/common/analysis.py` runs four synthetic checks and asserts:
1. **signal** present -> high AUROC, low permutation p,
2. **pure noise** -> permutation p not significant, 0 FDR-significant features,
3. **confounded** -> `covariates_only` ~ perfect while `features_residualized`
   collapses (confound caught),
4. **regression** sanity -> high Spearman, low p.

Latest run (seed 0):

```
CASE 1 signal:     AUROC=0.990  perm p=0.0033  sig_feats=3
CASE 2 noise:      AUROC=0.516  perm p=0.362   sig_feats=0
CASE 3 confound:   features AUROC=0.984  covariate-only=1.000  residualized=0.275
CASE 4 regression: Spearman=0.978  perm p=0.0033
ALL SELF-TESTS PASSED   (UMAP available: True)
```
