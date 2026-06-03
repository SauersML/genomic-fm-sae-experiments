# Aim 1 Adversarial Audit

## Findings

1. **Local artifact alignment passes.** `features.npy` has shape `(700, 32768)` with `float32` values, matching `docs/RESULTS_AIM1.md` and `results/aim1_sv/results.json`. `ids.txt`, `manifest.jsonl`, `labels.parquet`, and `covariates_extra.tsv` each contain 700 unique IDs; `ids.txt` order matches both manifest and label order. This does not disprove the reported row alignment.

2. **Feature matrix sanity passes basic checks.** All 22,937,600 feature values are finite, no rows are all-zero, and no exact duplicate feature rows were found. The matrix is sparse-ish, with 86.7% exact zeros, which is plausible for mean-pooled BatchTopK SAE ref-to-alt deltas. Row norms range from `0.00650` to `0.44320` with median `0.04908`.

3. **No obvious stale or synthetic artifact was detected locally.** `features.npy` and `ids.txt` were modified after `manifest.jsonl` and `labels.parquet`; `results/aim1_sv/results.json` was modified after the feature files; `docs/RESULTS_AIM1.md` was modified after results. `meta.json` reports `n=700`, `dim=32768`, `kind=delta_coord`, `pool=mean`, and `n_forward_seqs=1400`, consistent with ref/alt processing. However, this is provenance evidence, not proof that the remote sequence construction followed `MANIFEST_SPEC.md`.

4. **The headline numeric table matches `results.json` to rounding for the linear/logistic model.** The reported AUROC, CI, permutation p, covariate AUROC, residualized AUROC, sample sizes, and FDR feature counts correspond to `l2_logreg` entries in `results/aim1_sv/results.json`. The document appropriately omits but mentions stronger `hist_gbt` values, including primary residualized AUROC `0.653` and raw coding-vs-intergenic residualized AUROC `0.678`.

5. **Major methodological flaw: grouped permutation p-values are not valid for these data.** `src/common/analysis.py::permutation_test` uses the first label observed in each chromosome group, permutes those group labels, then assigns one label to every sample in that chromosome. But labels vary within every chromosome for the primary contrast, and within most chromosomes for length-matched contrasts. Example observed vs induced group-label positive rates:
   - primary: observed `0.286`, group-label induced `0.197`
   - raw coding/intergenic: observed `0.667`, induced `0.797`
   - length-matched coding/intergenic: observed `0.500`, induced `0.699`
   This null changes class balance and label structure, so the reported permutation p-values should be considered unreliable. The significant `splice_vs_intergenic_lenmatched` p=`0.014` is especially suspect because its null mean is `0.324`, far below chance.

6. **Chromosome group split itself is mostly appropriate, but not fully leak-free.** Held-out evaluation uses `GroupKFold` by chromosome, so no chromosome appears in both train and test. However, SVD is fit once on all rows in each contrast before CV, using test-fold feature distributions. This is unsupervised and not label leakage, but it is still cross-fold preprocessing leakage. ROC plots are also generated with non-grouped splits, though the reported metrics come from grouped splits.

7. **Length matching is valid only for `log_svlen`, not for broader confounding.** The matched coding/intergenic subset has closely aligned log-length medians (`2.086` coding vs `2.076` intergenic), reproducing `results.json`. But other covariates remain strongly separated: insertion fraction `0.410` vs `0.265`, repeat fraction mean `0.336` vs `0.583`, and gene density mean `0.850` vs `0.043`. This explains why covariates-only AUROC remains near-perfect (`0.991`) after length matching. The length-matched test is useful, but it is not a matched design for annotation context or SV type.

8. **Best alternate positive interpretation is weak and nonlinear.** The strongest pro-signal read is that tree models recover some residualized AUROC in broader contrasts after linear covariate residualization. That could indicate nonlinear SAE-delta signal not captured by linear residualization. It is not enough to overturn the reported conclusion because the critical length-matched coding/intergenic residualized tree model is only `0.394` AUROC with CI crossing chance, and covariates dominate all coding/intergenic contrasts.

## Residual Risks

- The local files cannot prove that `features.npy` was generated from the exact ref/alt windows required by `MANIFEST_SPEC.md`; there is no per-row sequence hash, feature extraction log, model/SAE hash, or manifest hash tying the artifact to the spec.
- `gc_window` is absent/all-NaN and therefore dropped. The analysis controls repeat, mappability, gene density, SV length, and insertion status, but not GC.
- Residualized AUROCs far below 0.5 suggest unstable or inverted cross-chromosome generalization rather than clean absence of signal. This supports skepticism, but it also means "collapse below chance" should not be overinterpreted as a calibrated negative effect.
- Since gene density is almost definitional for coding vs intergenic labels, covariate control may remove biologically meaningful location information as well as confounding. That is acceptable for the stated conservative claim, but it narrows the claim being tested.

## Verdict

I could not disprove the main reported interpretation that Aim 1 does not show a robust, non-confounded SAE-delta consequence signal. The local artifact alignment and result transcription are sound, and the critical length-matched linear feature result is weak.

The strongest adverse finding is that the reported permutation p-values are methodologically invalid under chromosome grouping and should not be used as evidence. Even ignoring permutation p-values, the substantive conclusion remains negative: covariates dominate, length matching only fixes length, and the residualized length-matched feature signal does not credibly beat chance.

## Addendum After Harness Patch

After this audit, `src/common/analysis.py::permutation_test` was patched so mixed-label groups are permuted within group rather than collapsed to the first group label. Aim 1 was rerun with the corrected null. The negative interpretation was unchanged: the broad primary contrast remains nominally significant (`p=0.0050`), but the critical coding/intergenic length-matched contrast is not significant (`p=0.296`) and remains dominated by covariates.
