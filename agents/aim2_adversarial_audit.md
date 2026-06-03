# Aim 2 Adversarial Audit

## Findings

1. I did not disprove the selective-sweeps positive claim. The reported logreg AUROC is 0.624 [0.580, 0.681] with within-chromosome permutation p=0.000999, and the residualized-feature model remains above chance at 0.615 [0.574, 0.667]. Covariates explain some of the signal (covariate-only AUROC=0.610), but not all of it under the stated decision rule.

2. The introgression result validates as null/not established. Feature AUROC is 0.537 [0.489, 0.592], permutation p=0.1189, covariates-only is stronger at 0.605, and residualized features collapse to 0.494 [0.446, 0.544]. The one FDR-significant component does not rescue the classification result and is not evidence beyond the covariates/null.

3. Feature/id/table/manifest alignment looks clean. `features.npy` is `(1200, 32768)` float32, all finite, with 1200 unique `ids.txt` entries, no `__w32` ids, and `ids.txt` exactly matches `manifest_w8.jsonl` order and set. Both task tables have 600 unique ids, all covered by ids, manifest_w8, and `covariates_extra.tsv`; coordinate checks against manifest_w8 had zero mismatches.

4. Feature shape/sanity is acceptable but sparse. Row norms are nonzero (3.18 to 5.16), no NaN/inf values were present, and the matrix has expected SAE sparsity (`~85.3%` exact zeros). There are 18,557 zero-variance raw columns, so the effective feature space is much smaller than 32,768; the SVD preprocessing is doing real compression rather than just convenience.

5. GC/covariate alignment looks internally consistent. `gc.npy` has 1200 finite values aligned to the same ids, range 0.273 to 0.656. Extra covariates cover all 1200 8 kb ids plus all 1200 `__w32` ids, with no extras outside `manifest.jsonl`. I did not independently recompute GC from FASTA, so this is an alignment/range check, not a sequence-level remeasurement.

6. I found no split leakage in the model evaluation. `GroupKFold` groups by chromosome, and the actual folds are class-balanced at 60 positives / 60 controls each. The table `split` column marks chr1/chr2 as test, but the harness does not use that predefined split; it evaluates all 22 chromosomes through 5-fold chromosome-held-out CV. The docs' held-out-chrom balance line only reports the chr1/chr2 table split, not every CV fold.

7. The permutation unit is the corrected one for mixed-label chromosome groups. Results JSON records `permutation_unit: within_group`, and the code shuffles labels within each chromosome because chromosomes contain both classes. This preserves per-chromosome class composition while breaking sample-level feature-label association.

8. The SVD preprocessing creates a caveat, but not an observed failure. `analyze.py` fits unsupervised TruncatedSVD once on the full aligned matrix before CV, so test chromosomes influence the learned basis. A local stricter no-write check fitting SVD inside each GroupKFold split gave sweeps AUROC=0.645 and residualized nested-SVD AUROC=0.638; introgression remained weak/null with nested-SVD AUROC=0.550 and nested residualized AUROC=0.536. This suggests the sweeps claim is not an artifact of transductive SVD, but the published pipeline should still disclose or fix the preprocessing placement.

9. The docs match the JSON. `summary.json` agrees with task-level `results.json` for AUROC, permutation p-value, and FDR feature count; the rounded values in `docs/RESULTS_AIM2.md` match those JSON values.

## Residual Risks

- The sweeps effect is modest and close to the covariate baseline. The claim should be phrased as evidence of residual SAE signal, not as a large or cleanly mechanistic separation.
- Differential features are SVD components (`svd_*`), not original SAE feature ids. They support a separability claim but should not be interpreted as individual Evo2-SAE dictionary features without back-projection or a no-SVD rerun on nonconstant raw dimensions.
- GC was only checked for id/shape/range alignment locally. A stronger audit would recompute GC from the extracted sequences and compare exact values.
- The predefined `split=train/test` column is not the split used for reported model performance. This is not leakage, but it is a documentation ambiguity that could confuse readers expecting a single chr1/chr2 held-out test.

## Verdict

Selective sweeps: the positive claim survives this adversarial audit, with the important caveat that the effect is modest, partially covariate-adjacent, and currently based on transductive unsupervised SVD components.

Introgression: the null verdict is validated. The apparent feature signal does not beat the covariate baseline, fails the permutation/null checks, and collapses after confound control.
