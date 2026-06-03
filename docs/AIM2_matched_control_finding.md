# Aim 2 — what is actually happening with the sweep "positive" (matched-control investigation)

The reported sweeps result (held-out AUROC 0.624, perm p=0.001, "survives confound")
used only **linear residualization on 4 scalar covariates**. That is too weak a control.
A stronger adversarial test — **1:1 nearest-neighbor matching** of controls to sweeps on
[repeat_frac, mappability, gene_density, GC], then the same GroupKFold(chrom) classifier —
substantially erodes the signal:

| Test | Features AUROC | Covariate-only AUROC |
|---|---:|---:|
| Unmatched (as reported) | 0.625 | 0.610 |
| **Composition-matched controls (n=534)** | **0.581** | 0.535 |

## Mechanism (what the classifier is really seeing)
- Sweep regions are a **biased genomic sample**: more **genic** (gene_density 0.68 vs 0.58)
  and more **repeat-rich** (0.58 vs 0.52) than the controls.
- Controls are **random background** (bedtools shuffle), so they are depleted of exactly those
  properties → trivially separable by any composition-aware model.
- The separating directions are the **top SVD components** (svd_1, svd_2) of the SAE space —
  generic sequence composition, not specific interpretable SAE dictionary features.
- Matching cannot fully close the gene_density gap (random controls lack enough genic regions;
  n falls to 534, gene_density still 0.68 vs 0.62), so the residual 0.581 is consistent with
  **un-matched genic context, not a selection signature**.

## Corrected verdict
**NOT established as selection-specific.** Evo2 SAE features distinguish the *genomic
neighborhood* where sweeps were called (genic, repeat-rich), not selection itself — the same
confound that nulled Aim 1 (where length-matching killed the apparent signal).

## What a valid test would require
Sweep-vs-**non-sweep** controls matched on gene density + recombination rate +
background-selection (B-statistic) + repeat content — NOT random genomic shuffles. Under such
matching I predict the residual 0.581 erodes further toward chance.

## Note on the complementary audit
A separate fold-local/nested-SVD audit (`agents/aim2_adversarial_audit.md`) ruled out
*transductive SVD leakage* (signal survived: 0.645). That is a different confound; it does not
address the control-selection confound documented here, which is the dominant effect.
