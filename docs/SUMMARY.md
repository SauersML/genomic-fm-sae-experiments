# Cross-aim summary

## Overall conclusion

All three pilots now have real Evo2-SAE feature artifacts and completed local
analyses. The program found one clear null/confounded result, one modest
population-genetic signal, and one strong expression-association signal that is
still not cleanly separated from genotype burden.

The recurring lesson is that Evo2-SAE features encode broad local sequence and
genomic context very strongly. That makes them useful predictive features, but
it also makes confound controls decisive.

## Aim-level findings

- **Aim 1 SV consequence:** negative/confounded. The broad coding-disrupting
  contrast is nominally above chance (AUROC 0.586, permutation p=0.005), but
  covariates are much stronger (AUROC 0.802) and the key length-matched
  coding/splice-vs-intergenic contrast is not significant (AUROC 0.547,
  p=0.296, residualized AUROC 0.049).
- **Aim 2 selective sweeps:** modest positive. Sweep windows separate from
  matched controls (AUROC 0.624, permutation p=0.000999) and remain slightly
  positive after the available length/GC/repeat/mappability/gene-density
  controls (residualized AUROC 0.615). The effect is close to the covariate
  baseline and should be treated as a candidate signal, not proof of selection
  specificity.
- **Aim 2 introgression:** null/not established. Feature AUROC is 0.537,
  permutation p=0.119, and residualized AUROC is 0.494.
- **Aim 3 expression association:** strongest pilot. All 8 genes pass the
  primary CV permutation/FDR criterion and ancestry residualization; 7/8 pass
  EUR-only sensitivity. However, ALT-count baselines are competitive or better
  on explicit held-out individuals for C17orf97, ERAP2, SLFN5, and ZNF266, and
  GSTM1 is weak in EUR-only. The conservative verdict is "promising, not
  mechanistically proven."

## Cross-aim readout

- `results/cross_aim/master_auroc.csv` and
  `plots/cross_auroc_forest.png` compare raw features, covariates-only, and
  residualized features across all reported contrasts.
- Aim 2 top SVD directions are partly composition-driven: composition covariates
  explain R2=0.76 for SVD1 and R2=0.72 for SVD3
  (`results/cross_aim/aim2_svd_composition_r2.csv`).
- SAE activation is not sparse at the pooled-row level after mean pooling:
  median nonzero features are 4438.5 (Aim1), 4918.5 (Aim2), and 4861.5 (Aim3).
  The top-10 active feature sets overlap heavily across aims, including 10/10
  overlap between Aim2 and Aim3, consistent with dominant generic sequence
  content axes.
- Aim1 delta magnitude differs by consequence (Kruskal p=1.53e-08; coding vs
  other Mann-Whitney p=4.55e-08), but magnitude does not rescue the matched
  confound-adjusted classification result.
- Adding composition covariates to the Aim2 sweep feature model leaves AUROC in
  the same narrow range (0.621-0.626), reinforcing that the sweep result is
  modest and near the composition baseline rather than a large independent
  selection signature.
- Aim3 feature-vs-ALT held-out differences are mixed: POMZP3 (+0.376), RPS26
  (+0.097), and PEX6 (+0.067) favor features, while C17orf97 (-0.102), ERAP2
  (-0.237), SLFN5 (-0.017), and ZNF266 (-0.002) do not.

## Caveats

- Aim2 and Aim3 use full-data unsupervised SVD before CV in the main pipeline.
  The Aim2 audit ran fold-local SVD stress checks and the sweep signal survived;
  Aim3 still needs the same fold-local confirmation.
- Significant SVD components are not directly named Goodfire SAE dictionary
  features. Dictionary-level interpretation remains future work.
- No full-scale expansion or ESM2/InterPLM protein-delta extra was run after the
  three pilots; the A100 was deallocated once real Aim3 inference completed to
  stop cost.
- These are pilots. The next valid scale-up needs richer matching on
  recombination, background selection, variant density, gene density, repeats,
  mappability, and GC, with all preprocessing fit inside training folds.

## Primary artifacts

- `docs/RESULTS_AIM1.md`
- `docs/RESULTS_AIM2.md`
- `docs/RESULTS_AIM3.md`
- `agents/aim1_adversarial_audit.md`
- `agents/aim2_adversarial_audit.md`
- `agents/aim3_adversarial_audit.md`
- `results/cross_aim/master_auroc.csv`
- `plots/cross_auroc_forest.png`
- `plots/cross_svd_vs_composition.png`
- `plots/cross_pca_umap.png`
- `plots/cross_sae_activation.png`
- `plots/aim1_delta_by_consequence.png`
- `plots/cross_sweeps_covariate_path.png`
- `plots/cross_aim3_vs_altcount.png`
