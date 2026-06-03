# Final verdict

The three requested pilots are complete with real Evo2-SAE features.

- Aim 1: **not validated**. Apparent coding-disruption signal is weaker than
  covariates and fails the critical length-matched control.
- Aim 2 sweeps: **candidate signal**. The sweep contrast beats chance and the
  available covariate controls, but the margin is modest and composition axes are
  prominent.
- Aim 2 introgression: **not validated**. The result does not beat chance or
  confound controls.
- Aim 3: **promising association signal**. Haplotype SAE features predict
  expression across the pilot genes and survive ancestry controls, but genotype
  ALT-count remains a hard baseline and several genes do not beat it on explicit
  held-out individuals.

The Azure A100 inference run is complete and the VM has been deallocated. No
additional GPU scale-up was run after Aim 3 because the pilot evidence did not
justify more paid inference before tightening controls and fold-local
preprocessing.
