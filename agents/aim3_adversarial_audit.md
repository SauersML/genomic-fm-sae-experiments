# Aim 3 adversarial audit

## Scope

This audit checks whether the Aim 3 expression-association result can be
explained by stale artifacts, row misalignment, ancestry, ALT-burden, or analysis
leakage. A separate `codex exec` audit was attempted, but the local Codex account
hit a usage limit before it could run; this document records the direct local
audit performed by the orchestrator.

## Artifact checks

- `data/aim3_assoc/features.npy` is real Azure output, not the older synthetic
  placeholder: shape `(4080, 32768)`, matching `meta.json`
  `{"n":4080,"dim":32768,"pool":"mean","kind":"region_seq","peak_mem_gb":17.32,
  "seqs_per_sec":0.686,"seconds":5951.8,"n_forward_seqs":4080}`.
- `FEATURES_READY` exists and records Azure validation.
- `ids.txt` has 4080 unique IDs and matches `manifest.jsonl` order exactly.
- Manifest structure is 8 genes x 255 samples x 2 haplotypes = 4080 rows.
- A feature sample was finite and nonzero during pull validation
  (`sample_norm_min=3.928`, `sample_meanabs=0.000872`).

## Controls and disproof attempts

- **Ancestry:** all 8 genes retain positive ancestry-residualized CV Spearman,
  and ancestry-only Spearman is near zero for most genes. This argues against a
  pure ancestry-confound explanation.
- **EUR-only sensitivity:** 7/8 genes pass the EUR-only CV signal criterion.
  GSTM1 does not: full CV Spearman is only 0.139 and EUR-only CI crosses zero.
  GSTM1 should not be counted as independently validated.
- **ALT-burden:** this is the strongest alternative explanation. ALT-count alone
  is competitive with or better than the SAE feature model on explicit held-out
  individuals for C17orf97, ERAP2, SLFN5, and ZNF266. Feature-over-ALT held-out
  advantages are clear mainly for POMZP3 (+0.376 Spearman), RPS26 (+0.097), and
  PEX6 (+0.067). Therefore the result is not yet a clean SAE-specific regulatory
  mechanism.
- **Preprocessing leakage:** per-gene TruncatedSVD is fit on the full feature
  matrix before CV. This is unsupervised and does not use expression labels, but
  it is transductive. A confirmation run should fit SVD inside each training fold
  and rerun the permutation/null analysis.
- **Feature interpretation:** reported significant dimensions are SVD components
  of the SAE matrix, not named Goodfire SAE dictionary features. Dictionary-level
  claims are unsupported.

## Verdict

Aim 3 is a real and promising pilot association: the haplotype SAE features
predict expression for most genes and survive ancestry controls. The result is
not yet sufficient for a mechanistic claim because local genotype/ALT burden
explains much of the same signal and beats features on several held-out genes.
The honest next step is fold-local preprocessing plus a stricter incremental
feature-over-ALT test with confidence intervals and gene-level FDR.
