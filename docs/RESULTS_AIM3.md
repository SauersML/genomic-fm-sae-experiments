# RESULTS -- Aim 3: expression association from Evo2-SAE haplotype features

## Verdict

The pilot beats chance and the ancestry-only baseline for all 8 genes under the current CV/permutation criterion, and the EUR-only sensitivity remains positive for 7/8 genes. However, this is **not yet a clean SAE-specific regulatory mechanism**: ALT-burden-only baselines are already strong for several genes, and the current aggregate only records a positive ALT-burden-residualized Spearman, not a CI/FDR-tested incremental effect.

This is the strongest pilot result in the program, but the conservative call is
"promising, not mechanistically proven." On the explicit held-out split,
ALT-count alone is competitive with or better than the SAE feature model for
C17orf97, ERAP2, SLFN5, and ZNF266. The clearest feature-over-ALT held-out
advantages are POMZP3, RPS26, and PEX6. GSTM1 should be treated as weak: it is
positive in the full CV aggregate but its EUR-only feature CI crosses zero.

The current analysis uses full-data unsupervised TruncatedSVD from 32768 SAE
features to 128 components before cross-validation. That is not label-supervised
leakage, but it is transductive preprocessing. A confirmation run should use
fold-local SVD and should test the incremental feature-over-ALT residual with a
CI/FDR rule, not just the sign of residualized Spearman.

## Key Counts

- genes analysed: 8
- CV signal after gene-level FDR: 8/8
- ancestry-residualized signal: 8/8
- positive residual after ALT-burden baseline: 8/8
- EUR-only CV signal: 7/8

## Per-Gene Summary

| gene | feature Spearman | ancestry-only | residualized ancestry | ALT-burden only | residualized ALT-burden | perm q |
|---|---:|---:|---:|---:|---:|---:|
| `C17orf97|ENSG00000187624` | 0.737 | 0.051 | 0.737 | 0.727 | 0.681 | 0.002 |
| `ERAP2|ENSG00000164308` | 0.605 | -0.141 | 0.570 | 0.580 | 0.457 | 0.002 |
| `GSTM1|ENSG00000134184` | 0.139 | -0.100 | 0.233 | 0.139 | 0.137 | 0.002 |
| `PEX6|ENSG00000124587` | 0.795 | -0.135 | 0.802 | 0.648 | 0.681 | 0.002 |
| `POMZP3|ENSG00000146707` | 0.754 | -0.048 | 0.767 | 0.580 | 0.751 | 0.002 |
| `RPS26|ENSG00000197728` | 0.825 | 0.049 | 0.818 | 0.493 | 0.779 | 0.002 |
| `SLFN5|ENSG00000166750` | 0.726 | -0.063 | 0.713 | 0.756 | 0.602 | 0.002 |
| `ZNF266|ENSG00000174652` | 0.777 | -0.007 | 0.775 | 0.756 | 0.645 | 0.002 |

## Reproduce

```bash
.venv/bin/python src/aim3_assoc/analyze.py --n-perm 500
```

The full generated harness report is in `results/aim3_assoc/report.md`.
Independent audit: `agents/aim3_adversarial_audit.md`.
