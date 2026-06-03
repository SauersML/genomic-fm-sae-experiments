# Aim 2 Popgen Analysis Outputs

This directory is reserved for the CPU-only Aim 2 analysis outputs.

Run after the GPU embedding job has copied all required files into
`data/aim2_popgen/`:

```bash
.venv/bin/python -m src.aim2_popgen.analyze --n-perm 1000
```

Required real-run inputs:

- `data/aim2_popgen/features.npy`: 2-D SAE feature matrix.
- `data/aim2_popgen/ids.txt`: one feature row id per line, aligned to
  `features.npy`.
- `data/aim2_popgen/gc.npy`: 1-D GC vector aligned to `ids.txt`.
- `data/aim2_popgen/table_sweeps.tsv`.
- `data/aim2_popgen/table_introgression.tsv`.
- `data/aim2_popgen/covariates_extra.tsv`: must contain `repeat_frac`,
  `mappability`, and `gene_density` for every analyzed 8 kb row.

Expected outputs from the command:

- `results/aim2_popgen/sweeps/`
- `results/aim2_popgen/introgression/`
- `results/aim2_popgen/summary.json`
- `docs/RESULTS_AIM2.md`
