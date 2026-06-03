# Genomic & protein foundation-model SAE experiments

Do sparse-autoencoder (SAE) features of genomic/protein foundation models carry
biologically meaningful signal? This repo runs three experiments using
**Evo 2** (genomic FM) and **ESM-2** (protein FM) with published SAEs, validated
against held-out truth with adversarial baselines.

## Aims

1. **Structural-variant functional consequence.** For each HPRC structural
   variant, run the **ref** and **alt** sequence windows through Evo 2 (7B),
   take layer-26 activations, encode with the
   [`Goodfire/Evo-2-Layer-26-Mixed`](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed)
   SAE, and compute the **ref→alt feature delta**. Question: do the deltas
   separate SVs by functional consequence (coding / regulatory / intergenic)?
   Validated against GENCODE / ENCODE cCRE / ClinVar labels.

2. **Selection & introgression.** Take regions population genetics has already
   flagged — selective sweeps and archaic-introgression segments — and ask
   whether their SAE feature content separates them from matched controls.
   Held out by chromosome. Two tasks: selection, introgression.

3. **Feature-based association.** Represent individual haplotypes by SAE feature
   profiles across loci and associate them with an outcome
   (Geuvadis LCL expression; 1000G GRCh38 phased haplotypes). Held out by
   individual.

## Models

| role | model | SAE |
|---|---|---|
| genomic FM | `evo2_7b` (StripedHyena2, d_model 4096) | `Goodfire/Evo-2-Layer-26-Mixed` — tied BatchTopK, k=64, d_sae=32768, hook `blocks.26.mlp.l3` |
| protein FM | `facebook/esm2_t33_650M_UR50D` | `Elana/InterPLM-esm2-650m` — ReLU SAE, d_sae=10240, layer 24 |

## Layout

```
src/evo2/extract.py     Evo2 layer-26 activations + SAE encode; feature_delta / features_for_regions
src/esm2/extract.py     ESM2 embeddings + InterPLM SAE; features_for_proteins
src/modal_app.py        Modal A100 app exposing the extraction functions
src/common/analysis.py  Held-out eval harness: group-aware CV, AUROC/AUPRC+CIs,
                        permutation nulls, confound residualization, differential features, plots
src/common/*            SV labeling, GENCODE/cCRE feature builders, SV-window fetch
src/aim3_assoc/*        haplotype reconstruction + association skeleton
docs/                   per-component references and data provenance
```

Data (HPRC SVs, 1000G haplotypes, Geuvadis, sweep/introgression tracks,
GENCODE/ENCODE annotations) is **public** and reproduced by the scripts here; it
is not committed. See `docs/DATA_*.md` for exact sources and commands.

## Compute

GPU inference runs on A100-80GB (Evo 2 7B in bf16) via Modal and/or a persistent
cloud A100. The analysis harness is pure-CPU.

## Methodological stance

Every reported number comes with a held-out test set, a bootstrap CI, a
label-permutation null, and a confounder-only baseline (e.g. SV length, GC,
ancestry) — so "the SAE features separate X" is only claimed when features beat
both chance and the obvious confound. Results are reviewed adversarially.

> Work in progress.
