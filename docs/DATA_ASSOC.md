# Aim-3 Association — DATA assembly

**Goal:** Assemble a public, individual-level dataset to test
*"haplotypes represented by SAE feature profiles across loci predict an outcome,"*
with a held-out test set. This document covers the **DATA skeleton**: outcome
choice + justification, exact sources/commands, sample/locus counts, how
haplotype sequences are reconstructed, and the split definition (with seed).

The Evo2-SAE feature extraction itself (running each reconstructed haplotype
sequence through Evo2-7B layer-26 + the Goodfire SAE) is the GPU agent's job;
here we produce everything needed to feed it.

---

## 1. Chosen outcome + justification

**Outcome = Geuvadis LCL gene expression (Outcome option A).**

Why this over the alternatives:

- **Public, individual-level, no controlled access.** Geuvadis RNA-seq (Lappalainen
  et al., *Nature* 2013) and the 1000 Genomes 30x phased panel are both fully
  public. GTEx genotypes are controlled-access (dbGaP), so GTEx was rejected for
  the individual-level haplotype→outcome design.
- **Phased haplotypes available for the same individuals.** The 1000G 30x NYGC
  panel provides **phased** SNV/indel/SV genotypes on **GRCh38** for 3202
  individuals; **449** of the Geuvadis 462 expression samples are in it (see §3).
  Phasing is essential — the hypothesis is about *haplotypes*, and we need to
  reconstruct each of an individual's two haplotype sequences per locus.
- **Strong, well-characterized cis-eQTLs give a real signal to detect.** We pick
  the 100 genes with the strongest cis-eQTL (e.g. ERAP2, RPS26, GSTM1) so that
  haplotype identity genuinely carries expression-predictive information — a
  fair test of whether SAE feature profiles capture it.
- **Per-gene continuous target** (expression level) is a clean regression target,
  one outcome vector per locus, directly comparable across train/val/test.

Outcome B (a selection statistic like iHS/nSL as a per-locus continuous target)
remains a viable *secondary* design and is compatible with the same locus/
haplotype machinery; it is **not** built here. Outcome C (phenotype) was
deprioritized (no public individual-level genotype+phenotype at this scale).

---

## 2. Sources & exact commands

All access is public (HTTP + remote tabix; `--no-sign-request` not even needed
since these are EBI HTTP endpoints, not S3). No AWS credentials used.

### 2a. 1000G 30x **phased** panel — GRCh38 (NYGC, via EBI)
Base URL:
```
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
file pattern: 1kGP_high_coverage_Illumina.<chr>.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
```
- 3202 samples, contigs are **chr1..chr22** (chr-prefixed), GRCh38.
- We **never download whole chromosomes**; we stream only each cis-window for the
  449 individuals with remote `bcftools view -r <region> -S samples.txt`.

### 2b. Geuvadis expression + cis-eQTL (EBI BioStudies / ArrayExpress E-GEUV-1)
Base URL:
```
https://ftp.ebi.ac.uk/biostudies/fire/E-GEUV-/001/E-GEUV-1/Files/E-GEUV-1/analysis_results/
```
Files used (downloaded into `data/assoc/`):
- `GD462.GeneQuantRPKM.50FN.samplename.resk10.txt.gz` → **expression matrix**,
  23 722 genes × 462 samples. Normalization = library-depth-scaled RPKM, units
  with 0 counts in >50% samples removed, **PEER-corrected** (per README; values
  may be slightly negative — expected, not an error). This is the canonical
  Geuvadis gene-level matrix (the extra standard-normal transform was reverted in
  the 2013-11-05 update).
- `EUR373.gene.cis.FDR5.best.rs137.txt.gz` → **best cis-eQTL per gene** (EUR373),
  used to rank genes by cis-eQTL strength. 3258 genes at FDR<5%.
  (YRI89 equivalent also downloaded for reference; 500 genes.)
  Columns (no header; see Geuvadis README): `SNP_ID, ID(null), GENE_ID, PROBE_ID,
  CHR_SNP, CHR_GENE, SNPpos, TSSpos, Distance, rvalue(Spearman rho, signed by
  non-ref allele), pvalue, log10pvalue`.

### 2c. GRCh38 gene model (GENCODE v44) — for cis-window coordinates
```
https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.basic.annotation.gtf.gz
```
Streamed and reduced to gene-level rows (`/tmp/gencode_v44_genes.tsv`):
`ensembl_id  chr  start  end  strand  symbol  gene_type`.

### 2d. 1000G population panel (for split composition / QC)
```
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt
```

> **Coordinate provenance — important.** Geuvadis expression/eQTL coordinates are
> **hg19 / GENCODE v12**. The phased panel is **GRCh38**. We therefore do **not**
> trust the hg19 positions for windowing. We key everything by **Ensembl gene ID**
> (version-stripped) and take the cis-window from **GENCODE v44 GRCh38** TSS, so
> the window and the VCF are both GRCh38-consistent. Expression is matched to a
> gene purely by Ensembl ID (no coordinate dependence). The hg19 eQTL SNP/TSS
> positions are retained in `loci.tsv` for reference only.

---

## 3. Counts

| Quantity | Value |
|---|---|
| Geuvadis expression samples | 462 |
| 1000G 30x phased panel samples | 3202 |
| **Overlap (usable individuals)** | **449** |
| Superpopulations of the 449 | EUR 360 (GBR 86, FIN 92, CEU 91, TSI 91), AFR 89 (YRI 89) |
| Loci (genes) selected | 100 |
| Locus selection | top-100 by EUR373 best-cis-eQTL `log10p`, protein-coding, autosomal, Ensembl-ID mappable to GENCODE v44 |
| cis-window | TSS ± 100 kb (200 kb total) |
| Phased variants per window | ~4 500–6 000 biallelic SNV/indel (SVs and `*` spanning-dels dropped) |
| (individual × haplotype × locus) cells | 449 × 2 × 100 = **89 800** haplotype-window sequences |

The 13 Geuvadis samples absent from the 30x panel are dropped (not re-sequenced
at 30x / ID mismatch); the panel→Geuvadis join is by exact sample ID.

---

## 4. Output files (`data/assoc/`)

| File | Contents |
|---|---|
| `loci.bed` | GRCh38 cis-windows, `chr<N>  start0  end  locus_id` (chr-prefixed to match panel) |
| `loci.tsv` | full locus metadata: GRCh38 window/TSS/strand, gene id+symbol, eQTL SNP, `log10p`, `rvalue`, hg19 eQTL coords (reference only) |
| `samples.txt` | the 449 individuals (intersection), one per line |
| `samples_pop.tsv` | `sample  population  superpopulation` for the 449 |
| `expression.tsv` | 100 loci × 449 samples, PEER-normalized RPKM, aligned to `samples.txt`; cols `locus_id, ens_gene, <449 sample cols>` |
| `splits.json` | train/val/test individual lists + seed + counts |
| `haplotypes/<locus>.vcf.gz(.csi)` | per-locus phased VCF (449 samples) — the variants to reconstruct haplotype sequences |
| `haplotypes_manifest.tsv` | per-locus record counts + paths |
| `GD462.GeneQuantRPKM.txt.gz`, `EUR373/YRI89.gene.cis.FDR5.best.txt.gz` | raw source files (kept for provenance) |

Scripts (`src/aim3_assoc/`):
- `build_assoc_skeleton.py` — selects loci, builds `loci.{bed,tsv}`, `expression.tsv`, `samples.txt`, `splits.json`.
- `extract_haplotypes.sh` — streams per-locus phased VCFs from the remote panel (resumable).
- `reconstruct_haplotypes.py` — applies phased haplotypes to the GRCh38 reference → haplotype window FASTAs (GPU agent input).

---

## 5. Haplotype sequence reconstruction (ref FASTA + phased variants)

The modeling table is per **(individual, haplotype, locus)**. The Evo2-SAE
feature profile is computed on the *DNA sequence* of that haplotype's cis-window.
We reconstruct it by applying the individual's phased alleles to the GRCh38
reference window — the standard, indel-correct way via `bcftools consensus`:

```
# one-time: GRCh38 primary reference with chr-prefixed contigs, e.g.
#   GRCh38_full_analysis_set_plus_decoy_hla.fa  (the panel's reference)
#   from .../technical/reference/GRCh38_reference_genome/
samtools faidx GRCh38.fa

# per locus window <chr:start-end> and per sample S, haplotype H in {1,2}:
samtools faidx GRCh38.fa chr12:55941351-56141351 > window.fa
bcftools consensus -H H -s S -f window.fa data/assoc/haplotypes/RPS26__ENSG00000197728.vcf.gz
#   -> the S/H haplotype sequence of the window (ref with that haplotype's
#      phased SNVs/indels applied). FASTA header rewritten to  S|hH|locus_id.
```

`reconstruct_haplotypes.py --ref GRCh38.fa --out-dir data/assoc/hap_seqs`
does this for all 100 loci × 449 samples × 2 haplotypes, emitting one
`*.haps.fa` per locus (89 800 sequences total). Run with no `--ref` to print the
plan/counts without building sequences.

> Scope note: only biallelic SNV/indel variants are applied (SVs and spanning-
> deletion `*` alleles are dropped at extraction). Indels change window length
> slightly per haplotype, which is fine for Evo2 (variable-length input);
> structural variants are out of scope for this aim (they're Aim-1).

The GPU/Evo2 agent then: for each FASTA record → Evo2-7B layer-26 activations →
Goodfire SAE feature profile → that vector is the per-(individual, haplotype,
locus) feature. Per-individual features for a locus = combine the two haplotype
vectors (e.g. sum/concat — modeling agent's choice).

---

## 6. Train / val / test split (held-out test set)

- **Split unit = individual.** No individual appears in more than one split
  (both of a person's haplotypes, and all 100 loci for that person, stay in the
  same split). This prevents leakage of individual-level genetic background and
  of the shared-haplotype structure across loci.
- **Fractions:** train 0.70, val 0.10, test 0.20.
- **Seed:** `SEED = 20260603` (in `build_assoc_skeleton.py`; recorded in
  `splits.json`). Random shuffle of the 449 individuals, then sliced.
- **Resulting counts:** train **314**, val **45**, test **90** (total 449).
- **Population composition is preserved approximately** by the random split
  (EUR/AFR ≈ 80/20 in each): train EUR 247/AFR 67, val EUR 38/AFR 7,
  test EUR 75/AFR 15. If the modeling agent wants exact stratification by
  superpopulation, re-run with a stratified shuffle (same seed) — the population
  labels are in `samples_pop.tsv`.

> Test set is fully held out and must not be touched during feature selection,
> model selection, or hyperparameter tuning (use `val` for that).

---

## 7. Blockers / caveats

- **None blocking.** All data is public and reachable; remote region subsetting
  works (verified). The full per-locus VCF extraction (100 loci) runs in
  ~8–10 min over the network; it is resumable.
- **GRCh38 reference download (~3 GB)** is required *only* for the sequence-
  reconstruction step and should be done on the GPU VM (where Evo2 runs), not
  locally — it's large and only needed adjacent to inference. The per-locus
  phased VCFs (small) are produced here.
- **Coordinate-build mismatch** (Geuvadis hg19 vs panel GRCh38) is handled by
  ID-based joins + GENCODE v44 windows (see §2 note). No liftOver needed.
- **PEER-normalized expression** has the cis-eQTL signal preserved but global
  technical variation removed; this is the appropriate target for an eQTL-style
  haplotype→expression test. Raw counts (`GD660.GeneQuantCount.txt.gz`) are
  available upstream if an alternative normalization is later wanted.
- **Ancestry confound:** EUR vs AFR differ in both haplotype frequencies and
  expression; a model could exploit ancestry rather than locus-specific
  haplotype effects. Mitigations for the modeling stage (not data): include
  superpopulation as a covariate / regress it out, or run EUR-only
  (360 individuals) as a sensitivity analysis. Flagged here for the modeling agent.
