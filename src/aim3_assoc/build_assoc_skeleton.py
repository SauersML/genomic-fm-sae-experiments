#!/usr/bin/env python3
"""
Aim-3 association DATA skeleton builder.

Assembles the modeling table to test:
  "haplotypes represented by SAE feature profiles across loci predict an outcome"
with held-out test sets.

Outcome: Geuvadis LCL gene expression (GD462 PEER-normalized RPKM).
Genotypes: 1000G 30x phased panel, GRCh38 (NYGC/EBI), reconstructed per cis-window.

Pure-Python (csv/gzip) to avoid fragile pandas env. CPU/network only.

Inputs (already downloaded into data/assoc/):
  - EUR373.gene.cis.FDR5.best.txt.gz  (best cis-eQTL per gene, EUR373, hg19 coords/IDs)
  - YRI89.gene.cis.FDR5.best.txt.gz   (best cis-eQTL per gene, YRI89)
  - GD462.GeneQuantRPKM.txt.gz        (462-sample gene x sample expression matrix)
External (fetched once):
  - /tmp/gencode_v44_genes.tsv        (GRCh38 gene coords; ensembl_id chr start end strand symbol type)
  - /tmp/geuv_samples.txt             (462 Geuvadis expression sample IDs)
  - /tmp/kgp_samples.txt              (3202 1000G 30x panel sample IDs)

Outputs (data/assoc/):
  - loci.bed                 GRCh38 cis-windows (TSS +/- FLANK), one per selected gene
  - loci.tsv                 locus metadata incl. eQTL stats, GRCh38 + hg19 coords
  - samples.txt              the N individuals present in BOTH Geuvadis expr and 1000G panel
  - expression.tsv           genes(rows) x samples(cols), aligned to samples.txt, log2(RPKM+1)? no: raw matrix values
  - splits.json              train/val/test split by individual (seed-documented)
"""
import csv, gzip, json, os, random, sys

DATA = "/Users/user/bio-interp-experiments/data/assoc"
GENCODE = "/tmp/gencode_v44_genes.tsv"
GEUV_SAMPLES = "/tmp/geuv_samples.txt"
KGP_SAMPLES = "/tmp/kgp_samples.txt"

N_LOCI = 100          # number of cis-eQTL genes (loci) to include
FLANK = 100_000       # cis-window half-width around TSS (bp)
SEED = 20260603       # documented RNG seed for splits + locus selection tie-breaks
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20

AUTOSOMES = ["chr%d" % i for i in range(1, 23)]

EQTL_COLS = ["snp_id","idnull","gene_id","probe_id","chr_snp","chr_gene",
             "snp_pos","tss_pos","dist","rvalue","pvalue","log10p"]


def read_eqtl(path):
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            d = dict(zip(EQTL_COLS, p))
            d["ens"] = d["gene_id"].split(".")[0]
            try:
                d["log10p"] = float(d["log10p"])
                d["rvalue"] = float(d["rvalue"])
            except ValueError:
                continue
            rows.append(d)
    return rows


def read_gencode(path):
    g = {}  # ens -> dict
    with open(path) as fh:
        for line in fh:
            gid, chrom, start, end, strand, sym, gtype = line.rstrip("\n").split("\t")
            ens = gid.split(".")[0]
            g[ens] = dict(gid=gid, chr=chrom, start=int(start), end=int(end),
                          strand=strand, sym=sym, gtype=gtype)
    return g


def tss_grch38(rec):
    return rec["start"] if rec["strand"] == "+" else rec["end"]


def main():
    rng = random.Random(SEED)

    eqtl = read_eqtl(os.path.join(DATA, "EUR373.gene.cis.FDR5.best.txt.gz"))
    gencode = read_gencode(GENCODE)
    geuv = [s for s in open(GEUV_SAMPLES).read().split() if s]
    kgp = set(s for s in open(KGP_SAMPLES).read().split() if s)

    # individuals usable: in Geuvadis expression AND 1000G phased panel
    overlap = [s for s in geuv if s in kgp]
    overlap_set = set(overlap)
    print(f"[samples] Geuvadis expr={len(geuv)}  1000G panel={len(kgp)}  overlap={len(overlap)}")

    # --- locus selection ---
    # Strongest cis-eQTL genes (largest log10p), mappable to GRCh38 Gencode,
    # autosomal, protein_coding, one locus per gene, dedup by gene.
    cand = []
    seen = set()
    for d in sorted(eqtl, key=lambda x: -x["log10p"]):
        ens = d["ens"]
        if ens in seen:
            continue
        rec = gencode.get(ens)
        if rec is None:
            continue
        if rec["chr"] not in AUTOSOMES:
            continue
        if rec["gtype"] != "protein_coding":
            continue
        seen.add(ens)
        cand.append((d, rec))
        if len(cand) >= N_LOCI:
            break
    print(f"[loci] selected {len(cand)} strong cis-eQTL protein-coding autosomal genes "
          f"(of {len(eqtl)} EUR best-eQTL records)")

    # --- write loci.bed + loci.tsv ---
    bed_path = os.path.join(DATA, "loci.bed")
    tsv_path = os.path.join(DATA, "loci.tsv")
    with open(bed_path, "w") as bed, open(tsv_path, "w") as tsv:
        tw = csv.writer(tsv, delimiter="\t")
        tw.writerow(["locus_id","ens_gene","gencode_gene_id","symbol","chr_grch38",
                     "win_start","win_end","tss_grch38","strand","gene_start","gene_end",
                     "eqtl_snp_id","eqtl_log10p","eqtl_rvalue","eqtl_chr_hg19","eqtl_tss_hg19"])
        for d, rec in cand:
            tss = tss_grch38(rec)
            ws = max(0, tss - FLANK)
            we = tss + FLANK
            locus_id = rec["sym"] + "|" + d["ens"]
            # BED: 0-based start. Keep 'chr' prefix to match the 1000G 30x panel
            # contig naming (##contig=<ID=chr12,...>; records use CHROM=chr12).
            bed.write(f"{rec['chr']}\t{max(0,ws-1)}\t{we}\t{locus_id}\n")
            tw.writerow([locus_id, d["ens"], rec["gid"], rec["sym"], rec["chr"],
                         ws, we, tss, rec["strand"], rec["start"], rec["end"],
                         d["snp_id"], f"{d['log10p']:.4f}", f"{d['rvalue']:.4f}",
                         d["chr_gene"], d["tss_pos"]])
    print(f"[write] {bed_path}  {tsv_path}")

    # --- aligned expression matrix (selected genes x overlap samples) ---
    sel_ens = {d["ens"]: rec["sym"] + "|" + d["ens"] for d, rec in cand}
    expr_path = os.path.join(DATA, "expression.tsv")
    with gzip.open(os.path.join(DATA, "GD462.GeneQuantRPKM.txt.gz"), "rt") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader)
        sample_cols = header[4:]
        # column indices for overlap samples, in overlap order
        col_idx = {s: 4 + i for i, s in enumerate(sample_cols)}
        kept_samples = [s for s in overlap if s in col_idx]
        idxs = [col_idx[s] for s in kept_samples]
        rows_written = 0
        with open(expr_path, "w") as out:
            w = csv.writer(out, delimiter="\t")
            w.writerow(["locus_id", "ens_gene"] + kept_samples)
            for row in reader:
                tgt = row[0]  # TargetID (gene id with version)
                ens = tgt.split(".")[0]
                if ens in sel_ens:
                    vals = [row[i] for i in idxs]
                    w.writerow([sel_ens[ens], ens] + vals)
                    rows_written += 1
    print(f"[write] {expr_path}  genes={rows_written}  samples={len(kept_samples)}")
    if rows_written != len(cand):
        print(f"[warn] {len(cand)-rows_written} selected genes not found in expression matrix")

    # --- splits by individual (no individual in two splits) ---
    samples = list(kept_samples)
    rng.shuffle(samples)
    n = len(samples)
    n_test = round(n * TEST_FRAC)
    n_val = round(n * VAL_FRAC)
    test = sorted(samples[:n_test])
    val = sorted(samples[n_test:n_test + n_val])
    train = sorted(samples[n_test + n_val:])
    assert set(test).isdisjoint(val) and set(test).isdisjoint(train) and set(val).isdisjoint(train)

    with open(os.path.join(DATA, "samples.txt"), "w") as fh:
        fh.write("\n".join(kept_samples) + "\n")

    splits = dict(seed=SEED, split_by="individual",
                  fractions=dict(train=TRAIN_FRAC, val=VAL_FRAC, test=TEST_FRAC),
                  counts=dict(total=n, train=len(train), val=len(val), test=len(test)),
                  train=train, val=val, test=test)
    with open(os.path.join(DATA, "splits.json"), "w") as fh:
        json.dump(splits, fh, indent=1)
    print(f"[split] total={n} train={len(train)} val={len(val)} test={len(test)} seed={SEED}")
    print("[done]")


if __name__ == "__main__":
    main()
