#!/usr/bin/env python3
"""
Aim-3 association PILOT manifest + aligned-table builder.

Produces, under data/aim3_assoc/:
  - manifest.jsonl        one record per (gene, individual, hap) consensus+embed job
  - MANIFEST_SPEC.md      schema doc (written separately, see file)
  - genes.tsv             the pilot gene set + window coords + eqtl strength
  - samples_pilot.txt     the pilot individual subset (test+val intact, train subset)
  - outcome.tsv           per-(gene,sample) expression target (long), + split + ancestry
  - covariates.tsv        per-sample ancestry dummy (afr=1) and split label
  - altcount.tsv          per-(gene,sample,hap) and per-(gene,sample) ALT-allele count
                          in the window = simple variant-burden baseline
  - splits_pilot.json     train/val/test individual lists restricted to the pilot subset

Design (see docs/RESULTS_AIM3.md once features exist):
  * Genes      = top-N by EUR373 best-cis-eQTL log10p (strongest cis signal -> fair test).
  * Window     = TSS(GRCh38) +/- WIN_HALF bp (default 3000 => 6 kb total). The GRCh38 TSS
                 is the trustworthy anchor (Geuvadis eQTL SNP coords are hg19 -> not used
                 for windowing; see docs/DATA_ASSOC.md s2 note).
  * Individuals= ALL test + ALL val individuals kept intact (held-out integrity), train
                 subset to TRAIN_KEEP individuals (seed-stratified by superpop to preserve
                 EUR/AFR ratio) to stay within the pilot token budget.
  * Record id  = "<locus_id>|<sample>|h<h>"  with locus_id = "SYMBOL|ENSG..." (the '|' inside
                 the locus_id is kept; the id has exactly 4 '|'-separated logical fields:
                 SYMBOL, ENSG, sample, hH -- parse by rsplit('|',2)).

The VCF file on disk is named  locus_id.replace('|','_') + '.vcf.gz'  (verified against
data/assoc/haplotypes/), e.g. RPS26|ENSG00000197728 -> RPS26_ENSG00000197728.vcf.gz.
"""
import argparse, csv, json, os, random, subprocess, sys
from collections import defaultdict

ROOT = "/Users/user/bio-interp-experiments"
ASSOC = os.path.join(ROOT, "data/assoc")
OUT = os.path.join(ROOT, "data/aim3_assoc")
HAPDIR = os.path.join(ASSOC, "haplotypes")
HAPDIR_REL = "data/assoc/haplotypes"  # path written into the manifest (box-relative)


def vcf_path(locus_id):
    return locus_id.replace("|", "_") + ".vcf.gz"


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout


def load_loci():
    with open(os.path.join(ASSOC, "loci.tsv")) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def pick_genes(loci, n):
    loci = sorted(loci, key=lambda r: -float(r["eqtl_log10p"]))
    return loci[:n]


def load_pop():
    pop = {}
    with open(os.path.join(ASSOC, "samples_pop.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            pop[r["sample"]] = r["superpopulation"]
    return pop


def stratified_subset(samples, pop, keep, seed):
    """Subsample `samples` down to ~keep, preserving EUR/AFR proportions."""
    if keep >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in samples:
        by[pop.get(s, "NA")].append(s)
    frac = keep / len(samples)
    out = []
    for grp, members in sorted(by.items()):
        members = sorted(members)
        rng.shuffle(members)
        k = max(1, round(len(members) * frac))
        out.extend(members[:k])
    return sorted(out)


def altcount_for_locus(locus_id, chrom, start0, end0):
    """Per-(sample,hap) ALT-allele count in [start0,end0) from the phased VCF.
    Returns dict sample -> (h1_alt, h2_alt). 1-based region for bcftools = start0+1..end0."""
    vcf = os.path.join(HAPDIR, vcf_path(locus_id))
    region = f"{chrom}:{start0+1}-{end0}"
    # query phased GTs for all samples in the window
    samples = sh(f"bcftools query -l {vcf}").split()
    out = {s: [0, 0] for s in samples}
    txt = sh(f'bcftools query -r {region} -f "[%GT\\t]\\n" {vcf}')
    for line in txt.splitlines():
        gts = line.rstrip("\t").split("\t")
        for s, gt in zip(samples, gts):
            # phased GT like "0|1"; count alt (any allele index >0) per haplotype
            sep = "|" if "|" in gt else "/"
            parts = gt.split(sep)
            if len(parts) == 2:
                for hi, a in enumerate(parts):
                    if a not in (".", "0"):
                        out[s][hi] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-genes", type=int, default=8)
    ap.add_argument("--win-half", type=int, default=3000, help="half-window bp around TSS")
    ap.add_argument("--train-keep", type=int, default=120,
                    help="how many TRAIN individuals to keep (test+val kept whole)")
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--no-altcount", action="store_true",
                    help="skip the (bcftools) ALT-count baseline computation")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    loci = load_loci()
    genes = pick_genes(loci, args.n_genes)
    pop = load_pop()
    splits = json.load(open(os.path.join(ASSOC, "splits.json")))

    test = list(splits["test"])
    val = list(splits["val"])
    train_full = list(splits["train"])
    train_sub = stratified_subset(train_full, pop, args.train_keep, args.seed)
    pilot_samples = sorted(set(test) | set(val) | set(train_sub))

    # split label per pilot sample
    split_of = {}
    for s in test: split_of[s] = "test"
    for s in val: split_of[s] = "val"
    for s in train_sub: split_of[s] = "train"

    # ---- genes.tsv + window coords ----
    gene_rows = []
    for r in genes:
        tss = int(r["tss_grch38"])
        start0 = tss - args.win_half          # 0-based, inclusive
        end0 = tss + args.win_half            # 0-based, exclusive (bcftools 1-based: start0+1..end0)
        gene_rows.append({
            "locus_id": r["locus_id"], "ens_gene": r["ens_gene"], "symbol": r["symbol"],
            "chrom": r["chr_grch38"], "tss": tss, "strand": r["strand"],
            "start0": start0, "end0": end0,
            "eqtl_log10p": r["eqtl_log10p"], "eqtl_rvalue": r["eqtl_rvalue"],
            "vcf": f"{HAPDIR_REL}/{vcf_path(r['locus_id'])}",
        })
    with open(os.path.join(OUT, "genes.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(gene_rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(gene_rows)

    # ---- samples_pilot.txt + splits_pilot.json ----
    with open(os.path.join(OUT, "samples_pilot.txt"), "w") as fh:
        fh.write("\n".join(pilot_samples) + "\n")
    json.dump({
        "seed": args.seed, "split_by": "individual",
        "source_splits": "data/assoc/splits.json (seed %s)" % splits.get("seed"),
        "note": "test+val kept whole; train subset to ~%d (superpop-stratified)" % args.train_keep,
        "counts": {"train": sum(1 for s in pilot_samples if split_of[s] == "train"),
                   "val": sum(1 for s in pilot_samples if split_of[s] == "val"),
                   "test": sum(1 for s in pilot_samples if split_of[s] == "test"),
                   "total": len(pilot_samples)},
        "train": [s for s in pilot_samples if split_of[s] == "train"],
        "val": [s for s in pilot_samples if split_of[s] == "val"],
        "test": [s for s in pilot_samples if split_of[s] == "test"],
    }, open(os.path.join(OUT, "splits_pilot.json"), "w"), indent=2)

    # ---- manifest.jsonl ----
    n_jobs = 0
    with open(os.path.join(OUT, "manifest.jsonl"), "w") as fh:
        for g in gene_rows:
            for s in pilot_samples:
                for h in (1, 2):
                    rec = {
                        "id": f"{g['locus_id']}|{s}|h{h}",
                        "gene": g["locus_id"], "sample": s, "hap": h,
                        "vcf": g["vcf"], "chrom": g["chrom"],
                        "start0": g["start0"], "end0": g["end0"],
                    }
                    fh.write(json.dumps(rec) + "\n")
                    n_jobs += 1

    # ---- outcome.tsv  (long: locus_id, sample, split, superpop, afr_dummy, expression) ----
    expr = {}
    with open(os.path.join(ASSOC, "expression.tsv")) as efh:
        rd = csv.DictReader(efh, delimiter="\t")
        for row in rd:
            if row["locus_id"] in {g["locus_id"] for g in gene_rows}:
                expr[row["locus_id"]] = row
    with open(os.path.join(OUT, "outcome.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["locus_id", "sample", "split", "superpop", "afr_dummy", "expression"])
        for g in gene_rows:
            erow = expr[g["locus_id"]]
            for s in pilot_samples:
                sp = pop.get(s, "NA")
                w.writerow([g["locus_id"], s, split_of[s], sp,
                            1 if sp == "AFR" else 0, erow[s]])

    # ---- covariates.tsv (per-sample, deduped) ----
    with open(os.path.join(OUT, "covariates.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sample", "split", "superpop", "afr_dummy"])
        for s in pilot_samples:
            sp = pop.get(s, "NA")
            w.writerow([s, split_of[s], sp, 1 if sp == "AFR" else 0])

    # ---- altcount.tsv (variant-burden baseline) ----
    if not args.no_altcount:
        with open(os.path.join(OUT, "altcount.tsv"), "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["locus_id", "sample", "h1_alt", "h2_alt", "mean_alt", "sum_alt"])
            pilot_set = set(pilot_samples)
            for g in gene_rows:
                ac = altcount_for_locus(g["locus_id"], g["chrom"], g["start0"], g["end0"])
                for s in pilot_samples:
                    h1, h2 = ac.get(s, (0, 0))
                    w.writerow([g["locus_id"], s, h1, h2, (h1 + h2) / 2.0, h1 + h2])
                print(f"[altcount] {g['locus_id']}: done", file=sys.stderr)

    print(json.dumps({
        "n_genes": len(gene_rows),
        "n_individuals": len(pilot_samples),
        "split_counts": {"train": sum(1 for s in pilot_samples if split_of[s] == "train"),
                         "val": sum(1 for s in pilot_samples if split_of[s] == "val"),
                         "test": sum(1 for s in pilot_samples if split_of[s] == "test")},
        "n_embed_jobs": n_jobs,
        "window_bp": 2 * args.win_half,
        "genes": [g["locus_id"] for g in gene_rows],
    }, indent=2))


if __name__ == "__main__":
    main()
