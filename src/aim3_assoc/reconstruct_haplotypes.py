#!/usr/bin/env python3
"""
Reconstruct per-(individual, haplotype, locus) cis-window DNA sequences for the
Aim-3 association task, by applying each phased haplotype's SNV/indel alleles to
the GRCh38 reference within the locus window.

This is the bridge between the DATA skeleton (this dir) and the Evo2-SAE feature
extraction (GPU agent): the GPU agent runs each reconstructed haplotype sequence
through Evo2, takes the layer-26 SAE feature profile, and that becomes the
per-(individual, haplotype, locus) feature vector used to predict expression.

Approach (uses bcftools consensus, the standard, correct way to apply phased
variants incl. indels to a reference):
  for each locus window:
    samtools faidx REF chr:win_start-win_end > window.fa            # ref window
    for hap in 1,2:
        bcftools consensus -H <hap> -s <sample> -f window.fa locus.vcf.gz
  -> one FASTA record per (sample, hap).

We don't run Evo2 here; we emit either:
  (a) FASTA of all haplotype window sequences (if --ref given), or
  (b) a plan/manifest the GPU agent consumes.

Requirements: bcftools, samtools, bgzip, tabix; a GRCh38 reference FASTA whose
contig names match the panel (chr1..chr22). The 1000G 30x panel is on the
GRCh38 primary assembly (hs38DH / GRCh38_full_analysis_set). Recommended ref:
  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/GRCh38_reference_genome/GRCh38_full_analysis_set_plus_decoy_hla.fa
(download once; ~3GB. Or any GRCh38 primary FASTA with chr-prefixed contigs.)

Usage:
  reconstruct_haplotypes.py --ref GRCh38.fa --out-dir data/assoc/hap_seqs \
      [--loci data/assoc/loci.bed] [--hapdir data/assoc/haplotypes] \
      [--samples data/assoc/samples.txt]
If --ref is omitted, only writes the reconstruction plan (no sequences).
"""
import argparse, csv, os, subprocess, sys

ASSOC = "/Users/user/bio-interp-experiments/data/assoc"


def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="GRCh38 reference FASTA (chr-prefixed contigs)")
    ap.add_argument("--loci", default=os.path.join(ASSOC, "loci.bed"))
    ap.add_argument("--hapdir", default=os.path.join(ASSOC, "haplotypes"))
    ap.add_argument("--samples", default=os.path.join(ASSOC, "samples.txt"))
    ap.add_argument("--out-dir", default=os.path.join(ASSOC, "hap_seqs"))
    args = ap.parse_args()

    samples = [s for s in open(args.samples).read().split() if s]
    loci = []
    with open(args.loci) as fh:
        for chrom, start, end, locus_id in csv.reader(fh, delimiter="\t"):
            loci.append((chrom, int(start), int(end), locus_id))

    os.makedirs(args.out_dir, exist_ok=True)

    if args.ref is None:
        print("[plan] No --ref given; emitting reconstruction plan only.")
        print(f"[plan] {len(loci)} loci x {len(samples)} individuals x 2 haplotypes "
              f"= {len(loci)*len(samples)*2} haplotype-window sequences to build.")
        print("[plan] For each locus VCF in", args.hapdir,
              "run: samtools faidx REF chr:start-end | bcftools consensus -H {1,2} -s SAMPLE")
        return

    if not os.path.exists(args.ref + ".fai"):
        print(f"[setup] indexing reference {args.ref}")
        sh(f"samtools faidx {args.ref}")

    for chrom, start, end, locus_id in loci:
        safe = locus_id.replace("|", "__").replace("/", "_")
        vcf = os.path.join(args.hapdir, safe + ".vcf.gz")
        if not os.path.exists(vcf):
            print(f"[skip] missing VCF for {locus_id}")
            continue
        region = f"{chrom}:{start+1}-{end}"
        win_fa = os.path.join(args.out_dir, safe + ".window.fa")
        sh(f"samtools faidx {args.ref} {region} > {win_fa}")
        out_fa = os.path.join(args.out_dir, safe + ".haps.fa")
        with open(out_fa, "w") as out:
            for s in samples:
                for hap in (1, 2):
                    try:
                        r = sh(f"bcftools consensus -H {hap} -s {s} -f {win_fa} {vcf}")
                    except subprocess.CalledProcessError:
                        continue
                    # rename FASTA header to sample|hap|locus
                    seq = "".join(l for l in r.stdout.splitlines() if not l.startswith(">"))
                    out.write(f">{s}|h{hap}|{locus_id}\n{seq}\n")
        print(f"[ok] {locus_id}: wrote {out_fa}")
    print("[done]")


if __name__ == "__main__":
    main()
