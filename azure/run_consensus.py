#!/usr/bin/env python
"""Aim-3 helper: build a haplotype consensus sequence over a region from a
per-locus phased VCF + hg38.fa, using `bcftools consensus`.

For a phased diploid sample, -H1 / -H2 select the two haplotypes. Output is the
reference sequence over [chrom:start1-end1] with that sample's phased variants
applied.

Usage (single):
    python azure/run_consensus.py --vcf locus.vcf.gz --fasta ~/hf_cache/hg38.fa \
        --region chr12:55941350-56141351 --sample HG00096 --hap 1

Programmatic:
    from azure.run_consensus import consensus
    seq = consensus(vcf, fasta, "chr12", 55941350, 56141351, "HG00096", hap=1)
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile


def consensus(vcf: str, fasta: str, chrom: str, start1: int, end1: int,
              sample: str, hap: int = 1) -> str:
    """Return the haplotype `hap` (1 or 2) consensus sequence for `sample` over
    the 1-based inclusive region chrom:start1-end1. bcftools applies that
    sample's phased ALT alleles onto the reference slice."""
    assert hap in (1, 2)
    region = f"{chrom}:{start1}-{end1}"
    # samtools faidx emits the ref slice; bcftools consensus splices variants.
    faidx = subprocess.run(
        ["samtools", "faidx", fasta, region],
        capture_output=True, text=True, check=True).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".fa", delete=False) as fh:
        fh.write(faidx)
        ref_fa = fh.name
    out = subprocess.run(
        ["bcftools", "consensus",
         "-f", ref_fa,
         "-H", str(hap),
         "-s", sample,
         "-r", region,
         vcf],
        capture_output=True, text=True, check=True).stdout
    return "".join(out.splitlines()[1:]).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--region", required=True, help="chrom:start1-end1 (1-based)")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--hap", type=int, default=1, choices=[1, 2])
    args = ap.parse_args()
    chrom, rng = args.region.split(":")
    s1, e1 = rng.split("-")
    seq = consensus(args.vcf, args.fasta, chrom, int(s1), int(e1),
                    args.sample, hap=args.hap)
    print(f">{args.sample}_H{args.hap}_{args.region}")
    print(seq)


if __name__ == "__main__":
    main()
