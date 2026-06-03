#!/usr/bin/env python3
"""Helper: build ref & alt sequence windows for Evo2 from the curated SV table.

This does NOT download anything by itself; it needs a local GRCh38 FASTA (indexed
with `samtools faidx`). See docs/DATA_SV.md for which FASTA to use.

For each SV row we have (0-based, half-open):
    ref_start0, ref_end0   : reference allele footprint  (= POS-1 .. POS-1+len(REF))
    window_start0/end0     : footprint +/- flank_bp
    ref, alt               : sequence-resolved alleles (anchor base included, as in VCF)

REF window  = genome[window_start0 : window_end0]
ALT window  = genome[window_start0 : ref_start0] + ALT + genome[ref_end0 : window_end0]

i.e. splice the ALT allele into the reference flanks. Because POS is the VCF
1-based anchor, genome[ref_start0] should equal REF[0] (assert to validate the
FASTA matches the VCF coordinate system).

Usage:
    python fetch_sv_windows.py <genome.fa> <svs.parquet> <out.parquet> [n]
Requires: pysam (pip install pysam) for FASTA random access.
"""
import sys
import pandas as pd

def main():
    fa, table, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else None
    import pysam
    g = pysam.FastaFile(fa)
    df = pd.read_parquet(table) if table.endswith(".parquet") else pd.read_csv(table, sep="\t")
    if n:
        df = df.head(n)
    ref_w, alt_w = [], []
    bad = 0
    for r in df.itertuples():
        ws = max(0, r.window_start0)
        we = r.window_end0
        try:
            left = g.fetch(r.chrom, ws, r.ref_start0)
            right = g.fetch(r.chrom, r.ref_end0, we)
            ref_seq = g.fetch(r.chrom, ws, we)
        except Exception:
            ref_w.append(None); alt_w.append(None); bad += 1; continue
        # validate anchor
        anchor = g.fetch(r.chrom, r.ref_start0, r.ref_start0 + 1).upper()
        if anchor and r.ref and anchor != r.ref[0].upper():
            bad += 1
        alt_seq = left + r.alt + right
        ref_w.append(ref_seq.upper())
        alt_w.append(alt_seq.upper())
    df["ref_window"] = ref_w
    df["alt_window"] = alt_w
    df.to_parquet(out, index=False)
    print(f"wrote {out} rows={len(df)} coord_mismatches={bad}")

if __name__ == "__main__":
    main()
