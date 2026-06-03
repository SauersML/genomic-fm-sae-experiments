#!/usr/bin/env python3
"""Parse GENCODE v44 GTF (GRCh38) into per-chromosome sorted interval arrays for
fast SV overlap labeling. Produces a compact pickle of numpy arrays per chrom for
several feature classes:

  - cds         : coding sequence exons
  - utr         : UTR exons
  - exon        : all exons (any biotype) -- used for "exonic noncoding" / general exon
  - gene        : gene span (for intronic vs intergenic determination)
  - splice      : splice-region windows (+/- SPLICE_PAD bp around each exon boundary)

We keep separate gene spans for protein_coding genes so we can distinguish
"intronic in a coding gene" from "intronic in a noncoding gene".

Output: data/annotations/gencode_v44_features.pkl
"""
import gzip, pickle, sys
from collections import defaultdict
import numpy as np

GTF = sys.argv[1] if len(sys.argv) > 1 else "data/annotations/gencode.v44.annotation.gtf.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/annotations/gencode_v44_features.pkl"
SPLICE_PAD = 8  # bp around exon boundaries considered splice-region (covers canonical donor/acceptor)

def attr(field, key):
    # field is the 9th GTF column; attributes like: gene_type "protein_coding";
    i = field.find(key + ' "')
    if i < 0:
        return None
    i += len(key) + 2
    j = field.find('"', i)
    return field[i:j]

feats = {
    "cds": defaultdict(list),
    "utr": defaultdict(list),
    "exon": defaultdict(list),
    "gene_coding": defaultdict(list),
    "gene_any": defaultdict(list),
    "splice": defaultdict(list),
}

n = 0
op = gzip.open(GTF, "rt")
for line in op:
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) < 9:
        continue
    chrom, src, ftype, start, end, score, strand, frame, info = f
    s = int(start) - 1  # GTF is 1-based inclusive -> 0-based half-open
    e = int(end)
    if ftype == "gene":
        gt = attr(info, "gene_type")
        feats["gene_any"][chrom].append((s, e))
        if gt == "protein_coding":
            feats["gene_coding"][chrom].append((s, e))
    elif ftype == "CDS":
        feats["cds"][chrom].append((s, e))
    elif ftype in ("five_prime_utr", "three_prime_utr", "UTR"):
        feats["utr"][chrom].append((s, e))
    elif ftype == "exon":
        feats["exon"][chrom].append((s, e))
        # splice-region windows around both exon boundaries
        feats["splice"][chrom].append((s - SPLICE_PAD, s + SPLICE_PAD))
        feats["splice"][chrom].append((e - SPLICE_PAD, e + SPLICE_PAD))
    n += 1
op.close()

# Merge overlapping intervals per class/chrom and store as sorted numpy arrays.
def merge_sorted(intervals):
    if not intervals:
        return np.empty((0, 2), dtype=np.int64)
    iv = sorted(intervals)
    out = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            if e > ce:
                ce = e
        else:
            out.append((cs, ce)); cs, ce = s, e
    out.append((cs, ce))
    return np.asarray(out, dtype=np.int64)

result = {}
for cls, d in feats.items():
    result[cls] = {chrom: merge_sorted(ivs) for chrom, ivs in d.items()}

with open(OUT, "wb") as fh:
    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

# report
print(f"parsed {n} GTF feature lines")
for cls in result:
    tot = sum(len(a) for a in result[cls].values())
    print(f"  {cls:12s}: {tot} merged intervals across {len(result[cls])} contigs")
print(f"wrote {OUT}")
