#!/usr/bin/env python3
"""Build merged ENCODE cCRE interval arrays per chrom for SV overlap.
Keeps a coarse class per element (PLS/ELS/CTCF/other) for an optional finer label,
plus an 'any cCRE' merged set for the binary regulatory flag.

Output: data/annotations/encode_ccre_features.pkl
"""
import pickle, sys
from collections import defaultdict
import numpy as np

BED = sys.argv[1] if len(sys.argv) > 1 else "data/annotations/encode_ccre_grch38.bed"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/annotations/encode_ccre_features.pkl"

def coarse(cls):
    if cls.startswith("PLS"):
        return "promoter"
    if "ELS" in cls:
        return "enhancer"
    if "CTCF" in cls:
        return "ctcf"
    return "other"

by_class = defaultdict(lambda: defaultdict(list))  # coarse -> chrom -> intervals
any_ccre = defaultdict(list)
n = 0
with open(BED) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 6:
            continue
        chrom = f[0]; s = int(f[1]); e = int(f[2]); cls = f[5]
        any_ccre[chrom].append((s, e))
        by_class[coarse(cls)][chrom].append((s, e))
        n += 1

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

result = {"any": {c: merge_sorted(v) for c, v in any_ccre.items()}}
for cls, d in by_class.items():
    result[cls] = {c: merge_sorted(v) for c, v in d.items()}

with open(OUT, "wb") as fh:
    pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

print(f"parsed {n} cCRE elements")
for cls in result:
    tot = sum(len(a) for a in result[cls].values())
    print(f"  {cls:10s}: {tot} merged intervals")
print(f"wrote {OUT}")
