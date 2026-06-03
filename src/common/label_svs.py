#!/usr/bin/env python3
"""Label HPRC release2 wave SVs with functional consequence via GENCODE v44 +
ENCODE cCRE interval overlap, and emit the curated SV table.

Input  : data/hprc_sv/sv_raw.tsv   (CHROM POS ID REF ALT INFO ; SVs only, indel
         length delta >= 50, biallelic, sequence-resolved)
Refs   : data/annotations/gencode_v44_features.pkl
         data/annotations/encode_ccre_features.pkl

SV interval on GRCh38 (0-based half-open):
  POS in VCF is 1-based, REF[0] is the anchor base shared by ref & alt for indels.
  affected reference span start = POS (0-based POS, i.e. POS-1+1 = the base after
    anchor) ... we use a robust span:
      ref_start0 = POS-1            (0-based pos of anchor)
      ref_end0   = POS-1 + len(REF) (end of reference allele)
  For an insertion (len REF==1) this is a near-point interval [POS-1, POS); we
  still query a window of >=1bp. For deletions the deleted reference span is
  [POS, POS-1+len(REF)) (excludes anchor). We label using the full ref allele
  span which is the conservative footprint on the reference.

Consequence hierarchy (most -> least severe), assigned as the first match:
  cds         : overlaps a CDS interval
  splice      : overlaps a splice-region window (+/-8bp of an exon boundary)
  utr         : overlaps a UTR exon
  exon_noncod : overlaps an exon but not CDS/UTR (noncoding-gene exon / ncRNA)
  intronic    : within a gene span but not in any exon
  regulatory  : not genic, but overlaps an ENCODE cCRE
  intergenic  : none of the above

Targets:
  consequence            : the 7-class label above (multiclass)
  consequence_coarse     : {coding, noncoding_genic, regulatory, intergenic}
  is_coding_disrupting   : binary 1 if consequence in {cds, splice}
                           (the cleanest "function-altering" proxy vs the rest)

Outputs (data/hprc_sv/):
  svs.parquet / svs.tsv  : full table
  svs_pilot.parquet/tsv  : balanced, deduplicated ~2-5k subset across classes
"""
import pickle, sys, hashlib
import numpy as np
import pandas as pd

RAW = "data/hprc_sv/sv_raw.tsv"
GENCODE = "data/annotations/gencode_v44_features.pkl"
CCRE = "data/annotations/encode_ccre_features.pkl"
OUTDIR = "data/hprc_sv"
FLANK = 4096  # bp flank each side -> stored so Evo2 windows can be built later

# ---- load interval refs ----
with open(GENCODE, "rb") as f:
    G = pickle.load(f)
with open(CCRE, "rb") as f:
    C = pickle.load(f)

def overlaps(arr, s, e):
    """Does query [s,e) overlap any merged interval in sorted Nx2 arr?"""
    if arr is None or len(arr) == 0:
        return False
    starts = arr[:, 0]
    # first interval whose start < e ; check the candidate just before insertion of e
    idx = np.searchsorted(starts, e, side="right") - 1
    if idx < 0:
        return False
    # walk back a few (merged intervals are disjoint & sorted, so the candidate
    # with largest start < e is the only one that can overlap)
    if arr[idx, 1] > s and arr[idx, 0] < e:
        return True
    return False

def feat(d, chrom):
    return d.get(chrom)

# ---- parse INFO ----
def parse_info(info):
    af = ac = an = ns = None
    typ = None
    ln = None
    inv = False
    for kv in info.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k == "AF":
                # could be comma list if somehow multiallelic; take max
                try:
                    af = max(float(x) for x in v.split(","))
                except ValueError:
                    af = None
            elif k == "AC":
                try: ac = max(int(x) for x in v.split(","))
                except ValueError: ac = None
            elif k == "AN":
                try: an = int(v)
                except ValueError: an = None
            elif k == "NS":
                try: ns = int(v)
                except ValueError: ns = None
            elif k == "TYPE":
                typ = v
            elif k == "LEN":
                try: ln = max(abs(int(x)) for x in v.split(","))
                except ValueError: ln = None
        elif kv == "INV":
            inv = True
    return af, ac, an, ns, typ, ln, inv

rows = []
n = 0
with open(RAW) as fh:
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 6:
            continue
        chrom, pos, vid, ref, alt, info = f[0], int(f[1]), f[2], f[3], f[4], f[5]
        rl = len(ref); al = len(alt)
        svlen = al - rl  # signed: +insertion, -deletion
        af, ac, an, ns, typ, ln, inv = parse_info(info)
        # svtype
        if inv:
            svtype = "INV"
        elif svlen > 0:
            svtype = "INS"
        elif svlen < 0:
            svtype = "DEL"
        else:
            svtype = "COMPLEX"  # equal length but >=50 diff impossible; guard

        # reference footprint (0-based half-open)
        ref_start0 = pos - 1
        ref_end0 = pos - 1 + rl
        # query span: ensure at least 1bp width for pure insertions
        qs, qe = ref_start0, max(ref_end0, ref_start0 + 1)

        ga = feat(G["cds"], chrom)
        cds = overlaps(ga, qs, qe)
        spl = overlaps(feat(G["splice"], chrom), qs, qe)
        utr = overlaps(feat(G["utr"], chrom), qs, qe)
        exn = overlaps(feat(G["exon"], chrom), qs, qe)
        gen_any = overlaps(feat(G["gene_any"], chrom), qs, qe)
        gen_cod = overlaps(feat(G["gene_coding"], chrom), qs, qe)
        reg = overlaps(feat(C["any"], chrom), qs, qe)

        if cds:
            cons = "cds"
        elif spl:
            cons = "splice"
        elif utr:
            cons = "utr"
        elif exn:
            cons = "exon_noncod"
        elif gen_any:
            cons = "intronic"
        elif reg:
            cons = "regulatory"
        else:
            cons = "intergenic"

        coarse = {
            "cds": "coding", "splice": "coding",
            "utr": "noncoding_genic", "exon_noncod": "noncoding_genic",
            "intronic": "noncoding_genic",
            "regulatory": "regulatory", "intergenic": "intergenic",
        }[cons]
        is_cd = 1 if cons in ("cds", "splice") else 0

        # stable id
        h = hashlib.sha1(f"{chrom}:{pos}:{ref}:{alt}".encode()).hexdigest()[:12]
        sv_id = f"hprcv2_{chrom}_{pos}_{svtype}_{h}"

        rows.append((
            sv_id, chrom, pos, vid, ref, alt, svlen, abs(svlen), svtype,
            af, ac, an, ns, inv,
            ref_start0, ref_end0,
            ref_start0 - FLANK, ref_end0 + FLANK, FLANK,
            cds, spl, utr, exn, gen_any, gen_cod, reg,
            cons, coarse, is_cd,
        ))
        n += 1
        if n % 50000 == 0:
            print(f"  labeled {n} ...", file=sys.stderr)

cols = [
    "sv_id", "chrom", "pos", "vcf_id", "ref", "alt", "svlen_signed", "svlen_abs",
    "svtype", "af", "ac", "an", "ns", "inv_flag",
    "ref_start0", "ref_end0",
    "window_start0", "window_end0", "flank_bp",
    "ov_cds", "ov_splice", "ov_utr", "ov_exon", "ov_gene_any", "ov_gene_coding",
    "ov_ccre",
    "consequence", "consequence_coarse", "is_coding_disrupting",
]
df = pd.DataFrame(rows, columns=cols)
print(f"total SVs labeled: {len(df)}")

# write full table
df.to_parquet(f"{OUTDIR}/svs.parquet", index=False)
df.to_csv(f"{OUTDIR}/svs.tsv", sep="\t", index=False)

print("\n=== consequence (multiclass) ===")
print(df["consequence"].value_counts().to_string())
print("\n=== consequence_coarse ===")
print(df["consequence_coarse"].value_counts().to_string())
print("\n=== is_coding_disrupting ===")
print(df["is_coding_disrupting"].value_counts().to_string())
print("\n=== svtype ===")
print(df["svtype"].value_counts().to_string())
print(f"\nAF present: {df['af'].notna().sum()} / {len(df)}")
print(f"svlen_abs quantiles: {df['svlen_abs'].quantile([.5,.9,.99]).to_dict()}")

# ---- balanced, deduplicated pilot subset across the 7-class consequence ----
# dedup on (chrom,pos,ref,alt) already unique by construction; also drop exact
# coordinate duplicates just in case.
dd = df.drop_duplicates(subset=["chrom", "pos", "ref", "alt"]).copy()
rng = np.random.default_rng(42)
PER_CLASS = 700  # 7 classes -> up to ~4900
parts = []
for cls, grp in dd.groupby("consequence"):
    take = min(PER_CLASS, len(grp))
    parts.append(grp.sample(n=take, random_state=int(rng.integers(1e9))))
pilot = pd.concat(parts).sample(frac=1.0, random_state=7).reset_index(drop=True)
pilot.to_parquet(f"{OUTDIR}/svs_pilot.parquet", index=False)
pilot.to_csv(f"{OUTDIR}/svs_pilot.tsv", sep="\t", index=False)
print(f"\npilot subset: {len(pilot)} rows")
print(pilot["consequence"].value_counts().to_string())
print("\npilot svlen_abs by class (median) -- leakage check:")
print(pilot.groupby("consequence")["svlen_abs"].median().to_string())
