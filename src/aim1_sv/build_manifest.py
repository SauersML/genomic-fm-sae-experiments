#!/usr/bin/env python
"""Build the Aim-1 ref/alt-delta manifest + id-aligned label/covariate/split tables.

Local / CPU only. Reads data/hprc_sv/svs_pilot.parquet, selects a balanced pilot
subset (~600-800 SVs across the 7 consequence classes, both DEL and INS), and
writes:
  data/aim1_sv/manifest.jsonl   - one record per SV (coords + alleles), see MANIFEST_SPEC.md
  data/aim1_sv/labels.parquet   - id-aligned y / covariates / groups for analyze.py

Window construction is documented in data/aim1_sv/MANIFEST_SPEC.md and matched by
azure/embed_evo2.py. We do NOT extract FASTA locally (no genome here); azure builds
the windows. GC% of the window is therefore filled by azure (gc_window column is
NaN here and merged later); svlen comes from the parquet.

Usage:
    .venv/bin/python src/aim1_sv/build_manifest.py [--per-class N] [--seed S]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PILOT = os.path.join(ROOT, "data/hprc_sv/svs_pilot.parquet")
OUTDIR = os.path.join(ROOT, "data/aim1_sv")

FLANK = 3072
MAX_ALLELE = 1024
CLASSES = ["cds", "splice", "utr", "exon_noncod", "intronic", "regulatory", "intergenic"]


def select_pilot(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Balanced selection across the 7 classes; within each class, keep both
    DEL and INS represented (sample proportionally to availability but force a
    floor of each svtype when both exist)."""
    rng = np.random.default_rng(seed)
    picks = []
    for cls in CLASSES:
        sub = df[df["consequence"] == cls]
        dels = sub[sub["svtype"] == "DEL"]
        inss = sub[sub["svtype"] == "INS"]
        n_avail = len(sub)
        take = min(per_class, n_avail)
        # Aim for both types present. If both exist, reserve >=25% for the
        # minority type (capped by availability) so neither vanishes.
        if len(dels) and len(inss):
            n_minor = min(len(dels), len(inss))
            minor_is_del = len(dels) <= len(inss)
            n_minor_take = min(int(np.ceil(0.25 * take)), n_minor)
            n_major_take = take - n_minor_take
            if minor_is_del:
                d_take, i_take = n_minor_take, n_major_take
            else:
                d_take, i_take = n_major_take, n_minor_take
            d_take = min(d_take, len(dels)); i_take = min(i_take, len(inss))
            # top up if rounding left us short
            short = take - d_take - i_take
            if short > 0:
                if len(dels) - d_take >= len(inss) - i_take:
                    d_take += min(short, len(dels) - d_take)
                else:
                    i_take += min(short, len(inss) - i_take)
            chosen = pd.concat([
                dels.sample(d_take, random_state=int(rng.integers(1 << 31))),
                inss.sample(i_take, random_state=int(rng.integers(1 << 31))),
            ])
        else:
            chosen = sub.sample(take, random_state=int(rng.integers(1 << 31)))
        picks.append(chosen)
    out = pd.concat(picks).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def cap_len(n: int, m: int = MAX_ALLELE) -> int:
    return min(n, m)


def build(per_class: int, seed: int):
    df = pd.read_parquet(PILOT)
    pilot = select_pilot(df, per_class, seed)

    # ---- manifest.jsonl (coords + alleles; azure builds windows) ----
    os.makedirs(OUTDIR, exist_ok=True)
    mpath = os.path.join(OUTDIR, "manifest.jsonl")
    with open(mpath, "w") as fh:
        for r in pilot.itertuples(index=False):
            rec = {
                "id": r.sv_id,
                "chrom": r.chrom,
                "start0": int(r.ref_start0),   # L (footprint start, 0-based)
                "end0": int(r.ref_end0),       # R (footprint end,   0-based)
                "ref": r.ref,
                "alt": r.alt,
                "flank": FLANK,
                "max_allele": MAX_ALLELE,
            }
            fh.write(json.dumps(rec) + "\n")

    # ---- id-aligned label/covariate/split table ----
    lab = pd.DataFrame({
        "id": pilot["sv_id"].values,
        "chrom": pilot["chrom"].values,            # groups for GroupKFold
        "svtype": pilot["svtype"].values,
        "consequence": pilot["consequence"].values,
        "consequence_coarse": pilot["consequence_coarse"].values,
        "y_binary": pilot["is_coding_disrupting"].astype(int).values,  # primary target
        "svlen_abs": pilot["svlen_abs"].astype(float).values,
        "svlen_signed": pilot["svlen_signed"].astype(float).values,
        "af": pilot["af"].astype(float).values,
    })
    lab["log_svlen"] = np.log10(lab["svlen_abs"].clip(lower=1.0))
    # GC% of the Evo2 window: requires the genomic flanks -> filled by azure-compute
    # from hg38.fa and merged back (see MANIFEST_SPEC.md). NaN placeholder for now.
    lab["gc_window"] = np.nan
    # svtype dummy (DEL=0, INS=1) as a covariate column
    lab["svtype_ins"] = (lab["svtype"] == "INS").astype(int)

    lab.to_parquet(os.path.join(OUTDIR, "labels.parquet"), index=False)

    # ---- summary ----
    print(f"[build_manifest] wrote {len(pilot)} records -> {mpath}")
    print("[build_manifest] per consequence x svtype:")
    print(pd.crosstab(lab["consequence"], lab["svtype"]).to_string())
    print("\n[build_manifest] y_binary (is_coding_disrupting):")
    print(lab["y_binary"].value_counts().to_string())
    print("\n[build_manifest] median svlen_abs by consequence:")
    print(lab.groupby("consequence")["svlen_abs"].median().round(0).to_string())
    print(f"\n[build_manifest] n chroms (groups): {lab['chrom'].nunique()}")
    print("[build_manifest] labels.parquet cols:", list(lab.columns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=100,
                    help="SVs per consequence class (7 classes -> ~700 total)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.per_class, args.seed)
