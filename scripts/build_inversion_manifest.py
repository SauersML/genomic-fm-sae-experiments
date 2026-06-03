#!/usr/bin/env python3
"""Build the Evo2/SAE ref-vs-alt manifest for HPRC inversion alleles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INDIR = ROOT / "data/inversions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(INDIR / "hprc_inversions_pilot.parquet"))
    ap.add_argument("--outdir", default=str(INDIR))
    ap.add_argument("--flank", type=int, default=3072)
    ap.add_argument("--max-allele", type=int, default=4096)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input).copy()
    df = df.sort_values(["chrom", "pos", "allele_index"]).reset_index(drop=True)

    with open(outdir / "manifest.jsonl", "w") as fh:
        for r in df.itertuples(index=False):
            rec = {
                "id": r.sv_id,
                "chrom": r.chrom,
                "start0": int(r.ref_start0),
                "end0": int(r.ref_end0),
                "ref": r.ref,
                "alt": r.alt,
                "flank": int(args.flank),
                "max_allele": int(args.max_allele),
            }
            fh.write(json.dumps(rec) + "\n")

    labels = pd.DataFrame({
        "id": df["sv_id"],
        "chrom": df["chrom"],
        "pos": df["pos"],
        "allele_index": df["allele_index"],
        "svtype": "INV",
        "type_allele": df["type_allele"].fillna(""),
        "consequence": df["consequence"],
        "consequence_coarse": df["consequence_coarse"],
        "y_binary": df["is_coding_disrupting"].astype(int),
        "inv_len": df["inv_len"].astype(float),
        "log_inv_len": df["log_inv_len"].astype(float),
        "af": pd.to_numeric(df["af"], errors="coerce"),
        "ac": pd.to_numeric(df["ac"], errors="coerce"),
        "len_delta": df["len_delta"].astype(float),
        "ref_len": df["ref_len"].astype(float),
        "alt_len": df["alt_len"].astype(float),
        "is_balanced": df["is_balanced"].astype(int),
    })
    labels.to_parquet(outdir / "labels.parquet", index=False)

    print(f"wrote {len(df)} records")
    print("manifest:", outdir / "manifest.jsonl")
    print(labels["consequence"].value_counts().to_string())
    print("\ntype_allele:")
    print(labels["type_allele"].value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
