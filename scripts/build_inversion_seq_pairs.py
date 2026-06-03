#!/usr/bin/env python3
"""Build ref/alt sequence-pair manifest for inversion-core SAE deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def cap(seq: str, max_len: int) -> str:
    seq = seq.upper()
    if len(seq) <= max_len:
        return seq
    left = max_len // 2
    return seq[:left] + seq[-(max_len - left):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/inversions/hprc_inversions_pilot.parquet")
    ap.add_argument("--output", default="data/inversions/seq_pairs.jsonl")
    ap.add_argument("--max-allele", type=int, default=4096)
    args = ap.parse_args()

    df = pd.read_parquet(ROOT / args.input).sort_values(
        ["chrom", "pos", "allele_index"]
    ).reset_index(drop=True)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in df.itertuples(index=False):
            rec = {
                "id": r.sv_id,
                "ref_seq": cap(r.ref, args.max_allele),
                "alt_seq": cap(r.alt, args.max_allele),
                "chrom": r.chrom,
                "pos": int(r.pos),
                "allele_index": int(r.allele_index),
                "max_allele": int(args.max_allele),
            }
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(df)} records -> {out}")
    print(f"max_allele={args.max_allele}")


if __name__ == "__main__":
    main()
