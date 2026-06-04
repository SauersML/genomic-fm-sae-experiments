#!/usr/bin/env python3
"""Build Evo2-SAE manifests for the ancient-selection SNP pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATADIR = ROOT / "data" / "ancient_selection"
DEFAULT_SNPS = DATADIR / "snps_pilot.tsv"
REGION_MANIFEST = DATADIR / "manifest_regions.jsonl"
DELTA_MANIFEST = DATADIR / "manifest_delta.jsonl"


def _clean_id(value: object, idx: int) -> str:
    sid = str(value).strip()
    if sid and sid.lower() != "nan" and sid != ".":
        return sid
    return f"ancient_snp_{idx:05d}"


def build(snps_path: Path, outdir: Path, flank: int, max_allele: int) -> dict:
    df = pd.read_csv(snps_path, sep="\t")
    required = {"rsid", "chrom", "start0", "end0", "pos_hg38", "ref", "alt"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{snps_path} missing required columns: {missing}")

    outdir.mkdir(parents=True, exist_ok=True)
    region_path = outdir / REGION_MANIFEST.name
    delta_path = outdir / DELTA_MANIFEST.name

    ids: list[str] = []
    with region_path.open("w") as rfh, delta_path.open("w") as dfh:
        for i, row in df.reset_index(drop=True).iterrows():
            sid = _clean_id(row["rsid"], i)
            ids.append(sid)
            chrom = str(row["chrom"])
            region = {
                "id": sid,
                "chrom": chrom,
                "start0": int(row["start0"]),
                "end0": int(row["end0"]),
            }
            pos0 = int(row["pos_hg38"]) - 1
            delta = {
                "id": sid,
                "chrom": chrom,
                "start0": pos0,
                "end0": pos0 + 1,
                "ref": str(row["ref"]).upper(),
                "alt": str(row["alt"]).upper(),
                "flank": int(flank),
                "max_allele": int(max_allele),
            }
            rfh.write(json.dumps(region, separators=(",", ":")) + "\n")
            dfh.write(json.dumps(delta, separators=(",", ":")) + "\n")

    if len(set(ids)) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})[:10]
        raise ValueError(f"duplicate manifest ids; examples={dupes}")

    summary = {
        "snps": str(snps_path),
        "n": int(len(df)),
        "region_manifest": str(region_path),
        "delta_manifest": str(delta_path),
        "flank": int(flank),
        "max_allele": int(max_allele),
    }
    (outdir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snps", type=Path, default=DEFAULT_SNPS)
    ap.add_argument("--outdir", type=Path, default=DATADIR)
    ap.add_argument("--flank", type=int, default=2500)
    ap.add_argument("--max-allele", type=int, default=1)
    args = ap.parse_args()
    summary = build(args.snps, args.outdir, args.flank, args.max_allele)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
