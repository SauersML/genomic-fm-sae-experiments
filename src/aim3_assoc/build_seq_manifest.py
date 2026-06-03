#!/usr/bin/env python3
"""Build Aim-3 haplotype consensus sequences for Evo2 embedding.

Consumes data/aim3_assoc/manifest.jsonl records with per-sample phased VCF
coordinates, applies the requested haplotype with bcftools consensus, and writes
sequence records {"id": ..., "seq": ...}. The output order matches the input
manifest order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from azure.run_consensus import consensus  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    recs: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _build_one(args: tuple[int, dict, str, str]) -> tuple[int, dict]:
    i, rec, fasta, root = args
    vcf = rec["vcf"]
    if not os.path.isabs(vcf):
        vcf = os.path.join(root, vcf)
    seq = consensus(
        vcf=vcf,
        fasta=fasta,
        chrom=rec["chrom"],
        start1=int(rec["start0"]) + 1,
        end1=int(rec["end0"]),
        sample=rec["sample"],
        hap=int(rec["hap"]),
    )
    return i, {"id": rec["id"], "seq": seq}


def _write_jsonl(path: Path, recs: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for rec in recs:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _validate(recs: list[dict], expected_n: int) -> dict:
    if len(recs) != expected_n:
        raise ValueError(f"sequence count mismatch: got {len(recs)} expected {expected_n}")
    lengths = [len(rec["seq"]) for rec in recs]
    bad = [rec["id"] for rec in recs if set(rec["seq"]) - set("ACGT")]
    if bad:
        raise ValueError(f"non-ACGT sequence(s), first={bad[0]}")
    return {
        "n": len(recs),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "mean_len": round(sum(lengths) / len(lengths), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/aim3_assoc/manifest.jsonl")
    ap.add_argument("--out", default="data/aim3_assoc/seq_manifest.jsonl")
    ap.add_argument("--job-dir", default="data/aim3_assoc/evo2_seq_job")
    ap.add_argument("--fasta", default=os.path.expanduser("~/hf_cache/hg38.fa"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)
    job_dir = Path(args.job_dir)
    recs = _read_jsonl(manifest)
    if not recs:
        raise ValueError(f"empty manifest: {manifest}")

    fasta = os.path.expanduser(args.fasta)
    if not Path(fasta).exists():
        raise FileNotFoundError(fasta)
    if not Path(fasta + ".fai").exists():
        raise FileNotFoundError(fasta + ".fai")

    rows: list[dict | None] = [None] * len(recs)
    work = [(i, rec, fasta, str(ROOT)) for i, rec in enumerate(recs)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for done, (i, row) in enumerate(ex.map(_build_one, work), start=1):
            rows[i] = row
            if done % 100 == 0 or done == len(recs):
                print(f"[seq] {done}/{len(recs)}", flush=True)

    seq_recs = [row for row in rows if row is not None]
    stats = _validate(seq_recs, len(recs))
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out, seq_recs)
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(job_dir / "manifest.jsonl", seq_recs)
    with (job_dir / "seq_meta.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
