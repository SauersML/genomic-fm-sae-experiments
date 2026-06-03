#!/usr/bin/env python3
"""Annotate Evo2 manifest rows with reusable genomic covariate sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

try:
    from src.common import region_covariates as cov
except ImportError:  # Allows direct execution: python src/common/annotate_manifest.py
    import region_covariates as cov  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write id-keyed repeat/mappability covariates for a manifest JSONL."
    )
    parser.add_argument("--manifest", required=True, help="Input manifest JSONL.")
    parser.add_argument("--out", required=True, help="Output TSV sidecar.")
    parser.add_argument(
        "--no-gene-density",
        action="store_true",
        help="Do not include GENCODE gene_density even if the cache is present.",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    include_gene_density = not args.no_gene_density and cov.GENCODE_FEATURES.exists()

    rows = []
    with manifest.open("r") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(_annotate_record(record, line_number, include_gene_density))

    columns = ["id", "chrom", "cov_start0", "cov_end0", "cov_bases", "repeat_frac", "mappability"]
    if include_gene_density:
        columns.append("gene_density")

    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")


def _annotate_record(
    record: dict[str, Any],
    line_number: int,
    include_gene_density: bool,
) -> dict[str, Any]:
    record_id = record.get("id")
    if record_id is None:
        raise ValueError(f"manifest line {line_number} has no id")
    chrom = record.get("chrom")
    intervals = manifest_reference_intervals(record)
    cov_start0 = min(start for start, _ in intervals)
    cov_end0 = max(end for _, end in intervals)
    cov_bases = sum(end - start for start, end in intervals)

    row = {
        "id": record_id,
        "chrom": chrom if chrom is not None else "",
        "cov_start0": cov_start0,
        "cov_end0": cov_end0,
        "cov_bases": cov_bases,
        "repeat_frac": _format_float(cov.interval_mean(cov.repeat_fraction, chrom, intervals)),
        "mappability": _format_float(cov.interval_mean(cov.mappability, chrom, intervals)),
    }
    if include_gene_density:
        row["gene_density"] = _format_float(cov.interval_mean(cov.gene_density, chrom, intervals))
    return row


def manifest_reference_intervals(record: dict[str, Any]) -> list[tuple[int, int]]:
    """Return reference-backed manifest window pieces for track lookup.

    Region records use [start0, end0). SV records with flank/max_allele use the
    same reference-backed pieces as the Evo2 ref window: left flank, capped ref
    interior, and right flank. ALT inserted sequence is not reference-backed and
    therefore cannot be assigned RepeatMasker or mappability track values here.
    """

    if "chrom" not in record:
        raise ValueError(f"record {record.get('id', '<unknown>')} has no chrom")
    if "start0" not in record or "end0" not in record:
        raise ValueError(f"record {record.get('id', '<unknown>')} has no start0/end0")

    start0 = int(record["start0"])
    end0 = int(record["end0"])
    if start0 < 0 or end0 <= start0:
        raise ValueError(f"record {record.get('id', '<unknown>')} has invalid coordinates")

    is_sv = "ref" in record or "alt" in record or "flank" in record or "max_allele" in record
    if not is_sv:
        return [(start0, end0)]

    flank = int(record.get("flank", 0))
    max_allele = int(record.get("max_allele", end0 - start0))
    intervals: list[tuple[int, int]] = []

    left_start = max(0, start0 - flank)
    if left_start < start0:
        intervals.append((left_start, start0))

    ref_len = end0 - start0
    if ref_len <= max_allele:
        intervals.append((start0, end0))
    elif max_allele > 0:
        head = max_allele // 2
        tail = max_allele - head
        if head > 0:
            intervals.append((start0, start0 + head))
        if tail > 0:
            intervals.append((end0 - tail, end0))

    if flank > 0:
        intervals.append((end0, end0 + flank))

    merged = cov._merge_intervals(intervals)
    if not merged:
        raise ValueError(f"record {record.get('id', '<unknown>')} has no annotatable bases")
    return merged


def _format_float(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.8g}"


if __name__ == "__main__":
    main()

