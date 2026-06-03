#!/usr/bin/env python3
"""Fast hg38 region covariates used to harden SAE feature analyses.

Coordinates are UCSC-style 0-based half-open intervals on chr-prefixed GRCh38.
The public tracks live under data/annotations:

  - rmsk.txt.gz from UCSC hg38 database dumps.
  - k100.Unique.Mappability.bb from the UCSC Umap track.
  - gencode_v44_features.pkl produced by build_gencode_features.py.
"""

from __future__ import annotations

import gzip
import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = REPO_ROOT / "data" / "annotations"
RMSK_TXT_GZ = ANNOTATION_DIR / "rmsk.txt.gz"
RMSK_CACHE = ANNOTATION_DIR / "rmsk_intervals.pkl"
UMAP_K100_BB = ANNOTATION_DIR / "umap_k100_unique_mappability.bb"
GENCODE_FEATURES = ANNOTATION_DIR / "gencode_v44_features.pkl"

Interval = tuple[int, int]


def repeat_fraction(chrom: str, start0: int, end0: int) -> float:
    """Return the fraction of an interval overlapped by RepeatMasker calls."""

    chrom, start0, end0 = _validate_interval(chrom, start0, end0)
    tracks = _load_repeat_intervals()
    return _coverage_fraction(tracks.get(chrom), start0, end0)


def mappability(chrom: str, start0: int, end0: int) -> float:
    """Return Umap k100 unique-mappability coverage fraction, or NaN if absent."""

    chrom, start0, end0 = _validate_interval(chrom, start0, end0)
    if not UMAP_K100_BB.exists():
        return math.nan
    try:
        bb = _load_umap_bigbed()
    except (ImportError, RuntimeError):
        return math.nan
    if chrom not in bb.chroms():
        return math.nan
    entries = bb.entries(chrom, start0, end0)
    if not entries:
        return 0.0
    intervals = _merge_intervals((max(start0, s), min(end0, e)) for s, e, _ in entries)
    return _covered_bases(intervals, start0, end0) / (end0 - start0)


def gene_density(chrom: str, start0: int, end0: int) -> float:
    """Return the fraction of an interval covered by any GENCODE v44 gene span."""

    chrom, start0, end0 = _validate_interval(chrom, start0, end0)
    tracks = _load_gene_intervals()
    return _coverage_fraction(tracks.get(chrom), start0, end0)


def gc_from_fasta(chrom: str, start0: int, end0: int, fasta_path: str | Path) -> float:
    """Return GC fraction from an indexed FASTA using pysam, or NaN if unavailable."""

    chrom, start0, end0 = _validate_interval(chrom, start0, end0)
    try:
        import pysam  # type: ignore
    except ImportError:
        return math.nan

    with pysam.FastaFile(str(fasta_path)) as fasta:
        seq = fasta.fetch(chrom, start0, end0).upper()
    called = sum(base in "ACGT" for base in seq)
    if called == 0:
        return math.nan
    return (seq.count("G") + seq.count("C")) / called


def interval_mean(
    func,
    chrom: str,
    intervals: Iterable[Interval],
) -> float:
    """Length-weighted mean of a single-interval covariate over interval pieces."""

    total = 0
    weighted = 0.0
    for start0, end0 in _merge_intervals(intervals):
        length = end0 - start0
        if length <= 0:
            continue
        value = func(chrom, start0, end0)
        if math.isnan(value):
            return math.nan
        total += length
        weighted += value * length
    if total == 0:
        return math.nan
    return weighted / total


def _validate_interval(chrom: str, start0: int, end0: int) -> tuple[str, int, int]:
    if not chrom:
        raise ValueError("chrom is required")
    chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    start0 = int(start0)
    end0 = int(end0)
    if start0 < 0:
        raise ValueError(f"start0 must be >= 0, got {start0}")
    if end0 <= start0:
        raise ValueError(f"end0 must be greater than start0, got {start0}-{end0}")
    return chrom, start0, end0


@lru_cache(maxsize=1)
def _load_repeat_intervals() -> dict[str, np.ndarray]:
    if RMSK_CACHE.exists() and RMSK_CACHE.stat().st_mtime >= RMSK_TXT_GZ.stat().st_mtime:
        with RMSK_CACHE.open("rb") as handle:
            return pickle.load(handle)
    if not RMSK_TXT_GZ.exists():
        raise FileNotFoundError(f"missing RepeatMasker track: {RMSK_TXT_GZ}")

    by_chrom: dict[str, list[Interval]] = {}
    with gzip.open(RMSK_TXT_GZ, "rt") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom = fields[5]
            if not chrom.startswith("chr"):
                continue
            start0 = int(fields[6])
            end0 = int(fields[7])
            if end0 > start0:
                by_chrom.setdefault(chrom, []).append((start0, end0))

    merged = {chrom: _interval_array(intervals) for chrom, intervals in by_chrom.items()}
    with RMSK_CACHE.open("wb") as handle:
        pickle.dump(merged, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return merged


@lru_cache(maxsize=1)
def _load_gene_intervals() -> dict[str, np.ndarray]:
    if not GENCODE_FEATURES.exists():
        return {}
    with GENCODE_FEATURES.open("rb") as handle:
        features = pickle.load(handle)
    genes = features.get("gene_any", {})
    return {chrom: np.asarray(intervals, dtype=np.int64) for chrom, intervals in genes.items()}


@lru_cache(maxsize=1)
def _load_umap_bigbed():
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise ImportError("pyBigWig is required to read the Umap bigBed") from exc
    bb = pyBigWig.open(str(UMAP_K100_BB))
    if bb is None or not bb.isBigBed():
        raise RuntimeError(f"could not open Umap bigBed: {UMAP_K100_BB}")
    return bb


def _coverage_fraction(intervals: np.ndarray | None, start0: int, end0: int) -> float:
    if intervals is None or len(intervals) == 0:
        return 0.0
    return _covered_bases_array(intervals, start0, end0) / (end0 - start0)


def _covered_bases_array(intervals: np.ndarray, start0: int, end0: int) -> int:
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    idx = int(np.searchsorted(ends, start0, side="right"))
    covered = 0
    while idx < len(intervals) and starts[idx] < end0:
        covered += max(0, min(int(ends[idx]), end0) - max(int(starts[idx]), start0))
        idx += 1
    return covered


def _covered_bases(intervals: list[Interval], start0: int, end0: int) -> int:
    covered = 0
    for start, end in intervals:
        if end <= start0:
            continue
        if start >= end0:
            break
        covered += max(0, min(end, end0) - max(start, start0))
    return covered


def _interval_array(intervals: Iterable[Interval]) -> np.ndarray:
    merged = _merge_intervals(intervals)
    if not merged:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(merged, dtype=np.int64)


def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    sorted_intervals = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    if not sorted_intervals:
        return []
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged

