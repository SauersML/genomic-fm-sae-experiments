"""Shared job I/O + sequence building for the Azure warm-batch entrypoints.

A "job" is a directory containing `manifest.jsonl` (one JSON record per line).
The embed_* entrypoints read it, produce one feature vector per record (rows
aligned to `ids.txt`), and write `features.npy` (float32) + `meta.json`.

This module owns:
  - manifest reading
  - hg38.fa access (pysam-free: faidx via samtools through a tiny cache, or
    pyfaidx if available) for coord->sequence extraction
  - the Aim-1 ref/alt window splice (exact MANIFEST_SPEC.md geometry)
  - output writers
"""
from __future__ import annotations

import json
import os
import subprocess
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

HG38 = os.environ.get("HG38_FA", os.path.expanduser("~/hf_cache/hg38.fa"))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def read_manifest(jobdir: str) -> List[dict]:
    path = os.path.join(jobdir, "manifest.jsonl")
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


# --------------------------------------------------------------------------- #
# hg38 FASTA access
# --------------------------------------------------------------------------- #
class Genome:
    """0-based half-open sequence fetch from an indexed FASTA.

    Uses pyfaidx if present (fast, pure-python), else shells out to
    `samtools faidx` (1-based inclusive -> convert).
    """

    def __init__(self, fa: str = HG38):
        self.fa = fa
        if not os.path.exists(fa):
            raise FileNotFoundError(f"hg38 FASTA not found at {fa}")
        if not os.path.exists(fa + ".fai"):
            raise FileNotFoundError(
                f"{fa}.fai missing - run `samtools faidx {fa}`")
        self._fai = None
        try:
            from pyfaidx import Fasta  # type: ignore
            self._fai = Fasta(fa, sequence_always_upper=True, as_raw=True)
            self._mode = "pyfaidx"
        except Exception:
            self._mode = "samtools"

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        """genome[start0:end0], 0-based half-open, uppercase. Clamps at 0."""
        if end0 <= start0:
            return ""
        start0 = max(0, start0)
        if self._mode == "pyfaidx":
            return str(self._fai[chrom][start0:end0]).upper()
        # samtools faidx is 1-based inclusive
        region = f"{chrom}:{start0 + 1}-{end0}"
        out = subprocess.run(
            ["samtools", "faidx", self.fa, region],
            capture_output=True, text=True, check=True).stdout
        return "".join(out.splitlines()[1:]).upper()


@lru_cache(maxsize=1)
def get_genome() -> Genome:
    return Genome()


# --------------------------------------------------------------------------- #
# Aim-1 ref/alt window splice (MANIFEST_SPEC.md geometry)
# --------------------------------------------------------------------------- #
def _cap(s: str, m: int) -> str:
    if len(s) <= m:
        return s
    h = m // 2
    return s[:h] + s[-(m - h):]


def build_ref_alt_windows(rec: dict, genome: Genome) -> Tuple[str, str]:
    """Build (ref_window, alt_window) for an Aim-1 SV record per the spec.

    Record keys: chrom, start0(=L), end0(=R), ref, alt, flank, max_allele.
    Shared flanks; interior = capped REF span vs capped ALT.
    Asserts anchor base genome[L] == REF[0] when len(REF)>=1.
    """
    chrom = rec["chrom"]
    L = int(rec["start0"])
    R = int(rec["end0"])
    ref = rec["ref"]
    alt = rec["alt"]
    flank = int(rec.get("flank", 3072))
    max_allele = int(rec.get("max_allele", 1024))

    left_flank = genome.fetch(chrom, L - flank, L)
    right_flank = genome.fetch(chrom, R, R + flank)
    ref_interior = genome.fetch(chrom, L, R)  # == REF allele span

    # coordinate sanity: anchor base
    if len(ref) >= 1 and len(ref_interior) >= 1 and ref_interior[0] != ref[0].upper():
        raise ValueError(
            f"anchor mismatch {rec['id']}: genome[L]={ref_interior[0]} "
            f"REF[0]={ref[0]} (FASTA coordinate frame wrong)")

    ref_window = left_flank + _cap(ref_interior, max_allele) + right_flank
    alt_window = left_flank + _cap(alt.upper(), max_allele) + right_flank
    return ref_window, alt_window


# --------------------------------------------------------------------------- #
# Record classification -> what the embedder should do
# --------------------------------------------------------------------------- #
def record_kind(rec: dict) -> str:
    """Return one of: 'delta_seq', 'delta_coord', 'region_seq', 'region_coord'."""
    if "ref_seq" in rec and "alt_seq" in rec:
        return "delta_seq"
    if "ref" in rec and "alt" in rec and "chrom" in rec:
        return "delta_coord"
    if "seq" in rec:
        return "region_seq"
    if "chrom" in rec and "start0" in rec and "end0" in rec:
        return "region_coord"
    raise ValueError(f"cannot classify manifest record: keys={list(rec.keys())}")


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def write_outputs(jobdir: str, ids: List[str], feats: np.ndarray,
                  meta: dict) -> None:
    feats = np.asarray(feats, dtype=np.float32)
    assert feats.shape[0] == len(ids), (feats.shape, len(ids))
    np.save(os.path.join(jobdir, "features.npy"), feats)
    with open(os.path.join(jobdir, "ids.txt"), "w") as f:
        f.write("\n".join(ids) + "\n")
    with open(os.path.join(jobdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
