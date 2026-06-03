#!/usr/bin/env python3
"""Per-token Evo2/Goodfire SAE analysis of genomic inversions.

This is designed to run on the Azure A100 box from the repository root. It
streams dense per-token SAE activations in chunks and only saves compact
statistics, selected feature traces, and plots.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
import time
import urllib.request
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False
    plt = None


AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# Coarse GRCh38 centromere spans. Windows are also rejected if they contain N.
CENTROMERES = {
    "chr1": (121_700_000, 125_100_000),
    "chr2": (91_800_000, 96_000_000),
    "chr3": (87_800_000, 94_000_000),
    "chr4": (48_200_000, 51_800_000),
    "chr5": (46_100_000, 51_400_000),
    "chr6": (58_500_000, 62_600_000),
    "chr7": (58_100_000, 62_100_000),
    "chr8": (43_200_000, 47_200_000),
    "chr9": (42_200_000, 45_500_000),
    "chr10": (38_000_000, 41_600_000),
    "chr11": (51_000_000, 55_800_000),
    "chr12": (33_200_000, 37_800_000),
    "chr13": (16_500_000, 18_900_000),
    "chr14": (16_100_000, 18_200_000),
    "chr15": (17_500_000, 20_500_000),
    "chr16": (35_300_000, 38_400_000),
    "chr17": (22_700_000, 27_400_000),
    "chr18": (15_400_000, 21_500_000),
    "chr19": (24_200_000, 28_100_000),
    "chr20": (25_700_000, 30_400_000),
    "chr21": (10_900_000, 13_000_000),
    "chr22": (13_700_000, 17_400_000),
}

RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass
class Window:
    id: str
    chrom: str
    start0: int
    end0: int
    seq: str
    gc: float
    repeat_frac: float | None
    genic: bool | None


@dataclass
class PairStats:
    pooled_delta: np.ndarray
    pos_l1: np.ndarray | None = None
    pos_l2: np.ndarray | None = None
    bp_l1_mean: float | None = None
    bp_l2_mean: float | None = None
    bp_abs_sum: np.ndarray | None = None
    bp_n: int = 0
    bin_abs_sum: np.ndarray | None = None
    bin_n: np.ndarray | None = None
    strand_pearson: float | None = None
    strand_spearman: float | None = None
    strand_active_pearson: float | None = None
    strand_active_spearman: float | None = None
    strand_cosine_mean: float | None = None
    scatter: np.ndarray | None = None


class Genome:
    def __init__(self, fasta: Path):
        self.fasta = Path(fasta)
        self.fai = Path(str(fasta) + ".fai")
        if not self.fasta.exists() or not self.fai.exists():
            raise FileNotFoundError(f"missing FASTA or index: {self.fasta}")
        self.lengths: dict[str, int] = {}
        with self.fai.open() as fh:
            for line in fh:
                fields = line.rstrip().split("\t")
                self.lengths[fields[0]] = int(fields[1])
        try:
            from pyfaidx import Fasta  # type: ignore

            self._fa = Fasta(str(fasta), sequence_always_upper=True, as_raw=True)
            self._mode = "pyfaidx"
        except Exception:
            self._fa = None
            self._mode = "samtools"

    def fetch(self, chrom: str, start0: int, end0: int) -> str:
        start0 = max(0, start0)
        end0 = min(self.lengths[chrom], end0)
        if end0 <= start0:
            return ""
        if self._mode == "pyfaidx":
            return str(self._fa[chrom][start0:end0]).upper()
        import subprocess

        region = f"{chrom}:{start0 + 1}-{end0}"
        out = subprocess.run(
            ["samtools", "faidx", str(self.fasta), region],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return "".join(out.splitlines()[1:]).upper()


def revcomp(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def gc_frac(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def overlaps_centromere(chrom: str, start0: int, end0: int, pad: int = 1_000_000) -> bool:
    cen = CENTROMERES.get(chrom)
    if cen is None:
        return False
    return start0 < cen[1] + pad and end0 > cen[0] - pad


def load_gene_intervals(gtf: Path | None) -> dict[str, list[tuple[int, int]]]:
    intervals = {chrom: [] for chrom in AUTOSOMES}
    if gtf is None or not gtf.exists():
        return intervals
    opener = gzip.open if str(gtf).endswith(".gz") else open
    with opener(gtf, "rt") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or fields[2] != "gene" or fields[0] not in intervals:
                continue
            intervals[fields[0]].append((int(fields[3]) - 1, int(fields[4])))
    return {chrom: merge_intervals(vals) for chrom, vals in intervals.items()}


def load_repeat_intervals(rmsk: Path | None) -> dict[str, list[tuple[int, int]]]:
    intervals = {chrom: [] for chrom in AUTOSOMES}
    if rmsk is None or not rmsk.exists():
        return intervals
    opener = gzip.open if str(rmsk).endswith(".gz") else open
    with opener(rmsk, "rt") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or fields[5] not in intervals:
                continue
            intervals[fields[5]].append((int(fields[6]), int(fields[7])))
    return {chrom: merge_intervals(vals) for chrom, vals in intervals.items()}


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        elif end > merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
    return merged


def interval_overlap_bp(intervals: list[tuple[int, int]], start0: int, end0: int) -> int:
    if not intervals:
        return 0
    starts = [x[0] for x in intervals]
    i = max(0, bisect_left(starts, start0) - 1)
    total = 0
    while i < len(intervals) and intervals[i][0] < end0:
        total += max(0, min(end0, intervals[i][1]) - max(start0, intervals[i][0]))
        i += 1
    return total


def sample_windows(
    genome: Genome,
    n: int,
    win_len: int,
    gene_intervals: dict[str, list[tuple[int, int]]],
    repeat_intervals: dict[str, list[tuple[int, int]]],
    seed: int,
) -> list[Window]:
    rng = random.Random(seed)
    chrom_weights = np.array([genome.lengths[c] for c in AUTOSOMES], dtype=np.float64)
    chrom_weights /= chrom_weights.sum()
    targets = {True: n // 2, False: n - n // 2}
    got: dict[bool, list[Window]] = {True: [], False: []}
    attempts = 0
    max_attempts = n * 3000
    while sum(len(v) for v in got.values()) < n and attempts < max_attempts:
        attempts += 1
        chrom = rng.choices(AUTOSOMES, weights=chrom_weights, k=1)[0]
        length = genome.lengths[chrom]
        start = rng.randint(1_000_000, length - win_len - 1_000_000)
        end = start + win_len
        if overlaps_centromere(chrom, start, end):
            continue
        seq = genome.fetch(chrom, start, end)
        if len(seq) != win_len or any(base not in "ACGT" for base in seq):
            continue
        genic = interval_overlap_bp(gene_intervals.get(chrom, []), start, end) > 0
        if len(got[genic]) >= targets[genic]:
            continue
        repeat_bp = interval_overlap_bp(repeat_intervals.get(chrom, []), start, end)
        got[genic].append(
            Window(
                id=f"syn_{len(got[True]) + len(got[False]):04d}_{chrom}_{start}",
                chrom=chrom,
                start0=start,
                end0=end,
                seq=seq,
                gc=gc_frac(seq),
                repeat_frac=repeat_bp / win_len if repeat_intervals.get(chrom) else None,
                genic=genic if any(gene_intervals.values()) else None,
            )
        )
    windows = got[True] + got[False]
    if len(windows) < n:
        raise RuntimeError(f"only sampled {len(windows)} clean windows after {attempts} attempts")
    rng.shuffle(windows)
    return windows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = math.sqrt(float(np.dot(xm, xm) * np.dot(ym, ym)))
    if denom == 0:
        return float("nan")
    return float(np.dot(xm, ym) / denom)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata_average(np.asarray(x)), rankdata_average(np.asarray(y)))


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 1000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def make_alt(seq: str, inv_size: int) -> tuple[str, int, int]:
    left = (len(seq) - inv_size) // 2
    right = left + inv_size
    return seq[:left] + revcomp(seq[left:right]) + seq[right:], left, right


@torch.no_grad()
def encode_dense(sae, acts: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return sae.encode(acts[start:end], topk=True)


@torch.no_grad()
def analyze_pair(
    evo2x,
    sae,
    ref_seq: str,
    alt_seq: str,
    inv_start: int,
    inv_end: int,
    *,
    collect_profile: bool,
    collect_breakpoint: bool,
    collect_bins: bool,
    collect_strand: bool,
    chunk: int,
    breakpoint_width: int,
    nbins: int,
    scatter_features: int,
) -> PairStats:
    ref_acts, alt_acts = evo2x.embed_dna([ref_seq, alt_seq], device="cuda")
    L = min(ref_acts.shape[0], alt_acts.shape[0], len(ref_seq), len(alt_seq))
    d_sae = sae.d_sae
    pooled_sum = torch.zeros(d_sae, dtype=torch.float32, device="cpu")
    pos_l1 = np.zeros(L, dtype=np.float32) if collect_profile else None
    pos_l2 = np.zeros(L, dtype=np.float32) if collect_profile else None
    bp_abs_sum = torch.zeros(d_sae, dtype=torch.float32, device="cpu") if collect_breakpoint else None
    bp_l1_vals: list[np.ndarray] = []
    bp_l2_vals: list[np.ndarray] = []
    bp_n = 0
    bin_abs_sum = (
        np.zeros((nbins, d_sae), dtype=np.float32) if collect_bins else None
    )
    bin_n = np.zeros(nbins, dtype=np.int64) if collect_bins else None

    for start in range(0, L, chunk):
        end = min(L, start + chunk)
        rf = encode_dense(sae, ref_acts, start, end)
        af = encode_dense(sae, alt_acts, start, end)
        delta = (af - rf).float()
        pooled_sum += delta.sum(0).cpu()
        if collect_profile:
            abs_delta = delta.abs()
            l1 = abs_delta.sum(1)
            l2 = torch.linalg.vector_norm(delta, dim=1)
            pos_l1[start:end] = l1.cpu().numpy()
            pos_l2[start:end] = l2.cpu().numpy()
        if collect_breakpoint:
            positions = torch.arange(start, end, device=delta.device)
            mask = (positions - inv_start).abs() <= breakpoint_width
            mask |= (positions - inv_end).abs() <= breakpoint_width
            if bool(mask.any()):
                d_bp = delta[mask]
                abs_bp = d_bp.abs()
                bp_abs_sum += abs_bp.sum(0).cpu()
                bp_l1_vals.append(abs_bp.sum(1).cpu().numpy())
                bp_l2_vals.append(torch.linalg.vector_norm(d_bp, dim=1).cpu().numpy())
                bp_n += int(mask.sum().item())
        if collect_bins:
            abs_np = delta.abs().cpu().numpy()
            bins = np.minimum((np.arange(start, end) * nbins) // L, nbins - 1)
            for b in np.unique(bins):
                sel = bins == b
                bin_abs_sum[b] += abs_np[sel].sum(axis=0)
                bin_n[b] += int(sel.sum())
        del rf, af, delta
    pooled_delta = (pooled_sum / L).numpy().astype(np.float32)

    strand_pearson = None
    strand_spearman = None
    strand_active_pearson = None
    strand_active_spearman = None
    strand_cosine_mean = None
    scatter = None
    if collect_strand:
        ref_sum = torch.zeros(d_sae, dtype=torch.float32, device="cpu")
        alt_sum = torch.zeros(d_sae, dtype=torch.float32, device="cpu")
        cos_sum = 0.0
        cos_n = 0
        for start in range(inv_start, inv_end, chunk):
            end = min(inv_end, start + chunk)
            alt_start = inv_start + (inv_end - end)
            alt_end = inv_start + (inv_end - start)
            rf = encode_dense(sae, ref_acts, start, end).float()
            af = torch.flip(encode_dense(sae, alt_acts, alt_start, alt_end), dims=[0]).float()
            ref_sum += rf.sum(0).cpu()
            alt_sum += af.sum(0).cpu()
            denom = torch.linalg.vector_norm(rf, dim=1) * torch.linalg.vector_norm(af, dim=1)
            dots = (rf * af).sum(1)
            valid = denom > 0
            if bool(valid.any()):
                cos_sum += float((dots[valid] / denom[valid]).sum().item())
                cos_n += int(valid.sum().item())
            del rf, af
        x = (ref_sum / (inv_end - inv_start)).numpy()
        y = (alt_sum / (inv_end - inv_start)).numpy()
        strand_pearson = pearson(x, y)
        strand_spearman = spearman(x, y)
        active = (x > 0) | (y > 0)
        if int(active.sum()) > 3:
            strand_active_pearson = pearson(x[active], y[active])
            strand_active_spearman = spearman(x[active], y[active])
        strand_cosine_mean = cos_sum / cos_n if cos_n else float("nan")
        if scatter_features > 0:
            score = x + y
            take = np.argsort(score)[-min(scatter_features, len(score)) :]
            scatter = np.stack([x[take], y[take]], axis=1).astype(np.float32)

    del ref_acts, alt_acts
    torch.cuda.empty_cache()
    return PairStats(
        pooled_delta=pooled_delta,
        pos_l1=pos_l1,
        pos_l2=pos_l2,
        bp_l1_mean=float(np.concatenate(bp_l1_vals).mean()) if bp_l1_vals else None,
        bp_l2_mean=float(np.concatenate(bp_l2_vals).mean()) if bp_l2_vals else None,
        bp_abs_sum=bp_abs_sum.numpy().astype(np.float32) if bp_abs_sum is not None else None,
        bp_n=bp_n,
        bin_abs_sum=bin_abs_sum,
        bin_n=bin_n,
        strand_pearson=strand_pearson,
        strand_spearman=strand_spearman,
        strand_active_pearson=strand_active_pearson,
        strand_active_spearman=strand_active_spearman,
        strand_cosine_mean=strand_cosine_mean,
        scatter=scatter,
    )


def load_evo2(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from src.evo2 import extract as evo2x

    sae = evo2x.load_sae(device="cuda", dtype=torch.bfloat16)
    evo2x._load_evo2()
    return evo2x, sae


def parse_gnomad_inversions(url: str, max_records: int) -> list[dict]:
    rows: list[dict] = []
    with urllib.request.urlopen(url, timeout=120) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            text = (line.decode("utf-8") for line in gz)
            reader = csv.DictReader(text, delimiter="\t")
            for rec in reader:
                if rec["svtype"] != "INV" or rec["#chrom"] not in AUTOSOMES:
                    continue
                svlen = abs(int(float(rec.get("SVLEN") or rec["end"])))
                if 500 <= svlen <= 4000:
                    rows.append(
                        {
                            "id": rec["name"],
                            "chrom": rec["#chrom"],
                            "start0": int(rec["start"]),
                            "end0": int(rec["end"]),
                            "svlen": svlen,
                            "af": float(rec.get("AF") or "nan"),
                            "source": "gnomAD v4.1 non-neuro controls sites BED",
                        }
                    )
                if len(rows) >= max_records * 10:
                    break
    rows.sort(key=lambda x: (-(x["af"] if math.isfinite(x["af"]) else 0.0), x["chrom"], x["start0"]))
    return rows[:max_records]


def pick_matched_control(
    real: dict,
    windows: list[Window],
    used: set[str],
    repeat_available: bool,
) -> Window | None:
    best = None
    best_score = float("inf")
    for w in windows:
        if w.id in used:
            continue
        score = abs(w.gc - real["gc"])
        if repeat_available and real.get("repeat_frac") is not None and w.repeat_frac is not None:
            score += abs(w.repeat_frac - real["repeat_frac"])
        if score < best_score:
            best_score = score
            best = w
    if best is not None:
        used.add(best.id)
    return best


def load_aim1_norms(repo_root: Path) -> dict[str, np.ndarray]:
    data_dir = repo_root / "data" / "aim1_sv"
    features_path = data_dir / "features.npy"
    labels_path = data_dir / "labels.parquet"
    labels_csv_path = data_dir / "labels.csv"
    ids_path = data_dir / "ids.txt"
    if not features_path.exists() or not ids_path.exists():
        return {}
    import pandas as pd

    X = np.load(features_path, mmap_mode="r")
    ids = ids_path.read_text().splitlines()
    if labels_csv_path.exists():
        labels = pd.read_csv(labels_csv_path).set_index("id").loc[ids]
    elif labels_path.exists():
        labels = pd.read_parquet(labels_path).set_index("id").loc[ids]
    else:
        return {}
    norms = {}
    l2 = np.linalg.norm(np.asarray(X), axis=1)
    l1 = np.abs(np.asarray(X)).sum(axis=1)
    for svtype in ("DEL", "INS"):
        mask = labels["svtype"].to_numpy() == svtype
        size = labels["svlen_abs"].to_numpy(dtype=float)
        matched = mask & (size >= 500) & (size <= 8000)
        norms[f"{svtype}_l2"] = l2[matched].astype(np.float32)
        norms[f"{svtype}_l1"] = l1[matched].astype(np.float32)
        norms[f"{svtype}_size"] = size[matched].astype(np.float32)
    return norms


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_windows(path: Path, windows: list[Window]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "chrom", "start0", "end0", "gc", "repeat_frac", "genic"],
        )
        writer.writeheader()
        for w in windows:
            writer.writerow(
                {
                    "id": w.id,
                    "chrom": w.chrom,
                    "start0": w.start0,
                    "end0": w.end0,
                    "gc": w.gc,
                    "repeat_frac": w.repeat_frac,
                    "genic": w.genic,
                }
            )


def plot_profile(plot_dir: Path, mean_l1: np.ndarray, mean_l2: np.ndarray, inv_start: int, inv_end: int) -> None:
    if not HAVE_MPL:
        return
    x = np.arange(len(mean_l1))
    fig, ax = plt.subplots(figsize=(10, 4), dpi=180)
    ax.plot(x, mean_l1, lw=1.2, label="mean per-token L1")
    ax2 = ax.twinx()
    ax2.plot(x, mean_l2, lw=1.0, color="#C44536", label="mean per-token L2")
    ax.axvspan(inv_start, inv_end, color="#E9C46A", alpha=0.18, label="inverted interval")
    for pos in (inv_start, inv_end):
        ax.axvline(pos, color="black", lw=0.9, ls="--")
    ax.set_title("Synthetic 4 kb inversion: per-token SAE delta localizes to breakpoints")
    ax.set_xlabel("8 kb window position")
    ax.set_ylabel("L1 delta")
    ax2.set_ylabel("L2 delta")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "inversion_delta_profile.png")
    plt.close(fig)


def plot_strand(plot_dir: Path, scatter: np.ndarray, pearsons: np.ndarray, active_pearsons: np.ndarray) -> None:
    if not HAVE_MPL:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    if len(scatter):
        axes[0].hexbin(scatter[:, 0], scatter[:, 1], gridsize=60, mincnt=1, cmap="viridis")
    lim = float(np.nanpercentile(scatter, 99.5)) if len(scatter) else 1.0
    axes[0].plot([0, lim], [0, lim], color="white", lw=0.8, alpha=0.8)
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel("ref forward interior pooled SAE")
    axes[0].set_ylabel("alt revcomp interior pooled SAE")
    axes[0].set_title("Strand-symmetry scatter")
    axes[1].hist(pearsons, bins=30, alpha=0.7, label="all features")
    axes[1].hist(active_pearsons[np.isfinite(active_pearsons)], bins=30, alpha=0.7, label="active union")
    axes[1].set_xlabel("per-window Pearson r")
    axes[1].set_ylabel("windows")
    axes[1].set_title("Interior ref vs revcomp correlation")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "strand_symmetry_scatter.png")
    plt.close(fig)


def plot_size(plot_dir: Path, size_rows: list[dict]) -> None:
    if not HAVE_MPL:
        return
    sizes = sorted({r["size"] for r in size_rows})
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=180)
    means = []
    lows = []
    highs = []
    for size in sizes:
        vals = np.array([r["pooled_l2"] for r in size_rows if r["size"] == size])
        ci = bootstrap_ci(vals, seed=100 + size)
        means.append(float(vals.mean()))
        lows.append(ci[0])
        highs.append(ci[1])
    ax.errorbar(sizes, means, yerr=[np.array(means) - lows, np.array(highs) - means], marker="o", capsize=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [f"{s//1000} kb" for s in sizes])
    ax.set_xlabel("inverted segment size")
    ax.set_ylabel("pooled SAE delta L2")
    ax.set_title("Synthetic inversion visibility grows with size")
    fig.tight_layout()
    fig.savefig(plot_dir / "inversion_size_dependence.png")
    plt.close(fig)


def plot_breakpoint_heatmap(plot_dir: Path, heat: np.ndarray, top_features: list[int], inv_start: int, inv_end: int, win_len: int) -> None:
    if not HAVE_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5), dpi=180)
    im = ax.imshow(heat, aspect="auto", cmap="magma", interpolation="nearest")
    nbins = heat.shape[1]
    for pos in (inv_start, inv_end):
        ax.axvline(pos / win_len * nbins, color="white", lw=0.9, ls="--")
    ax.set_yticks(np.arange(len(top_features)), [str(f) for f in top_features], fontsize=7)
    ax.set_xlabel("position bins across 8 kb window")
    ax.set_ylabel("SAE feature id")
    ax.set_title("Top breakpoint SAE-feature signatures")
    fig.colorbar(im, ax=ax, label="mean |delta|")
    fig.tight_layout()
    fig.savefig(plot_dir / "breakpoint_feature_heatmap.png")
    plt.close(fig)


def plot_indel(plot_dir: Path, inv_l2: np.ndarray, aim1_norms: dict[str, np.ndarray]) -> None:
    if not HAVE_MPL:
        return
    data = [inv_l2]
    labels = ["synthetic INV\n1-4 kb"]
    for svtype in ("DEL", "INS"):
        vals = aim1_norms.get(f"{svtype}_l2")
        if vals is not None and len(vals):
            data.append(vals)
            labels.append(f"Aim1 {svtype}\n0.5-8 kb")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("pooled SAE delta L2")
    ax.set_title("Inversion pooled deltas compared with Aim1 indels")
    fig.tight_layout()
    fig.savefig(plot_dir / "inversion_vs_indel_delta_distribution.png")
    plt.close(fig)


def plot_real(plot_dir: Path, rows: list[dict]) -> None:
    if not HAVE_MPL:
        return
    if not rows:
        return
    real = np.array([r["real_bp_l2"] for r in rows], dtype=np.float64)
    ctrl = np.array([r["control_bp_l2"] for r in rows], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=180)
    axes[0].scatter(ctrl, real, s=22, alpha=0.8)
    lim = max(float(np.nanmax(real)), float(np.nanmax(ctrl)))
    axes[0].plot([0, lim], [0, lim], color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("matched control breakpoint L2")
    axes[0].set_ylabel("real gnomAD INV breakpoint L2")
    axes[0].set_title("Real inversions vs matched controls")
    axes[1].hist(real - ctrl, bins=20, color="#386641", alpha=0.8)
    axes[1].axvline(0, color="black", lw=0.9)
    axes[1].set_xlabel("real - control breakpoint L2")
    axes[1].set_ylabel("pairs")
    axes[1].set_title("Paired null")
    fig.tight_layout()
    fig.savefig(plot_dir / "real_inversions_matched_controls.png")
    plt.close(fig)


def permutation_paired_p(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=np.float64)
    obs = abs(float(diffs.mean()))
    if len(diffs) == 0:
        return float("nan")
    rng = np.random.default_rng(123)
    n_perm = 5000
    ge = 1
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        stat = abs(float((diffs * signs).mean()))
        if stat >= obs:
            ge += 1
    return ge / (n_perm + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--fasta", default=str(Path.home() / "hf_cache" / "hg38.fa"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2603)
    ap.add_argument("--win-len", type=int, default=8000)
    ap.add_argument("--sizes", default="1000,2000,4000")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--nbins", type=int, default=160)
    ap.add_argument("--breakpoint-width", type=int, default=64)
    ap.add_argument("--scatter-features", type=int, default=256)
    ap.add_argument("--real-n", type=int, default=50)
    ap.add_argument("--outdir", default="results/inversions")
    ap.add_argument("--plotdir", default="plots")
    ap.add_argument("--gnomad-url", default="https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/genome_sv/gnomad.v4.1.sv.non_neuro_controls.sites.bed.gz")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    outdir = repo_root / args.outdir
    plotdir = repo_root / args.plotdir
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)
    genome = Genome(Path(args.fasta))
    gtf = repo_root / "data" / "annotations" / "gencode.v44.annotation.gtf.gz"
    rmsk = repo_root / "data" / "annotations" / "rmsk.txt.gz"
    print("[load] annotations", flush=True)
    gene_intervals = load_gene_intervals(gtf)
    repeat_intervals = load_repeat_intervals(rmsk)
    repeat_available = any(repeat_intervals.values())
    print("[sample] synthetic windows", flush=True)
    windows = sample_windows(
        genome,
        args.n,
        args.win_len,
        gene_intervals,
        repeat_intervals,
        args.seed,
    )
    write_windows(outdir / "synthetic_windows.csv", windows)

    print("[load] Evo2 + Goodfire SAE", flush=True)
    evo2x, sae = load_evo2(repo_root)
    sizes = [int(x) for x in args.sizes.split(",") if x]
    main_size = max(sizes)
    main_left = (args.win_len - main_size) // 2
    main_right = main_left + main_size

    profile_l1_sum = np.zeros(args.win_len, dtype=np.float64)
    profile_l2_sum = np.zeros(args.win_len, dtype=np.float64)
    bp_abs_sum = np.zeros(sae.d_sae, dtype=np.float64)
    bp_n = 0
    bin_abs_sum = np.zeros((args.nbins, sae.d_sae), dtype=np.float64)
    bin_n = np.zeros(args.nbins, dtype=np.int64)
    size_rows: list[dict] = []
    main_pooled = []
    strand_rows = []
    scatter_parts = []

    for wi, w in enumerate(windows, start=1):
        for size in sizes:
            alt, inv_start, inv_end = make_alt(w.seq, size)
            collect = size == main_size
            stats = analyze_pair(
                evo2x,
                sae,
                w.seq,
                alt,
                inv_start,
                inv_end,
                collect_profile=collect,
                collect_breakpoint=collect,
                collect_bins=collect,
                collect_strand=collect,
                chunk=args.chunk,
                breakpoint_width=args.breakpoint_width,
                nbins=args.nbins,
                scatter_features=args.scatter_features if collect and wi <= 120 else 0,
            )
            pooled_l1 = float(np.abs(stats.pooled_delta).sum())
            pooled_l2 = float(np.linalg.norm(stats.pooled_delta))
            size_rows.append(
                {
                    "id": w.id,
                    "size": size,
                    "pooled_l1": pooled_l1,
                    "pooled_l2": pooled_l2,
                    "gc": w.gc,
                    "repeat_frac": w.repeat_frac,
                    "genic": w.genic,
                }
            )
            if collect:
                main_pooled.append(stats.pooled_delta)
                profile_l1_sum += stats.pos_l1[: args.win_len]
                profile_l2_sum += stats.pos_l2[: args.win_len]
                bp_abs_sum += stats.bp_abs_sum
                bp_n += stats.bp_n
                bin_abs_sum += stats.bin_abs_sum
                bin_n += stats.bin_n
                strand_rows.append(
                    {
                        "id": w.id,
                        "pearson": stats.strand_pearson,
                        "spearman": stats.strand_spearman,
                        "active_pearson": stats.strand_active_pearson,
                        "active_spearman": stats.strand_active_spearman,
                        "token_cosine_mean": stats.strand_cosine_mean,
                    }
                )
                if stats.scatter is not None:
                    scatter_parts.append(stats.scatter)
        if wi % 10 == 0:
            elapsed = time.time() - t0
            print(f"[synthetic] {wi}/{len(windows)} windows elapsed={elapsed/60:.1f} min", flush=True)

    mean_l1 = profile_l1_sum / len(windows)
    mean_l2 = profile_l2_sum / len(windows)
    bp_mean_abs = bp_abs_sum / max(1, bp_n)
    top_features = np.argsort(bp_mean_abs)[-20:][::-1].astype(int).tolist()
    heat = (bin_abs_sum[:, top_features] / np.maximum(bin_n[:, None], 1)).T

    np.save(outdir / "synthetic_4kb_pooled_delta.npy", np.stack(main_pooled).astype(np.float32))
    np.save(outdir / "synthetic_profile_l1.npy", mean_l1.astype(np.float32))
    np.save(outdir / "synthetic_profile_l2.npy", mean_l2.astype(np.float32))
    np.save(outdir / "breakpoint_feature_heatmap.npy", heat.astype(np.float32))
    np.save(outdir / "breakpoint_mean_abs_delta.npy", bp_mean_abs.astype(np.float32))

    with (outdir / "size_dependence.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(size_rows[0]))
        writer.writeheader()
        writer.writerows(size_rows)
    with (outdir / "strand_symmetry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(strand_rows[0]))
        writer.writeheader()
        writer.writerows(strand_rows)
    with (outdir / "top_breakpoint_features.tsv").open("w") as fh:
        fh.write("rank\tfeature\tmean_abs_delta_at_breakpoints\n")
        for rank, feat in enumerate(top_features, start=1):
            fh.write(f"{rank}\t{feat}\t{bp_mean_abs[feat]:.8g}\n")

    scatter = np.concatenate(scatter_parts, axis=0) if scatter_parts else np.zeros((0, 2), dtype=np.float32)
    np.save(outdir / "strand_scatter_points.npy", scatter.astype(np.float32))

    strand_pearson = np.array([r["pearson"] for r in strand_rows], dtype=np.float64)
    strand_spearman = np.array([r["spearman"] for r in strand_rows], dtype=np.float64)
    strand_active_pearson = np.array([r["active_pearson"] for r in strand_rows], dtype=np.float64)
    strand_active_spearman = np.array([r["active_spearman"] for r in strand_rows], dtype=np.float64)
    strand_cos = np.array([r["token_cosine_mean"] for r in strand_rows], dtype=np.float64)

    aim1_norms = load_aim1_norms(repo_root)
    inv_all_l2 = np.array([r["pooled_l2"] for r in size_rows], dtype=np.float32)
    if aim1_norms:
        np.savez(outdir / "aim1_indel_norms_matched_size.npz", **aim1_norms)

    real_rows = []
    real_source_status = {"attempted": True, "source": args.gnomad_url}
    try:
        candidates = parse_gnomad_inversions(args.gnomad_url, args.real_n * 3)
        clean_reals = []
        for rec in candidates:
            flank = (args.win_len - rec["svlen"]) // 2
            start = rec["start0"] - flank
            end = rec["end0"] + (args.win_len - rec["svlen"] - flank)
            if start < 0 or end > genome.lengths[rec["chrom"]] or overlaps_centromere(rec["chrom"], start, end):
                continue
            seq = genome.fetch(rec["chrom"], start, end)
            if len(seq) != args.win_len or any(base not in "ACGT" for base in seq):
                continue
            rec["window_start0"] = start
            rec["window_end0"] = end
            rec["seq"] = seq
            rec["gc"] = gc_frac(seq)
            rec["repeat_frac"] = (
                interval_overlap_bp(repeat_intervals.get(rec["chrom"], []), start, end) / args.win_len
                if repeat_available
                else None
            )
            clean_reals.append(rec)
            if len(clean_reals) >= args.real_n:
                break
        used_controls: set[str] = set()
        for ri, rec in enumerate(clean_reals, start=1):
            control = pick_matched_control(rec, windows, used_controls, repeat_available)
            if control is None:
                continue
            real_alt = rec["seq"][: rec["start0"] - rec["window_start0"]]
            rel_start = rec["start0"] - rec["window_start0"]
            rel_end = rec["end0"] - rec["window_start0"]
            real_alt = rec["seq"][:rel_start] + revcomp(rec["seq"][rel_start:rel_end]) + rec["seq"][rel_end:]
            real_stats = analyze_pair(
                evo2x,
                sae,
                rec["seq"],
                real_alt,
                rel_start,
                rel_end,
                collect_profile=False,
                collect_breakpoint=True,
                collect_bins=False,
                collect_strand=False,
                chunk=args.chunk,
                breakpoint_width=args.breakpoint_width,
                nbins=args.nbins,
                scatter_features=0,
            )
            ctrl_alt, ctrl_start, ctrl_end = make_alt(control.seq, rec["svlen"])
            ctrl_stats = analyze_pair(
                evo2x,
                sae,
                control.seq,
                ctrl_alt,
                ctrl_start,
                ctrl_end,
                collect_profile=False,
                collect_breakpoint=True,
                collect_bins=False,
                collect_strand=False,
                chunk=args.chunk,
                breakpoint_width=args.breakpoint_width,
                nbins=args.nbins,
                scatter_features=0,
            )
            real_rows.append(
                {
                    "id": rec["id"],
                    "chrom": rec["chrom"],
                    "start0": rec["start0"],
                    "end0": rec["end0"],
                    "svlen": rec["svlen"],
                    "af": rec["af"],
                    "gc": rec["gc"],
                    "repeat_frac": rec["repeat_frac"],
                    "control_id": control.id,
                    "control_gc": control.gc,
                    "control_repeat_frac": control.repeat_frac,
                    "real_bp_l2": real_stats.bp_l2_mean,
                    "control_bp_l2": ctrl_stats.bp_l2_mean,
                    "real_bp_l1": real_stats.bp_l1_mean,
                    "control_bp_l1": ctrl_stats.bp_l1_mean,
                }
            )
            if ri % 10 == 0:
                print(f"[real] {ri}/{len(clean_reals)}", flush=True)
        real_source_status.update({"n_candidates": len(candidates), "n_clean": len(clean_reals), "n_tested": len(real_rows)})
    except Exception as exc:
        real_source_status.update({"error": repr(exc), "n_tested": 0})
        print(f"[real] skipped: {exc!r}", flush=True)

    if real_rows:
        with (outdir / "real_inversions_matched_controls.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(real_rows[0]))
            writer.writeheader()
            writer.writerows(real_rows)

    plot_profile(plotdir, mean_l1, mean_l2, main_left, main_right)
    plot_profile(outdir, mean_l1, mean_l2, main_left, main_right)
    plot_strand(plotdir, scatter, strand_pearson, strand_active_pearson)
    plot_strand(outdir, scatter, strand_pearson, strand_active_pearson)
    plot_size(plotdir, size_rows)
    plot_size(outdir, size_rows)
    plot_breakpoint_heatmap(plotdir, heat, top_features, main_left, main_right, args.win_len)
    plot_breakpoint_heatmap(outdir, heat, top_features, main_left, main_right, args.win_len)
    if aim1_norms:
        plot_indel(plotdir, inv_all_l2, aim1_norms)
        plot_indel(outdir, inv_all_l2, aim1_norms)
    plot_real(plotdir, real_rows)
    plot_real(outdir, real_rows)

    size_summary = {}
    for size in sizes:
        vals_l2 = np.array([r["pooled_l2"] for r in size_rows if r["size"] == size], dtype=np.float64)
        vals_l1 = np.array([r["pooled_l1"] for r in size_rows if r["size"] == size], dtype=np.float64)
        size_summary[str(size)] = {
            "n": int(len(vals_l2)),
            "pooled_l2_mean": float(vals_l2.mean()),
            "pooled_l2_ci95": bootstrap_ci(vals_l2, seed=args.seed + size),
            "pooled_l1_mean": float(vals_l1.mean()),
            "pooled_l1_ci95": bootstrap_ci(vals_l1, seed=args.seed + size + 1),
        }

    breakpoint = {
        "breakpoint_width_tokens": args.breakpoint_width,
        "n_breakpoint_tokens": int(bp_n),
        "mean_l1_breakpoint": float(np.r_[mean_l1[main_left - args.breakpoint_width : main_left + args.breakpoint_width + 1], mean_l1[main_right - args.breakpoint_width : main_right + args.breakpoint_width + 1]].mean()),
        "mean_l1_interior_mid": float(mean_l1[main_left + 512 : main_right - 512].mean()),
        "mean_l1_flanks": float(np.r_[mean_l1[: main_left - 512], mean_l1[main_right + 512 :]].mean()),
        "top_features": top_features,
    }

    real_summary = {"source_status": real_source_status}
    if real_rows:
        diffs = np.array([r["real_bp_l2"] - r["control_bp_l2"] for r in real_rows], dtype=np.float64)
        real_summary.update(
            {
                "n": len(real_rows),
                "real_bp_l2_mean": float(np.mean([r["real_bp_l2"] for r in real_rows])),
                "control_bp_l2_mean": float(np.mean([r["control_bp_l2"] for r in real_rows])),
                "paired_diff_mean": float(diffs.mean()),
                "paired_diff_ci95": bootstrap_ci(diffs, seed=42),
                "paired_signflip_p": permutation_paired_p(diffs),
            }
        )

    aim1_summary = {}
    for svtype in ("DEL", "INS"):
        vals = aim1_norms.get(f"{svtype}_l2")
        if vals is not None and len(vals):
            aim1_summary[svtype] = {
                "n_0p5_to_8kb": int(len(vals)),
                "pooled_l2_mean": float(vals.mean()),
                "pooled_l2_ci95": bootstrap_ci(vals, seed=args.seed + len(vals)),
            }

    summary = {
        "seed": args.seed,
        "n_synthetic_windows": args.n,
        "window_length": args.win_len,
        "sizes": sizes,
        "model": "evo2_7b",
        "sae": "Goodfire/Evo-2-Layer-26-Mixed layer 26",
        "sae_dtype": "bfloat16",
        "chunk_tokens": args.chunk,
        "sample_metadata": {
            "genic_windows": int(sum(1 for w in windows if w.genic is True)),
            "intergenic_windows": int(sum(1 for w in windows if w.genic is False)),
            "mean_gc": float(np.mean([w.gc for w in windows])),
            "mean_repeat_frac": float(np.mean([w.repeat_frac for w in windows if w.repeat_frac is not None])) if repeat_available else None,
        },
        "delta_profile": breakpoint,
        "strand_symmetry": {
            "pearson_mean": float(np.nanmean(strand_pearson)),
            "pearson_ci95": bootstrap_ci(strand_pearson[np.isfinite(strand_pearson)], seed=11),
            "spearman_mean": float(np.nanmean(strand_spearman)),
            "spearman_ci95": bootstrap_ci(strand_spearman[np.isfinite(strand_spearman)], seed=12),
            "active_pearson_mean": float(np.nanmean(strand_active_pearson)),
            "active_pearson_ci95": bootstrap_ci(strand_active_pearson[np.isfinite(strand_active_pearson)], seed=13),
            "active_spearman_mean": float(np.nanmean(strand_active_spearman)),
            "active_spearman_ci95": bootstrap_ci(strand_active_spearman[np.isfinite(strand_active_spearman)], seed=14),
            "matched_token_cosine_mean": float(np.nanmean(strand_cos)),
            "matched_token_cosine_ci95": bootstrap_ci(strand_cos[np.isfinite(strand_cos)], seed=15),
        },
        "size_dependence": size_summary,
        "aim1_indel_comparison": aim1_summary,
        "real_inversions": real_summary,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    save_json(outdir / "summary.json", summary)

    report = f"""# Evo2 Layer-26 Goodfire SAE Inversion Analysis

Run completed in {summary['runtime_seconds'] / 60:.1f} minutes on Azure A100.

## Controlled Synthetic Inversions

- Sampled {args.n} clean hg38 autosomal 8 kb windows: {summary['sample_metadata']['genic_windows']} genic and {summary['sample_metadata']['intergenic_windows']} intergenic.
- Mean GC = {summary['sample_metadata']['mean_gc']:.3f}; mean RepeatMasker overlap = {summary['sample_metadata']['mean_repeat_frac'] if summary['sample_metadata']['mean_repeat_frac'] is not None else 'NA'}.
- The 4 kb inversion profile uses dense per-token SAE deltas streamed in {args.chunk}-token chunks.

Breakpoint localization:

- Mean L1 delta at +/-{args.breakpoint_width} token breakpoint bands: {breakpoint['mean_l1_breakpoint']:.4g}
- Mean L1 delta in the central interior away from breakpoints: {breakpoint['mean_l1_interior_mid']:.4g}
- Mean L1 delta in distant flanks: {breakpoint['mean_l1_flanks']:.4g}
- Top breakpoint features: {', '.join(map(str, top_features[:20]))}

Strand-symmetry:

- Interior pooled ref-vs-revcomp Pearson r = {summary['strand_symmetry']['pearson_mean']:.4f} (95% bootstrap CI {summary['strand_symmetry']['pearson_ci95'][0]:.4f}, {summary['strand_symmetry']['pearson_ci95'][1]:.4f})
- Active-union Pearson r = {summary['strand_symmetry']['active_pearson_mean']:.4f} (95% CI {summary['strand_symmetry']['active_pearson_ci95'][0]:.4f}, {summary['strand_symmetry']['active_pearson_ci95'][1]:.4f})
- Matched-token cosine = {summary['strand_symmetry']['matched_token_cosine_mean']:.4f} (95% CI {summary['strand_symmetry']['matched_token_cosine_ci95'][0]:.4f}, {summary['strand_symmetry']['matched_token_cosine_ci95'][1]:.4f})

Size dependence:

"""
    for size in sizes:
        ss = size_summary[str(size)]
        report += f"- {size} bp: pooled L2 mean {ss['pooled_l2_mean']:.4g} (95% CI {ss['pooled_l2_ci95'][0]:.4g}, {ss['pooled_l2_ci95'][1]:.4g})\n"
    report += "\n## Aim1 DEL/INS Comparison\n\n"
    if aim1_summary:
        for svtype, vals in aim1_summary.items():
            report += f"- Aim1 {svtype} 0.5-8 kb: n={vals['n_0p5_to_8kb']}, pooled L2 mean {vals['pooled_l2_mean']:.4g} (95% CI {vals['pooled_l2_ci95'][0]:.4g}, {vals['pooled_l2_ci95'][1]:.4g})\n"
    else:
        report += "- Aim1 feature artifacts were not present on this machine; comparison skipped.\n"
    report += "\n## Real gnomAD v4.1 Inversions\n\n"
    if real_rows:
        report += f"- Tested {len(real_rows)} clean gnomAD v4.1 INV sites with 0.5-4 kb span and full 8 kb clean hg38 windows.\n"
        report += f"- Real breakpoint L2 mean {real_summary['real_bp_l2_mean']:.4g}; matched control mean {real_summary['control_bp_l2_mean']:.4g}; paired difference {real_summary['paired_diff_mean']:.4g} (95% CI {real_summary['paired_diff_ci95'][0]:.4g}, {real_summary['paired_diff_ci95'][1]:.4g}); sign-flip p={real_summary['paired_signflip_p']:.4g}.\n"
    else:
        report += f"- Real inversion arm skipped or empty. Source status: `{json.dumps(real_source_status)}`\n"
    report += """
## Interpretation

This report is intentionally quantitative rather than categorical. A strand-aware
SAE should show high interior ref-vs-revcomp similarity while still producing
localized breakpoint/junction deltas. The plots and JSON summary contain the
actual run values used for that judgment.

## Figures

- `inversion_delta_profile.png`
- `strand_symmetry_scatter.png`
- `inversion_size_dependence.png`
- `breakpoint_feature_heatmap.png`
- `inversion_vs_indel_delta_distribution.png`
- `real_inversions_matched_controls.png` when the real arm has data
"""
    (outdir / "report.md").write_text(report)
    print(f"[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
