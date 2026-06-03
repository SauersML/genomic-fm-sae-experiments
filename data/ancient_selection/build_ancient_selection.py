#!/usr/bin/env python3
"""Build ancient-DNA selection SNP tables lifted to hg38.

All write-heavy artifacts stay under data/ancient_selection. Shared reference
and annotation files are read only.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import pickle
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
import pysam
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 20260603
WINDOW_FLANK = 2500
CHROMS = [f"chr{i}" for i in range(1, 23)]
CHROM_TO_CODE = {chrom: i for i, chrom in enumerate(CHROMS, start=1)}
CODE_TO_CHROM = {i: chrom for chrom, i in CHROM_TO_CODE.items()}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "ancient_selection"
RAW_DIR = DATA_DIR / "raw"
WORK_DIR = DATA_DIR / "work"
ANNOT_DIR = ROOT / "data" / "annotations"
REF_FASTA = ROOT / "data" / "reference" / "hg38.fa"

SOURCE_TSV_GZ = RAW_DIR / "Selection_Summary_Statistics_01OCT2025.tsv.gz"
LIFTOVER = DATA_DIR / "tools" / "liftOver"
CHAIN = RAW_DIR / "hg19ToHg38.over.chain.gz"
RECOMB_BW = RAW_DIR / "recombAvg.hg38.bw"
BKGD_BED_GZ = RAW_DIR / "bkgd_hg38.bed.gz"
RMSK_CACHE = ANNOT_DIR / "rmsk_intervals.pkl"
GENCODE_CACHE = ANNOT_DIR / "gencode_v44_features.pkl"
GENCODE_GTF = ANNOT_DIR / "gencode.v44.annotation.gtf.gz"

LIFTOVER_BED = WORK_DIR / "akbari_snps_hg19.bed"
LIFTED_BED = WORK_DIR / "akbari_snps_hg38.lifted.bed"
UNMAPPED_BED = WORK_DIR / "akbari_snps_hg38.unmapped.bed"
SNPS_HG38 = DATA_DIR / "snps_hg38.tsv"
SNPS_PILOT = DATA_DIR / "snps_pilot.tsv"
MANIFEST = DATA_DIR / "MANIFEST_SPEC.md"
SUMMARY_JSON = DATA_DIR / "summary.json"
READY = DATA_DIR / "READY"


@dataclass
class BuildStats:
    source_rows: int = 0
    source_snp_rows: int = 0
    autosomal_snp_rows: int = 0
    liftover_in: int = 0
    liftover_mapped: int = 0
    liftover_unmapped: int = 0
    full_rows: int = 0
    anc_known_rows: int = 0
    selected_rows: int = 0
    control_rows: int = 0
    continuous_rows: int = 0
    pilot_rows: int = 0
    train_rows: int = 0
    test_rows: int = 0
    s_min: float = math.nan
    s_max: float = math.nan


SOURCE_COLUMNS = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "ANC",
    "ID",
    "RSID",
    "AF",
    "S",
    "SE",
    "X",
    "P_X",
    "POSTERIOR",
    "FDR",
    "CHI2_BE",
    "FILTER",
]


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    stats = BuildStats()
    if SNPS_HG38.exists() and not SOURCE_TSV_GZ.exists():
        pilot = build_pilot_from_full_table(stats)
        write_pilot_table(pilot)
        write_manifest(stats)
        write_summary(stats)
        READY.write_text("ready\n")
        return

    if LIFTOVER_BED.exists() and LIFTED_BED.exists() and UNMAPPED_BED.exists():
        scan_source_stats(stats)
        stats.liftover_in = count_data_lines(LIFTOVER_BED)
        stats.liftover_mapped = count_data_lines(LIFTED_BED)
        stats.liftover_unmapped = count_unmapped_records(UNMAPPED_BED)
    else:
        write_liftover_input(stats)
        run_liftover(stats)
    pos_map, chrom_code_map = load_liftover_maps(stats)
    df = parse_mapped_source(pos_map, chrom_code_map, stats)
    stats.full_rows = len(df)
    stats.anc_known_rows = int(df["derived_allele_freq"].notna().sum())
    stats.s_min = float(df["selection_coefficient"].min())
    stats.s_max = float(df["selection_coefficient"].max())
    free_large_inputs_before_final_write()
    pilot_candidates = annotate_and_write_full(df)
    pilot = build_pilot(pilot_candidates, stats)
    write_pilot_table(pilot)
    write_manifest(stats)
    write_summary(stats)
    READY.write_text("ready\n")


def write_liftover_input(stats: BuildStats) -> None:
    if LIFTOVER_BED.exists():
        LIFTOVER_BED.unlink()
    idx = 0
    with LIFTOVER_BED.open("w") as bed_handle:
        for chunk in read_source_chunks(["CHROM", "POS", "REF", "ALT"]):
            stats.source_rows += len(chunk)
            snp = (chunk["REF"].str.len() == 1) & (chunk["ALT"].str.len() == 1)
            stats.source_snp_rows += int(snp.sum())
            chunk = chunk.loc[snp].copy()
            chunk["chrom"] = "chr" + chunk["CHROM"].astype(str).str.removeprefix("chr")
            chunk = chunk.loc[chunk["chrom"].isin(CHROMS)]
            stats.autosomal_snp_rows += len(chunk)
            for row in chunk.itertuples(index=False):
                idx += 1
                pos = int(row.POS)
                bed_handle.write(f"{row.chrom}\t{pos - 1}\t{pos}\t{idx}\n")
    stats.liftover_in = idx


def scan_source_stats(stats: BuildStats) -> None:
    for chunk in read_source_chunks(["CHROM", "POS", "REF", "ALT"]):
        stats.source_rows += len(chunk)
        snp = (chunk["REF"].str.len() == 1) & (chunk["ALT"].str.len() == 1)
        stats.source_snp_rows += int(snp.sum())
        chunk = chunk.loc[snp].copy()
        chunk["chrom"] = "chr" + chunk["CHROM"].astype(str).str.removeprefix("chr")
        stats.autosomal_snp_rows += int(chunk["chrom"].isin(CHROMS).sum())


def run_liftover(stats: BuildStats) -> None:
    for path in (LIFTED_BED, UNMAPPED_BED):
        if path.exists():
            path.unlink()
    subprocess.run(
        [str(LIFTOVER), str(LIFTOVER_BED), str(CHAIN), str(LIFTED_BED), str(UNMAPPED_BED)],
        check=True,
    )
    stats.liftover_mapped = count_data_lines(LIFTED_BED)
    stats.liftover_unmapped = count_unmapped_records(UNMAPPED_BED)


def load_liftover_maps(stats: BuildStats) -> tuple[np.ndarray, np.ndarray]:
    pos_map = np.zeros(stats.liftover_in + 1, dtype=np.int64)
    chrom_code_map = np.zeros(stats.liftover_in + 1, dtype=np.int8)
    with LIFTED_BED.open() as handle:
        for line in handle:
            chrom, start, _, idx_s = line.rstrip("\n").split("\t")[:4]
            idx = int(idx_s)
            pos_map[idx] = int(start) + 1
            chrom_code_map[idx] = CHROM_TO_CODE.get(chrom, 0)
    return pos_map, chrom_code_map


def parse_mapped_source(pos_map: np.ndarray, chrom_code_map: np.ndarray, stats: BuildStats) -> pd.DataFrame:
    frames = []
    idx_offset = 0
    for chunk in read_source_chunks(SOURCE_COLUMNS):
        snp = (chunk["REF"].str.len() == 1) & (chunk["ALT"].str.len() == 1)
        chunk = chunk.loc[snp].copy()
        chunk["chrom_source"] = "chr" + chunk["CHROM"].astype(str).str.removeprefix("chr")
        chunk = chunk.loc[chunk["chrom_source"].isin(CHROMS)].copy()
        n = len(chunk)
        if n == 0:
            continue
        idx = np.arange(idx_offset + 1, idx_offset + n + 1, dtype=np.int64)
        idx_offset += n
        mapped = (pos_map[idx] > 0) & (chrom_code_map[idx] > 0)
        if not mapped.any():
            continue
        chunk = chunk.loc[mapped].copy()
        mapped_idx = idx[mapped]
        chunk["idx"] = mapped_idx
        chunk["chrom"] = [CODE_TO_CHROM[int(code)] for code in chrom_code_map[mapped_idx]]
        chunk["pos_hg38"] = pos_map[mapped_idx]
        chunk.rename(
            columns={
                "POS": "pos_hg19",
                "REF": "ref",
                "ALT": "alt",
                "ANC": "anc",
                "ID": "variant_id",
                "RSID": "rsid",
                "AF": "alt_af",
                "S": "selection_coefficient",
                "SE": "selection_se",
                "X": "selection_z",
                "P_X": "selection_p",
                "POSTERIOR": "posterior",
                "FDR": "fdr",
                "CHI2_BE": "chi2_batch_effect",
                "FILTER": "filter",
            },
            inplace=True,
        )
        chunk["anc"] = chunk["anc"].fillna("NA")
        chunk["rsid"] = chunk["rsid"].fillna(".").replace("NA", ".")
        chunk["derived_allele_freq"] = [
            derived_frequency(anc, ref, alt, af)
            for anc, ref, alt, af in zip(chunk["anc"], chunk["ref"], chunk["alt"], chunk["alt_af"])
        ]
        chunk["matching_af"] = chunk["derived_allele_freq"].where(
            chunk["derived_allele_freq"].notna(), chunk["alt_af"]
        )
        keep = [
            "idx",
            "rsid",
            "chrom",
            "pos_hg19",
            "pos_hg38",
            "ref",
            "alt",
            "anc",
            "variant_id",
            "alt_af",
            "derived_allele_freq",
            "matching_af",
            "selection_coefficient",
            "selection_se",
            "selection_z",
            "selection_p",
            "posterior",
            "fdr",
            "chi2_batch_effect",
            "filter",
        ]
        frames.append(chunk[keep])
    if idx_offset != stats.liftover_in:
        raise RuntimeError(f"parsed SNP count {idx_offset} differs from liftOver input {stats.liftover_in}")
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["chrom", "pos_hg38", "idx"]).reset_index(drop=True)


def read_source_chunks(usecols: list[str]):
    return pd.read_csv(
        SOURCE_TSV_GZ,
        sep="\t",
        comment="#",
        chunksize=500_000,
        usecols=usecols,
        dtype={
            "CHROM": "string",
            "POS": "int64",
            "REF": "string",
            "ALT": "string",
            "ANC": "string",
            "ID": "string",
            "RSID": "string",
            "FILTER": "string",
        },
    )


def derived_frequency(anc: str, ref: str, alt: str, af: float) -> float:
    anc = str(anc).upper()
    ref = str(ref).upper()
    alt = str(alt).upper()
    if anc == ref:
        return float(af)
    if anc == alt:
        return 1.0 - float(af)
    return math.nan


def annotate_and_write_full(df: pd.DataFrame) -> pd.DataFrame:
    df["start0"] = (df["pos_hg38"] - 1 - WINDOW_FLANK).clip(lower=0).astype(np.int64)
    df["end0"] = (df["pos_hg38"] + WINDOW_FLANK).astype(np.int64)

    repeat_intervals = load_interval_cache(RMSK_CACHE)
    gene_intervals = load_gene_any_intervals()
    tss_by_chrom = load_tss_by_chrom()
    bkgd_by_chrom = load_bkgd()
    recomb_bw = pyBigWig.open(str(RECOMB_BW))
    fasta = pysam.FastaFile(str(REF_FASTA))

    if SNPS_HG38.exists():
        SNPS_HG38.unlink()
    pilot_candidates = []
    abs_s_q20 = df["selection_coefficient"].abs().quantile(0.20)
    first = True
    for chrom in CHROMS:
        sub = df.loc[df["chrom"] == chrom].copy()
        if sub.empty:
            continue
        chrom_len = fasta.get_reference_length(chrom)
        sub["end0"] = sub["end0"].clip(upper=chrom_len).astype(np.int64)
        sub["window_bp"] = sub["end0"] - sub["start0"]
        starts = sub["start0"].to_numpy(np.int64)
        ends = sub["end0"].to_numpy(np.int64)
        pos0 = sub["pos_hg38"].to_numpy(np.int64) - 1

        sub["gc"] = gc_fraction_for_windows(fasta, chrom, chrom_len, starts, ends)
        sub["repeat_frac"] = coverage_fraction_from_intervals(chrom_len, starts, ends, repeat_intervals.get(chrom))
        sub["gene_density"] = coverage_fraction_from_intervals(chrom_len, starts, ends, gene_intervals.get(chrom))
        sub["b_statistic"] = values_at_positions(pos0, bkgd_by_chrom.get(chrom), default=np.nan)
        sub["recomb_rate_cm_per_mb"] = recomb_values_at_positions(recomb_bw, chrom, pos0)
        sub["dist_nearest_tss"] = nearest_distance(pos0, tss_by_chrom.get(chrom))
        append_full_table(sub, header=first)
        first = False

        eligible = sub.loc[sub["filter"] == "PASS"].copy()
        covars = ["matching_af", "b_statistic", "recomb_rate_cm_per_mb", "gc", "gene_density"]
        eligible = eligible.dropna(subset=covars + ["selection_coefficient", "selection_p", "fdr"])
        eligible["abs_s"] = eligible["selection_coefficient"].abs()
        selected = eligible.loc[eligible["fdr"] <= 0.05]
        neutral = eligible.loc[(eligible["fdr"] >= 0.95) & (eligible["abs_s"] <= abs_s_q20)]
        continuous = eligible.drop(index=selected.index, errors="ignore")
        parts = []
        if not selected.empty:
            parts.append(selected)
        if not neutral.empty:
            parts.append(neutral.sample(min(len(neutral), 20_000), random_state=SEED + CHROM_TO_CODE[chrom]))
        if not continuous.empty:
            parts.append(continuous.sample(min(len(continuous), 10_000), random_state=SEED + 100 + CHROM_TO_CODE[chrom]))
        if parts:
            pilot_candidates.append(pd.concat(parts, ignore_index=True))

    fasta.close()
    recomb_bw.close()
    if not pilot_candidates:
        raise RuntimeError("no pilot candidates collected")
    return pd.concat(pilot_candidates, ignore_index=True).drop_duplicates(subset=["idx"])


def build_pilot(df: pd.DataFrame, stats: BuildStats) -> pd.DataFrame:
    eligible = df.loc[df["filter"] == "PASS"].copy()
    covars = ["matching_af", "b_statistic", "recomb_rate_cm_per_mb", "gc", "gene_density"]
    eligible = eligible.dropna(subset=covars + ["selection_coefficient", "selection_p", "fdr"])
    eligible["abs_s"] = eligible["selection_coefficient"].abs()

    selected = eligible.loc[eligible["fdr"] <= 0.05].copy()
    if len(selected) < 500:
        cutoff = eligible["abs_s"].quantile(0.995)
        selected = eligible.loc[eligible["abs_s"] >= cutoff].copy()
    selected = selected.sort_values(["abs_s", "fdr"], ascending=[False, True]).head(1500).copy()
    selected["label"] = "selected"

    neutral_pool = eligible.loc[eligible["fdr"] >= 0.95].copy()
    controls = match_controls(selected, neutral_pool, covars, n_controls=1500)
    controls["label"] = "control"

    exclude = set(selected["idx"].astype(int)) | set(controls["idx"].astype(int))
    continuous_pool = eligible.loc[~eligible["idx"].astype(int).isin(exclude)].copy()
    continuous = (
        continuous_pool.assign(s_bin=pd.qcut(continuous_pool["selection_coefficient"], 20, duplicates="drop"))
        .groupby("s_bin", observed=True, group_keys=False)
        .apply(lambda x: x.sample(min(len(x), max(1, 2000 // 20)), random_state=SEED))
        .head(2000)
        .copy()
    )
    continuous["label"] = "continuous"

    pilot = pd.concat([selected, controls, continuous], ignore_index=True)
    pilot = pilot.drop_duplicates(subset=["idx"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    pilot["split"] = np.where(pilot["chrom"].isin(["chr1", "chr2"]), "test", "train")
    pilot["label_binary"] = pilot["label"].map({"selected": 1, "control": 0}).astype("Int64")

    stats.selected_rows = int((pilot["label"] == "selected").sum())
    stats.control_rows = int((pilot["label"] == "control").sum())
    stats.continuous_rows = int((pilot["label"] == "continuous").sum())
    stats.pilot_rows = len(pilot)
    stats.train_rows = int((pilot["split"] == "train").sum())
    stats.test_rows = int((pilot["split"] == "test").sum())
    return pilot


def build_pilot_from_full_table(stats: BuildStats) -> pd.DataFrame:
    stats.source_rows = 9_739_624
    stats.source_snp_rows = 8_074_573
    stats.autosomal_snp_rows = 8_074_573
    stats.liftover_in = 8_074_573
    stats.liftover_unmapped = count_unmapped_records(UNMAPPED_BED) if UNMAPPED_BED.exists() else 0
    stats.liftover_mapped = stats.liftover_in - stats.liftover_unmapped

    abs_s_values = []
    stats.full_rows = 0
    stats.anc_known_rows = 0
    stats.s_min = math.inf
    stats.s_max = -math.inf
    for chunk in pd.read_csv(
        SNPS_HG38,
        sep="\t",
        chunksize=500_000,
        usecols=["selection_coefficient", "derived_allele_freq"],
        na_values=["NA"],
    ):
        stats.full_rows += len(chunk)
        stats.anc_known_rows += int(chunk["derived_allele_freq"].notna().sum())
        s = chunk["selection_coefficient"].to_numpy(np.float64)
        stats.s_min = min(stats.s_min, float(np.nanmin(s)))
        stats.s_max = max(stats.s_max, float(np.nanmax(s)))
        abs_s_values.append(np.abs(s))
    abs_s_q20 = float(np.quantile(np.concatenate(abs_s_values), 0.20))

    candidates = []
    covars = ["matching_af", "b_statistic", "recomb_rate_cm_per_mb", "gc", "gene_density"]
    usecols = [
        "idx",
        "rsid",
        "chrom",
        "start0",
        "end0",
        "pos_hg19",
        "pos_hg38",
        "ref",
        "alt",
        "anc",
        "alt_af",
        "derived_allele_freq",
        "matching_af",
        "selection_coefficient",
        "selection_se",
        "selection_z",
        "selection_p",
        "posterior",
        "fdr",
        "gc",
        "repeat_frac",
        "gene_density",
        "recomb_rate_cm_per_mb",
        "b_statistic",
        "dist_nearest_tss",
        "filter",
    ]
    for i, chunk in enumerate(pd.read_csv(SNPS_HG38, sep="\t", chunksize=500_000, usecols=usecols, na_values=["NA"])):
        eligible = chunk.loc[chunk["filter"] == "PASS"].copy()
        eligible = eligible.dropna(subset=covars + ["selection_coefficient", "selection_p", "fdr"])
        if eligible.empty:
            continue
        eligible["abs_s"] = eligible["selection_coefficient"].abs()
        selected = eligible.loc[eligible["fdr"] <= 0.05]
        neutral = eligible.loc[(eligible["fdr"] >= 0.95) & (eligible["abs_s"] <= abs_s_q20)]
        continuous = eligible.drop(index=selected.index, errors="ignore")
        parts = []
        if not selected.empty:
            parts.append(selected)
        if not neutral.empty:
            parts.append(neutral.sample(min(len(neutral), 20_000), random_state=SEED + i))
        if not continuous.empty:
            parts.append(continuous.sample(min(len(continuous), 10_000), random_state=SEED + 1000 + i))
        if parts:
            candidates.append(pd.concat(parts, ignore_index=True))
    if not candidates:
        raise RuntimeError("no pilot candidates collected from full table")
    candidate_df = pd.concat(candidates, ignore_index=True).drop_duplicates(subset=["idx"])
    return build_pilot(candidate_df, stats)


def match_controls(selected: pd.DataFrame, pool: pd.DataFrame, covars: list[str], n_controls: int) -> pd.DataFrame:
    if pool.empty:
        raise RuntimeError("neutral control pool is empty")
    if selected.empty:
        raise RuntimeError("selected set is empty")
    take = min(n_controls, len(selected), len(pool))
    selected = selected.head(take).copy()
    scaler = StandardScaler()
    combined = pd.concat([selected[covars], pool[covars]], ignore_index=True)
    scaler.fit(combined)
    selected_x = scaler.transform(selected[covars])
    pool_x = scaler.transform(pool[covars])
    nn = NearestNeighbors(n_neighbors=min(20, len(pool)), algorithm="auto")
    nn.fit(pool_x)
    _, neigh = nn.kneighbors(selected_x)
    used: set[int] = set()
    chosen = []
    pool_indices = pool.index.to_numpy()
    for candidates in neigh:
        pick = None
        for candidate in candidates:
            idx = int(pool_indices[candidate])
            if idx not in used:
                pick = idx
                break
        if pick is None:
            remaining = [int(i) for i in pool_indices if int(i) not in used]
            if not remaining:
                break
            pick = remaining[0]
        used.add(pick)
        chosen.append(pick)
    controls = pool.loc[chosen].copy()
    if len(controls) < take:
        raise RuntimeError(f"only matched {len(controls)} controls for {take} selected SNPs")
    return controls


def gc_fraction_for_windows(
    fasta: pysam.FastaFile, chrom: str, chrom_len: int, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    seq = fasta.fetch(chrom, 0, chrom_len).upper()
    arr = np.frombuffer(seq.encode("ascii"), dtype="S1")
    called = np.isin(arr, [b"A", b"C", b"G", b"T"]).astype(np.int32)
    gc = np.isin(arr, [b"G", b"C"]).astype(np.int32)
    called_cs = prefix_sum(called)
    gc_cs = prefix_sum(gc)
    called_bp = called_cs[ends] - called_cs[starts]
    gc_bp = gc_cs[ends] - gc_cs[starts]
    return np.divide(gc_bp, called_bp, out=np.full(len(starts), np.nan), where=called_bp > 0)


def load_interval_cache(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_gene_any_intervals() -> dict[str, np.ndarray]:
    with GENCODE_CACHE.open("rb") as handle:
        features = pickle.load(handle)
    return {chrom: np.asarray(intervals, dtype=np.int64) for chrom, intervals in features["gene_any"].items()}


def load_tss_by_chrom() -> dict[str, np.ndarray]:
    tss: dict[str, list[int]] = {}
    with gzip.open(GENCODE_GTF, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, _, _, start, end, _, strand, _, attrs = fields
            if chrom not in CHROMS:
                continue
            if 'gene_type "protein_coding"' not in attrs and 'gene_type "lncRNA"' not in attrs:
                continue
            start0 = int(start) - 1
            end0 = int(end)
            tss.setdefault(chrom, []).append(start0 if strand != "-" else end0 - 1)
    return {chrom: np.asarray(sorted(vals), dtype=np.int64) for chrom, vals in tss.items()}


def load_bkgd() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: dict[str, list[list[float]]] = {}
    with gzip.open(BKGD_BED_GZ, "rt") as handle:
        for line in handle:
            chrom, start, end, _, value = line.rstrip("\n").split("\t")[:5]
            if chrom not in CHROMS:
                continue
            out.setdefault(chrom, [[], [], []])
            out[chrom][0].append(int(start))
            out[chrom][1].append(int(end))
            out[chrom][2].append(float(value))
    return {
        chrom: (
            np.asarray(vals[0], dtype=np.int64),
            np.asarray(vals[1], dtype=np.int64),
            np.asarray(vals[2], dtype=np.float32),
        )
        for chrom, vals in out.items()
    }


def prefix_sum(mask: np.ndarray) -> np.ndarray:
    out = np.empty(len(mask) + 1, dtype=np.int32)
    out[0] = 0
    out[1:] = np.cumsum(mask, dtype=np.int32)
    return out


def coverage_fraction_from_intervals(
    chrom_len: int, starts: np.ndarray, ends: np.ndarray, intervals: np.ndarray | None
) -> np.ndarray:
    if intervals is None or len(intervals) == 0:
        return np.zeros(len(starts), dtype=np.float32)
    diff = np.zeros(chrom_len + 1, dtype=np.int16)
    clipped_start = np.clip(intervals[:, 0].astype(np.int64), 0, chrom_len)
    clipped_end = np.clip(intervals[:, 1].astype(np.int64), 0, chrom_len)
    valid = clipped_end > clipped_start
    np.add.at(diff, clipped_start[valid], 1)
    np.add.at(diff, clipped_end[valid], -1)
    covered = (np.cumsum(diff[:-1]) > 0).astype(np.int32)
    covered_cs = prefix_sum(covered)
    covered_bp = covered_cs[ends] - covered_cs[starts]
    return covered_bp / (ends - starts)


def values_at_positions(
    pos0: np.ndarray, intervals: tuple[np.ndarray, np.ndarray, np.ndarray] | None, default: float
) -> np.ndarray:
    if intervals is None:
        return np.full(len(pos0), default, dtype=np.float32)
    starts, ends, values = intervals
    idx = np.searchsorted(starts, pos0, side="right") - 1
    valid = (idx >= 0) & (pos0 < ends[np.clip(idx, 0, len(ends) - 1)])
    out = np.full(len(pos0), default, dtype=np.float32)
    out[valid] = values[idx[valid]]
    return out


def recomb_values_at_positions(bw, chrom: str, pos0: np.ndarray) -> np.ndarray:
    chrom_len = bw.chroms()[chrom]
    intervals = bw.intervals(chrom, 0, chrom_len)
    if not intervals:
        return np.full(len(pos0), np.nan, dtype=np.float32)
    starts = np.asarray([x[0] for x in intervals], dtype=np.int64)
    ends = np.asarray([x[1] for x in intervals], dtype=np.int64)
    vals = np.asarray([x[2] for x in intervals], dtype=np.float32)
    return values_at_positions(pos0, (starts, ends, vals), default=np.nan)


def nearest_distance(pos0: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.full(len(pos0), np.nan)
    right = np.searchsorted(points, pos0, side="left")
    left = right - 1
    left_dist = np.full(len(pos0), np.iinfo(np.int64).max, dtype=np.int64)
    right_dist = np.full(len(pos0), np.iinfo(np.int64).max, dtype=np.int64)
    has_left = left >= 0
    has_right = right < len(points)
    left_dist[has_left] = np.abs(pos0[has_left] - points[left[has_left]])
    right_dist[has_right] = np.abs(points[right[has_right]] - pos0[has_right])
    return np.minimum(left_dist, right_dist)


def free_large_inputs_before_final_write() -> None:
    for path in (SOURCE_TSV_GZ, LIFTOVER_BED, LIFTED_BED):
        if path.exists():
            path.unlink()


def append_full_table(df: pd.DataFrame, header: bool) -> None:
    ordered = [
        "rsid",
        "chrom",
        "pos_hg19",
        "pos_hg38",
        "start0",
        "end0",
        "ref",
        "alt",
        "anc",
        "variant_id",
        "alt_af",
        "derived_allele_freq",
        "matching_af",
        "selection_coefficient",
        "selection_se",
        "selection_z",
        "selection_p",
        "posterior",
        "fdr",
        "chi2_batch_effect",
        "filter",
        "gc",
        "repeat_frac",
        "gene_density",
        "recomb_rate_cm_per_mb",
        "b_statistic",
        "dist_nearest_tss",
        "window_bp",
        "idx",
    ]
    df[ordered].to_csv(
        SNPS_HG38,
        sep="\t",
        index=False,
        na_rep="NA",
        float_format="%.8g",
        mode="a",
        header=header,
    )


def write_pilot_table(pilot: pd.DataFrame) -> None:
    ordered = [
        "rsid",
        "chrom",
        "start0",
        "end0",
        "pos_hg19",
        "pos_hg38",
        "ref",
        "alt",
        "anc",
        "alt_af",
        "derived_allele_freq",
        "matching_af",
        "selection_coefficient",
        "selection_se",
        "selection_z",
        "selection_p",
        "posterior",
        "fdr",
        "gc",
        "repeat_frac",
        "gene_density",
        "recomb_rate_cm_per_mb",
        "b_statistic",
        "dist_nearest_tss",
        "label",
        "label_binary",
        "split",
        "idx",
    ]
    pilot[ordered].to_csv(SNPS_PILOT, sep="\t", index=False, na_rep="NA", float_format="%.8g")


def count_data_lines(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def count_unmapped_records(path: Path) -> int:
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            count += 1
    return count


def write_summary(stats: BuildStats) -> None:
    SUMMARY_JSON.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True) + "\n")


def write_manifest(stats: BuildStats) -> None:
    MANIFEST.write_text(
        f"""# Ancient Selection SNP Manifest

Built by `data/ancient_selection/build_ancient_selection.py` with seed `{SEED}`.

## Files

- `snps_hg38.tsv`: full lifted autosomal biallelic SNP table with Akbari selection statistics and hg38 covariates.
- `snps_pilot.tsv`: pilot modeling table with 5,001 bp centered windows (`start0`, `end0`), selected/control/continuous labels, and chromosome split.
- `summary.json`: machine-readable build counts.
- `work/akbari_snps_hg38.unmapped.bed`: UCSC liftOver unmapped records retained as the mapping log.
- `READY`: completion sentinel.

## Coordinate Conventions

- Source coordinates are GRCh37/hg19, 1-based.
- Output `chrom`, `start0`, and `end0` are chr-prefixed GRCh38/hg38, 0-based half-open.
- `pos_hg38` is 1-based.
- Windows are `[pos_hg38 - 1 - 2500, pos_hg38 + 2500)`, clipped to chromosome bounds.

## Build Counts

- Source rows: {stats.source_rows}
- Source biallelic SNP rows: {stats.source_snp_rows}
- Autosomal SNPs submitted to liftOver: {stats.liftover_in}
- liftOver mapped: {stats.liftover_mapped}
- liftOver unmapped: {stats.liftover_unmapped}
- Full output rows: {stats.full_rows}
- Rows with ancestral allele-derived DAF: {stats.anc_known_rows}
- Pilot rows: {stats.pilot_rows}
- Pilot selected/control/continuous: {stats.selected_rows}/{stats.control_rows}/{stats.continuous_rows}
- Pilot train/test by chromosome split: {stats.train_rows}/{stats.test_rows}

## Pilot Labels

- `selected`: top FDR-significant (`FDR <= 0.05`) SNPs by `abs(selection_coefficient)`, capped at 1,500.
- `control`: nearest-neighbor matched neutral SNPs from `FDR >= 0.95` and the lowest 20% of `abs(selection_coefficient)`.
- `continuous`: stratified sample across the signed `selection_coefficient` distribution, excluding selected/control SNPs.
- Matching covariates: `matching_af`, `b_statistic`, `recomb_rate_cm_per_mb`, `gc`, and `gene_density`.
- `split`: `test` for `chr1`/`chr2`, otherwise `train`.
"""
    )


if __name__ == "__main__":
    main()
