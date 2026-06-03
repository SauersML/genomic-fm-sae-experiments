#!/usr/bin/env python3
"""Focused real-inversion follow-up for the Evo2/Goodfire SAE inversion run."""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch


def load_inv_module(repo_root: Path):
    path = repo_root / "scripts" / "run_inversion_analysis.py"
    spec = importlib.util.spec_from_file_location("inv_analysis", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["inv_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_real_candidates(url: str, genome, inv, repeat_intervals, repeat_available: bool, n: int, win_len: int) -> list[dict]:
    clean: list[dict] = []
    seen = 0
    with urllib.request.urlopen(url, timeout=180) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            reader = csv.DictReader((line.decode("utf-8") for line in gz), delimiter="\t")
            for rec in reader:
                chrom = rec["#chrom"]
                if rec["svtype"] != "INV" or chrom not in inv.AUTOSOMES:
                    continue
                start0 = int(rec["start"])
                end0 = int(rec["end"])
                span = end0 - start0
                if not (500 <= span <= 4000):
                    continue
                seen += 1
                flank = (win_len - span) // 2
                right = win_len - span - flank
                win_start = start0 - flank
                win_end = end0 + right
                if win_start < 0 or win_end > genome.lengths[chrom] or inv.overlaps_centromere(chrom, win_start, win_end):
                    continue
                seq = genome.fetch(chrom, win_start, win_end)
                if len(seq) != win_len or any(base not in "ACGT" for base in seq):
                    continue
                repeat_frac = (
                    inv.interval_overlap_bp(repeat_intervals.get(chrom, []), win_start, win_end) / win_len
                    if repeat_available
                    else None
                )
                clean.append(
                    {
                        "id": rec["name"],
                        "chrom": chrom,
                        "start0": start0,
                        "end0": end0,
                        "span": span,
                        "svlen_field": abs(int(float(rec.get("SVLEN") or span))),
                        "af": float(rec.get("AF") or "nan"),
                        "window_start0": win_start,
                        "window_end0": win_end,
                        "rel_start": start0 - win_start,
                        "rel_end": end0 - win_start,
                        "seq": seq,
                        "gc": inv.gc_frac(seq),
                        "repeat_frac": repeat_frac,
                    }
                )
                if len(clean) >= n:
                    break
    return clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--fasta", default=str(Path.home() / "hf_cache" / "hg38.fa"))
    ap.add_argument("--outdir", default="results/inversions")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--win-len", type=int, default=8000)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--breakpoint-width", type=int, default=64)
    ap.add_argument("--gnomad-url", default="https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/genome_sv/gnomad.v4.1.sv.non_neuro_controls.sites.bed.gz")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    repo_root = Path(args.repo_root).resolve()
    outdir = repo_root / args.outdir
    inv = load_inv_module(repo_root)
    genome = inv.Genome(Path(args.fasta))
    repeat_intervals = inv.load_repeat_intervals(repo_root / "data" / "annotations" / "rmsk.txt.gz")
    repeat_available = any(repeat_intervals.values())

    windows = []
    with (outdir / "synthetic_windows.csv").open() as fh:
        for row in csv.DictReader(fh):
            seq = genome.fetch(row["chrom"], int(row["start0"]), int(row["end0"]))
            windows.append(
                inv.Window(
                    id=row["id"],
                    chrom=row["chrom"],
                    start0=int(row["start0"]),
                    end0=int(row["end0"]),
                    seq=seq,
                    gc=float(row["gc"]),
                    repeat_frac=float(row["repeat_frac"]) if row["repeat_frac"] else None,
                    genic=row["genic"] == "True" if row["genic"] else None,
                )
            )

    reals = parse_real_candidates(
        args.gnomad_url,
        genome,
        inv,
        repeat_intervals,
        repeat_available,
        args.n,
        args.win_len,
    )
    evo2x, sae = inv.load_evo2(repo_root)
    used_controls: set[str] = set()
    rows = []
    for i, rec in enumerate(reals, start=1):
        control = inv.pick_matched_control(rec, windows, used_controls, repeat_available)
        if control is None:
            continue
        real_alt = (
            rec["seq"][: rec["rel_start"]]
            + inv.revcomp(rec["seq"][rec["rel_start"] : rec["rel_end"]])
            + rec["seq"][rec["rel_end"] :]
        )
        real_stats = inv.analyze_pair(
            evo2x,
            sae,
            rec["seq"],
            real_alt,
            rec["rel_start"],
            rec["rel_end"],
            collect_profile=False,
            collect_breakpoint=True,
            collect_bins=False,
            collect_strand=False,
            chunk=args.chunk,
            breakpoint_width=args.breakpoint_width,
            nbins=160,
            scatter_features=0,
        )
        ctrl_alt, ctrl_start, ctrl_end = inv.make_alt(control.seq, rec["span"])
        ctrl_stats = inv.analyze_pair(
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
            nbins=160,
            scatter_features=0,
        )
        rows.append(
            {
                "id": rec["id"],
                "chrom": rec["chrom"],
                "start0": rec["start0"],
                "end0": rec["end0"],
                "span": rec["span"],
                "svlen_field": rec["svlen_field"],
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
        if i % 10 == 0:
            print(f"[real-followup] {i}/{len(reals)}", flush=True)

    if rows:
        with (outdir / "real_inversions_matched_controls.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    diffs = np.array([r["real_bp_l2"] - r["control_bp_l2"] for r in rows], dtype=np.float64)
    summary = {
        "source": args.gnomad_url,
        "n_clean": len(reals),
        "n_tested": len(rows),
        "real_bp_l2_mean": float(np.mean([r["real_bp_l2"] for r in rows])) if rows else None,
        "control_bp_l2_mean": float(np.mean([r["control_bp_l2"] for r in rows])) if rows else None,
        "paired_diff_mean": float(diffs.mean()) if len(diffs) else None,
        "paired_diff_ci95": inv.bootstrap_ci(diffs, seed=4242) if len(diffs) else None,
        "paired_signflip_p": inv.permutation_paired_p(diffs) if len(diffs) else None,
    }
    (outdir / "real_inversions_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("[done] real follow-up", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
