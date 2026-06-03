#!/usr/bin/env python3
"""Build the Aim 2 (popgen) Evo2-SAE manifest + aligned label/covariate/split tables.

For each selected region (positives + matched controls, two tasks) we emit one
manifest record describing a fixed 8 kb window centered on the region
(center +- 4096 bp; already precomputed as window8_start/end in the source TSV).
A 32 kb variant id (suffix ``__w32``) is also emitted cheaply for later use.

azure-compute reads ``manifest.jsonl`` -> extracts sequence from hg38.fa ->
returns ``features.npy`` (pooled Evo2 layer-26 SAE feature vector per row) +
``ids.txt`` (row order) and (ideally) ``gc.npy`` (per-window GC fraction).

Outputs (all under data/aim2_popgen/):
  manifest.jsonl        one JSON record per row (8 kb windows; +32 kb variants)
  manifest_w8.jsonl     just the 8 kb records (primary)
  table_sweeps.tsv      aligned id,label,chrom(group),length,log_length,split,...
  table_introgression.tsv
  pilot_ids.txt         the pilot subset of ids (8 kb), both tasks
  README is in MANIFEST_SPEC.md

The 8 kb window id == region_id. The 32 kb id == region_id + "__w32".
analyze.py keys features.npy rows by ids.txt (the order azure-compute returns).
"""
from __future__ import annotations
import csv
import json
import math
import os
import random
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = {
    "sweeps": os.path.join(ROOT, "data/popgen/sweeps_regions.grch38.tsv"),
    "introgression": os.path.join(ROOT, "data/popgen/introgression_regions.grch38.tsv"),
}
POS_LABEL = {"sweeps": "sweep", "introgression": "introgression"}
OUTDIR = os.path.join(ROOT, "data/aim2_popgen")

WIN8 = 8192
WIN32 = 32768
PILOT_PER_SIDE = 300      # ~300 positives + ~300 controls per task
SEED = 20260603


def load_rows(task):
    with open(SRC[task]) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def make_record(row, win, suffix=""):
    """8 kb (win=='8') or 32 kb (win=='32') window record for azure-compute."""
    if win == "8":
        s, e = int(row["window8_start"]), int(row["window8_end"])
    else:
        s, e = int(row["window32_start"]), int(row["window32_end"])
    return {
        "id": row["region_id"] + suffix,
        "chrom": row["chrom"],
        "start0": s,   # 0-based half-open, matches BED / hg38.fa extraction
        "end0": e,
    }


def pilot_select(rows, task, rng):
    """Balanced ~PILOT_PER_SIDE positives + controls, spread across chromosomes,
    and guaranteeing the held-out test chroms (chr1, chr2) are represented so the
    by-chromosome GroupKFold + held-out evaluation is meaningful."""
    pos_label = POS_LABEL[task]
    by = defaultdict(list)  # (split,label) -> rows
    for r in rows:
        by[(r["split"], r["label"])].append(r)

    def spread_sample(pool, n):
        """Sample n rows spread as evenly as possible over chromosomes."""
        bych = defaultdict(list)
        for r in pool:
            bych[r["chrom"]].append(r)
        for ch in bych:
            rng.shuffle(bych[ch])
        chroms = sorted(bych, key=lambda c: (int(c.replace("chr", "")) ))
        out, i = [], 0
        # round-robin across chromosomes
        while len(out) < n and any(bych[c] for c in chroms):
            ch = chroms[i % len(chroms)]
            if bych[ch]:
                out.append(bych[ch].pop())
            i += 1
        return out

    # Take ~1/3 of the pilot from held-out test chroms, rest from train chroms.
    n_test = max(2, PILOT_PER_SIDE // 3)
    n_train = PILOT_PER_SIDE - n_test
    selected = []
    for label in (pos_label, "control"):
        test_pool = by[("test", label)]
        train_pool = by[("train", label)]
        sel_test = spread_sample(test_pool, min(n_test, len(test_pool)))
        sel_train = spread_sample(train_pool, n_train)
        selected.extend(sel_test + sel_train)
    return selected


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(SEED)

    manifest_all = []   # all 8kb records across both tasks (+32kb variants)
    w8_records = []
    pilot_ids = []

    for task in ("sweeps", "introgression"):
        rows = load_rows(task)
        pos_label = POS_LABEL[task]
        sel = pilot_select(rows, task, rng)

        # aligned table
        tpath = os.path.join(OUTDIR, f"table_{task}.tsv")
        cols = ["id", "task", "label", "y", "chrom", "split",
                "start0", "end0", "length", "log_length",
                "population", "stat_name", "stat_value", "archaic_best"]
        with open(tpath, "w", newline="") as out:
            w = csv.DictWriter(out, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in sel:
                rid = r["region_id"]
                length = int(r["end"]) - int(r["start"])  # original region length
                y = 1 if r["label"] == pos_label else 0
                w.writerow({
                    "id": rid,
                    "task": task,
                    "label": r["label"],
                    "y": y,
                    "chrom": r["chrom"],
                    "split": r["split"],
                    "start0": r["window8_start"],
                    "end0": r["window8_end"],
                    "length": length,
                    "log_length": f"{math.log10(max(length,1)):.6f}",
                    "population": r.get("population", ""),
                    "stat_name": r.get("stat_name", ""),
                    "stat_value": r.get("stat_value", ""),
                    "archaic_best": r.get("archaic_best", ""),
                })
                # manifest records
                rec8 = make_record(r, "8")
                rec32 = make_record(r, "32", suffix="__w32")
                w8_records.append(rec8)
                manifest_all.append(rec8)
                manifest_all.append(rec32)
                pilot_ids.append(rid)

    # write manifests
    with open(os.path.join(OUTDIR, "manifest.jsonl"), "w") as fh:
        for rec in manifest_all:
            fh.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUTDIR, "manifest_w8.jsonl"), "w") as fh:
        for rec in w8_records:
            fh.write(json.dumps(rec) + "\n")
    with open(os.path.join(OUTDIR, "pilot_ids.txt"), "w") as fh:
        fh.write("\n".join(pilot_ids) + "\n")

    # counts summary
    n8 = len(w8_records)
    n32 = sum(1 for r in manifest_all if r["id"].endswith("__w32"))
    print(f"manifest.jsonl: {len(manifest_all)} records ({n8} x 8kb + {n32} x 32kb)")
    print(f"manifest_w8.jsonl: {n8} records")
    for task in ("sweeps", "introgression"):
        import collections
        c = collections.Counter()
        with open(os.path.join(OUTDIR, f"table_{task}.tsv")) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                c[(row["label"], row["split"])] += 1
        print(f"  {task}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
