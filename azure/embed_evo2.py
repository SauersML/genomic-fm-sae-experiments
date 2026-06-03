#!/usr/bin/env python
"""Warm-batch Evo2-7B + Goodfire layer-26 SAE feature extractor.

Loads Evo2-7B (bf16) + the SAE ONCE, then processes a job dir's manifest.jsonl,
writing features.npy (rows aligned to ids.txt) + meta.json.

Manifest record forms (auto-detected per record, see azure/jobio.record_kind):
  region (pooled SAE feature vector):
    {"id":.., "seq":"ACGT..."}                         -> pooled features
    {"id":.., "chrom":.., "start0":.., "end0":..}      -> extract from hg38, pooled
  ref/alt delta (pooled(alt) - pooled(ref)):
    {"id":.., "ref_seq":.., "alt_seq":..}              -> delta
    {"id":.., "chrom":.., "start0":.., "end0":.., "ref":.., "alt":..,
     "flank":.., "max_allele":..}                      -> build windows, delta

Pooling defaults to 'mean' (the Aim-1 spec uses mean-pool for the delta).

Usage:
    python azure/embed_evo2.py --job <jobdir> [--pool mean|max]

Can also run as a persistent server loop polling a queue dir (keeps model warm):
    python azure/embed_evo2.py --serve <queue_dir>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jobio  # noqa: E402
from src.evo2 import extract as evo2x  # noqa: E402


def _pool_vec(feats: torch.Tensor, pool: str) -> np.ndarray:
    if pool == "mean":
        v = feats.mean(0)
    elif pool == "max":
        v = feats.max(0).values
    else:
        raise ValueError(pool)
    return v.float().cpu().numpy()


@torch.no_grad()
def _features_for_seq(seq: str, sae, pool: str) -> np.ndarray:
    acts = evo2x.embed_dna([seq], device="cuda")[0]   # [L, 4096]
    feats = sae.encode(acts, topk=True)               # [L, 32768]
    out = _pool_vec(feats, pool)
    del acts, feats
    return out


def run_job(jobdir: str, pool: str = "mean") -> dict:
    recs = jobio.read_manifest(jobdir)
    genome = None
    sae = evo2x._load_sae_cached("cuda")
    # warm the base model
    evo2x._load_evo2()

    ids: list[str] = []
    rows: list[np.ndarray] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    n_seqs_run = 0

    for i, rec in enumerate(recs):
        kind = jobio.record_kind(rec)
        ids.append(str(rec["id"]))

        if kind in ("delta_coord", "delta_seq"):
            if kind == "delta_seq":
                ref_w, alt_w = rec["ref_seq"], rec["alt_seq"]
            else:
                if genome is None:
                    genome = jobio.get_genome()
                ref_w, alt_w = jobio.build_ref_alt_windows(rec, genome)
            f_ref = _features_for_seq(ref_w, sae, pool)
            f_alt = _features_for_seq(alt_w, sae, pool)
            rows.append((f_alt - f_ref).astype(np.float32))
            n_seqs_run += 2
        else:  # region
            if kind == "region_seq":
                seq = rec["seq"]
            else:
                if genome is None:
                    genome = jobio.get_genome()
                seq = genome.fetch(rec["chrom"], int(rec["start0"]),
                                   int(rec["end0"]))
            rows.append(_features_for_seq(seq, sae, pool))
            n_seqs_run += 1

        if (i + 1) % 25 == 0:
            torch.cuda.empty_cache()
            print(f"  [{i+1}/{len(recs)}] {time.time()-t0:.1f}s", flush=True)

    dt = time.time() - t0
    feats = np.stack(rows, axis=0).astype(np.float32)
    peak = torch.cuda.max_memory_allocated() / 1e9
    meta = {
        "n": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "pool": pool,
        "kind": jobio.record_kind(recs[0]),
        "peak_mem_gb": round(peak, 2),
        "seqs_per_sec": round(n_seqs_run / dt, 3) if dt > 0 else None,
        "seconds": round(dt, 1),
        "n_forward_seqs": n_seqs_run,
    }
    jobio.write_outputs(jobdir, ids, feats, meta)
    torch.cuda.empty_cache()
    print(f"DONE job={jobdir} -> {os.path.join(jobdir,'features.npy')} "
          f"shape={feats.shape} meta={meta}", flush=True)
    return meta


def serve(queue_dir: str, pool: str = "mean") -> None:
    """Poll queue_dir for *.job files (each contains a jobdir path); process in
    order, keeping the model warm. Writes <name>.done with the meta."""
    import glob
    import json
    evo2x._load_evo2()
    evo2x._load_sae_cached("cuda")
    print(f"[serve] model warm; polling {queue_dir}", flush=True)
    seen: set[str] = set()
    while True:
        for jf in sorted(glob.glob(os.path.join(queue_dir, "*.job"))):
            if jf in seen:
                continue
            seen.add(jf)
            jobdir = open(jf).read().strip()
            print(f"[serve] running {jobdir}", flush=True)
            try:
                meta = run_job(jobdir, pool=pool)
                open(jf + ".done", "w").write(json.dumps(meta))
            except Exception as e:  # report, keep serving
                open(jf + ".error", "w").write(repr(e))
                print(f"[serve] ERROR {jobdir}: {e!r}", flush=True)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", help="job dir with manifest.jsonl")
    ap.add_argument("--serve", help="queue dir to poll (persistent warm server)")
    ap.add_argument("--pool", default="mean", choices=["mean", "max"])
    args = ap.parse_args()
    if args.serve:
        serve(args.serve, pool=args.pool)
    elif args.job:
        run_job(args.job, pool=args.pool)
    else:
        ap.error("need --job or --serve")


if __name__ == "__main__":
    main()
