#!/usr/bin/env python
"""Warm-batch ESM2-650M + InterPLM layer-24 SAE feature extractor.

Loads ESM2-650M + the InterPLM SAE ONCE, then processes a job dir's
manifest.jsonl of protein sequences, writing features.npy (rows aligned to
ids.txt) + meta.json.

Manifest record form:
    {"id":.., "seq":"MRWQ..."}     protein amino-acid sequence -> pooled features

Usage:
    python azure/embed_esm2.py --job <jobdir> [--pool mean|max]
    python azure/embed_esm2.py --serve <queue_dir>
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
from src.esm2 import extract as esmx  # noqa: E402


@torch.no_grad()
def _features_for_protein(seq: str, sae, pool: str) -> np.ndarray:
    emb = esmx.embed_protein([seq], device="cuda")[0]   # [L, 1280]
    feats = sae.encode(emb)                              # [L, 10240]
    if pool == "mean":
        v = feats.mean(0)
    elif pool == "max":
        v = feats.max(0).values
    else:
        raise ValueError(pool)
    out = v.float().cpu().numpy()
    del emb, feats
    return out


def run_job(jobdir: str, pool: str = "mean") -> dict:
    recs = jobio.read_manifest(jobdir)
    sae = esmx._load_sae_cached("cuda")
    esmx._load_esm("cuda")

    ids: list[str] = []
    rows: list[np.ndarray] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    for i, rec in enumerate(recs):
        if "seq" not in rec:
            raise ValueError(f"ESM2 record needs 'seq': {list(rec.keys())}")
        ids.append(str(rec["id"]))
        rows.append(_features_for_protein(rec["seq"], sae, pool))
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            print(f"  [{i+1}/{len(recs)}] {time.time()-t0:.1f}s", flush=True)

    dt = time.time() - t0
    feats = np.stack(rows, axis=0).astype(np.float32)
    peak = torch.cuda.max_memory_allocated() / 1e9
    meta = {
        "n": int(feats.shape[0]),
        "dim": int(feats.shape[1]),
        "pool": pool,
        "peak_mem_gb": round(peak, 2),
        "seqs_per_sec": round(len(recs) / dt, 3) if dt > 0 else None,
        "seconds": round(dt, 1),
    }
    jobio.write_outputs(jobdir, ids, feats, meta)
    torch.cuda.empty_cache()
    print(f"DONE job={jobdir} -> {os.path.join(jobdir,'features.npy')} "
          f"shape={feats.shape} meta={meta}", flush=True)
    return meta


def serve(queue_dir: str, pool: str = "mean") -> None:
    import glob
    import json
    esmx._load_esm("cuda")
    esmx._load_sae_cached("cuda")
    print(f"[serve] esm2 model warm; polling {queue_dir}", flush=True)
    seen: set[str] = set()
    while True:
        for jf in sorted(glob.glob(os.path.join(queue_dir, "*.job"))):
            if jf in seen:
                continue
            seen.add(jf)
            jobdir = open(jf).read().strip()
            try:
                meta = run_job(jobdir, pool=pool)
                open(jf + ".done", "w").write(json.dumps(meta))
            except Exception as e:
                open(jf + ".error", "w").write(repr(e))
                print(f"[serve] ERROR {jobdir}: {e!r}", flush=True)
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--serve")
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
