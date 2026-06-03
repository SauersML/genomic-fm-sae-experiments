"""Chunked Modal runner for Aim 3 Evo2/Goodfire SAE features.

This reuses src/modal_app.py's Modal image/function definitions and keeps all
large artifacts local. It writes chunk files first, then assembles and validates
data/aim3_assoc/features.npy only when every manifest row is complete.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal
import numpy as np

from src.modal_app import app, evo2_features


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/aim3_assoc"
SEQ_MANIFEST = DATA / "seq_manifest.jsonl"
CHUNK_DIR = DATA / "modal_chunks"
FEATURES = DATA / "features.npy"
IDS = DATA / "ids.txt"
META = DATA / "meta.json"
READY = DATA / "FEATURES_READY"
SAE_DIM = 32768


def _read_records() -> list[dict]:
    records = []
    with SEQ_MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"empty manifest: {SEQ_MANIFEST}")
    ids = [rec["id"] for rec in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in seq_manifest")
    bad = [rec["id"] for rec in records if set(rec["seq"]) - set("ACGT")]
    if bad:
        raise ValueError(f"non-ACGT sequence, first={bad[0]}")
    return records


def _chunk_path(start: int, end: int) -> Path:
    return CHUNK_DIR / f"features_{start:05d}_{end:05d}.npy"


def _validate_matrix(X: np.ndarray, ids: list[str]) -> None:
    if X.shape != (len(ids), SAE_DIM):
        raise ValueError(f"bad feature shape {X.shape}, expected {(len(ids), SAE_DIM)}")
    if not np.isfinite(X).all():
        raise ValueError("features contain NaN or Inf")
    zero_rows = np.where(np.all(X == 0, axis=1))[0]
    if len(zero_rows):
        raise ValueError(f"all-zero feature rows, first index {int(zero_rows[0])}")


@app.local_entrypoint(name="aim3")
def aim3(chunk_size: int = 8, pool: str = "mean") -> None:
    records = _read_records()
    ids = [rec["id"] for rec in records]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for start in range(0, len(records), chunk_size):
        end = min(start + chunk_size, len(records))
        out = _chunk_path(start, end)
        if out.exists():
            X = np.load(out)
            if X.shape == (end - start, SAE_DIM) and np.isfinite(X).all():
                print(f"[aim3-modal] skip complete chunk {start}:{end}", flush=True)
                continue
            out.unlink()

        seqs = [rec["seq"] for rec in records[start:end]]
        X = np.asarray(evo2_features.remote(seqs, pool=pool), dtype=np.float32)
        if X.shape != (end - start, SAE_DIM):
            raise ValueError(f"chunk {start}:{end} bad shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError(f"chunk {start}:{end} contains NaN or Inf")
        if np.any(np.all(X == 0, axis=1)):
            raise ValueError(f"chunk {start}:{end} contains all-zero rows")
        np.save(out, X)
        done = end
        elapsed = time.time() - t0
        print(
            f"[aim3-modal] wrote chunk {start}:{end} "
            f"done={done}/{len(records)} elapsed={elapsed:.1f}s",
            flush=True,
        )

    parts = []
    for start in range(0, len(records), chunk_size):
        end = min(start + chunk_size, len(records))
        path = _chunk_path(start, end)
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(np.load(path))
    X = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    _validate_matrix(X, ids)

    np.save(FEATURES, X)
    IDS.write_text("\n".join(ids) + "\n")
    meta = {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "pool": pool,
        "kind": "region_seq",
        "backend": "modal",
        "chunk_size": chunk_size,
        "seconds": round(time.time() - t0, 1),
        "mean_abs": float(np.mean(np.abs(X))),
    }
    META.write_text(json.dumps(meta, indent=2) + "\n")
    READY.write_text(json.dumps(meta, sort_keys=True) + "\n")
    print(f"AIM3 FEATURES READY {json.dumps(meta, sort_keys=True)}", flush=True)
