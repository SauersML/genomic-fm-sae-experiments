"""Modal app: Evo 2 (7B) + Goodfire SAE and ESM-2 650M + InterPLM SAE on A100-80GB.

This is the PRIMARY GPU backend for feature extraction. Weights live in a
persistent Modal Volume mounted at /cache (HF_HOME=/cache/hf), so they download
once and are reused across runs.

Setup (local, already authed, profile sauersml):
    cd /Users/user/bio-interp-experiments
    . .venv-modal/bin/activate

One-time weight download into the volume:
    modal run src/modal_app.py::download

Smoke test (1kb DNA -> Evo2 SAE, 200aa protein -> InterPLM SAE):
    modal run src/modal_app.py::smoke

Batch feature extraction from Python (data agents):
    import modal
    f = modal.Function.from_name("bio-interp-fm", "evo2_features")
    feats = f.remote(list_of_dna_seqs)        # -> np.ndarray [N, 32768]
    d = modal.Function.from_name("bio-interp-fm", "evo2_feature_deltas")
    deltas = d.remote(list_of_(ref,alt)_tuples)  # -> {'delta_mean','delta_max'}
    p = modal.Function.from_name("bio-interp-fm", "esm2_features")
    pf = p.remote(list_of_protein_seqs)       # -> np.ndarray [N, 10240]

The extraction logic lives in src/evo2/extract.py and src/esm2/extract.py and is
mounted into the container so it stays backend-agnostic and editable.
"""
from __future__ import annotations

import time

import modal

APP_NAME = "bio-interp-fm"
CACHE_DIR = "/cache"
HF_HOME = "/cache/hf"

app = modal.App(APP_NAME)

# Persistent volume for HF weights (download once, reuse).
vol = modal.Volume.from_name("bio-interp-hf-cache", create_if_missing=True)

# ---- Image: torch + evo2 (StripedHyena2/vortex) + transformers + InterPLM deps.
# Use a prebuilt FlashAttention wheel matching torch 2.6 / py3.11. Source-building
# flash-attn on Modal previously stalled long enough to be a bad fallback path.
# Torch 2.6 + flash-attn wheels hit a known binary-compatibility failure in this
# image family. Use the Torch 2.7/CUDA12 wheel pair instead.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "build-essential", "ninja-build")
    .env({"HF_HOME": HF_HOME, "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(
        "pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126",
        "pip install https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.7/cu12/abiFALSE/cp311/flash_attn-2.8.3%2Bcu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
        "pip install 'numpy<2' huggingface_hub transformers==4.44.2 einops packaging wheel evo2==0.5.3",
    )
    # mount our backend-agnostic extraction libs
    .add_local_dir(
        "/Users/user/bio-interp-experiments/src",
        remote_path="/root/src",
    )
)

GPU = "A100-80GB"


def _import_extract():
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")
    from src.evo2 import extract as evo2x
    from src.esm2 import extract as esmx
    return evo2x, esmx


# --------------------------------------------------------------------------- #
# One-time download into the volume
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={CACHE_DIR: vol}, timeout=60 * 60,
              gpu=GPU)
def download():
    """Pull all 4 model/SAE repos into the persistent volume."""
    import os
    os.environ["HF_HOME"] = HF_HOME
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading Evo2-7B (via evo2 package)...")
    # evo2 downloads arcinstitute/evo2_7b on first construct; do it here so the
    # weights land in the volume.
    from evo2 import Evo2
    _ = Evo2("evo2_7b")
    print("Evo2-7B ready.")

    print("Downloading Goodfire SAE...")
    hf_hub_download("Goodfire/Evo-2-Layer-26-Mixed",
                    "sae-layer26-mixed-expansion_8-k_64.pt")
    print("Downloading ESM2-650M...")
    snapshot_download("facebook/esm2_t33_650M_UR50D")
    print("Downloading InterPLM SAE (layer 24)...")
    hf_hub_download("Elana/InterPLM-esm2-650m", "layer_24/ae_normalized.pt")
    vol.commit()
    print("All weights cached in volume.")


# --------------------------------------------------------------------------- #
# Feature functions (batched) — return numpy arrays
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={CACHE_DIR: vol}, gpu=GPU,
              timeout=60 * 30, max_containers=4)
def evo2_features(seqs: list[str], pool: str = "mean"):
    """DNA seqs -> [N, 32768] pooled Evo2 layer-26 SAE feature matrix."""
    import os
    os.environ["HF_HOME"] = HF_HOME
    evo2x, _ = _import_extract()
    return evo2x.features_for_regions(seqs, pool=pool, device="cuda")


@app.function(image=image, volumes={CACHE_DIR: vol}, gpu=GPU,
              timeout=60 * 30, max_containers=4)
def evo2_feature_deltas(pairs: list[tuple[str, str]]):
    """(ref,alt) pairs -> {'delta_mean':[N,32768], 'delta_max':[N,32768]}."""
    import os
    os.environ["HF_HOME"] = HF_HOME
    evo2x, _ = _import_extract()
    return evo2x.feature_deltas(pairs, device="cuda")


@app.function(image=image, volumes={CACHE_DIR: vol}, gpu=GPU,
              timeout=60 * 30)
def esm2_features(seqs: list[str], pool: str = "mean"):
    """Protein seqs -> [N, 10240] pooled InterPLM SAE feature matrix."""
    import os
    os.environ["HF_HOME"] = HF_HOME
    _, esmx = _import_extract()
    return esmx.features_for_proteins(seqs, pool=pool, device="cuda")


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={CACHE_DIR: vol}, gpu=GPU,
              timeout=60 * 30)
def smoke():
    import os
    os.environ["HF_HOME"] = HF_HOME
    import numpy as np
    import torch
    evo2x, esmx = _import_extract()

    results = {}

    # ---- Evo2: 1kb DNA ----
    rng = np.random.default_rng(0)
    dna = "".join(rng.choice(list("ACGT"), size=1000))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    acts = evo2x.embed_dna([dna], device="cuda")[0]          # [L, 4096]
    sae = evo2x._load_sae_cached("cuda")
    feats = sae.encode(acts, topk=True)                       # [L, 32768]
    torch.cuda.synchronize()
    dt = time.time() - t0
    L = acts.shape[0]
    peak = torch.cuda.max_memory_allocated() / 1e9
    pooled = feats.mean(0)
    results["evo2"] = {
        "dna_len": len(dna), "tokens": int(L),
        "acts_shape": tuple(acts.shape),
        "feat_shape": tuple(feats.shape),
        "pooled_shape": tuple(pooled.shape),
        "nnz_per_token_mean": float((feats != 0).sum(-1).float().mean().item()),
        "pooled_nnz": int((pooled != 0).sum().item()),
        "peak_gpu_gb": round(peak, 2),
        "seconds": round(dt, 3),
        "tokens_per_sec": round(L / dt, 1),
    }
    print("EVO2:", results["evo2"])

    # ---- ESM2: 200aa protein ----
    aa = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=200))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    emb = esmx.embed_protein([aa], device="cuda")[0]          # [L, 1280]
    psae = esmx._load_sae_cached("cuda")
    pfeats = psae.encode(emb)                                  # [L, 10240]
    torch.cuda.synchronize()
    dt2 = time.time() - t0
    peak2 = torch.cuda.max_memory_allocated() / 1e9
    results["esm2"] = {
        "aa_len": len(aa),
        "emb_shape": tuple(emb.shape),
        "feat_shape": tuple(pfeats.shape),
        "pooled_nnz": int((pfeats.mean(0) != 0).sum().item()),
        "peak_gpu_gb": round(peak2, 2),
        "seconds": round(dt2, 3),
    }
    print("ESM2:", results["esm2"])

    # persist a copy in the volume
    import json
    os.makedirs("/cache/smoke", exist_ok=True)
    with open("/cache/smoke/smoke_results.json", "w") as f:
        json.dump(results, f, indent=2)
    vol.commit()
    return results


@app.local_entrypoint()
def main():
    """`modal run src/modal_app.py` -> run the smoke test and print results."""
    import json
    res = smoke.remote()
    print(json.dumps(res, indent=2))


@app.local_entrypoint()
def aim3_embed(
    manifest: str = "data/aim3_assoc/seq_manifest.jsonl",
    outdir: str = "data/aim3_assoc",
    batch_size: int = 16,
    limit: int = 0,
):
    """Embed Aim-3 seq_manifest rows on Modal and write outputs locally."""
    import json
    import time
    from pathlib import Path

    import numpy as np

    root = Path("/Users/user/bio-interp-experiments")
    mpath = root / manifest
    odir = root / outdir

    ids: list[str] = []
    seqs: list[str] = []
    with mpath.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(str(rec["id"]))
            seqs.append(str(rec["seq"]).upper())
    if limit:
        ids = ids[:limit]
        seqs = seqs[:limit]
    if not ids:
        raise ValueError(f"empty manifest: {mpath}")

    batches = [seqs[i:i + batch_size] for i in range(0, len(seqs), batch_size)]
    print(f"[modal-aim3] rows={len(ids)} batches={len(batches)} batch={batch_size}",
          flush=True)

    rows: list[np.ndarray] = []
    t0 = time.time()
    for i, arr in enumerate(evo2_features.map(batches), start=1):
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 32768:
            raise ValueError(f"bad batch shape at {i}: {a.shape}")
        if np.isnan(a).any():
            raise ValueError(f"NaN in batch {i}")
        rows.append(a)
        done = sum(x.shape[0] for x in rows)
        print(f"[modal-aim3] batch {i}/{len(batches)} rows={done}/{len(ids)} "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    X = np.concatenate(rows, axis=0)
    if X.shape != (len(ids), 32768):
        raise ValueError(f"final shape {X.shape} != ({len(ids)}, 32768)")
    if (np.abs(X).sum(axis=1) == 0).any():
        raise ValueError("one or more all-zero feature rows")

    odir.mkdir(parents=True, exist_ok=True)
    np.save(odir / "features.npy", X)
    with (odir / "ids.txt").open("w") as f:
        for rid in ids:
            f.write(rid + "\n")
    meta = {
        "backend": "modal",
        "model": "evo2_7b",
        "sae": "Goodfire/Evo-2-Layer-26-Mixed",
        "pool": "mean",
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "batch_size": batch_size,
        "seconds": round(time.time() - t0, 1),
        "source_manifest": str(mpath),
    }
    with (odir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    (odir / "FEATURES_READY").write_text(json.dumps(meta) + "\n")
    print(f"[modal-aim3] DONE shape={X.shape} meta={meta}", flush=True)


@app.local_entrypoint()
def embed_deltas(
    manifest: str = "data/inversions/seq_pairs.jsonl",
    outdir: str = "data/inversions",
    batch_size: int = 8,
    limit: int = 0,
):
    """Embed ref/alt sequence-pair deltas on Modal and write mean deltas locally."""
    import json
    import time
    from pathlib import Path

    import numpy as np

    root = Path("/Users/user/bio-interp-experiments")
    mpath = root / manifest
    odir = root / outdir

    ids: list[str] = []
    pairs: list[tuple[str, str]] = []
    with mpath.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(str(rec["id"]))
            pairs.append((str(rec["ref_seq"]).upper(), str(rec["alt_seq"]).upper()))
    if limit:
        ids = ids[:limit]
        pairs = pairs[:limit]
    if not ids:
        raise ValueError(f"empty manifest: {mpath}")

    batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
    print(f"[modal-deltas] rows={len(ids)} batches={len(batches)} batch={batch_size}",
          flush=True)

    rows: list[np.ndarray] = []
    t0 = time.time()
    for i, out in enumerate(evo2_feature_deltas.map(batches), start=1):
        a = np.asarray(out["delta_mean"], dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 32768:
            raise ValueError(f"bad batch shape at {i}: {a.shape}")
        if np.isnan(a).any():
            raise ValueError(f"NaN in batch {i}")
        rows.append(a)
        done = sum(x.shape[0] for x in rows)
        print(f"[modal-deltas] batch {i}/{len(batches)} rows={done}/{len(ids)} "
              f"elapsed={time.time() - t0:.1f}s", flush=True)

    X = np.concatenate(rows, axis=0)
    if X.shape != (len(ids), 32768):
        raise ValueError(f"final shape {X.shape} != ({len(ids)}, 32768)")
    if (np.abs(X).sum(axis=1) == 0).any():
        raise ValueError("one or more all-zero feature rows")

    odir.mkdir(parents=True, exist_ok=True)
    np.save(odir / "features.npy", X)
    with (odir / "ids.txt").open("w") as f:
        for rid in ids:
            f.write(rid + "\n")
    meta = {
        "backend": "modal",
        "model": "evo2_7b",
        "sae": "Goodfire/Evo-2-Layer-26-Mixed",
        "pool": "mean_delta",
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "batch_size": batch_size,
        "seconds": round(time.time() - t0, 1),
        "source_manifest": str(mpath),
    }
    with (odir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    (odir / "FEATURES_READY").write_text(json.dumps(meta) + "\n")
    print(f"[modal-deltas] DONE shape={X.shape} meta={meta}", flush=True)
