"""Evo 2 (7B) layer-26 activation extraction + Goodfire BatchTopK SAE encoding.

Backend-agnostic: this module only needs `torch`, the `evo2` package, and
`huggingface_hub`. It is imported and run *inside* the Modal A100 container
(see src/modal_app.py), but can also run on any GPU box with the evo2 stack.

Verified specs (docs/MODELS_REF.md; keys read from the live HF checkpoints):
  base FM   : evo2_7b  (StripedHyena2, d_model=4096, 32 blocks)
  hook      : blocks.26.mlp.l3   -> activations [1, L, 4096]
  SAE       : Goodfire/Evo-2-Layer-26-Mixed
              file  sae-layer26-mixed-expansion_8-k_64.pt
              tied BatchTopK, k=64, d_sae=32768
              state-dict keys (strip the torch.compile `_orig_mod.` prefix):
                  W      [4096, 32768]   (tied weight, d_model x d_sae)
                  b_enc  [32768]
                  b_dec  [4096]
  encode    : topk_64( relu( (x - b_dec) @ W + b_enc ) )   per token
  decode    : f @ W.T + b_dec
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EVO2_MODEL_NAME = "evo2_7b"
EVO2_LAYER = "blocks.26.mlp.l3"
EVO2_D_MODEL = 4096
SAE_REPO = "Goodfire/Evo-2-Layer-26-Mixed"
SAE_FILE = "sae-layer26-mixed-expansion_8-k_64.pt"
SAE_D = 32768
SAE_K = 64


# --------------------------------------------------------------------------- #
# SAE
# --------------------------------------------------------------------------- #
class Evo2SAE(torch.nn.Module):
    """Tied BatchTopK SAE for Evo 2 layer 26 (per-token top-k at inference)."""

    def __init__(self, W: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor,
                 k: int = SAE_K):
        super().__init__()
        self.register_buffer("W", W)          # [d_model, d_sae]
        self.register_buffer("b_enc", b_enc)  # [d_sae]
        self.register_buffer("b_dec", b_dec)  # [d_model]
        self.k = k
        self.d_model = W.shape[0]
        self.d_sae = W.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor, topk: bool = True) -> torch.Tensor:
        """x: [..., d_model] -> feature acts [..., d_sae] (per-token top-k sparse)."""
        x = x.to(self.W.dtype)
        pre = (x - self.b_dec) @ self.W + self.b_enc
        acts = F.relu(pre)
        if not topk:
            return acts
        topv, topi = acts.topk(self.k, dim=-1)
        out = torch.zeros_like(acts)
        out.scatter_(-1, topi, topv)
        return out

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W.t() + self.b_dec


def load_sae(device: str = "cuda", dtype: torch.dtype = torch.float32) -> Evo2SAE:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SAE_REPO, SAE_FILE)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    sae = Evo2SAE(sd["W"].to(dtype), sd["b_enc"].to(dtype), sd["b_dec"].to(dtype))
    return sae.to(device).eval()


# --------------------------------------------------------------------------- #
# Base model
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_evo2():
    """Load Evo2-7B once per process (bf16). Returns the evo2.Evo2 wrapper."""
    from evo2 import Evo2

    model = Evo2(EVO2_MODEL_NAME)
    return model


@lru_cache(maxsize=1)
def _load_sae_cached(device: str = "cuda") -> Evo2SAE:
    return load_sae(device=device)


@torch.no_grad()
def embed_dna(seqs: Sequence[str], device: str = "cuda") -> List[torch.Tensor]:
    """Run DNA sequence(s) through Evo2-7B and return layer-26 activations.

    Returns a list (one per input seq) of tensors [L, 4096] on `device`.
    Sequences are processed one at a time (batch=1) because evo2's tokenizer
    yields variable lengths; this keeps memory bounded and avoids padding noise.
    """
    model = _load_evo2()
    out: List[torch.Tensor] = []
    for seq in seqs:
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int,
                           device=device).unsqueeze(0)  # [1, L]
        _, embs = model(ids, return_embeddings=True, layer_names=[EVO2_LAYER])
        acts = embs[EVO2_LAYER][0]  # [L, 4096]
        out.append(acts)
        del ids, embs
    return out


@torch.no_grad()
def sae_features(acts: torch.Tensor, sae: Evo2SAE | None = None,
                 device: str = "cuda", topk: bool = True) -> torch.Tensor:
    """acts: [L, 4096] (or [..., 4096]) -> SAE feature acts [L, 32768]."""
    if sae is None:
        sae = _load_sae_cached(device)
    return sae.encode(acts.to(device), topk=topk)


# --------------------------------------------------------------------------- #
# High-level helpers used by the data agents
# --------------------------------------------------------------------------- #
@torch.no_grad()
def features_for_regions(seqs: Sequence[str], pool: str = "mean",
                         device: str = "cuda") -> np.ndarray:
    """List of DNA seqs -> [N, 32768] pooled SAE feature matrix (numpy float32).

    pool: 'mean' or 'max' over the sequence length (per-token top-k applied first).
    """
    sae = _load_sae_cached(device)
    rows = []
    for acts in embed_dna(seqs, device=device):
        feats = sae.encode(acts, topk=True)          # [L, 32768]
        rows.append(_pool(feats, pool).float().cpu().numpy())
        del acts, feats
    return np.stack(rows, axis=0)


def _pool(feats: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "mean":
        return feats.mean(dim=0)
    if pool == "max":
        return feats.max(dim=0).values
    raise ValueError(f"unknown pool {pool!r}")


@torch.no_grad()
def feature_delta(ref_seq: str, alt_seq: str, device: str = "cuda",
                  ) -> dict:
    """Ref vs alt SAE feature delta for one variant.

    Returns a dict with both mean- and max-pooled deltas over the full window:
        {
          'delta_mean': [32768]  (alt mean-pool  - ref mean-pool),
          'delta_max':  [32768]  (alt max-pool   - ref max-pool),
          'ref_mean', 'alt_mean', 'ref_max', 'alt_max': [32768] each,
          'nnz_delta_mean': int,
        }
    All vectors are numpy float32. Use mean for diffuse effects, max for a
    localized strong feature. (The two seqs may differ in length, e.g. indels;
    pooling over length makes the delta well-defined regardless.)
    """
    sae = _load_sae_cached(device)
    ref_acts, alt_acts = embed_dna([ref_seq, alt_seq], device=device)
    rf = sae.encode(ref_acts, topk=True)
    af = sae.encode(alt_acts, topk=True)
    ref_mean = rf.mean(0); alt_mean = af.mean(0)
    ref_max = rf.max(0).values; alt_max = af.max(0).values
    d_mean = (alt_mean - ref_mean)
    d_max = (alt_max - ref_max)
    return {
        "delta_mean": d_mean.float().cpu().numpy(),
        "delta_max": d_max.float().cpu().numpy(),
        "ref_mean": ref_mean.float().cpu().numpy(),
        "alt_mean": alt_mean.float().cpu().numpy(),
        "ref_max": ref_max.float().cpu().numpy(),
        "alt_max": alt_max.float().cpu().numpy(),
        "nnz_delta_mean": int((d_mean != 0).sum().item()),
    }


@torch.no_grad()
def feature_deltas(pairs: Iterable[Tuple[str, str]], device: str = "cuda",
                   ) -> dict:
    """Batched feature_delta over many (ref, alt) pairs.

    Returns {'delta_mean': [N,32768], 'delta_max': [N,32768]} numpy arrays.
    """
    dm, dx = [], []
    for ref, alt in pairs:
        d = feature_delta(ref, alt, device=device)
        dm.append(d["delta_mean"]); dx.append(d["delta_max"])
    return {"delta_mean": np.stack(dm), "delta_max": np.stack(dx)}
