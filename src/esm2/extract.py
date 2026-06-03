"""ESM-2 650M layer-24 embeddings + InterPLM ReLU SAE encoding.

Backend-agnostic: needs `torch`, `transformers`, `huggingface_hub`. Imported
inside the Modal A100 container (src/modal_app.py); runs on any GPU/CPU box.

Verified specs (docs/MODELS_REF.md; keys read from the live HF checkpoint):
  base FM   : facebook/esm2_t33_650M_UR50D  (33 layers, hidden 1280)
  layer     : 24  (one of {1,9,18,24,30,33}; mid-depth)
  SAE       : Elana/InterPLM-esm2-650m, file layer_24/ae_normalized.pt
              ReLUSAE (untied), d_sae=10240, expansion 8
              state-dict keys:
                  bias           [1280]   (pre-encoder centering)
                  encoder.weight [10240, 1280]
                  encoder.bias   [10240]
                  decoder.weight [1280, 10240]   (no decoder bias)
              `ae_normalized.pt` bakes the Swiss-Prot rescale into the weights
              (NO activation_rescale_factor buffer shipped) -> plain encode
              returns features already scaled to ~[0,1].
  encode    : relu( (x - bias) @ encoder.weight.T + encoder.bias )
  decode    : f @ decoder.weight.T + bias
  tokenizer : adds <cls> (pos 0) and <eos> (last) -> drop them for per-residue.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM_LAYER = 24
ESM_D = 1280
SAE_REPO = "Elana/InterPLM-esm2-650m"
SAE_FILE = f"layer_{ESM_LAYER}/ae_normalized.pt"
SAE_D = 10240


class InterPLMSAE(torch.nn.Module):
    """Untied ReLU SAE (InterPLM). encode = relu((x-bias)@Wenc.T + benc)."""

    def __init__(self, bias, enc_w, enc_b, dec_w):
        super().__init__()
        self.register_buffer("bias", bias)        # [d_model]
        self.register_buffer("enc_w", enc_w)      # [d_sae, d_model]
        self.register_buffer("enc_b", enc_b)      # [d_sae]
        self.register_buffer("dec_w", dec_w)      # [d_model, d_sae]
        self.d_model = bias.shape[0]
        self.d_sae = enc_b.shape[0]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.bias.dtype)
        return F.relu((x - self.bias) @ self.enc_w.t() + self.enc_b)

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.dec_w.t() + self.bias


def load_sae(device: str = "cuda", layer: int = ESM_LAYER,
             dtype: torch.dtype = torch.float32) -> InterPLMSAE:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SAE_REPO, f"layer_{layer}/ae_normalized.pt")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sae = InterPLMSAE(
        sd["bias"].to(dtype),
        sd["encoder.weight"].to(dtype),
        sd["encoder.bias"].to(dtype),
        sd["decoder.weight"].to(dtype),
    )
    return sae.to(device).eval()


@lru_cache(maxsize=1)
def _load_esm(device: str = "cuda"):
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM_MODEL_NAME, torch_dtype=torch.float32)
    model = model.to(device).eval()
    return tok, model


@lru_cache(maxsize=1)
def _load_sae_cached(device: str = "cuda") -> InterPLMSAE:
    return load_sae(device=device)


@torch.no_grad()
def embed_protein(seqs: Sequence[str], device: str = "cuda",
                  layer: int = ESM_LAYER) -> List[torch.Tensor]:
    """Protein seq(s) -> per-residue ESM-2 layer-`layer` embeddings.

    Returns list of [L_residues, 1280] tensors (cls/eos stripped) on `device`.
    """
    tok, model = _load_esm(device)
    out: List[torch.Tensor] = []
    for seq in seqs:
        enc = tok(seq, return_tensors="pt", add_special_tokens=True).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer][0]  # [T,1280]
        # drop cls (0) and eos (last real token); attention_mask marks padding
        n = int(enc["attention_mask"][0].sum().item())
        out.append(hs[1:n - 1])  # per-residue
        del enc, hs
    return out


@torch.no_grad()
def sae_features(emb: torch.Tensor, sae: InterPLMSAE | None = None,
                 device: str = "cuda") -> torch.Tensor:
    """emb: [L, 1280] -> InterPLM feature acts [L, 10240]."""
    if sae is None:
        sae = _load_sae_cached(device)
    return sae.encode(emb.to(device))


@torch.no_grad()
def features_for_proteins(seqs: Sequence[str], pool: str = "mean",
                          device: str = "cuda") -> np.ndarray:
    """List of protein seqs -> [N, 10240] pooled feature matrix (numpy float32)."""
    sae = _load_sae_cached(device)
    rows = []
    for emb in embed_protein(seqs, device=device):
        feats = sae.encode(emb)  # [L, 10240]
        if pool == "mean":
            v = feats.mean(0)
        elif pool == "max":
            v = feats.max(0).values
        else:
            raise ValueError(f"unknown pool {pool!r}")
        rows.append(v.float().cpu().numpy())
        del emb, feats
    return np.stack(rows, axis=0)
