# MODELS_REF — loading the two SAEs + base FMs

Precise, copy-pasteable reference for the genomic (Evo 2) and protein (ESM-2) SAEs and their base
foundation models. All dims/keys below were verified against the live HF repos (HF API + raw pickle
inspection) and source code, not just docs. CPU-only to read this; GPU only to run the base FMs.

---

## 1. Genomic: `Goodfire/Evo-2-Layer-26-Mixed` (SAE) + Evo 2 7B (base FM)

### Base foundation model
- **Model:** `evo2_7b` (Evo 2, 7B params). The SAE is trained on the **7B** model, *not* 40B or 1B.
  - Evo 2 7B = StripedHyena2, **model width (d_model) = 4096**, **32 blocks**, single-nucleotide
    resolution, context up to ~1M tokens (the released 7B main checkpoint is 1M ctx; `evo2_7b_base`
    is the 8192-ctx pretrain checkpoint). Use `evo2_7b`.
  - Source: Goodfire blog "the layer 26 SAE ... Evo 2 paper"; Evo 2 README model table; arch/width
    from the StripedHyena 2 paper (model width 4096). [G1][E1][E3]
- **Python package:** `evo2` (PyPI / ArcInstitute/evo2). Requires GPU (FlashAttention/Hyena kernels).
- **Tokenizer:** byte/single-nucleotide. `evo2_model.tokenizer.tokenize("ACGT...")` → list[int] of
  per-nucleotide token ids (uppercase ACGT; the tokenizer is byte-level so it maps each char to a byte
  id). No special BOS/EOS is added by `tokenize`; you feed the raw nucleotide string. [E1]

### Which layer / hook point feeds the SAE
- **Layer 26**, MLP output sublayer. The exact `evo2` hook/layer string is **`blocks.26.mlp.l3`**
  (the Evo 2 README demonstrates the identical pattern with `blocks.28.mlp.l3`; `.mlp.l3` is the
  final linear of the block MLP and is the residual-stream-width activation the SAE consumes). [E1][G1]
- **Activation dim into SAE = 4096** (= d_model). Confirmed by the SAE weight shapes below.

### Getting activations from the `evo2` package
```python
import torch
from evo2 import Evo2

model = Evo2('evo2_7b')                 # downloads arcinstitute/evo2_7b weights
seq = 'ACGTACGT...'                      # raw nucleotides, uppercase
input_ids = torch.tensor(
    model.tokenizer.tokenize(seq), dtype=torch.int,
).unsqueeze(0).to('cuda:0')             # shape [1, L]

LAYER = 'blocks.26.mlp.l3'
_, embs = model(input_ids, return_embeddings=True, layer_names=[LAYER])
acts = embs[LAYER]                       # shape [1, L, 4096]  <-- SAE input x
```
Source: Evo 2 README "Embeddings" example (swap `blocks.28.mlp.l3` → `blocks.26.mlp.l3`). [E1]

### SAE: file, format, exact state-dict, encode math
- **File (only weight file in repo):** `sae-layer26-mixed-expansion_8-k_64.pt`
  (PyTorch zip-pickle, ~537 MB, FloatStorage/fp32). [G1 HF API]
- **Architecture (from filename + pickle dir name):** `batch-topk-tied`, **expansion 8**, **k = 64**.
  BatchTopK SAE with **tied weights** (encoder = decoderᵀ). "Mixed" = trained on an even mix of
  **eukaryotic + prokaryotic** high-quality reference genomes. [G1][G2]
- **Dictionary size d_sae = 32768** (= 4096 × expansion 8).
- **state_dict keys + shapes** (verified by reading the pickle directly; note the `_orig_mod.`
  prefix from `torch.compile` — strip it on load):

  | key                | shape          | meaning                         |
  |--------------------|----------------|---------------------------------|
  | `_orig_mod.W`      | `[4096, 32768]`| tied weight, d_model × d_sae    |
  | `_orig_mod.b_enc`  | `[32768]`      | encoder (pre-activation) bias   |
  | `_orig_mod.b_dec`  | `[4096]`       | decoder / input centering bias  |

  There is **no stored `k` or `threshold` tensor** — k=64 comes from the filename. At inference you
  either re-apply TopK (k=64) per token, or apply a JumpReLU threshold (the standard BatchTopK
  inference trick); for feature-delta work, top-k per token is the faithful choice. [BTK]

- **Minimal load + encode** (tied: W_enc = Wᵀ, W_dec = W):
```python
import torch, torch.nn.functional as F
from huggingface_hub import hf_hub_download

p = hf_hub_download("Goodfire/Evo-2-Layer-26-Mixed",
                    "sae-layer26-mixed-expansion_8-k_64.pt")
sd = torch.load(p, map_location="cpu")
sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
W     = sd["W"]        # [4096, 32768]  (d_model, d_sae)
b_enc = sd["b_enc"]    # [32768]
b_dec = sd["b_dec"]    # [4096]
K = 64

def encode(x):                      # x: [..., 4096] from blocks.26.mlp.l3
    pre = (x - b_dec) @ W + b_enc   # [..., 32768]   (W acts as W_encᵀ since tied)
    acts = F.relu(pre)
    # BatchTopK -> per-token top-k at inference:
    topv, topi = acts.topk(K, dim=-1)
    out = torch.zeros_like(acts).scatter_(-1, topi, topv)
    return out                       # sparse feature activations [..., 32768]

def decode(f):                       # reconstruct
    return f @ W.T + b_dec           # [..., 4096]
```
> Centering by `b_dec` before the encoder and decoding with `W.T` + `b_dec` is the standard
> tied-BatchTopK convention (matches Bussmann et al. BatchTopK SAEs [BTK]). If a future
> reconstruction sanity-check (decode(encode(x)) ≈ x) fails, try k-per-token vs JumpReLU-threshold
> and verify whether activations need unit-norm scaling — the repo ships no normalization buffer, so
> none is expected.

### Goodfire docs / paper on these SAEs
- Goodfire blog **"Interpreting Evo 2"** (Feb 2025): BatchTopK SAEs on Evo 2 layer 26; discovered
  features for **intron/exon boundaries, transcription-factor motifs, protein-structure traits**;
  even euk/prok training mix. [G2]
- Features ship in the **Evo 2 paper** (Nature / bioRxiv 2025.02.18.638918) interpretability section. [E2]

---

## 2. Protein: `Elana/InterPLM-esm2-650m` (SAE) + ESM-2 650M (base FM)

### Base foundation model
- **Model:** `facebook/esm2_t33_650M_UR50D` — ESM-2, 650M params, **33 layers**, **hidden dim 1280**.
- **Tokenizer:** HF `EsmTokenizer` / `AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")`.
  Adds **`<cls>` (BOS) at start and `<eos>` at end**; one token per amino acid. **Max context 1024**
  tokens (incl. cls/eos). Drop the cls/eos positions before/after SAE if you want per-residue features. [I1]

### Which layer feeds the SAE
- HF repo provides SAEs for ESM-2-650M layers **{1, 9, 18, 24, 30, 33}** (subdirs `layer_<n>/`).
- The InterPLM paper / interplm.ai analysis centers on a **mid layer**; the package README example uses
  **layer 18**, and **layer 24** is the other commonly-referenced mid-depth choice. All six share the
  same architecture and dict size. **Pick layer 24** (mid-depth, in the example set, the headline
  layer in the repo config) unless an experiment dictates otherwise; layer is a tunable knob here. [I2][I3]
- **Activation dim into SAE = 1280** (ESM-2-650M hidden size). [I2][I3]

### SAE: files, format, exact architecture, encode math
- **Per-layer files:** `layer_<n>/ae_normalized.pt`, `layer_<n>/ae_unnormalized.pt`, `layer_<n>/config.json`.
- **Use `ae_normalized.pt` by default** — features rescaled to ~[0,1] by Swiss-Prot max activations so
  features are comparable. Use `ae_unnormalized.pt` only for custom normalization. [I3]
- **Architecture (from `layer_24/config.json`):** `ReLUSAE`, **esm_dim 1280, expansion_factor 8,
  feature_dim (d_sae) = 10240**. Untied encoder/decoder; pre-encoder bias; unit-norm decoder rows.
  Same dict size 10240 for all six layers. [I2][I4]
- **state_dict keys** (`interplm.sae.dictionary.ReLUSAE`):
  `bias` `[1280]` (pre-encoder centering), `encoder.weight` `[10240,1280]`, `encoder.bias` `[10240]`,
  `decoder.weight` `[1280,10240]` (no decoder bias), buffer `activation_rescale_factor` `[10240]`. [I4]
- **Encode (exactly the InterPLM source):**
  ```
  features = ReLU( (x - bias) @ encoder.weight.T + encoder.bias )   # x: [..., 1280] -> [..., 10240]
  # if you loaded the *unnormalized* SAE and want comparable features:
  #   features = features / activation_rescale_factor
  ```
  `ae_normalized.pt` already has weights baked so plain `encode(x)` returns 0–1-scaled features
  (`normalize_features=False`). Decode: `x_hat = features @ decoder.weight.T + bias`. [I4]

### Recommended loading path (their package handles everything)
```python
from interplm.sae.inference import load_sae_from_hf
from interplm.esm.embed import embed_single_sequence

emb = embed_single_sequence(            # ESM-2 embeddings, per-residue [L, 1280]
    sequence="MRWQEMGYIFYPRKLR",
    model_name="esm2_t33_650M_UR50D",
    layer=24)                            # one of 1,9,18,24,30,33
sae = load_sae_from_hf(plm_model="esm2-650m", plm_layer=24)   # downloads ae_normalized.pt
features = sae.encode(emb)               # [L, 10240] sparse feature activations
```
`load_sae_from_hf(... unnormalized=True)` to get the raw SAE. Repo: `pip install` the InterPLM
GitHub package (ElanaPearl/InterPLM). [I3][I4]

### Paper / code
- **InterPLM** — Simon & Zou, "InterPLM: Discovering Interpretable Features in Protein Language Models
  via Sparse Autoencoders," bioRxiv 2024.11.14.623630 (now Nature Methods 2025). Up to ~2,548
  interpretable features/layer mapping to ~143 biological concepts (binding sites, structural motifs,
  domains). Features browsable at interplm.ai. [I2][I5]
- Code/training: github.com/ElanaPearl/InterPLM. [I3]

---

## Quick reference table

| | Genomic SAE | Protein SAE |
|---|---|---|
| SAE repo | `Goodfire/Evo-2-Layer-26-Mixed` | `Elana/InterPLM-esm2-650m` |
| Base FM | `evo2_7b` (Evo 2 7B, StripedHyena2) | `facebook/esm2_t33_650M_UR50D` |
| Hook / layer | `blocks.26.mlp.l3` (layer 26 MLP out) | hidden layer 24 (of {1,9,18,24,30,33}) |
| d_model (SAE input) | 4096 | 1280 |
| d_sae (dict) | 32768 (×8) | 10240 (×8) |
| SAE type | BatchTopK, **tied**, k=64 | ReLU SAE, untied |
| Weight file | `sae-layer26-mixed-expansion_8-k_64.pt` | `layer_24/ae_normalized.pt` |
| state-dict keys | `_orig_mod.{W[4096,32768], b_enc[32768], b_dec[4096]}` | `bias[1280], encoder.{weight[10240,1280],bias[10240]}, decoder.weight[1280,10240], activation_rescale_factor[10240]` |
| Encode | `topk_64(relu((x-b_dec)@W + b_enc))` | `relu((x-bias)@enc.Wᵀ + enc.b)` |
| Tokenizer | single-nucleotide byte tokenizer | EsmTokenizer (cls/eos, 1/residue) |
| Max context | ~1M nt (use what fits L4 mem) | 1024 tokens |
| Normalization | none in repo | normalized SAE = features ~[0,1] (Swiss-Prot) |

---

## Sources
- [G1] HF repo + files: https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed  (README + `https://huggingface.co/api/models/Goodfire/Evo-2-Layer-26-Mixed`; state-dict keys/shapes read from `sae-layer26-mixed-expansion_8-k_64.pt` pickle header)
- [G2] Goodfire blog "Interpreting Evo 2": https://www.goodfire.ai/research/interpreting-evo-2
- [E1] Evo 2 README (embeddings + tokenizer + model list): https://github.com/ArcInstitute/evo2/blob/main/README.md
- [E2] Evo 2 paper (Nature / bioRxiv): https://www.nature.com/articles/s41586-026-10176-5 · https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1
- [E3] StripedHyena 2 / multi-hybrid paper (model width 4096): https://arxiv.org/pdf/2503.01868 ; Evo 2 7B config: https://huggingface.co/arcinstitute/evo2_7b
- [BTK] Bussmann et al., "BatchTopK Sparse Autoencoders": https://arxiv.org/abs/2412.06410
- [I1] ESM-2 650M base model: https://huggingface.co/facebook/esm2_t33_650M_UR50D
- [I2] InterPLM SAE HF README: https://huggingface.co/Elana/InterPLM-esm2-650m
- [I3] InterPLM GitHub (package, usage, normalize.py): https://github.com/ElanaPearl/InterPLM
- [I4] InterPLM source (ReLUSAE encode, load_sae_from_hf): https://raw.githubusercontent.com/ElanaPearl/InterPLM/main/interplm/sae/dictionary.py · .../interplm/sae/inference.py ; per-layer arch: https://huggingface.co/Elana/InterPLM-esm2-650m/raw/main/layer_24/config.json
- [I5] InterPLM paper (Simon & Zou): https://www.biorxiv.org/content/10.1101/2024.11.14.623630v1 · https://www.nature.com/articles/s41592-025-02836-7
