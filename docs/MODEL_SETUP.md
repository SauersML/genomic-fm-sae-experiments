# MODEL_SETUP — foundation-model + SAE extraction pipelines

Primary GPU backend = **Modal** (serverless A100-80GB / H100, profile `sauersml`).
This solves the Evo2-7B memory problem: full **bf16 on 80 GB**, no FP8-on-L4
compromise. The L4 VM `evo-gpu-1` is kept idle as a fallback only.

All model/SAE specs below were verified against the **live HF checkpoints**
(state-dict keys + shapes read directly) — see also `docs/MODELS_REF.md`.

---

## 1. What runs where

| Layer | Location |
|---|---|
| Backend-agnostic extraction logic | `src/evo2/extract.py`, `src/esm2/extract.py` |
| Modal app (image, volume, GPU functions) | `src/modal_app.py` |
| Persistent HF weight cache | Modal Volume `bio-interp-hf-cache` mounted at `/cache`, `HF_HOME=/cache/hf` |

The extraction modules only need `torch` + `evo2`/`transformers` + `huggingface_hub`,
so they import and run inside the Modal container *and* on any GPU box.

---

## 2. Environment / invocation

Local Modal client venv (already authenticated, profile `sauersml`):

```bash
cd /Users/user/bio-interp-experiments
. .venv-modal/bin/activate         # modal 1.4.3
```

One-time weight download into the persistent volume (also builds the image):

```bash
modal run src/modal_app.py::download
```

Smoke test:

```bash
modal run src/modal_app.py::smoke          # or: modal run src/modal_app.py
```

### How the data agents call it (get features for a list of sequences)

```python
import modal

# DNA -> Evo2 layer-26 SAE features, pooled [N, 32768]
evo2_features = modal.Function.from_name("bio-interp-fm", "evo2_features")
feats = evo2_features.remote(list_of_dna_seqs, pool="mean")   # numpy [N, 32768]

# Ref/alt variant deltas
evo2_deltas = modal.Function.from_name("bio-interp-fm", "evo2_feature_deltas")
d = evo2_deltas.remote(list_of_(ref_seq, alt_seq)_tuples)
# d == {"delta_mean": [N,32768], "delta_max": [N,32768]}  (numpy)

# Protein -> ESM2 layer-24 InterPLM SAE features, pooled [N, 10240]
esm2_features = modal.Function.from_name("bio-interp-fm", "esm2_features")
pf = esm2_features.remote(list_of_protein_seqs, pool="mean")  # numpy [N, 10240]
```

`pool` is `"mean"` (default) or `"max"` over sequence length. Functions are
batched (pass the whole list in one `.remote()` call to amortize model load).

Importable Python API inside the container (or any GPU box), from
`src/evo2/extract.py`:

```python
embed_dna(seqs)              -> list[ Tensor[L, 4096] ]      # layer-26 acts
sae_features(acts)           -> Tensor[L, 32768]            # per-token top-k
features_for_regions(seqs, pool="mean"|"max") -> np.ndarray[N, 32768]
feature_delta(ref_seq, alt_seq) -> {delta_mean, delta_max, ref_*, alt_*, nnz_*}
feature_deltas(pairs)        -> {delta_mean:[N,32768], delta_max:[N,32768]}
```

from `src/esm2/extract.py`:

```python
embed_protein(seqs)          -> list[ Tensor[L, 1280] ]     # layer-24, per-residue
sae_features(emb)            -> Tensor[L, 10240]
features_for_proteins(seqs, pool="mean"|"max") -> np.ndarray[N, 10240]
```

---

## 3. Exact layer / hook names + SAE dims

### Genomic — Evo 2

| | value |
|---|---|
| Base FM | `evo2_7b` (StripedHyena2, d_model **4096**, 32 blocks), bf16 |
| Hook / layer | **`blocks.26.mlp.l3`** -> activations `[1, L, 4096]` |
| Get acts | `model(input_ids, return_embeddings=True, layer_names=['blocks.26.mlp.l3'])` |
| SAE repo / file | `Goodfire/Evo-2-Layer-26-Mixed` / `sae-layer26-mixed-expansion_8-k_64.pt` |
| SAE type | tied **BatchTopK**, k=64, d_sae **32768** (×8) |
| state-dict keys | `_orig_mod.W[4096,32768]`, `_orig_mod.b_enc[32768]`, `_orig_mod.b_dec[4096]` — strip `_orig_mod.` |
| Encode | `topk_64( relu( (x - b_dec) @ W + b_enc ) )` per token |
| Decode | `f @ W.T + b_dec` |
| Tokenizer | `model.tokenizer.tokenize(seq)`, single-nucleotide, no BOS/EOS |

### Protein — ESM-2 / InterPLM

| | value |
|---|---|
| Base FM | `facebook/esm2_t33_650M_UR50D` (33 layers, hidden **1280**) |
| Layer | **24** (of {1,9,18,24,30,33}; mid-depth — tunable knob) |
| Get emb | `output_hidden_states=True` -> `hidden_states[24]`, drop `<cls>`/`<eos>` -> per-residue `[L,1280]` |
| SAE repo / file | `Elana/InterPLM-esm2-650m` / `layer_24/ae_normalized.pt` |
| SAE type | untied **ReLU SAE**, d_sae **10240** (×8) |
| state-dict keys | `bias[1280]`, `encoder.weight[10240,1280]`, `encoder.bias[10240]`, `decoder.weight[1280,10240]` |
| Encode | `relu( (x - bias) @ encoder.weight.T + encoder.bias )` |
| Decode | `f @ decoder.weight.T + bias` |
| Normalization | `ae_normalized.pt` bakes Swiss-Prot rescale into weights (NO `activation_rescale_factor` buffer shipped) -> features already ~[0,1] |

---

## 4. Evo2-7B feasibility verdict + memory / throughput

**Verdict: Evo2-7B runs in full bf16 on the Modal A100-80GB — the L4 FP8
compromise is moot and abandoned.** (Note for the record: Arc's own guidance is
that **7B runs in bf16 without Transformer Engine on any supported GPU**; FP8 via
TE is only *required* for the 20B/40B variants and only numerically validated on
Hopper. On the L4's Ada GPU, FP8 via TE is not a supported path for 7B, so bf16
on a bigger GPU was always the right call — Modal provides it.)

**Modal build status (honest): NOT yet validated end-to-end.** The Modal image
build for `download()`/`smoke()` **failed at the flash-attn link step** — every
CUDA `.o` compiled fine, but flash-attn's `setup.py` invoked `clang++` for the
final `.so` link and the CUDA `devel` base (with Modal `add_python`) has no
clang installed:

```
error: command 'clang++' failed: No such file or directory
ERROR: Failed building wheel for flash-attn
```

So `download()` never populated the volume and `smoke()` did not run — **no
A100 mem/throughput numbers were captured by this agent.** This is a packaging
issue, not a model/algorithm problem. The extraction libs (`src/evo2/extract.py`,
`src/esm2/extract.py`) are syntactically valid and used by the Azure backend,
but were NOT executed end-to-end on Modal.

Fix (one-liner) for whoever resumes Modal: force the GNU compiler for the
flash-attn build, e.g. add `clang` to the image (`.apt_install("clang")`) OR set
`.run_commands("CC=gcc CXX=g++ pip install flash-attn==2.6.3 --no-build-isolation")`.
Arc also publishes prebuilt evo2/flash-attn paths; using a torch+CUDA base that
matches a prebuilt flash-attn wheel avoids the source compile entirely.

Smoke numbers (A100-80GB, batch=1, no_grad): _not captured — see build status above._

- Evo2 1kb DNA: peak GPU mem = (pending) · tokens/sec = (pending) · feat shape [L, 32768]
- ESM2 200aa: peak GPU mem = (pending) · feat shape [L, 10240]

> Note: the Azure A100 box (`a100box`) is now the primary backend and reuses
> these same extraction modules; capture the live numbers there.

---

## 5. Notes for scaling

- The volume caches all 4 repos after the first `download`; later runs skip
  download and just pay container cold-start + model load (~tens of sec).
- Evo2 acts are `[L, 4096]` bf16; the SAE runs in fp32 (cast inside `encode`).
  For long windows, feed batch=1 and pool over length to keep `[N, 32768]` small.
- Per-token top-k (k=64) is applied before pooling so ref/alt feature deltas are
  faithful (matches BatchTopK inference convention).
- Fallback L4 VM `evo-gpu-1`: venv at `/opt/venv` (`--system-site-packages`,
  inherits system torch 2.9.1+cu129), HF cache `/opt/hf_cache`. Source it with
  `source /opt/venv/bin/activate; export HF_HOME=/opt/hf_cache`. Not the primary path.
