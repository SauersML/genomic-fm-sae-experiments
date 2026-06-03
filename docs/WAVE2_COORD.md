# Wave 2 coordination protocol

Goal: run all three aims end-to-end (sequences → Evo2/ESM2 → SAE features →
held-out analysis → plots/stats), in parallel where possible, on a single A100
without GPU contention.

## Backends
- **PRIMARY = Azure A100 box `a100box`** (RG `RG-GPU-A100`, southcentralus, already RUNNING & billing — use it, then deallocate). One warm GPU. Has disk for hg38.fa + VCFs. Owned by the **azure-compute** agent.
- **FALLBACK = Modal** (`src/modal_app.py`, autoscaling A100, $30 cap). Use only if Azure setup fails.

## Roles
- **azure-compute** (compute server): owns the box. Installs the Evo2/ESM2 stack, downloads hg38.fa (+ .fai), syncs repo `src/` + `data/` manifests to the box, and exposes two batch entrypoints that keep the model warm:
  - `azure/embed_evo2.py --job <jobdir>`: reads `jobdir/manifest.jsonl` (one record per item) → writes `jobdir/features.npy` (+ `ids.txt`). Manifest record forms:
    - regions: `{"id":..., "seq":"ACGT..."}` OR `{"id":..., "chrom":..., "start0":..., "end0":...}` (extract from hg38.fa on the box).
    - ref/alt deltas: `{"id":..., "ref_seq":..., "alt_seq":...}` OR coords for both → writes the delta vector.
  - `azure/embed_esm2.py --job <jobdir>`: protein seqs → InterPLM features.
  Runs jobs **sequentially** (model loaded once). Reports the output path when each job is done.
- **aim1-sv / aim2-popgen / aim3-assoc** (prep+analysis): in parallel and CPU-only locally, each:
  1. Build `data/<aim>/manifest.jsonl` (sequences-as-coords + the small label/covariate/split tables). Do NOT extract FASTA locally unless trivial — coords are enough; the box extracts.
  2. Write + unit-test (on synthetic features) the analysis driver that calls `src/common/analysis.py::run_report` with the right `groups`, `covariates`, `task`.
  3. Signal "job ready" → orchestrator routes it to azure-compute.
  4. When `features.npy` exists, run the analysis driver → write `results/<aim>/` (results.json, report.md, plots) + a short `docs/RESULTS_<AIM>.md`.

## Held-out / confound rules (non-negotiable, quality)
- Aim1: split by **SV (random, but report by chromosome too)**; covariates = svlen (log), GC%; also run a coding-vs-intergenic test **matched on length**. Compare vs covariate-only + permutation null.
- Aim2: split **by chromosome** (hold out whole chroms); covariates = region length, GC%; positives vs matched controls; permutation null.
- Aim3: split **by individual**; covariate = ancestry (EUR/AFR from samples_pop.tsv) + a EUR-only sensitivity run; outcome = expression (regression).
- Never report a metric without a bootstrap CI, a permutation p, and a confound baseline.

## Budget
Pilots first (Aim1 ~400–800 windows, Aim2 ~600 regions/side, Aim3 ~6–10 loci × subset individuals). Confirm signal, then scale the promising ones. Deallocate the A100 the moment Wave 2 inference is done.
