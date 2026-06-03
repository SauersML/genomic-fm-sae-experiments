#!/usr/bin/env python3
"""Finish Aim 2/Aim 3 once Azure Evo2 jobs complete.

This is intentionally operational glue, not a library. It waits for the active
Azure A100 jobs to finish, pulls the small feature artifacts back, validates
them, runs the already-written analyses, writes a final cross-aim verdict, and
deallocates the Azure VM only after all remaining analyses succeed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/Users/user/bio-interp-experiments")
BOX = "azuser@20.225.193.70"
KEY = Path.home() / ".ssh/id_ed25519"
AZ_RG = "RG-GPU-A100"
AZ_VM = "a100box"
SSH_BASE = [
    "ssh",
    "-i",
    str(KEY),
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "ConnectTimeout=15",
    BOX,
]
SCP_BASE = ["scp", "-i", str(KEY), "-o", "StrictHostKeyChecking=no"]


def run(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=check)


def capture(cmd: list[str], *, cwd: Path = ROOT, check: bool = True) -> str:
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, text=True, check=check, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.stdout:
        print(p.stdout.rstrip(), flush=True)
    return p.stdout


def ssh(script: str, *, check: bool = True) -> str:
    return capture(SSH_BASE + [script], check=check)


def remote_exists(path: str) -> bool:
    p = subprocess.run(SSH_BASE + [f"test -e {path}"], text=True)
    return p.returncode == 0


def wait_for_done(name: str, done: str, err: str, progress_script: str, *, sleep_s: int = 120) -> None:
    print(f"[wait] {name}: waiting for {done}", flush=True)
    while True:
        if remote_exists(done):
            print(f"[done] {name}", flush=True)
            return
        if remote_exists(err):
            ssh(f"echo '[error file]'; cat {err}; echo '[recent logs]'; {progress_script}", check=False)
            raise RuntimeError(f"{name} failed; see {err}")
        ssh(progress_script, check=False)
        time.sleep(sleep_s)


def pull_files(remote_dir: str, local_dir: Path, files: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for fn in files:
        run(SCP_BASE + [f"{BOX}:{remote_dir}/{fn}", str(local_dir / fn)])


def validate_features(name: str, datadir: Path, expected_rows: int, expected_dim: int = 32768) -> dict:
    X = np.load(datadir / "features.npy", mmap_mode="r")
    ids = [line.strip() for line in (datadir / "ids.txt").read_text().splitlines() if line.strip()]
    meta = json.loads((datadir / "meta.json").read_text())
    print(f"[validate] {name}: X={X.shape}, ids={len(ids)}, meta={meta}", flush=True)
    if X.shape != (expected_rows, expected_dim):
        raise ValueError(f"{name}: expected {(expected_rows, expected_dim)}, got {X.shape}")
    if len(ids) != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} ids, got {len(ids)}")
    sample = np.asarray(X[: min(64, expected_rows)])
    if not np.isfinite(sample).all():
        raise ValueError(f"{name}: non-finite values in feature sample")
    row_norm = np.linalg.norm(sample, axis=1)
    if np.any(row_norm == 0):
        raise ValueError(f"{name}: all-zero feature row in sample")
    return {"shape": list(X.shape), "ids": len(ids), "meta": meta}


def compute_gc_for_aim2() -> None:
    """Compute GC aligned to Aim 2 ids.txt.

    The Azure embedder intentionally writes only SAE features/ids/meta. Aim 2's
    adversarial analysis requires GC as an explicit covariate, so derive it from
    the exact manifest rows that produced the feature IDs. The original Aim 2
    manifest includes both base 8kb regions and `__w32` 32kb companion windows;
    the final reported job uses the base 8kb IDs consumed by the analysis.
    """
    datadir = ROOT / "data/aim2_popgen"
    gpath = datadir / "gc.npy"
    if gpath.exists():
        return
    manifest = datadir / "manifest.jsonl"
    ids = [line.strip() for line in (datadir / "ids.txt").read_text().splitlines() if line.strip()]
    seq_by_id: dict[str, str] = {}
    for line_no, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        rid = str(rec["id"])
        seq = rec.get("seq")
        if seq is None:
            raise ValueError(f"{manifest}:{line_no}: cannot compute GC; row {rid} has no seq")
        seq_by_id[rid] = str(seq).upper()
    missing = [rid for rid in ids if rid not in seq_by_id]
    if missing:
        raise ValueError(f"Aim 2 GC: {len(missing)} ids missing from manifest, first={missing[:3]}")
    gc = []
    for rid in ids:
        seq = seq_by_id[rid]
        denom = sum(base in "ACGT" for base in seq)
        if denom == 0:
            raise ValueError(f"Aim 2 GC: row {rid} has no A/C/G/T bases")
        gc.append((seq.count("G") + seq.count("C")) / denom)
    arr = np.asarray(gc, dtype=np.float32)
    if arr.shape[0] != len(ids) or not np.isfinite(arr).all():
        raise ValueError("Aim 2 GC: invalid computed vector")
    np.save(gpath, arr)
    print(f"[aim2] computed {gpath} shape={arr.shape}", flush=True)


def run_aim2() -> dict:
    datadir = ROOT / "data/aim2_popgen"
    if not (datadir / "features.npy").exists():
        wait_for_done(
            "Aim 2",
            "~/aim2_w8_direct.done",
            "~/aim2_w8_direct.error",
            "echo '[gpu]'; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader; "
            "echo '[proc]'; pgrep -af 'data/aim2_popgen_w8|embed_evo2.py --serve' || true; "
            "echo '[tail]'; tail -8 ~/aim2_w8_direct.log 2>/dev/null || true; "
            "echo '[outputs]'; ls -lh ~/genomic-fm-sae-experiments/data/aim2_popgen/features.npy "
            "~/genomic-fm-sae-experiments/data/aim2_popgen/meta.json "
            "~/genomic-fm-sae-experiments/data/aim2_popgen_w8/features.npy "
            "~/genomic-fm-sae-experiments/data/aim2_popgen_w8/meta.json 2>/dev/null || true",
        )
        pull_files(
            "~/genomic-fm-sae-experiments/data/aim2_popgen_w8",
            datadir,
            ["features.npy", "ids.txt", "meta.json"],
        )
        if remote_exists("~/genomic-fm-sae-experiments/data/aim2_popgen_w8/gc.npy"):
            pull_files("~/genomic-fm-sae-experiments/data/aim2_popgen_w8", datadir, ["gc.npy"])
    compute_gc_for_aim2()
    val = validate_features("Aim 2", datadir, expected_rows=1200)
    run([str(ROOT / ".venv/bin/python"), "-m", "src.aim2_popgen.analyze", "--n-perm", "1000"])
    return val


def run_aim3() -> dict:
    datadir = ROOT / "data/aim3_assoc"
    if not (datadir / "features.npy").exists():
        wait_for_done(
            "Aim 3",
            "~/aim3_direct.done",
            "~/aim3_direct.error",
            "echo '[gpu]'; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader; "
            "echo '[proc]'; pgrep -af 'data/aim3_assoc/evo2_seq_job' || true; "
            "echo '[tail]'; tail -8 ~/aim3_direct.log 2>/dev/null || true",
        )
        pull_files(
            "~/genomic-fm-sae-experiments/data/aim3_assoc/evo2_seq_job",
            datadir,
            ["features.npy", "ids.txt", "meta.json"],
        )
    val = validate_features("Aim 3", datadir, expected_rows=4080)
    ready_meta = dict(val["meta"])
    ready_meta.setdefault("backend", "azure")
    ready_meta.setdefault("validated_by", "scripts/finish_remaining.py")
    (datadir / "FEATURES_READY").write_text(json.dumps(ready_meta, indent=2) + "\n")
    run([str(ROOT / ".venv/bin/python"), "src/aim3_assoc/analyze.py", "--n-perm", "500"])
    report = ROOT / "results/aim3_assoc/report.md"
    if report.exists():
        (ROOT / "docs/RESULTS_AIM3.md").write_text(report.read_text())
    return val


def metric_get(path: Path, keys: list[str], default=None):
    obj = json.loads(path.read_text())
    for k in keys:
        obj = obj.get(k, default) if isinstance(obj, dict) else default
    return obj


def write_final_verdict(aim2_val: dict, aim3_val: dict) -> None:
    lines = [
        "# Final Verdict",
        "",
        "## Status",
        "",
        "- Aim 1 completed: HPRC SV ref/alt Evo2-SAE deltas, held-out tests, adversarial covariate controls.",
        "- Aim 2 completed: sweep and introgression region feature-content tests against matched controls.",
        "- Aim 3 completed: haplotype feature profiles associated with expression across held-out individuals.",
        "",
        "## Feature Artifacts",
        "",
        f"- Aim 2 features: `{aim2_val['shape']}`, ids={aim2_val['ids']}, meta={aim2_val['meta']}",
        f"- Aim 3 features: `{aim3_val['shape']}`, ids={aim3_val['ids']}, meta={aim3_val['meta']}",
        "",
        "## Per-Aim Result Documents",
        "",
        "- `docs/RESULTS_AIM1.md`",
        "- `docs/RESULTS_AIM2.md`",
        "- `docs/RESULTS_AIM3.md`",
        "",
        "## Conservative Interpretation",
        "",
        "Treat any apparent signal as real only when it survives held-out evaluation,",
        "permutation/null testing, and the relevant adversarial controls. Aim-specific",
        "details and caveats are in the result documents above.",
        "",
    ]
    (ROOT / "docs/FINAL_VERDICT.md").write_text("\n".join(lines))


def modal_status() -> None:
    if (ROOT / ".venv-modal/bin/modal").exists():
        capture(["bash", "-lc", ". .venv-modal/bin/activate && modal container list"], check=False)


def deallocate_azure() -> None:
    print("[cleanup] deallocating Azure A100 now that Aim 2 and Aim 3 are complete", flush=True)
    run(["az", "vm", "deallocate", "-g", AZ_RG, "-n", AZ_VM])
    capture(["az", "vm", "list", "-d", "-o", "table"], check=False)


def main() -> int:
    os.chdir(ROOT)
    aim2_val = run_aim2()
    aim3_val = run_aim3()
    write_final_verdict(aim2_val, aim3_val)
    modal_status()
    deallocate_azure()
    print("[complete] remaining aims analyzed, final verdict written, Azure deallocated", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; Azure was not deallocated by this script.", file=sys.stderr)
        raise
