#!/usr/bin/env python3
"""Robust long-running poller for the three genomic SAE aims."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path("/Users/user/bio-interp-experiments")
LOG = ROOT / "logs/final_poll.log"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with LOG.open("a") as f:
        f.write(f"[{ts}] {msg}\n")
        f.flush()


def run(cmd: list[str] | str, *, shell: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=shell, cwd=ROOT, text=True, capture_output=True, timeout=timeout)


def az_ip() -> str:
    p = run(["az", "vm", "show", "-d", "-g", "RG-GPU-A100", "-n", "a100box", "--query", "publicIps", "-o", "tsv"], timeout=60)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout.strip()


def ssh(cmd: str, timeout: int | None = 120) -> subprocess.CompletedProcess:
    ip = az_ip()
    return run(["ssh", "-i", str(Path.home() / ".ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=20", f"azuser@{ip}", cmd], timeout=timeout)


def scp(remote: str, local: Path, timeout: int | None = 600) -> bool:
    ip = az_ip()
    local.parent.mkdir(parents=True, exist_ok=True)
    p = run(["scp", "-q", "-i", str(Path.home() / ".ssh/id_ed25519"), "-o", "StrictHostKeyChecking=no",
             f"azuser@{ip}:{remote}", str(local)], timeout=timeout)
    if p.returncode != 0:
        log(f"scp failed {remote}: {p.stderr.strip()}")
        return False
    return True


def remote_exists(path: str) -> bool:
    return ssh(f"test -f {path}", timeout=60).returncode == 0


def pull_aim(aim: str, remote_dir: str | None = None) -> bool:
    remote_dir = remote_dir or f"~/genomic-fm-sae-experiments/data/{aim}"
    ok = True
    for name in ["features.npy", "ids.txt", "meta.json", "gc.npy", "gc_window.parquet", "gc_window.csv"]:
        ok = scp(f"{remote_dir}/{name}", ROOT / f"data/{aim}/{name}") or ok
    return (ROOT / f"data/{aim}/features.npy").exists() and (ROOT / f"data/{aim}/ids.txt").exists()


def validate_aim3() -> bool:
    d = ROOT / "data/aim3_assoc"
    try:
        X = np.load(d / "features.npy")
        ids = [x.strip() for x in (d / "ids.txt").read_text().splitlines() if x.strip()]
        if X.shape != (len(ids), 32768) or len(ids) != 4080:
            log(f"aim3 bad shape {X.shape}, ids={len(ids)}")
            return False
        if np.isnan(X).any() or (np.abs(X).sum(axis=1) == 0).any():
            log("aim3 invalid rows: nan or all-zero")
            return False
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        meta.update({"validated_local": True, "n": int(X.shape[0]), "dim": int(X.shape[1])})
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "FEATURES_READY").write_text(json.dumps(meta) + "\n")
        return True
    except Exception as e:
        log(f"aim3 validation error: {e!r}")
        return False


def analyze(aim: str) -> bool:
    done = ROOT / f"logs/{aim}.analysis.done"
    if done.exists():
        return True
    cmd = {
        "aim1_sv": [".venv/bin/python", "src/aim1_sv/analyze.py"],
        "aim2_popgen": [".venv/bin/python", "-m", "src.aim2_popgen.analyze"],
        "aim3_assoc": [".venv/bin/python", "src/aim3_assoc/analyze.py"],
    }[aim]
    p = run(cmd, timeout=7200)
    (ROOT / f"logs/{aim}.analysis.stdout").write_text(p.stdout)
    (ROOT / f"logs/{aim}.analysis.stderr").write_text(p.stderr)
    if p.returncode != 0:
        log(f"{aim} analyze failed rc={p.returncode}: {p.stderr[-1000:]}")
        return False
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
    log(f"{aim} analysis done")
    return True


def launch_aim3_after_aim1() -> None:
    if (ROOT / "data/aim3_assoc/FEATURES_READY").exists() or remote_exists("~/aim3_direct.done"):
        return
    if not remote_exists("~/aim1_direct.done"):
        return
    if ssh('pgrep -af "embed_evo2.py --job data/aim3_assoc/evo2_seq_job" >/dev/null', timeout=60).returncode == 0:
        return
    log("launching aim3 direct on Azure")
    cmd = r"""cd ~/genomic-fm-sae-experiments && nohup bash -lc '
set -e
PYTHONSAFEPATH=1 HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 HG38_FA=$HOME/hf_cache/hg38.fa \
  ~/miniconda3/envs/evo2/bin/python azure/embed_evo2.py --job data/aim3_assoc/evo2_seq_job --pool mean \
  > ~/aim3_direct.log 2>&1
echo DONE > ~/aim3_direct.done
' >/dev/null 2>&1 &"""
    p = ssh(cmd, timeout=60)
    log(f"aim3 launch rc={p.returncode} err={p.stderr.strip()}")


def stop_modal_apps() -> None:
    if not (ROOT / ".venv-modal/bin/modal").exists():
        return
    p = run(". .venv-modal/bin/activate && modal app list", shell=True, timeout=120)
    for line in p.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("ap-") and "ephemeral" in line:
            run(f". .venv-modal/bin/activate && modal app stop {parts[0]} --yes", shell=True, timeout=120)


def maybe_deallocate(all_done: bool) -> None:
    if not all_done:
        return
    stop_modal_apps()
    p = run(["az", "vm", "deallocate", "-g", "RG-GPU-A100", "-n", "a100box"], timeout=600)
    log(f"azure deallocate rc={p.returncode} out={p.stdout.strip()} err={p.stderr.strip()}")
    raise SystemExit(0)


def main() -> None:
    LOG.parent.mkdir(exist_ok=True)
    log("start")
    while True:
        try:
            status = ssh("nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader; "
                         "tail -2 ~/aim1_direct.log 2>/dev/null || true; "
                         "ls -la ~/aim1_direct.done ~/aim3_direct.done ~/queue_evo2/*.done ~/queue_evo2/*.error 2>/dev/null || true",
                         timeout=120)
            log("remote status:\n" + status.stdout.strip())

            if remote_exists("~/aim1_direct.done") and not (ROOT / "logs/aim1_sv.analysis.done").exists():
                if pull_aim("aim1_sv"):
                    analyze("aim1_sv")

            if remote_exists("~/queue_evo2/a_aim2.job.done") and not (ROOT / "logs/aim2_popgen.analysis.done").exists():
                if pull_aim("aim2_popgen"):
                    analyze("aim2_popgen")
            if remote_exists("~/queue_evo2/a_aim2.job.error"):
                err = ssh("cat ~/queue_evo2/a_aim2.job.error", timeout=60)
                log("AIM2 ERROR " + err.stdout.strip())

            launch_aim3_after_aim1()
            if remote_exists("~/aim3_direct.done") and not (ROOT / "data/aim3_assoc/FEATURES_READY").exists():
                if pull_aim("aim3_assoc", "~/genomic-fm-sae-experiments/data/aim3_assoc/evo2_seq_job") and validate_aim3():
                    analyze("aim3_assoc")
            elif (ROOT / "data/aim3_assoc/FEATURES_READY").exists() and not (ROOT / "logs/aim3_assoc.analysis.done").exists():
                analyze("aim3_assoc")

            all_done = all((ROOT / f"logs/{aim}.analysis.done").exists()
                           for aim in ["aim1_sv", "aim2_popgen", "aim3_assoc"])
            maybe_deallocate(all_done)
        except Exception as e:
            log(f"loop error: {e!r}")
        time.sleep(60)


if __name__ == "__main__":
    main()
