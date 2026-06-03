#!/usr/bin/env bash
set -uo pipefail

cd /Users/user/bio-interp-experiments
mkdir -p logs
exec >> logs/aim3_azure_after_aim1.log 2>&1

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }
ip() { az vm show -d -g RG-GPU-A100 -n a100box --query publicIps -o tsv; }
sshbox() { ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=20 "azuser@$(ip)" "$@"; }
scpbox() { scp -q -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no "$@"; }

echo "[aim3-azure] start $(now)"

# Kill the old conservative watcher that only submits Aim3 after Aim1 AND Aim2.
sshbox 'pkill -f "[A]IM3_EMBED_SUBMITTED" 2>/dev/null || true; rm -f ~/queue_evo2/aim3.job' || true

while true; do
  if [ -f data/aim3_assoc/FEATURES_READY ]; then
    echo "[aim3-azure] local features already ready; exiting $(now)"
    exit 0
  fi

  if sshbox 'test -f ~/aim3_direct.done'; then
    echo "[aim3-azure] pulling completed Aim3 $(now)"
    mkdir -p data/aim3_assoc
    for f in features.npy ids.txt meta.json; do
      scpbox "azuser@$(ip):~/genomic-fm-sae-experiments/data/aim3_assoc/evo2_seq_job/$f" "data/aim3_assoc/$f"
    done
    .venv/bin/python - <<'PY'
import json
import numpy as np
from pathlib import Path
p = Path("data/aim3_assoc")
X = np.load(p / "features.npy")
ids = [x.strip() for x in (p / "ids.txt").read_text().splitlines() if x.strip()]
if X.shape != (len(ids), 32768):
    raise SystemExit(f"bad Aim3 shape {X.shape} ids={len(ids)}")
if len(ids) != 4080:
    raise SystemExit(f"unexpected Aim3 row count {len(ids)}")
if np.isnan(X).any():
    raise SystemExit("NaNs in Aim3 features")
if (np.abs(X).sum(axis=1) == 0).any():
    raise SystemExit("all-zero Aim3 feature row")
meta_path = p / "meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
meta.update({"validated_local": True, "n": int(X.shape[0]), "dim": int(X.shape[1])})
meta_path.write_text(json.dumps(meta, indent=2))
(p / "FEATURES_READY").write_text(json.dumps(meta) + "\n")
print(meta)
PY
    echo "[aim3-azure] local FEATURES_READY written $(now)"
    exit 0
  fi

  if sshbox 'test -f ~/aim3_direct.error'; then
    echo "[aim3-azure] AIM3 ERROR $(now)"
    sshbox 'cat ~/aim3_direct.error; tail -80 ~/aim3_direct.log 2>/dev/null || true'
    sleep 300
  fi

  if sshbox 'test -f ~/aim1_direct.done'; then
    if ! sshbox 'pgrep -af "embed_evo2.py --job data/aim3_assoc/evo2_seq_job" >/dev/null'; then
      echo "[aim3-azure] Aim1 done; launching Aim3 direct $(now)"
      sshbox 'cd ~/genomic-fm-sae-experiments && nohup bash -lc '"'"'
        set -e
        PYTHONSAFEPATH=1 HF_HOME=$HOME/hf_cache HF_HUB_DISABLE_XET=1 HG38_FA=$HOME/hf_cache/hg38.fa \
          ~/miniconda3/envs/evo2/bin/python azure/embed_evo2.py --job data/aim3_assoc/evo2_seq_job --pool mean \
          > ~/aim3_direct.log 2>&1
        echo DONE > ~/aim3_direct.done
      '"'"' >/dev/null 2>&1 &'
    else
      echo "[aim3-azure] Aim3 already running $(now)"
      sshbox 'tail -2 ~/aim3_direct.log 2>/dev/null || true; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true'
    fi
  else
    echo "[aim3-azure] waiting for Aim1 $(now)"
    sshbox 'tail -2 ~/aim1_direct.log 2>/dev/null || true'
  fi

  sleep 60
done
