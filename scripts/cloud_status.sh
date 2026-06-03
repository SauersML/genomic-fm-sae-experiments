#!/usr/bin/env bash
set -euo pipefail
cd /Users/user/bio-interp-experiments
IP=$(az vm show -d -g RG-GPU-A100 -n a100box --query publicIps -o tsv 2>/dev/null || true)
echo "Azure IP: ${IP:-none}"
if [ -n "$IP" ]; then
  ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=15 azuser@$IP '
    echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true
    echo "=== Evo2 processes ==="; pgrep -af "embed_evo2.py" || true
    echo "=== Aim3 tail ==="; tail -8 ~/aim3_direct.log 2>/dev/null || true
    echo "=== Sentinels ==="; ls -la ~/aim1_direct.done ~/aim3_direct.done ~/aim3_direct.error ~/queue_evo2/*.done ~/queue_evo2/*.error 2>/dev/null || true
    echo "=== Outputs ==="; for d in aim1_sv aim2_popgen aim3_assoc/evo2_seq_job; do echo $d; ls -lh ~/genomic-fm-sae-experiments/data/$d/features.npy ~/genomic-fm-sae-experiments/data/$d/ids.txt ~/genomic-fm-sae-experiments/data/$d/meta.json 2>/dev/null || true; done
  '
fi
if [ -x .venv-modal/bin/modal ]; then
  . .venv-modal/bin/activate
  echo "=== Modal apps ==="; modal app list | head -80
  echo "=== Modal containers ==="; modal container list | head -40
fi
