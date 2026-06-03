#!/usr/bin/env bash
set -uo pipefail

cd /Users/user/bio-interp-experiments
mkdir -p logs
exec >> logs/final_supervisor.log 2>&1

now() { date '+%Y-%m-%dT%H:%M:%S%z'; }

echo "[supervisor] start $(now)"

ip() {
  az vm show -d -g RG-GPU-A100 -n a100box --query publicIps -o tsv
}

sshbox() {
  ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=20 "azuser@$(ip)" "$@"
}

scpbox() {
  scp -q -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no "$@"
}

pull_aim() {
  local aim="$1"
  mkdir -p "data/$aim"
  for f in features.npy ids.txt meta.json gc.npy gc_window.parquet gc_window.csv; do
    scpbox "azuser@$(ip):~/genomic-fm-sae-experiments/data/$aim/$f" "data/$aim/$f" 2>/dev/null || true
  done
  ls -lh "data/$aim"/features.npy "data/$aim"/ids.txt "data/$aim"/meta.json
}

run_analysis() {
  local aim="$1"
  case "$aim" in
    aim1_sv) .venv/bin/python src/aim1_sv/analyze.py ;;
    aim2_popgen) .venv/bin/python -m src.aim2_popgen.analyze ;;
    aim3_assoc) .venv/bin/python src/aim3_assoc/analyze.py ;;
  esac
}

mark_done() {
  touch "logs/$1.analysis.done"
  echo "[supervisor] $1 analysis done $(now)"
}

while true; do
  echo "[supervisor] tick $(now)"

  if [ ! -f logs/aim1_sv.analysis.done ]; then
    if sshbox 'test -f ~/aim1_direct.done'; then
      echo "[supervisor] pulling aim1"
      pull_aim aim1_sv
      run_analysis aim1_sv
      mark_done aim1_sv
    elif sshbox 'test -f ~/aim1_direct.error'; then
      echo "[supervisor] AIM1 ERROR"
      sshbox 'cat ~/aim1_direct.error'
      sleep 300
    else
      sshbox 'tail -2 ~/aim1_direct.log 2>/dev/null || true; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader || true'
    fi
  fi

  if [ ! -f logs/aim2_popgen.analysis.done ]; then
    if sshbox 'test -f ~/queue_evo2/a_aim2.job.done'; then
      echo "[supervisor] pulling aim2"
      pull_aim aim2_popgen
      run_analysis aim2_popgen
      mark_done aim2_popgen
    elif sshbox 'test -f ~/queue_evo2/a_aim2.job.error'; then
      echo "[supervisor] AIM2 ERROR"
      sshbox 'cat ~/queue_evo2/a_aim2.job.error'
      sleep 300
    fi
  fi

  if [ ! -f logs/aim3_assoc.analysis.done ]; then
    if [ -f data/aim3_assoc/FEATURES_READY ]; then
      echo "[supervisor] analyzing aim3"
      run_analysis aim3_assoc
      mark_done aim3_assoc
    fi
  fi

  if [ -f logs/aim1_sv.analysis.done ] && [ -f logs/aim2_popgen.analysis.done ] && [ -f logs/aim3_assoc.analysis.done ]; then
    echo "[supervisor] all analyses done; stopping unused Modal apps and deallocating Azure"
    if [ -x .venv-modal/bin/modal ]; then
      . .venv-modal/bin/activate
      for app in $(modal app list 2>/dev/null | awk '/ephemeral/ {print $1}'); do
        modal app stop "$app" --yes || true
      done
    fi
    az vm deallocate -g RG-GPU-A100 -n a100box
    echo "[supervisor] azure deallocated $(now)"
    exit 0
  fi

  sleep 60
done
