#!/bin/bash
#SBATCH --job-name=vlm-step0
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data2/shared/vinhtranduc/vlm-leakage-pilot/logs/step0_%j.out
#SBATCH --error=/data2/shared/vinhtranduc/vlm-leakage-pilot/logs/step0_%j.err

set -euo pipefail
PY=/data2/cmdir/home/toandq/.venvs/biomedclip/bin/python
REPO=/data2/shared/vinhtranduc/vlm-leakage-pilot
RUN=$(cat "$REPO/results/LATEST_STEP0")
export PYTHONUNBUFFERED=1
# Do not echo token
export HF_HUB_DISABLE_PROGRESS_BARS=1

echo "HOST=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
echo "RUN=$RUN"
date -Is

EMB=/data2/shared/vinhtranduc/data/embeddings/step0_$(basename "$RUN")
mkdir -p "$EMB/nih_biomedclip" "$EMB/camelyon_biomedclip" "$EMB/camelyon_conch"

run_one () {
  local task="$1"
  local out="$2"
  local tries=0
  while true; do
    tries=$((tries+1))
    echo "=== TASK $task try=$tries ==="
    set +e
    "$PY" "$REPO/scripts/extract_embeddings.py" --task "$task" --out-dir "$out" --batch-size 256 --num-workers 8
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      echo "OK $task"
      return 0
    fi
    if [[ $tries -ge 2 ]]; then
      echo "FAIL $task after retry" | tee -a "$RUN/extract_failures.txt"
      tail -n 30 "$out/EXTRACT_FAIL.txt" 2>/dev/null || true
      return $rc
    fi
    echo "retrying $task once"
    sleep 5
  done
}

run_one nih_biomedclip "$EMB/nih_biomedclip" || true
run_one camelyon_biomedclip "$EMB/camelyon_biomedclip" || true
run_one camelyon_conch "$EMB/camelyon_conch" || true

echo "=== TIES ==="
TIES_ARGS=(--prompts-nih "$REPO/decisions/prompts_nih.json" --prompts-camelyon "$REPO/decisions/prompts_camelyon.json" --out "$RUN/ties.json" --seed 0 --device cuda)
[[ -f $EMB/nih_biomedclip/embeddings.npy ]] && TIES_ARGS+=(--nih-emb "$EMB/nih_biomedclip")
[[ -f $EMB/camelyon_biomedclip/embeddings.npy ]] && TIES_ARGS+=(--camelyon-biomedclip-emb "$EMB/camelyon_biomedclip")
[[ -f $EMB/camelyon_conch/embeddings.npy ]] && TIES_ARGS+=(--camelyon-conch-emb "$EMB/camelyon_conch")

"$PY" "$REPO/scripts/step0_ties.py" "${TIES_ARGS[@]}" || echo "TIES_FAILED" | tee -a "$RUN/extract_failures.txt"

echo "$EMB" > "$RUN/embeddings_root.txt"
date -Is
echo DONE
