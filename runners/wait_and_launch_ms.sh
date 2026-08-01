#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
RESUME_CKPT=${RESUME_CKPT:?RESUME_CKPT is required}
PREPROCESSOR_PATH=${PREPROCESSOR_PATH:?PREPROCESSOR_PATH is required}
BATCH_SIZE=${BATCH_SIZE:-256}
ACC_BATCHES=${ACC_BATCHES:-1}
NUM_CPU=${NUM_CPU:-16}
EPOCHS=${EPOCHS:-50}

consecutive=0
while (( consecutive < 3 )); do
  read -r memory_used utilization < <(
    nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -1 | tr -d ','
  )
  if (( memory_used <= 1000 && utilization <= 10 )); then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  printf '[%s] waiting_for_idle memory_mib=%s util=%s idle_checks=%s/3\n' \
    "$(date '+%F %T %Z')" "$memory_used" "$utilization" "$consecutive"
  if (( consecutive < 3 )); then
    sleep 60
  fi
done

printf '[%s] A800 idle; launching MS extension\n' "$(date '+%F %T %Z')"
exec env \
  RUN_DIR="$RUN_DIR" \
  BATCH_SIZE="$BATCH_SIZE" \
  ACC_BATCHES="$ACC_BATCHES" \
  NUM_CPU="$NUM_CPU" \
  EPOCHS="$EPOCHS" \
  RESUME_CKPT="$RESUME_CKPT" \
  PREPROCESSOR_PATH="$PREPROCESSOR_PATH" \
  bash "$PROJECT_ROOT/runners/run_unified_hcir_msms_pretrain.sh"
