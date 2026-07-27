#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
EVAL_DATA_DIR=$PROJECT_ROOT/data/unified_hc_ir_modalities_400bin_train_eval2048_20260725
MODALITY=${1:?usage: $0 nmr|ir}
# These accounts may be shared. Require the selected GPU to be genuinely idle
# unless the launcher explicitly opts out.
WAIT_FOR_GPU_IDLE=${WAIT_FOR_GPU_IDLE:-1}
GPU_IDLE_MEMORY_MIB=${GPU_IDLE_MEMORY_MIB:-1000}
GPU_IDLE_UTIL_PERCENT=${GPU_IDLE_UTIL_PERCENT:-10}
GPU_IDLE_CHECKS=${GPU_IDLE_CHECKS:-3}

echo "[$(date '+%F %T %Z')] queued modality=$MODALITY host=$(hostname)"
while [[ ! -s "$EVAL_DATA_DIR/READY.json" ]]; do
  echo "[$(date '+%F %T %Z')] waiting for staged data"
  sleep 60
done

if [[ "$WAIT_FOR_GPU_IDLE" == 1 ]]; then
  consecutive=0
  while (( consecutive < GPU_IDLE_CHECKS )); do
    read -r memory_used utilization < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | head -1 | tr -d ','
    )
    if (( memory_used <= GPU_IDLE_MEMORY_MIB && utilization <= GPU_IDLE_UTIL_PERCENT )); then
      consecutive=$((consecutive + 1))
    else
      consecutive=0
    fi
    echo "[$(date '+%F %T %Z')] gpu memory_mib=$memory_used util=$utilization idle_checks=$consecutive/$GPU_IDLE_CHECKS"
    if (( consecutive < GPU_IDLE_CHECKS )); then
      sleep 60
    fi
  done
fi

export DATA_DIR=$EVAL_DATA_DIR
export EPOCHS=${EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-16}
export ACC_BATCHES=${ACC_BATCHES:-2}
export LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-128}
export NUM_CPU=${NUM_CPU:-8}
exec "$PROJECT_ROOT/run_hc_ir_modalities_pretrain.sh" "$MODALITY"
