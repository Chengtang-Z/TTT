#!/usr/bin/env bash
set -euo pipefail

# Run one independent modality per A40. The two processes do not use
# cross-node DDP because gpu4 and ai4s expose separate containers/hosts.

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
PAYLOAD_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/unified_payload_v1_latest_sim_h_c_ir_ms_random811_taware_inchikeyclean_original_ms_final_20260705
SOURCE_VIEW=$PAYLOAD_ROOT/opennmt/T_1H_13C_IR
DATA_DIR=${DATA_DIR:-$PROJECT_ROOT/data/unified_hc_ir_modalities_400bin_20260725}
MODALITY=${1:?usage: $0 nmr|ir}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
ACC_BATCHES=${ACC_BATCHES:-2}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-128}
NUM_CPU=${NUM_CPU:-8}
LR=${LR:-1e-4}
RESUME_CKPT=${RESUME_CKPT:-}
VENV_PYTHON=$PROJECT_ROOT/.venv/bin/python

case "$MODALITY" in
  nmr)
    DEFAULT_RUN_NAME=pretrain_1hnmr_20260725
    DATA_CONFIG=nmr/hnmr_text
    PORT=29561
    MODEL_OVERRIDES=()
    ;;
  ir)
    DEFAULT_RUN_NAME=pretrain_ir_400bin_20260725
    DATA_CONFIG=ir/patches
    PORT=29562
    MODEL_OVERRIDES=(data.IR.preprocessor_arguments.patch_size=25)
    ;;
  *)
    echo "unknown modality: $MODALITY" >&2
    exit 2
    ;;
esac

RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
RUN_DIR=$PROJECT_ROOT/runs/$RUN_NAME
mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/logs/pipeline.log") 2>&1

echo "[$(date '+%F %T %Z')] start modality=$MODALITY host=$(hostname) user=$(whoami)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$PORT
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T %Z')] training start data=$DATA_DIR epochs=$EPOCHS"
CMD=(
  "$VENV_PYTHON" -m analytical_fm.cli.training
  working_dir="$RUN_DIR"
  job_name=pt
  data_path="$DATA_DIR"
  "data=$DATA_CONFIG"
  model=custom_model
  "model.batch_size=$BATCH_SIZE"
  "model.lr=$LR"
  model.n_beams=1
  model.rejection_sampling=False
  "trainer.epochs=$EPOCHS"
  "trainer.acc_batches=$ACC_BATCHES"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  trainer.early_stopping_patience=100
  trainer.save_checkpoints=all
  finetuning=False
  eval_ckpt=last
  splitting=given_splits
  molecules=True
  "num_cpu=$NUM_CPU"
)
CMD+=("${MODEL_OVERRIDES[@]}")
if [[ -n "$RESUME_CKPT" ]]; then
  CMD+=(model.model_checkpoint_path="$RESUME_CKPT")
fi
"${CMD[@]}"
echo "[$(date '+%F %T %Z')] finished modality=$MODALITY"
