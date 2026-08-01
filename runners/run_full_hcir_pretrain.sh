#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
SOURCE_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/final_sim_hard_msd_h_c_ir_ms_e20_smiles_to_spectra_20260717
FULL_DATA_DIR=$PROJECT_ROOT/data/final_sim_hcir_e20_full_20260725
TRIAL_DATA_DIR=$PROJECT_ROOT/data/final_sim_hcir_e20_trial_20260725
EVAL_DATA_DIR=$PROJECT_ROOT/data/final_sim_hcir_e20_full_train_trialtest_20260725
RUN_DIR=$PROJECT_ROOT/runs/pretrain_hcir_e20_full_20260725
VENV_PYTHON=$PROJECT_ROOT/.venv/bin/python
PREPARE_SCRIPT=$PROJECT_ROOT/paper_replication/msms/data_preparation/prepare_hcir_simulated_msms.py

mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/logs/pipeline.log") 2>&1

echo "[$(date '+%F %T %Z')] pipeline start host=$(hostname) user=$(whoami)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

if [[ ! -f "$FULL_DATA_DIR/summary.json" ]]; then
  mkdir -p "$FULL_DATA_DIR"
  "$VENV_PYTHON" "$PREPARE_SCRIPT" \
    --source-root "$SOURCE_ROOT" \
    --output-dir "$FULL_DATA_DIR" \
    --seed 3245 \
    --max-train 557436 \
    --max-validation 104610 \
    --max-test 104087
else
  echo "[$(date '+%F %T %Z')] full data already exists; reusing it"
fi

test -s "$FULL_DATA_DIR/train.parquet"
test -s "$FULL_DATA_DIR/validation.parquet"
test -s "$FULL_DATA_DIR/test.parquet"
test -s "$FULL_DATA_DIR/summary.json"
cat "$FULL_DATA_DIR/summary.json"

mkdir -p "$EVAL_DATA_DIR"
for split_name in train validation; do
  if [[ -L "$EVAL_DATA_DIR/$split_name.parquet" ]]; then
    unlink "$EVAL_DATA_DIR/$split_name.parquet"
  fi
  if [[ ! -e "$EVAL_DATA_DIR/$split_name.parquet" ]]; then
    ln "$FULL_DATA_DIR/$split_name.parquet" "$EVAL_DATA_DIR/$split_name.parquet"
  fi
done
if [[ -L "$EVAL_DATA_DIR/test.parquet" ]]; then
  unlink "$EVAL_DATA_DIR/test.parquet"
fi
if [[ ! -e "$EVAL_DATA_DIR/test.parquet" ]]; then
  ln "$TRIAL_DATA_DIR/test.parquet" "$EVAL_DATA_DIR/test.parquet"
fi
if [[ -L "$EVAL_DATA_DIR/summary.json" ]]; then
  unlink "$EVAL_DATA_DIR/summary.json"
fi
if [[ ! -e "$EVAL_DATA_DIR/summary.json" ]]; then
  ln "$FULL_DATA_DIR/summary.json" "$EVAL_DATA_DIR/summary.json"
fi

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29537
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T %Z')] training start data=$EVAL_DATA_DIR"
"$VENV_PYTHON" -m analytical_fm.cli.training \
  working_dir="$RUN_DIR" \
  job_name=pt_full \
  data_path="$EVAL_DATA_DIR" \
  data=msms/text_fingerprint \
  model=custom_model_align \
  model.batch_size=16 \
  model.lr=1e-4 \
  model.n_beams=1 \
  model.rejection_sampling=False \
  trainer.epochs=1 \
  trainer.acc_batches=1 \
  trainer.limit_val_batches=128 \
  trainer.early_stopping_patience=100 \
  trainer.save_checkpoints=every_5_epochs \
  finetuning=False \
  eval_ckpt=last \
  splitting=given_splits \
  molecules=True \
  num_cpu=4

echo "[$(date '+%F %T %Z')] pipeline finished exit=$?"
