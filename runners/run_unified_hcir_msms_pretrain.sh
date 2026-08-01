#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
PAYLOAD_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/unified_payload_v1_latest_sim_h_c_ir_ms_random811_taware_inchikeyclean_original_ms_final_20260705
SOURCE_VIEW=$PAYLOAD_ROOT/opennmt/T_1H_13C_MS_POS_IR
DATA_DIR=$PROJECT_ROOT/data/unified_hcir_ms_3energy_full_20260725
EVAL_DATA_DIR=$PROJECT_ROOT/data/unified_hcir_ms_3energy_train_eval2048_20260725
RUN_DIR=${RUN_DIR:-$PROJECT_ROOT/runs/pretrain_unified_hcir_ms_3energy_5ep_20260725}
VENV_PYTHON=$PROJECT_ROOT/.venv/bin/python
PREPARE_SCRIPT=$PROJECT_ROOT/paper_replication/msms/data_preparation/prepare_unified_hcir_msms.py
BATCH_SIZE=${BATCH_SIZE:-32}
ACC_BATCHES=${ACC_BATCHES:-2}
NUM_CPU=${NUM_CPU:-8}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-4}
RESUME_CKPT=${RESUME_CKPT:-}
PREPROCESSOR_PATH=${PREPROCESSOR_PATH:-}

EXPECTED_TRAIN_MOLECULES=4981273
EXPECTED_VALIDATION_MOLECULES=625640
EXPECTED_TEST_MOLECULES=627448
EXPECTED_TRAIN_SPECTRA=14943819
EXPECTED_VALIDATION_SPECTRA=1876920
EXPECTED_TEST_SPECTRA=1882344

mkdir -p "$RUN_DIR/logs"
exec > >(tee -a "$RUN_DIR/logs/pipeline.log") 2>&1

echo "[$(date '+%F %T %Z')] pipeline start host=$(hostname) user=$(whoami)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

if [[ ! -f "$DATA_DIR/summary.json" ]]; then
  mkdir -p "$DATA_DIR"
  "$VENV_PYTHON" "$PREPARE_SCRIPT" \
    --source-view "$SOURCE_VIEW" \
    --output-dir "$DATA_DIR" \
    --batch-molecules 4096 \
    --expected-train-molecules "$EXPECTED_TRAIN_MOLECULES" \
    --expected-validation-molecules "$EXPECTED_VALIDATION_MOLECULES" \
    --expected-test-molecules "$EXPECTED_TEST_MOLECULES"
else
  echo "[$(date '+%F %T %Z')] full converted data already exists; verifying it"
fi

for split_name in train validation test; do
  test -s "$DATA_DIR/$split_name.parquet"
done
test -s "$DATA_DIR/summary.json"

"$VENV_PYTHON" - "$DATA_DIR/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "train": (4_981_273, 14_943_819),
    "validation": (625_640, 1_876_920),
    "test": (627_448, 1_882_344),
}
for split, (molecules, spectra) in expected.items():
    counts = summary["splits"][split]["counts"]
    assert counts["molecules_written"] == molecules, (split, counts)
    assert counts["spectra_written"] == spectra, (split, counts)
    assert counts.get("formula_mismatches", 0) == 0, (split, counts)
print("full conversion summary verified")
PY

mkdir -p "$EVAL_DATA_DIR"
for split_name in train validation; do
  if [[ ! -e "$EVAL_DATA_DIR/$split_name.parquet" ]]; then
    ln "$DATA_DIR/$split_name.parquet" "$EVAL_DATA_DIR/$split_name.parquet"
  fi
done

if [[ ! -s "$EVAL_DATA_DIR/test.parquet" ]]; then
  "$VENV_PYTHON" - "$DATA_DIR/test.parquet" "$EVAL_DATA_DIR/test.parquet" <<'PY'
import sys

import pyarrow as pa
import pyarrow.parquet as pq

source, output = sys.argv[1:]
batch = next(pq.ParquetFile(source).iter_batches(batch_size=2048))
table = pa.Table.from_batches([batch]).slice(0, 2048)
assert table.num_rows == 2048
pq.write_table(table, output, compression="zstd")
print(f"wrote deterministic evaluation subset: {table.num_rows} rows")
PY
fi

"$VENV_PYTHON" - "$EVAL_DATA_DIR" <<'PY'
import sys

import pyarrow.parquet as pq

root = sys.argv[1]
expected = {"train": 14_943_819, "validation": 1_876_920, "test": 2_048}
for split, rows in expected.items():
    actual = pq.ParquetFile(f"{root}/{split}.parquet").metadata.num_rows
    assert actual == rows, (split, actual, rows)
print("training/evaluation parquet row counts verified")
PY

export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29547
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0

cd "$PROJECT_ROOT"
echo "[$(date '+%F %T %Z')] training start data=$EVAL_DATA_DIR"
TRAIN_ARGS=(
  working_dir="$RUN_DIR" \
  job_name=pt_full \
  data_path="$EVAL_DATA_DIR" \
  data=msms/text_fingerprint \
  model=custom_model_align \
  model.batch_size="$BATCH_SIZE" \
  model.lr="$LR" \
  model.n_beams=1 \
  model.rejection_sampling=False \
  trainer.epochs="$EPOCHS" \
  trainer.acc_batches="$ACC_BATCHES" \
  trainer.limit_val_batches=128 \
  trainer.early_stopping_patience=100 \
  trainer.save_checkpoints=all \
  finetuning=False \
  eval_ckpt=last \
  splitting=given_splits \
  molecules=True \
  num_cpu="$NUM_CPU"
)
if [[ -n "$RESUME_CKPT" ]]; then
  test -s "$RESUME_CKPT"
  TRAIN_ARGS+=(model.model_checkpoint_path="$RESUME_CKPT")
fi
if [[ -n "$PREPROCESSOR_PATH" ]]; then
  test -s "$PREPROCESSOR_PATH"
  TRAIN_ARGS+=(preprocessor_path="$PREPROCESSOR_PATH")
fi
"$VENV_PYTHON" -m analytical_fm.cli.training "${TRAIN_ARGS[@]}"

FINAL_CHECKPOINT=$RUN_DIR/pt_full/version_0/checkpoints/last.ckpt
test -s "$FINAL_CHECKPOINT"
"$VENV_PYTHON" - "$FINAL_CHECKPOINT" "$((EPOCHS - 1))" <<'PY'
import sys

import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu")
expected_epoch = int(sys.argv[2])
assert checkpoint["epoch"] == expected_epoch, checkpoint.get("epoch")
print(
    f"verified final checkpoint epoch={checkpoint['epoch']} "
    f"global_step={checkpoint['global_step']}"
)
PY
echo "[$(date '+%F %T %Z')] pipeline finished successfully"
