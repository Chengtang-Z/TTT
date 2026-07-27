#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/ttt
PAYLOAD_ROOT=/hpc2hdd/home/aimslab/ChengtangZhan/Dataset/spectra_tokenizer_project/data/transformer/unified_payload_v1_latest_sim_h_c_ir_ms_random811_taware_inchikeyclean_original_ms_final_20260705
SOURCE_VIEW=$PAYLOAD_ROOT/opennmt/T_1H_13C_IR
FULL_DATA_DIR=$PROJECT_ROOT/data/unified_hc_ir_modalities_400bin_full_20260725
EVAL_DATA_DIR=$PROJECT_ROOT/data/unified_hc_ir_modalities_400bin_train_eval2048_20260725
PYTHON=$PROJECT_ROOT/.venv/bin/python

EXPECTED_TRAIN=5042465
EXPECTED_VALIDATION=633622
EXPECTED_TEST=635423

echo "[$(date '+%F %T %Z')] preparation start host=$(hostname) user=$(whoami)"

if [[ ! -s "$FULL_DATA_DIR/summary.json" ]]; then
  mkdir -p "$FULL_DATA_DIR"
  if find "$FULL_DATA_DIR" -mindepth 1 -print -quit | grep -q .; then
    echo "incomplete non-empty output directory: $FULL_DATA_DIR" >&2
    exit 1
  fi
  PYTHONNOUSERSITE=1 "$PYTHON" "$PROJECT_ROOT/prepare_unified_hc_ir_modalities.py" \
    --source-view "$SOURCE_VIEW" \
    --output-dir "$FULL_DATA_DIR" \
    --batch-rows 4096 \
    --expected-ir-length 400
fi

PYTHONNOUSERSITE=1 "$PYTHON" - "$FULL_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected = {"train": 5_042_465, "validation": 633_622, "test": 635_423}
summary = json.loads((root / "summary.json").read_text())
for split, count in expected.items():
    actual = pq.ParquetFile(root / f"{split}.parquet").metadata.num_rows
    assert actual == count, (split, actual, count)
    assert summary["splits"][split]["ir_lengths"] == {"400": count}
print("full paired dataset verified")
PY

mkdir -p "$EVAL_DATA_DIR"
for split in train validation; do
  if [[ ! -e "$EVAL_DATA_DIR/$split.parquet" ]]; then
    ln "$FULL_DATA_DIR/$split.parquet" "$EVAL_DATA_DIR/$split.parquet"
  fi
done

if [[ ! -s "$EVAL_DATA_DIR/test.parquet" ]]; then
  PYTHONNOUSERSITE=1 "$PYTHON" - "$FULL_DATA_DIR/test.parquet" "$EVAL_DATA_DIR/test.parquet" <<'PY'
import sys
import pyarrow as pa
import pyarrow.parquet as pq

source, output = sys.argv[1:]
batch = next(pq.ParquetFile(source).iter_batches(batch_size=2048))
table = pa.Table.from_batches([batch]).slice(0, 2048)
assert table.num_rows == 2048
pq.write_table(table, output, compression="zstd")
print("wrote deterministic 2048-row test subset")
PY
fi

PYTHONNOUSERSITE=1 "$PYTHON" - "$EVAL_DATA_DIR" <<'PY'
import json
import sys
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sys.argv[1])
expected = {"train": 5_042_465, "validation": 633_622, "test": 2_048}
for split, count in expected.items():
    actual = pq.ParquetFile(root / f"{split}.parquet").metadata.num_rows
    assert actual == count, (split, actual, count)
(root / "summary.json").write_text(
    json.dumps(
        {
            "dataset": "unified_hc_ir_modalities_400bin_train_eval2048_20260725",
            "split_policy": "Full train/validation; deterministic first 2048 test rows",
            "rows": expected,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
(root / "READY.json").write_text(json.dumps({"rows": expected}, indent=2) + "\n")
print("staged training/evaluation dataset verified")
PY

echo "[$(date '+%F %T %Z')] preparation finished"
