#!/usr/bin/env python3
"""Add the numeric 400-bin view to derived parquet chunks in parallel."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def process_one(job: tuple[str, int, int]) -> tuple[int, str, int]:
    filename, index, total = job
    path = Path(filename)
    table = pq.read_table(path)
    if "ir_400" in table.column_names:
        return index, f"reuse {path.name}", table.num_rows
    rows = []
    for text in table["ir_tokens"].to_pylist():
        pieces = str(text).split()
        if len(pieces) != 401 or pieces[0] != "IR":
            raise ValueError(f"{path.name}: invalid IR token row length={len(pieces)}")
        rows.append([int(value) for value in pieces[1:]])
    updated = table.append_column("ir_400", pa.array(rows, type=pa.list_(pa.int16())))
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(updated, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, path)
    return index, f"added {path.name}", updated.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files found in {args.data_dir}")
    jobs = [(str(path), index, len(paths)) for index, path in enumerate(paths, start=1)]
    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, message, rows in executor.map(process_one, jobs):
            total_rows += rows
            print(f"[{index}/{len(paths)}] {message} rows={rows}", flush=True)
    (args.data_dir / "_IR400_NUMERIC_COMPLETE").write_text(
        f"rows={total_rows}\nfiles={len(paths)}\n", encoding="utf-8"
    )
    print(f"complete rows={total_rows} files={len(paths)}", flush=True)


if __name__ == "__main__":
    main()
