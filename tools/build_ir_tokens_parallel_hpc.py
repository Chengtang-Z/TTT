#!/usr/bin/env python3
"""Parallel HPC-only builder for the legacy 400-bin IR token view."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def legacy_ir_tokens(values: object) -> str:
    y = np.asarray(values, dtype=float)
    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("IR spectrum is empty or contains non-finite values")
    x = np.linspace(400.0, 4000.0, num=y.size)
    target_x = np.linspace(400.0, 4000.0, num=400)
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    if unique_x.size != x_sorted.size:
        summed = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(summed, inverse, y_sorted)
        np.add.at(counts, inverse, 1.0)
        x_sorted = unique_x
        y_sorted = summed / np.maximum(counts, 1.0)
    y_interp = np.interp(target_x, x_sorted, y_sorted)
    y_interp = y_interp + abs(float(np.nanmin(y_interp)))
    max_y = float(np.nanmax(y_interp))
    if max_y <= 0:
        raise ValueError("IR spectrum has zero intensity after interpolation")
    values_0_100 = np.clip(np.rint(y_interp / max_y * 100.0), 0, 100).astype(int)
    return "IR " + " ".join(str(int(value)) for value in values_0_100)


def process_one(args: tuple[str, str, int, int]) -> tuple[int, str, int]:
    input_name, output_name, index, total = args
    input_path = Path(input_name)
    output_path = Path(output_name)
    expected_rows = pq.ParquetFile(input_path).metadata.num_rows
    if output_path.exists():
        existing = pq.read_table(output_path, columns=["ir_tokens"])
        if existing.num_rows == expected_rows:
            return index, f"reuse {input_path.name}", expected_rows
        output_path.unlink()

    table = pq.read_table(input_path)
    ir_tokens = [legacy_ir_tokens(values) for values in table["ir_spectra"].to_pylist()]
    table = table.append_column("ir_tokens", pa.array(ir_tokens, type=pa.string()))
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, output_path)
    return index, f"wrote {input_path.name}", table.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files found in {args.input_dir}")

    jobs = [
        (str(path), str(args.output_dir / path.name), index, len(paths))
        for index, path in enumerate(paths, start=1)
    ]
    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, message, rows in executor.map(process_one, jobs):
            total_rows += rows
            print(f"[{index}/{len(paths)}] {message} rows={rows}", flush=True)

    (args.output_dir / "_IR400_COMPLETE").write_text(
        f"rows={total_rows}\nfiles={len(paths)}\n", encoding="utf-8"
    )
    print(f"complete rows={total_rows} files={len(paths)}", flush=True)


if __name__ == "__main__":
    main()
