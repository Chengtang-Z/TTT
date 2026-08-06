#!/usr/bin/env python3
"""Add the legacy Dataset IR 400-bin tokenization to parquet chunks.

The input data remains on the HPC. This script is uploaded and run there; it
only creates a separate derived parquet directory and never changes the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def legacy_ir_tokens(values: object) -> str:
    """Match spectra_tokenizer.route_builders.ir_400bin_tokens for dense IR."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(args.input_dir.glob("*.parquet"))
    if not input_paths:
        raise SystemExit(f"no parquet files found in {args.input_dir}")

    total_rows = 0
    for index, input_path in enumerate(input_paths, start=1):
        output_path = args.output_dir / input_path.name
        if output_path.exists():
            existing = pq.read_table(output_path, columns=["ir_tokens"])
            if existing.num_rows == pq.ParquetFile(input_path).metadata.num_rows:
                total_rows += existing.num_rows
                print(f"[{index}/{len(input_paths)}] reuse {input_path.name} rows={existing.num_rows}", flush=True)
                continue
            output_path.unlink()

        table = pq.read_table(input_path)
        ir_tokens = [legacy_ir_tokens(values) for values in table["ir_spectra"].to_pylist()]
        table = table.append_column("ir_tokens", pa.array(ir_tokens, type=pa.string()))
        pq.write_table(table, output_path, compression="zstd", use_dictionary=True)
        total_rows += table.num_rows
        print(f"[{index}/{len(input_paths)}] wrote {input_path.name} rows={table.num_rows}", flush=True)

    marker = args.output_dir / "_IR400_COMPLETE"
    marker.write_text(f"rows={total_rows}\nfiles={len(input_paths)}\n", encoding="utf-8")
    print(f"complete rows={total_rows} files={len(input_paths)}", flush=True)


if __name__ == "__main__":
    main()
