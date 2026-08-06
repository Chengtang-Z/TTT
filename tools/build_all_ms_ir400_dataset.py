#!/usr/bin/env python3
"""Build a compact parquet dataset containing all six MS/MS conditions."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


BASE_COLUMNS = [
    "smiles",
    "molecular_formula",
    "h_nmr_peaks",
    "c_nmr_peaks",
    "ir_400",
    "fingerprint",
]
MSMS_COLUMNS = [
    "msms_positive_10ev",
    "msms_positive_20ev",
    "msms_positive_40ev",
    "msms_negative_10ev",
    "msms_negative_20ev",
    "msms_negative_40ev",
]
OUTPUT_COLUMNS = BASE_COLUMNS + MSMS_COLUMNS


def process_file(job: tuple[Path, Path, Path]) -> tuple[str, int, str]:
    base_path, raw_path, output_path = job
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)

    expected_rows = pq.ParquetFile(base_path).metadata.num_rows
    if output_path.exists():
        existing = pq.ParquetFile(output_path)
        if (
            existing.metadata.num_rows == expected_rows
            and existing.schema_arrow.names == OUTPUT_COLUMNS
        ):
            return output_path.name, expected_rows, "reuse"
        raise RuntimeError(f"refusing to overwrite invalid output: {output_path}")

    base = pq.read_table(base_path, columns=BASE_COLUMNS)
    raw = pq.read_table(raw_path, columns=["smiles", *MSMS_COLUMNS])
    if base.num_rows != raw.num_rows or not base["smiles"].equals(raw["smiles"]):
        raise RuntimeError(f"row alignment mismatch: {base_path.name}")

    output = base
    for column in MSMS_COLUMNS:
        output = output.append_column(column, raw[column])

    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(output, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, output_path)
    return output_path.name, output.num_rows, "write"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    base_paths = sorted(args.base_dir.glob("aligned_chunk_*.parquet"))
    if len(base_paths) != 245:
        raise SystemExit(f"expected 245 base chunks, found {len(base_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (base_path, args.raw_dir / base_path.name, args.output_dir / base_path.name)
        for base_path in base_paths
    ]

    total_rows = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, job) for job in jobs]
        for future in as_completed(futures):
            name, rows, action = future.result()
            total_rows += rows
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(
                    f"[{completed}/{len(jobs)}] {action} {name} total_rows={total_rows}",
                    flush=True,
                )

    marker = args.output_dir / "_ALL_MS_COMPLETE"
    marker.write_text(
        f"rows={total_rows}\nfiles={len(jobs)}\ncolumns={','.join(OUTPUT_COLUMNS)}\n",
        encoding="utf-8",
    )
    print(f"complete rows={total_rows} files={len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
