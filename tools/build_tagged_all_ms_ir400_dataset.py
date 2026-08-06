#!/usr/bin/env python3
"""Build the six-condition MS/MS sequence used by multimodal pretraining."""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq


BASE_COLUMNS = [
    "smiles",
    "molecular_formula",
    "h_nmr_peaks",
    "c_nmr_peaks",
    "ir_400",
    "fingerprint",
]
MSMS_CONDITIONS = [
    ("E10Pos", "msms_positive_10ev"),
    ("E20Pos", "msms_positive_20ev"),
    ("E40Pos", "msms_positive_40ev"),
    ("E10Neg", "msms_negative_10ev"),
    ("E20Neg", "msms_negative_20ev"),
    ("E40Neg", "msms_negative_40ev"),
]
INPUT_COLUMNS = BASE_COLUMNS + [column for _, column in MSMS_CONDITIONS]
OUTPUT_COLUMNS = BASE_COLUMNS + ["spectrum"]


def format_tagged_msms(
    condition_spectra: Sequence[object],
    min_intensity: float = 1.0,
) -> str:
    """Serialize six MS/MS blocks as tagged m/z-intensity token pairs."""
    if len(condition_spectra) != len(MSMS_CONDITIONS):
        raise ValueError(
            f"expected {len(MSMS_CONDITIONS)} MS/MS conditions, "
            f"found {len(condition_spectra)}"
        )

    tokens: list[str] = []
    for (tag, _), peaks in zip(MSMS_CONDITIONS, condition_spectra):
        peak_tokens: list[str] = []
        for peak in peaks or []:
            if peak is None or len(peak) < 2:
                raise ValueError(f"{tag} contains an invalid peak: {peak!r}")
            mz = float(peak[0])
            intensity = float(peak[1])
            if not math.isfinite(mz) or not math.isfinite(intensity):
                raise ValueError(f"{tag} contains a non-finite peak: {peak!r}")
            if mz <= 0 or intensity < 0:
                raise ValueError(f"{tag} contains an out-of-range peak: {peak!r}")
            if intensity < min_intensity:
                continue
            peak_tokens.extend((f"{mz:.1f}", f"{intensity:.1f}"))

        if not peak_tokens:
            raise ValueError(f"{tag} has no peaks at intensity >= {min_intensity}")
        tokens.extend((tag, *peak_tokens))

    return " ".join(tokens)


def process_file(
    job: tuple[Path, Path, float],
) -> tuple[str, int, str, int, int]:
    input_path, output_path, min_intensity = job
    source = pq.ParquetFile(input_path)
    missing = [column for column in INPUT_COLUMNS if column not in source.schema_arrow.names]
    if missing:
        raise RuntimeError(f"{input_path.name} missing columns: {missing}")

    if output_path.exists():
        existing = pq.ParquetFile(output_path)
        if (
            existing.metadata.num_rows == source.metadata.num_rows
            and existing.schema_arrow.names == OUTPUT_COLUMNS
        ):
            table = pq.read_table(output_path, columns=["spectrum"])
            lengths = [len(value.split()) for value in table["spectrum"].to_pylist()]
            return (
                output_path.name,
                existing.metadata.num_rows,
                "reuse",
                max(lengths),
                sum(length > 924 for length in lengths),
            )
        raise RuntimeError(f"refusing to overwrite invalid output: {output_path}")

    table = pq.read_table(input_path, columns=INPUT_COLUMNS)
    condition_columns = [
        table[column].to_pylist() for _, column in MSMS_CONDITIONS
    ]
    spectra = [
        format_tagged_msms(
            [condition[row_index] for condition in condition_columns],
            min_intensity=min_intensity,
        )
        for row_index in range(table.num_rows)
    ]
    lengths = [len(spectrum.split()) for spectrum in spectra]

    output = table.select(BASE_COLUMNS).append_column(
        "spectrum",
        pa.array(spectra, type=pa.string()),
    )
    temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
    pq.write_table(output, temporary, compression="zstd", use_dictionary=True)
    os.replace(temporary, output_path)
    return (
        output_path.name,
        output.num_rows,
        "write",
        max(lengths),
        sum(length > 924 for length in lengths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-intensity", type=float, default=1.0)
    args = parser.parse_args()

    input_paths = sorted(args.input_dir.glob("aligned_chunk_*.parquet"))
    if len(input_paths) != 245:
        raise SystemExit(f"expected 245 input chunks, found {len(input_paths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (input_path, args.output_dir / input_path.name, args.min_intensity)
        for input_path in input_paths
    ]

    total_rows = 0
    max_tokens = 0
    over_924 = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, job) for job in jobs]
        for future in as_completed(futures):
            name, rows, action, file_max_tokens, file_over_924 = future.result()
            total_rows += rows
            max_tokens = max(max_tokens, file_max_tokens)
            over_924 += file_over_924
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(
                    f"[{completed}/{len(jobs)}] {action} {name} "
                    f"rows={total_rows} max_tokens={max_tokens} over_924={over_924}",
                    flush=True,
                )

    marker = args.output_dir / "_TAGGED_ALL_MS_COMPLETE"
    marker.write_text(
        f"rows={total_rows}\n"
        f"files={len(jobs)}\n"
        f"conditions={','.join(tag for tag, _ in MSMS_CONDITIONS)}\n"
        f"min_intensity={args.min_intensity}\n"
        f"max_tokens={max_tokens}\n"
        f"over_924={over_924}\n",
        encoding="utf-8",
    )
    print(
        f"complete rows={total_rows} files={len(jobs)} "
        f"max_tokens={max_tokens} over_924={over_924}",
        flush=True,
    )


if __name__ == "__main__":
    main()
