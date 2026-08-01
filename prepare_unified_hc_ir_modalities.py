#!/usr/bin/env python3
"""Convert the aligned OpenNMT H+13C+IR view to TTT multimodal parquet.

The TTT data loaders expect structured 1H multiplets and a numeric IR vector.
The source view keeps both modalities on the same line as the target SMILES;
this converter preserves that alignment and the source train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("molecular_formula", pa.string()),
        pa.field("h_nmr_text", pa.string()),
        pa.field("ir_spectra", pa.list_(pa.float32())),
        pa.field("smiles", pa.string()),
    ]
)
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Per-split limit for smoke tests; zero means all rows",
    )
    parser.add_argument(
        "--expected-ir-length",
        type=int,
        default=0,
        help="Require this many IR values in every row; zero disables the check",
    )
    return parser.parse_args()


def _marker_positions(tokens: list[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for marker in ("1HNMR", "13CNMR", "IR"):
        try:
            positions[marker] = tokens.index(marker)
        except ValueError as error:
            raise ValueError(f"missing marker {marker}") from error
    if not (
        positions["1HNMR"] < positions["13CNMR"] < positions["IR"]
    ):
        raise ValueError("source modality markers are out of order")
    return positions


def parse_formula(tokens: list[str]) -> str:
    formula = "".join(tokens)
    if not formula:
        raise ValueError("molecular formula is empty")
    return formula


def parse_h_nmr(tokens: list[str]) -> str:
    text = "1HNMR " + " ".join(tokens)
    if text == "1HNMR ":
        raise ValueError("1HNMR is empty")
    return text


def parse_source_line(
    line: str, expected_ir_length: int = 0
) -> tuple[str, str, list[float]]:
    tokens = line.strip().split()
    positions = _marker_positions(tokens)
    formula = parse_formula(tokens[: positions["1HNMR"]])
    h_nmr = parse_h_nmr(tokens[positions["1HNMR"] + 1 : positions["13CNMR"]])
    try:
        ir = [
            float(value)
            for value in tokens[positions["IR"] + 1 :]
        ]
    except ValueError as error:
        raise ValueError("IR contains a non-numeric value") from error
    if not ir:
        raise ValueError("IR is empty")
    if expected_ir_length and len(ir) != expected_ir_length:
        raise ValueError(
            f"IR length mismatch: expected {expected_ir_length}, got {len(ir)}"
        )
    return formula, h_nmr, ir


def flush_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
        rows.clear()


def convert_split(
    source_handle: TextIO,
    target_handle: TextIO,
    output_path: Path,
    split: str,
    batch_rows: int,
    max_molecules: int,
    expected_ir_length: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    ir_lengths: Counter[str] = Counter()
    started = time.monotonic()

    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd") as writer:
        for line_number, pair in enumerate(
            zip_longest(source_handle, target_handle), start=1
        ):
            source_line, target_line = pair
            if source_line is None or target_line is None:
                raise ValueError(f"{split} source/target mismatch at line {line_number}")
            if max_molecules and counts["molecules_written"] >= max_molecules:
                break
            try:
                formula, h_nmr_text, ir = parse_source_line(
                    source_line, expected_ir_length=expected_ir_length
                )
                smiles = "".join(target_line.split())
                if not smiles:
                    raise ValueError("target SMILES is empty")
            except Exception as error:
                raise ValueError(f"{split} line {line_number}: {error}") from error

            rows.append(
                {
                    "molecular_formula": formula,
                    "h_nmr_text": h_nmr_text,
                    "ir_spectra": ir,
                    "smiles": smiles,
                }
            )
            counts["molecules_written"] += 1
            counts["multiplets_written"] += h_nmr_text.count("|") + 1
            ir_lengths[str(len(ir))] += 1
            if len(rows) >= batch_rows:
                flush_rows(writer, rows)
            if counts["molecules_written"] % 100_000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"split={split} molecules={counts['molecules_written']} "
                    f"molecules_per_second={counts['molecules_written'] / elapsed:.1f}",
                    flush=True,
                )
        flush_rows(writer, rows)

    return {
        "counts": dict(counts),
        "ir_lengths": dict(ir_lengths),
        "elapsed_seconds": time.monotonic() - started,
    }


def main() -> None:
    args = parse_args()
    if args.batch_rows <= 0 or args.max_molecules < 0:
        raise ValueError("batch rows must be positive and max molecules non-negative")
    data_dir = args.source_view / "data"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    summaries: dict[str, Any] = {}
    for output_split, source_split in SPLITS.items():
        source_path = data_dir / f"src-{source_split}.txt"
        target_path = data_dir / f"tgt-{source_split}.txt"
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing {output_split} source/target files")
        print(f"starting split={output_split}", flush=True)
        with source_path.open("r", encoding="utf-8") as source_handle:
            with target_path.open("r", encoding="utf-8") as target_handle:
                summaries[output_split] = convert_split(
                    source_handle,
                    target_handle,
                    args.output_dir / f"{output_split}.parquet",
                    output_split,
                    args.batch_rows,
                    args.max_molecules,
                    args.expected_ir_length,
                )
        print(json.dumps({output_split: summaries[output_split]}, sort_keys=True))

    summary = {
        "source_view": str(args.source_view),
        "split_policy": "Preserve source train/val/test alignment",
        "max_molecules_per_split": args.max_molecules,
        "expected_ir_length": args.expected_ir_length,
        "splits": summaries,
        "schema": [field.name for field in OUTPUT_SCHEMA],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
