#!/usr/bin/env python3
"""Convert the unified 1H+13C+MS+IR OpenNMT view for TTT-MS/MS.

Each aligned molecule row contains E10Pos, E20Pos, and E40Pos spectra. The
paper's MS/MS pipeline treats collision energies as independent examples, so
this converter emits three rows per molecule while preserving the payload's
train/validation/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, TextIO

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


ENERGY_MARKERS = ("E10Pos", "E20Pos", "E40Pos")
SOURCE_MARKERS = ("1HNMR", "13CNMR", *ENERGY_MARKERS, "IR")
SPLITS = {"train": "train", "validation": "val", "test": "test"}
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("formula", pa.string()),
        pa.field("smiles", pa.string()),
        pa.field("spectrum", pa.list_(pa.list_(pa.float64()))),
        pa.field("fingerprint", pa.list_(pa.int8())),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-view",
        type=Path,
        required=True,
        help="Path ending in opennmt/T_1H_13C_MS_POS_IR",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-molecules", type=int, default=4096)
    parser.add_argument(
        "--max-molecules",
        type=int,
        default=0,
        help="Per-split molecule limit for smoke tests; zero means all rows",
    )
    parser.add_argument(
        "--expected-train-molecules", type=int, default=0
    )
    parser.add_argument(
        "--expected-validation-molecules", type=int, default=0
    )
    parser.add_argument(
        "--expected-test-molecules", type=int, default=0
    )
    return parser.parse_args()


def parse_spectrum(tokens: list[str], marker: str) -> tuple[list[list[float]], int]:
    if not tokens or len(tokens) % 2:
        raise ValueError(f"{marker} does not contain m/z-intensity pairs")

    spectrum: list[list[float]] = []
    zero_intensity_peaks = 0
    previous_mz = -math.inf
    for index in range(0, len(tokens), 2):
        mz = float(tokens[index])
        intensity = float(tokens[index + 1])
        if not math.isfinite(mz) or not math.isfinite(intensity):
            raise ValueError(f"{marker} contains a non-finite peak")
        if mz <= 0 or intensity < 0:
            raise ValueError(f"{marker} contains an invalid peak: {mz}, {intensity}")
        if mz < previous_mz:
            raise ValueError(f"{marker} m/z values are not sorted")
        previous_mz = mz
        if intensity == 0:
            zero_intensity_peaks += 1
            continue
        spectrum.append([mz, intensity])

    if not spectrum:
        raise ValueError(f"{marker} has no positive-intensity peaks")
    return spectrum, zero_intensity_peaks


def parse_source_line(line: str) -> tuple[str, dict[str, list[list[float]]], int]:
    tokens = line.split()
    positions: dict[str, int] = {}
    for marker in SOURCE_MARKERS:
        try:
            positions[marker] = tokens.index(marker)
        except ValueError as error:
            raise ValueError(f"missing marker {marker}") from error

    ordered_positions = [positions[marker] for marker in SOURCE_MARKERS]
    if ordered_positions != sorted(ordered_positions):
        raise ValueError("source modality markers are out of order")

    formula = "".join(tokens[: positions["1HNMR"]])
    if not formula:
        raise ValueError("molecular formula is empty")

    spectra: dict[str, list[list[float]]] = {}
    zero_intensity_peaks = 0
    next_markers = {"E10Pos": "E20Pos", "E20Pos": "E40Pos", "E40Pos": "IR"}
    for marker in ENERGY_MARKERS:
        peak_tokens = tokens[positions[marker] + 1 : positions[next_markers[marker]]]
        spectrum, removed = parse_spectrum(peak_tokens, marker)
        spectra[marker] = spectrum
        zero_intensity_peaks += removed
    return formula, spectra, zero_intensity_peaks


def parse_target_line(
    line: str, fingerprint_generator: Any
) -> tuple[str, list[int], str]:
    source_smiles = "".join(line.split())
    if not source_smiles:
        raise ValueError("target SMILES is empty")
    molecule = Chem.MolFromSmiles(source_smiles)
    if molecule is None:
        raise ValueError(f"invalid target SMILES: {source_smiles}")
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
    fingerprint = [
        int(bit)
        for bit in fingerprint_generator.GetFingerprint(molecule).ToBitString()
    ]
    calculated_formula = rdMolDescriptors.CalcMolFormula(molecule)
    return canonical_smiles, fingerprint, calculated_formula


def flush_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    writer.write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
    rows.clear()


def convert_split(
    source_handle: TextIO,
    target_handle: TextIO,
    output_path: Path,
    split: str,
    batch_molecules: int,
    max_molecules: int,
    fingerprint_generator: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    formula_mismatch_examples: list[dict[str, str]] = []
    started = time.monotonic()

    with pq.ParquetWriter(output_path, OUTPUT_SCHEMA, compression="zstd") as writer:
        for line_number, pair in enumerate(
            zip_longest(source_handle, target_handle), start=1
        ):
            source_line, target_line = pair
            if source_line is None or target_line is None:
                raise ValueError(
                    f"{split} source/target line count mismatch at line {line_number}"
                )
            if max_molecules and counts["molecules_written"] >= max_molecules:
                break

            try:
                formula, spectra, removed = parse_source_line(source_line)
                smiles, fingerprint, calculated_formula = parse_target_line(
                    target_line, fingerprint_generator
                )
            except Exception as error:
                raise ValueError(f"{split} line {line_number}: {error}") from error

            counts["molecules_scanned"] += 1
            counts["molecules_written"] += 1
            counts["zero_intensity_peaks_removed"] += removed
            if formula != calculated_formula:
                counts["formula_mismatches"] += 1
                if len(formula_mismatch_examples) < 10:
                    formula_mismatch_examples.append(
                        {
                            "payload": formula,
                            "rdkit": calculated_formula,
                            "smiles": smiles,
                        }
                    )

            for energy in ENERGY_MARKERS:
                rows.append(
                    {
                        "formula": formula,
                        "smiles": smiles,
                        "spectrum": spectra[energy],
                        "fingerprint": fingerprint,
                    }
                )
                counts[energy] += 1
                counts["spectra_written"] += 1

            if counts["molecules_written"] % batch_molecules == 0:
                flush_rows(writer, rows)
            if counts["molecules_written"] % 100_000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"split={split} molecules={counts['molecules_written']} "
                    f"spectra={counts['spectra_written']} "
                    f"molecules_per_second={counts['molecules_written'] / elapsed:.1f}",
                    flush=True,
                )
        flush_rows(writer, rows)

    elapsed = time.monotonic() - started
    return {
        "counts": dict(counts),
        "elapsed_seconds": elapsed,
        "molecules_per_second": counts["molecules_written"] / max(elapsed, 1e-9),
        "formula_mismatch_examples": formula_mismatch_examples,
    }


def main() -> None:
    args = parse_args()
    if args.batch_molecules <= 0 or args.max_molecules < 0:
        raise ValueError("batch and molecule limits must be non-negative")

    data_dir = args.source_view / "data"
    paths: dict[str, tuple[Path, Path]] = {}
    for output_split, payload_split in SPLITS.items():
        source_path = data_dir / f"src-{payload_split}.txt"
        target_path = data_dir / f"tgt-{payload_split}.txt"
        if not source_path.is_file() or not target_path.is_file():
            raise FileNotFoundError(f"missing aligned files for {output_split}")
        paths[output_split] = (source_path, target_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    RDLogger.DisableLog("rdApp.*")
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=128
    )
    expected = {
        "train": args.expected_train_molecules,
        "validation": args.expected_validation_molecules,
        "test": args.expected_test_molecules,
    }
    split_summaries: dict[str, Any] = {}
    started = time.monotonic()

    for split, (source_path, target_path) in paths.items():
        print(
            f"starting split={split} source={source_path} target={target_path}",
            flush=True,
        )
        with source_path.open("r", encoding="utf-8") as source_handle:
            with target_path.open("r", encoding="utf-8") as target_handle:
                split_summary = convert_split(
                    source_handle=source_handle,
                    target_handle=target_handle,
                    output_path=args.output_dir / f"{split}.parquet",
                    split=split,
                    batch_molecules=args.batch_molecules,
                    max_molecules=args.max_molecules,
                    fingerprint_generator=fingerprint_generator,
                )
        written = split_summary["counts"]["molecules_written"]
        if not args.max_molecules and expected[split] and written != expected[split]:
            raise ValueError(
                f"{split} count mismatch: expected {expected[split]}, wrote {written}"
            )
        split_summaries[split] = split_summary
        print(json.dumps({split: split_summary}, sort_keys=True), flush=True)

    total_molecules = sum(
        summary["counts"]["molecules_written"] for summary in split_summaries.values()
    )
    total_spectra = sum(
        summary["counts"]["spectra_written"] for summary in split_summaries.values()
    )
    source_manifest = args.source_view.parent.parent / "manifest.json"
    summary = {
        "source_view": str(args.source_view),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "source_contract": (
            "Each aligned payload row contains simulated 1H NMR, 13C NMR, "
            "E10Pos/E20Pos/E40Pos MS/MS, IR, and tokenized target SMILES"
        ),
        "split_policy": "Preserve canonical payload train/val/test split exactly",
        "sample_policy": "Emit E10Pos, E20Pos, and E40Pos as independent MS/MS examples",
        "energies": list(ENERGY_MARKERS),
        "max_molecules_per_split": args.max_molecules,
        "expected_molecules": expected,
        "total_molecules": total_molecules,
        "total_spectra": total_spectra,
        "splits": split_summaries,
        "schema": [field.name for field in OUTPUT_SCHEMA],
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": 128},
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
