#!/usr/bin/env python3
"""Convert the unified final MS payload into the TTT-MS/MS parquet schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("formula", pa.string()),
        pa.field("smiles", pa.string()),
        pa.field("spectrum", pa.list_(pa.list_(pa.float64()))),
        pa.field("fingerprint", pa.list_(pa.int8())),
    ]
)
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "real", "sim"), default="auto")
    parser.add_argument("--seed", type=int, default=3245)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-validation", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def split_for_key(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "validation"
    return "test"


def is_tandem_ms(row: dict[str, Any], mode: str) -> bool:
    category = str(row.get("ms_category") or "").lower()
    submodality = str(row.get("submodality") or "").lower()
    inferred_level = row.get("ms_level_inferred")
    level = row.get("ms_level")

    if mode == "real" or category:
        return category == "msms"
    return inferred_level == 2 or level == 2 or "msms" in submodality


def parse_spectrum(raw_peaks: Any) -> list[list[float]]:
    peaks = json.loads(raw_peaks) if isinstance(raw_peaks, str) else raw_peaks
    spectrum: list[list[float]] = []
    for peak in peaks or []:
        if isinstance(peak, dict):
            mz = peak.get("mz")
            intensity = peak.get("relative_intensity")
            if intensity is None:
                intensity = peak.get("intensity")
        elif isinstance(peak, (list, tuple)) and len(peak) >= 2:
            mz, intensity = peak[:2]
        else:
            continue
        try:
            mz_value = float(mz)
            intensity_value = float(intensity)
        except (TypeError, ValueError):
            continue
        if math.isfinite(mz_value) and math.isfinite(intensity_value) and intensity_value > 0:
            spectrum.append([mz_value, intensity_value])
    spectrum.sort(key=lambda peak: peak[0])
    return spectrum


def iter_rows(files: Iterable[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        required = {"canonical_smiles", "inchikey", "peaks", "submodality", "ms_level"}
        missing = required - available
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        optional = {"ms_category", "ms_level_inferred"}
        columns = sorted(required | (optional & available))
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield from batch.to_pylist()


def main() -> None:
    args = parse_args()
    files = sorted(args.source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [path for path in args.output_dir.iterdir() if path.name != ".gitkeep"]
    if occupied:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    limits = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    mode = args.mode
    if mode == "auto":
        mode = "real" if "real" in str(args.source_dir).lower() else "sim"

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    buffers: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    writers: dict[str, pq.ParquetWriter] = {}
    counts: Counter[str] = Counter()
    scanned = 0

    def flush(split: str) -> None:
        rows = buffers[split]
        if not rows:
            return
        if split not in writers:
            writers[split] = pq.ParquetWriter(
                args.output_dir / f"{split}.parquet", OUTPUT_SCHEMA, compression="zstd"
            )
        writers[split].write_table(pa.Table.from_pylist(rows, schema=OUTPUT_SCHEMA))
        rows.clear()

    try:
        for row in iter_rows(files, args.batch_size):
            scanned += 1
            if not is_tandem_ms(row, mode):
                counts["skipped_non_msms"] += 1
                continue

            smiles = row.get("canonical_smiles")
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            if mol is None:
                counts["skipped_invalid_smiles"] += 1
                continue
            try:
                spectrum = parse_spectrum(row.get("peaks"))
            except (TypeError, ValueError, json.JSONDecodeError):
                counts["skipped_invalid_peaks"] += 1
                continue
            if not spectrum:
                counts["skipped_empty_spectrum"] += 1
                continue

            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            split_key = str(row.get("inchikey") or canonical_smiles)
            split = split_for_key(split_key, args.seed)
            if limits[split] and counts[split] >= limits[split]:
                continue

            fingerprint = [int(bit) for bit in generator.GetFingerprint(mol).ToBitString()]
            buffers[split].append(
                {
                    "formula": Descriptors.rdMolDescriptors.CalcMolFormula(mol),
                    "smiles": canonical_smiles,
                    "spectrum": spectrum,
                    "fingerprint": fingerprint,
                }
            )
            counts[split] += 1
            if len(buffers[split]) >= args.batch_size:
                flush(split)

            finite_limits = [limits[name] for name in SPLITS if limits[name] > 0]
            if len(finite_limits) == len(SPLITS) and all(
                counts[name] >= limits[name] for name in SPLITS
            ):
                break
            if scanned % 100_000 == 0:
                print(f"scanned={scanned} counts={dict(counts)}", flush=True)
    finally:
        for split in SPLITS:
            flush(split)
        for writer in writers.values():
            writer.close()

    missing_splits = [split for split in SPLITS if counts[split] == 0]
    if missing_splits:
        raise RuntimeError(f"No rows were written for splits: {missing_splits}")

    summary = {
        "source_dir": str(args.source_dir),
        "mode": mode,
        "seed": args.seed,
        "scanned_rows": scanned,
        "counts": dict(counts),
        "schema": [field.name for field in OUTPUT_SCHEMA],
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": 128},
        "split_policy": "SHA-256 of seed and InChIKey; 80/10/10",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
