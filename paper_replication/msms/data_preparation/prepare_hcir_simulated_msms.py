#!/usr/bin/env python3
"""Build TTT-MS/MS parquets from the audited 1H+13C+IR+MS simulated cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("formula", pa.string()),
        pa.field("smiles", pa.string()),
        pa.field("spectrum", pa.list_(pa.list_(pa.float64()))),
        pa.field("fingerprint", pa.list_(pa.int8())),
    ]
)
SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3245)
    parser.add_argument("--max-train", type=int, default=2048)
    parser.add_argument("--max-validation", type=int, default=256)
    parser.add_argument("--max-test", type=int, default=256)
    return parser.parse_args()


def selection_score(seed: int, split: str, inchikey: str) -> int:
    payload = f"{seed}:{split}:{inchikey}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def parse_ms_line(line: str) -> tuple[list[list[float]], int]:
    tokens = line.strip().split()
    if not tokens or tokens[0] != "MS":
        raise ValueError("Expected an MS-prefixed spectrum")
    values = tokens[1:]
    if not values or len(values) % 2:
        raise ValueError("Expected one or more m/z-intensity pairs")
    spectrum: list[list[float]] = []
    zero_intensity_peaks = 0
    for i in range(0, len(values), 2):
        mz = float(values[i])
        intensity = float(values[i + 1])
        if mz <= 0:
            raise ValueError("Spectrum contains non-positive m/z")
        if intensity < 0:
            raise ValueError("Spectrum contains negative intensity")
        if intensity == 0:
            zero_intensity_peaks += 1
            continue
        spectrum.append([mz, intensity])
    if not spectrum:
        raise ValueError("Spectrum has no positive-intensity peaks")
    return spectrum, zero_intensity_peaks


def select_split(
    source_root: Path, source_split: str, output_split: str, limit: int, seed: int
) -> tuple[list[dict[str, str]], int]:
    provenance_path = source_root / "provenance" / f"{source_split}.tsv"
    spectra_path = source_root / "T_MS" / "data" / f"tgt-{source_split}.txt"
    heap: list[tuple[int, int, dict[str, str]]] = []
    scanned = 0

    with provenance_path.open("r", encoding="utf-8", newline="") as provenance_handle:
        provenance = csv.DictReader(provenance_handle, delimiter="\t")
        with spectra_path.open("r", encoding="utf-8") as spectra_handle:
            for index, pair in enumerate(zip_longest(provenance, spectra_handle)):
                row, spectrum_line = pair
                if row is None or spectrum_line is None:
                    raise ValueError(
                        f"Line count mismatch between {provenance_path} and {spectra_path}"
                    )
                scanned += 1
                inchikey = row["inchikey"]
                record = {
                    "inchikey": inchikey,
                    "canonical_smiles": row["canonical_smiles"],
                    "spectrum_line": spectrum_line,
                }
                score = selection_score(seed, output_split, inchikey)
                item = (-score, index, record)
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, item)

    selected = [item[2] for item in sorted(heap, key=lambda item: -item[0])]
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} {output_split} rows but selected {len(selected)}")
    return selected, scanned


def convert_records(records: list[dict[str, str]]) -> tuple[list[dict[str, Any]], int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    converted: list[dict[str, Any]] = []
    zero_intensity_peaks = 0
    for record in records:
        mol = Chem.MolFromSmiles(record["canonical_smiles"])
        if mol is None:
            raise ValueError(f"Invalid audited SMILES: {record['canonical_smiles']}")
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        fingerprint = [int(bit) for bit in generator.GetFingerprint(mol).ToBitString()]
        spectrum, filtered = parse_ms_line(record["spectrum_line"])
        zero_intensity_peaks += filtered
        converted.append(
            {
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "smiles": canonical_smiles,
                "spectrum": spectrum,
                "fingerprint": fingerprint,
            }
        )
    return converted, zero_intensity_peaks


def main() -> None:
    args = parse_args()
    limits = {
        "train": args.max_train,
        "validation": args.max_validation,
        "test": args.max_test,
    }
    if any(limit <= 0 for limit in limits.values()):
        raise ValueError("All split limits must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    selected_by_split: dict[str, list[dict[str, str]]] = {}
    scanned: dict[str, int] = {}
    for output_split, source_split in SPLIT_FILES.items():
        selected, source_count = select_split(
            args.source_root, source_split, output_split, limits[output_split], args.seed
        )
        selected_by_split[output_split] = selected
        scanned[output_split] = source_count

    identity_sets = {
        split: {record["inchikey"] for record in records}
        for split, records in selected_by_split.items()
    }
    for left_index, left in enumerate(SPLIT_FILES):
        for right in list(SPLIT_FILES)[left_index + 1 :]:
            overlap = identity_sets[left] & identity_sets[right]
            if overlap:
                raise ValueError(f"InChIKey leakage between {left} and {right}: {len(overlap)}")

    selection_hashes: dict[str, str] = {}
    zero_intensity_peaks_removed = 0
    for split, records in selected_by_split.items():
        converted, filtered = convert_records(records)
        zero_intensity_peaks_removed += filtered
        table = pa.Table.from_pylist(converted, schema=OUTPUT_SCHEMA)
        pq.write_table(table, args.output_dir / f"{split}.parquet", compression="zstd")
        selected_keys = "\n".join(sorted(identity_sets[split])).encode("utf-8")
        selection_hashes[split] = hashlib.sha256(selected_keys).hexdigest()

    summary = {
        "source_root": str(args.source_root),
        "source_contract": "same aligned row has 1H NMR, 13C NMR, IR, and positive 20 eV MS/MS",
        "task": "simulated MS/MS plus molecular formula to SMILES",
        "seed": args.seed,
        "source_rows": scanned,
        "selected_rows": limits,
        "selection_inchikey_sha256": selection_hashes,
        "cross_split_inchikey_overlap": 0,
        "spectrum_cleaning": {
            "zero_intensity_peaks_removed": zero_intensity_peaks_removed,
            "negative_intensity": "error",
            "non_positive_mz": "error",
        },
        "schema": [field.name for field in OUTPUT_SCHEMA],
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": 128},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
