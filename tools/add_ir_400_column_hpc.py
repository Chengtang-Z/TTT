#!/usr/bin/env python3
"""Add a numeric 400-bin view to the already-derived IR-token parquet files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no parquet files found in {args.data_dir}")

    total = 0
    for index, path in enumerate(paths, start=1):
        table = pq.read_table(path)
        if "ir_400" in table.column_names:
            total += table.num_rows
            print(f"[{index}/{len(paths)}] reuse {path.name}", flush=True)
            continue
        token_rows = table["ir_tokens"].to_pylist()
        bins = []
        for text in token_rows:
            pieces = str(text).split()
            if len(pieces) != 401 or pieces[0] != "IR":
                raise ValueError(f"{path.name}: invalid IR token row length={len(pieces)}")
            bins.append([int(value) for value in pieces[1:]])
        updated = table.append_column("ir_400", pa.array(bins, type=pa.list_(pa.int16())))
        temporary = path.with_suffix(path.suffix + ".tmp")
        pq.write_table(updated, temporary, compression="zstd", use_dictionary=True)
        os.replace(temporary, path)
        total += updated.num_rows
        print(f"[{index}/{len(paths)}] added {path.name} rows={updated.num_rows}", flush=True)

    marker = args.data_dir / "_IR400_NUMERIC_COMPLETE"
    marker.write_text(f"rows={total}\nfiles={len(paths)}\n", encoding="utf-8")
    print(f"complete rows={total} files={len(paths)}", flush=True)


if __name__ == "__main__":
    main()
