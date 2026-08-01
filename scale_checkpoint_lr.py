#!/usr/bin/env python3
"""Scale optimizer and scheduler learning rates in a Lightning checkpoint."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--target-base-lr", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source.resolve() == args.destination.resolve():
        raise ValueError("source and destination must be different")
    if args.destination.exists():
        raise FileExistsError(args.destination)

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    schedulers = checkpoint.get("lr_schedulers", [])
    if not schedulers or not schedulers[0].get("base_lrs"):
        raise ValueError("checkpoint has no scheduler base_lrs")

    old_base_lr = float(schedulers[0]["base_lrs"][0])
    scale = args.target_base_lr / old_base_lr

    for optimizer in checkpoint.get("optimizer_states", []):
        for group in optimizer.get("param_groups", []):
            group["lr"] = float(group["lr"]) * scale
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * scale

    for scheduler in schedulers:
        for key in ("base_lrs", "_last_lr"):
            if key in scheduler:
                scheduler[key] = [float(value) * scale for value in scheduler[key]]

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{args.destination.name}.",
        suffix=".tmp",
        dir=args.destination.parent,
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, args.destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    current_lrs = [
        group["lr"]
        for optimizer in checkpoint.get("optimizer_states", [])
        for group in optimizer.get("param_groups", [])
    ]
    print(
        f"wrote {args.destination} epoch={checkpoint.get('epoch')} "
        f"base_lr={args.target_base_lr:g} current_lrs={current_lrs}"
    )


if __name__ == "__main__":
    main()
