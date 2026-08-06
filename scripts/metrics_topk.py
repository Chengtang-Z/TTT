#!/usr/bin/env python3
"""Canonical exact-match and validity metrics for OpenNMT n-best output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--targets", required=True)
    p.add_argument("--n-best", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from rdkit import Chem, RDLogger, DataStructs
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    pred_lines = Path(args.predictions).read_text(encoding="utf-8").splitlines()
    tgt_lines = Path(args.targets).read_text(encoding="utf-8").splitlines()
    if len(pred_lines) != len(tgt_lines) * args.n_best:
        raise RuntimeError(
            f"expected {len(tgt_lines) * args.n_best} predictions, got {len(pred_lines)}"
        )

    def parse_smiles(line: str):
        text = "".join(line.split())
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return None
        return mol, Chem.MolToSmiles(mol, canonical=True)

    target_mols = []
    target_canon = []
    for line in tgt_lines:
        parsed = parse_smiles(line)
        if parsed is None:
            raise RuntimeError("invalid target SMILES encountered")
        target_mols.append(parsed[0])
        target_canon.append(parsed[1])

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    target_fp = [fpgen.GetFingerprint(mol) for mol in target_mols]
    summary = {
        "n_samples": len(tgt_lines),
        "n_best": args.n_best,
        "valid_prediction_rate_by_k": {},
        "canonical_exact_match_rate_by_k": {},
        "formula_match_rate_by_k": {},
        "best_morgan_r2_tanimoto_mean_by_k": {},
        "best_morgan_r2_tanimoto_median_by_k": {},
    }
    for k in (1, 5, 10):
        if k > args.n_best:
            continue
        valid_count = 0
        exact_count = 0
        formula_count = 0
        best_tanimoto = []
        for i, target in enumerate(target_mols):
            any_valid = False
            exact = False
            formula = False
            best = 0.0
            target_formula = rdMolDescriptors.CalcMolFormula(target)
            for line in pred_lines[i * args.n_best : i * args.n_best + k]:
                parsed = parse_smiles(line)
                if parsed is None:
                    continue
                any_valid = True
                mol, canonical = parsed
                exact = exact or canonical == target_canon[i]
                formula = formula or rdMolDescriptors.CalcMolFormula(mol) == target_formula
                best = max(best, DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol), target_fp[i]))
            valid_count += int(any_valid)
            exact_count += int(exact)
            formula_count += int(formula)
            best_tanimoto.append(best)
        arr = np.asarray(best_tanimoto, dtype=float)
        summary["valid_prediction_rate_by_k"][str(k)] = valid_count / len(tgt_lines)
        summary["canonical_exact_match_rate_by_k"][str(k)] = exact_count / len(tgt_lines)
        summary["formula_match_rate_by_k"][str(k)] = formula_count / len(tgt_lines)
        summary["best_morgan_r2_tanimoto_mean_by_k"][str(k)] = float(arr.mean())
        summary["best_morgan_r2_tanimoto_median_by_k"][str(k)] = float(np.median(arr))

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
