#!/usr/bin/env python3
"""A transparent TTT-inspired adapter for an OpenNMT checkpoint.

This is intentionally separate from the paper's official implementation.  The
checkpoint used here has no fingerprint head, so we fit a small Morgan-
fingerprint head on a disjoint labeled candidate pool, then follow the paper's
cluster/retrieve/one-step update schedule.  The validation targets are never
read by this program.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-src", required=True)
    p.add_argument("--train-tgt", required=True)
    p.add_argument("--test-src", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--candidate-limit", type=int, default=20000)
    p.add_argument("--test-limit", type=int, default=0)
    p.add_argument("--clusters", type=int, default=100)
    p.add_argument("--retrieval-batch", type=int, default=64)
    p.add_argument("--encode-batch-size", type=int, default=32)
    p.add_argument("--refresh-embeddings", type=int, default=20)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--fp-lambda", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=3245)
    return p.parse_args()


def read_parallel(src_path: str, tgt_path: str, limit: int) -> tuple[list[str], list[str]]:
    src_lines: list[str] = []
    tgt_lines: list[str] = []
    with open(src_path, encoding="utf-8") as fs, open(tgt_path, encoding="utf-8") as ft:
        for i, (src, tgt) in enumerate(zip(fs, ft)):
            if limit and i >= limit:
                break
            src_lines.append(src.rstrip("\n"))
            tgt_lines.append(tgt.rstrip("\n"))
        if not limit and (fs.readline() or ft.readline()):
            raise RuntimeError(f"parallel corpus files have different lengths: {src_path}, {tgt_path}")
    if not src_lines or len(src_lines) != len(tgt_lines):
        raise RuntimeError(f"parallel corpus is empty or misaligned: {src_path}, {tgt_path}")
    return src_lines, tgt_lines


def read_source(src_path: str, limit: int) -> list[str]:
    lines: list[str] = []
    with open(src_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            lines.append(line.rstrip("\n"))
    if not lines:
        raise RuntimeError(f"source corpus is empty: {src_path}")
    return lines


def make_examples(src_lines, tgt_lines, vocabs, start_index=0):
    """Build the same token-id dictionaries used by OpenNMT's tensorify."""
    from onmt.constants import DefaultTokens

    examples = []
    for offset, src_line in enumerate(src_lines):
        src_tokens = src_line.strip().split()
        ex = {
            "src": {"src": " ".join(src_tokens), "src_ids": vocabs["src"](src_tokens)},
            "tgt": None,
            "indices": start_index + offset,
        }
        if tgt_lines is not None:
            tgt_tokens = tgt_lines[offset].strip().split()
            ex["tgt"] = {
                "tgt": " ".join(tgt_tokens),
                "tgt_ids": vocabs["tgt"](
                    [DefaultTokens.BOS] + tgt_tokens + [DefaultTokens.EOS]
                ),
            }
        examples.append(ex)
    return examples


def tensor_batch(examples, vocabs):
    from onmt.inputters.text_utils import tensorify, text_sort_key

    examples = sorted(examples, key=text_sort_key, reverse=True)
    return tensorify(vocabs, examples)


def encode_lines(model, lines, vocabs, device, batch_size=64, with_targets=None):
    """Return mean encoder representations in original line order."""
    reps = torch.empty((len(lines), model.encoder.embeddings.embedding_size), dtype=torch.float32)
    model.eval()
    for start in range(0, len(lines), batch_size):
        stop = min(len(lines), start + batch_size)
        batch_targets = with_targets[start:stop] if with_targets is not None else None
        examples = make_examples(lines[start:stop], batch_targets, vocabs, start)
        batch = tensor_batch(examples, vocabs)
        src = batch["src"].to(device)
        src_len = batch["srclen"].to(device)
        with torch.no_grad():
            enc_out, _, enc_len = model.encoder(src, src_len)
            positions = torch.arange(enc_out.size(1), device=device).unsqueeze(0)
            mask = positions < enc_len.unsqueeze(1)
            rep = (enc_out * mask.unsqueeze(-1)).sum(dim=1) / enc_len.clamp_min(1).unsqueeze(1)
        for row, original_index in enumerate(batch["indices"].tolist()):
            reps[original_index] = rep[row].float().cpu()
        if (start // batch_size) % 50 == 0:
            print(f"encode {stop}/{len(lines)}", flush=True)
    return reps


def morgan_fingerprints(smiles_lines: list[str], fp_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import DataStructs, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    values = np.zeros((len(smiles_lines), fp_size), dtype=np.float32)
    valid = np.zeros(len(smiles_lines), dtype=bool)
    for i, line in enumerate(smiles_lines):
        mol = Chem.MolFromSmiles("".join(line.split()))
        if mol is None:
            continue
        arr = np.zeros(fp_size, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
        values[i] = arr
        valid[i] = True
    return values, valid


class FingerprintHead(nn.Module):
    def __init__(self, hidden_size: int, fp_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, fp_size),
        )

    def forward(self, x):
        return self.net(x)


def fit_head(head, reps, labels, valid, device, epochs, lr, seed):
    rng = np.random.default_rng(seed)
    train_idx = np.flatnonzero(valid)
    x = reps[train_idx]
    y = torch.from_numpy(labels[train_idx])
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    batch_size = 256
    for epoch in range(epochs):
        order = rng.permutation(len(train_idx))
        total = 0.0
        for begin in range(0, len(order), batch_size):
            ids = order[begin : begin + batch_size]
            xb = x[ids].to(device)
            yb = y[ids].to(device)
            loss = F.binary_cross_entropy_with_logits(head(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        print(f"fingerprint_head epoch={epoch + 1}/{epochs} loss={total / len(train_idx):.6f}", flush=True)


def predict_head(head, reps, device, batch_size=4096) -> torch.Tensor:
    head.eval()
    output = torch.empty((len(reps), head.net[-1].out_features), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(reps), batch_size):
            stop = min(len(reps), start + batch_size)
            output[start:stop] = head(reps[start:stop].to(device)).float().cpu()
    return output


def make_cluster_representatives(test_fp: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    normalized = test_fp / np.maximum(np.linalg.norm(test_fp, axis=1, keepdims=True), 1e-8)
    n_clusters = min(n_clusters, len(test_fp))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed, max_iter=100)
    labels = kmeans.fit_predict(normalized)
    chosen = []
    for cluster_id in range(n_clusters):
        indices = np.flatnonzero(labels == cluster_id)
        if len(indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        chosen.append(int(indices[np.argmax(normalized[indices] @ centroid)]))
    return np.asarray(chosen, dtype=np.int64)


def normalized_rows(logits: torch.Tensor) -> torch.Tensor:
    return logits / logits.norm(dim=1, keepdim=True).clamp_min(1e-8)


def train_step(model, head, optimizer, src_lines, tgt_lines, selected, train_fp, vocabs, device, fp_lambda):
    examples = make_examples([src_lines[i] for i in selected], [tgt_lines[i] for i in selected], vocabs)
    batch = tensor_batch(examples, vocabs)
    src = batch["src"].to(device)
    src_len = batch["srclen"].to(device)
    tgt = batch["tgt"].to(device)

    model.train()
    head.train()
    dec_in = tgt[:, :-1, :]
    enc_out, enc_final, enc_len = model.encoder(src, src_len)
    model.decoder.init_state(src, enc_out, enc_final)
    dec_out, _ = model.decoder(dec_in, enc_out, src_len=enc_len)
    logits = model.generator(dec_out)
    pad_idx = vocabs["tgt"]["<blank>"]
    gold = tgt[:, 1:, 0]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gold.reshape(-1), ignore_index=pad_idx)

    positions = torch.arange(enc_out.size(1), device=device).unsqueeze(0)
    mask = positions < enc_len.unsqueeze(1)
    rep = (enc_out * mask.unsqueeze(-1)).sum(dim=1) / enc_len.clamp_min(1).unsqueeze(1)
    fp_logits = head(rep)
    # tensorify sorts the selected examples by length, so use returned indices.
    local_positions = np.asarray(batch["indices"].tolist(), dtype=np.int64)
    global_ids = np.asarray(selected, dtype=np.int64)[local_positions]
    fp_target = torch.from_numpy(train_fp[global_ids]).to(device)
    fp_loss = F.binary_cross_entropy_with_logits(fp_logits, fp_target)
    loss = ce + fp_lambda * fp_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach()), float(ce.detach()), float(fp_loss.detach())


def save_checkpoint(ck, model, out_path: Path):
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "generator": {k: v.detach().cpu() for k, v in model.generator.state_dict().items()},
        "vocab": ck["vocab"],
        "opt": ck["opt"],
    }
    torch.save(payload, out_path)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    from onmt.inputters.inputter import dict_to_vocabs
    from onmt.model_builder import build_base_model

    train_src, train_tgt = read_parallel(args.train_src, args.train_tgt, args.candidate_limit)
    test_src = read_source(args.test_src, args.test_limit)
    print(f"candidate={len(train_src)} test={len(test_src)} device={device}", flush=True)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabs = dict_to_vocabs(ck["vocab"])
    model_opt = copy.deepcopy(ck["opt"])
    model = build_base_model(model_opt, vocabs, gpu=device.type == "cuda", checkpoint=ck)
    model.to(device)
    model.eval()

    train_fp, train_fp_valid = morgan_fingerprints(train_tgt)
    train_reps = encode_lines(model, train_src, vocabs, device, batch_size=args.encode_batch_size)
    test_reps = encode_lines(model, test_src, vocabs, device, batch_size=args.encode_batch_size)
    hidden_size = train_reps.shape[1]
    head = FingerprintHead(hidden_size, train_fp.shape[1]).to(device)
    fit_head(head, train_reps, train_fp, train_fp_valid, device, args.head_epochs, args.head_lr, args.seed)

    train_fp_logits = predict_head(head, train_reps, device)
    test_fp_logits = predict_head(head, test_reps, device)
    representatives = make_cluster_representatives(test_fp_logits.numpy(), args.clusters, args.seed)
    print(f"clusters={len(representatives)} retrieval_batch={args.retrieval_batch}", flush=True)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    train_norm = normalized_rows(train_fp_logits)
    test_norm = normalized_rows(test_fp_logits)
    history = []

    for step, test_index in enumerate(representatives):
        if step and args.refresh_embeddings and step % args.refresh_embeddings == 0:
            print(f"refresh embeddings at step={step}", flush=True)
            train_reps = encode_lines(model, train_src, vocabs, device, batch_size=args.encode_batch_size)
            test_reps = encode_lines(model, test_src, vocabs, device, batch_size=args.encode_batch_size)
            train_fp_logits = predict_head(head, train_reps, device)
            test_fp_logits = predict_head(head, test_reps, device)
            train_norm = normalized_rows(train_fp_logits)
            test_norm = normalized_rows(test_fp_logits)

        scores = torch.mv(train_norm, test_norm[int(test_index)].to(train_norm.device))
        k = min(args.retrieval_batch, len(train_src))
        selected = torch.topk(scores, k=k, largest=True).indices.cpu().numpy().astype(np.int64)
        loss, ce, fp_loss = train_step(
            model, head, optimizer, train_src, train_tgt, selected,
            train_fp, vocabs, device, args.fp_lambda
        )
        scheduler.step()
        record = {
            "step": step + 1,
            "test_index": int(test_index),
            "loss": loss,
            "ce": ce,
            "fingerprint_loss": fp_loss,
            "lr": scheduler.get_last_lr()[0],
            "selected": selected.tolist(),
        }
        history.append(record)
        print(
            f"ttt step={step + 1}/{len(representatives)} test_index={test_index} "
            f"loss={loss:.5f} ce={ce:.5f} fp={fp_loss:.5f}",
            flush=True,
        )
        if (step + 1) % max(1, args.refresh_embeddings) == 0:
            (out_dir / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    save_checkpoint(ck, model, out_dir / "model_step250000_ttt_proxy.pt")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, out_dir / "fingerprint_head.pt")
    (out_dir / "progress.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = vars(args) | {
        "candidate_count": len(train_src),
        "test_count": len(test_src),
        "cluster_count": len(representatives),
        "device": str(device),
        "note": "TTT-inspired proxy; OpenNMT checkpoint lacks the paper's native fingerprint head.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {out_dir / 'model_step250000_ttt_proxy.pt'}", flush=True)


if __name__ == "__main__":
    main()
