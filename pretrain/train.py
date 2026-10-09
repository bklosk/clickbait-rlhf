"""Pretrain the headline GPT on the M-series GPU (MPS).

Usage: uv run python train.py --data ../data --out ../data/ckpt.pt [--steps 20000]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config, target_mask


def batches(data: np.ndarray, batch: int, rng: np.random.Generator):
    while True:
        for start in range(0, len(data) - batch + 1, batch):
            if start == 0:
                order = rng.permutation(len(data))
            rows = data[order[start : start + batch]]
            width = int((rows != 0).sum(1).max()) + 2  # trim padding to the longest row
            yield torch.from_numpy(rows[:, :width].astype(np.int64))


def masked_nll(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    mask = target_mask(idx)
    return -(model.token_logprobs(idx) * mask).sum() / mask.sum()


@torch.no_grad()
def evaluate(model: GPT, val: np.ndarray, device: str) -> float:
    model.eval()
    total, count = 0.0, 0.0
    for start in range(0, len(val), 1024):
        idx = torch.from_numpy(val[start : start + 1024].astype(np.int64)).to(device)
        mask = target_mask(idx)
        total += -(model.token_logprobs(idx) * mask).sum().item()
        count += mask.sum().item()
    model.train()
    return total / count


@torch.no_grad()
def sample(model: GPT, vocab: list[str], n: int, device: str, temperature: float = 0.9) -> list[str]:
    model.eval()
    idx = torch.zeros((n, 1), dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(model.cfg.ctx - 1):
        probs = torch.softmax(model(idx)[:, -1] / temperature, dim=-1)
        nxt = torch.multinomial(probs, 1)
        nxt[done] = 0
        done |= nxt[:, 0] == 0
        idx = torch.cat([idx, nxt], 1)
        if done.all():
            break
    model.train()
    return ["".join(vocab[i] for i in row[1:] if i != 0).replace("▁", " ").strip() for row in idx.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../data/ckpt.pt"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)
    train = np.load(args.data / "train.npy")
    val = np.load(args.data / "val.npy")
    vocab = json.loads((args.data / "tokenizer.json").read_text())["vocab"]

    cfg = Config(vocab=len(vocab))
    model = GPT(cfg).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} params on {device}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        frac = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    stream = batches(train, args.batch, np.random.default_rng(0))
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = next(stream).to(device)
        loss = masked_nll(model, idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0:
            dt = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.3f}  lr {lr_at(step):.2e}  {dt:6.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            print(f"  val loss {evaluate(model, val, device):.4f}", flush=True)
            for s in sample(model, vocab, 4, device):
                print("   ", s)
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step + 1}, args.out)


if __name__ == "__main__":
    main()
