"""Quantize a checkpoint into the single-file browser format, and optionally write a parity
fixture (logprobs, grads, one AdamW step) for web/test/parity.mjs.

Usage: uv run python export.py --ckpt ../data/ckpt.pt --out ../web/public/headline-gpt.bin [--fixture ../data/fixture.bin]
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from model import GPT, Config, target_mask
from train import evaluate

QUANTIZED = ("wte.weight", "qkv.weight", "proj.weight", "fc.weight", "fc_proj.weight")


def write_container(path: Path, header: dict, blobs: list[tuple[dict, list[np.ndarray]]]) -> None:
    """blobs: (tensor entry, [data, optional per-row scales]) — offsets are filled in here."""
    body, offset = [], 0
    for entry, parts in blobs:
        for key, arr in zip(("offset", "scaleOffset"), parts):
            raw = arr.tobytes()
            entry[key] = offset
            body.append(raw + b"\0" * (-len(raw) % 4))
            offset += len(body[-1])
        header["tensors"].append(entry)
    head = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as f:
        f.write(b"HGPT" + struct.pack("<I", len(head)) + head + b"\0" * (-len(head) % 4))
        f.writelines(body)


def quantize_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("../data/ckpt.pt"))
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../web/public/headline-gpt.bin"))
    ap.add_argument("--fixture", type=Path)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = Config(**ck["cfg"])
    tok = json.loads((args.data / "tokenizer.json").read_text())
    state = {k: v.float().numpy() for k, v in ck["model"].items()}

    header = {"config": cfg.to_dict(), "tokenizer": {"vocab": tok["vocab"], "merges": tok["merges"]}, "tensors": [], "meta": {"step": ck["step"]}}
    blobs, deq = [], {}
    for name, w in state.items():
        if name.endswith(QUANTIZED):
            q, scale = quantize_rows(w)
            blobs.append(({"name": name, "shape": list(w.shape), "dtype": "int8"}, [q, scale]))
            deq[name] = q.astype(np.float32) * scale[:, None]
        else:
            blobs.append(({"name": name, "shape": list(w.shape), "dtype": "f32"}, [w.astype(np.float32)]))
            deq[name] = w

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    val = np.load(args.data / "val.npy")
    model = GPT(cfg).to(device)
    model.load_state_dict(ck["model"])
    fp32_loss = evaluate(model, val, device)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in deq.items()})
    q_loss = evaluate(model, val, device)
    header["meta"].update(val_loss_fp32=round(fp32_loss, 4), val_loss_int8=round(q_loss, 4))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_container(args.out, header, blobs)
    n_params = sum(v.size for v in state.values())
    print(f"{n_params:,} params -> {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"val loss fp32 {fp32_loss:.4f}  int8 {q_loss:.4f}")

    if args.fixture:
        write_fixture(args.fixture, cfg, deq, val)


def write_fixture(path: Path, cfg: Config, weights: dict[str, np.ndarray], val: np.ndarray) -> None:
    """Reference numbers for the WebGPU implementation, computed in float64 on CPU."""
    torch.manual_seed(0)
    model = GPT(cfg).double()
    model.load_state_dict({k: torch.from_numpy(v).double() for k, v in weights.items()})
    rows = val[:8]
    width = int((rows != 0).sum(1).max()) + 2
    idx = torch.from_numpy(rows[:, :width].astype(np.int64))
    mask = target_mask(idx).double()
    grad_logp = -mask / mask.sum()

    logp = model.token_logprobs(idx)
    (logp * grad_logp).sum().backward()
    grads = {k: p.grad.clone() for k, p in model.named_parameters()}
    opt_cfg = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weightDecay": 0.1, "maxGradNorm": 1.0}
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": opt_cfg["weightDecay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=opt_cfg["lr"], betas=(opt_cfg["beta1"], opt_cfg["beta2"]), eps=opt_cfg["eps"],
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), opt_cfg["maxGradNorm"])
    opt.step()
    after = {k: p.detach() for k, p in model.named_parameters()}

    header = {
        "config": cfg.to_dict(),
        "tensors": [],
        "meta": {"B": idx.shape[0], "T": idx.shape[1], "tokens": idx.flatten().tolist(), "opt": opt_cfg},
    }
    f32 = lambda t: np.ascontiguousarray(t.detach().numpy(), dtype=np.float32)  # noqa: E731
    blobs = [({"name": "logp", "shape": list(logp.shape), "dtype": "f32"}, [f32(logp)])]
    blobs.append(({"name": "grad_logp", "shape": list(grad_logp.shape), "dtype": "f32"}, [f32(grad_logp)]))
    for k, w in weights.items():
        blobs.append(({"name": f"w/{k}", "shape": list(w.shape), "dtype": "f32"}, [w.astype(np.float32)]))
        blobs.append(({"name": f"g/{k}", "shape": list(w.shape), "dtype": "f32"}, [f32(grads[k])]))
        blobs.append(({"name": f"after/{k}", "shape": list(w.shape), "dtype": "f32"}, [f32(after[k])]))
    write_container(path, header, blobs)
    print(f"fixture B={idx.shape[0]} T={idx.shape[1]} loss {-(logp * mask).sum().item() / mask.sum().item():.4f} -> {path}")


if __name__ == "__main__":
    main()
