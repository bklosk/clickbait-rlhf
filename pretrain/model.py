"""GPT-2-style model. Every op here has a hand-written WGSL twin in web/src/gpt.js,
so keep the two in lockstep: pre-LN, tanh GELU, LN eps 1e-5, tied input/output embedding."""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 4096
    ctx: int = 32
    d: int = 256
    layers: int = 4
    heads: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d
        self.heads = cfg.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc_proj = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, d))
        return x + self.fc_proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.d)
        self.wpe = nn.Embedding(cfg.ctx, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.lnf = nn.LayerNorm(cfg.d)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.lnf(x) @ self.wte.weight.T

    def token_logprobs(self, idx: torch.Tensor) -> torch.Tensor:
        """log p(idx[:, t+1] | idx[:, :t+1]) for every t; last column is 0."""
        logp = F.log_softmax(self(idx), dim=-1)[:, :-1]
        out = logp.gather(2, idx[:, 1:, None]).squeeze(2)
        return F.pad(out, (0, 1))


def target_mask(idx: torch.Tensor) -> torch.Tensor:
    """Rows look like [eot, w1..wn, eot, pad...]: train on predicting w1..wn and the closing eot."""
    n = (idx != 0).sum(1, keepdim=True)
    return (torch.arange(idx.shape[1], device=idx.device) <= n).float()
