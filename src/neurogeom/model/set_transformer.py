"""Permutation-invariant set encoders: Set Transformer (MAB / ISAB / PMA) and a DeepSets baseline.

Global context: the input is an unordered point cloud of 5..1024 points, so the encoder must be
invariant to point order and to padding. The Set Transformer blocks follow
juho-lee/set_transformer/modules.py (MIT license, Copyright (c) 2020 Juho Lee, reproduced in LICENSE;
https://github.com/juho-lee/set_transformer;
Lee et al., "Set Transformer", ICML 2019, arXiv:1810.00825) with two changes:
  1. a key-padding mask so batches can mix cloud sizes (padded points never receive attention);
  2. attention via torch.nn.functional.scaled_dot_product_attention with the standard per-head
     1/sqrt(d_head) scaling (Vaswani et al. 2017) instead of the original 1/sqrt(dim_V).
ISAB attends N points to m inducing points and back, O(N m) instead of O(N^2) (Lee et al. Sec. 3.2).
The DeepSets encoder (Zaheer et al. 2017, arXiv:1703.06114) is ablation A4.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead attention block: H = LN(Q + MHA(Q, K, K)); out = LN(H + relu(FC(H)))."""

    def __init__(self, dim_q: int, dim_k: int, dim_v: int, num_heads: int):
        super().__init__()
        self.dim_v, self.num_heads = dim_v, num_heads
        self.fc_q, self.fc_k, self.fc_v = nn.Linear(dim_q, dim_v), nn.Linear(dim_k, dim_v), nn.Linear(dim_k, dim_v)
        self.ln0, self.ln1 = nn.LayerNorm(dim_v), nn.LayerNorm(dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, q: torch.Tensor, k: torch.Tensor, k_mask: torch.Tensor | None = None) -> torch.Tensor:
        q = self.fc_q(q)
        kk, vv = self.fc_k(k), self.fc_v(k)
        b, nq, _ = q.shape
        h, dh = self.num_heads, self.dim_v // self.num_heads
        split = lambda t: t.view(b, t.shape[1], h, dh).transpose(1, 2)        # (B, h, n, dh)
        attn_mask = None if k_mask is None else k_mask[:, None, None, :]       # True = attend
        o = F.scaled_dot_product_attention(split(q), split(kk), split(vv), attn_mask=attn_mask)
        o = q + o.transpose(1, 2).reshape(b, nq, self.dim_v)                   # residual on the projected query
        o = self.ln0(o)
        o = self.ln1(o + F.relu(self.fc_o(o)))
        return o


class ISAB(nn.Module):
    """Induced set attention block: X -> H = MAB(I, X) -> MAB(X, H)."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.inducing)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        hidden = self.mab0(self.inducing.expand(x.shape[0], -1, -1), x, mask)   # inducing points read the set
        return self.mab1(x, hidden)                                              # every point reads the summary


class PMA(nn.Module):
    """Pooling by multihead attention with learned seed vectors."""

    def __init__(self, dim: int, num_heads: int, num_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = MAB(dim, dim, dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return self.mab(self.seeds.expand(x.shape[0], -1, -1), x, mask)


class SetTransformerEncoder(nn.Module):
    """points (B, N, F) + mask -> embedding (B, d). Stack: Linear -> ISAB x n_isab -> PMA(1)."""

    def __init__(self, in_dim: int, d_model: int, n_heads: int, n_inducing: int, n_isab: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([ISAB(d_model, d_model, n_heads, n_inducing) for _ in range(n_isab)])
        self.pool = PMA(d_model, n_heads, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.inp(feats)
        for blk in self.blocks:
            h = blk(h, mask)
        return self.pool(h, mask).squeeze(1)


class DeepSetsEncoder(nn.Module):
    """Ablation A4: per-point MLP, masked mean + max pooling, then an MLP (Zaheer et al. 2017)."""

    def __init__(self, in_dim: int, d_model: int, **_):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.ReLU(),
                                 nn.Linear(d_model, d_model))
        self.rho = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(feats)
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(1)
        return self.rho(torch.cat([mean, mx], -1))
