"""ShapeFitter: set encoder + family classifier + one parameter head per shape family.

Global context: given a normalised 3D cloud, the network outputs logits over the 13 families and, for
every family, a raw parameter vector that the family's `decode` turns into a valid shape. During
training only the ground-truth family's head receives geometric losses (teacher forcing, as in SPFN,
Li et al. 2019, arXiv:1811.08988); at inference the top-k families' heads initialise test-time
refinement. Each head's output bias is initialised to a unit canonical shape so the very first
predictions are valid, non-degenerate shapes (a zero 6D vector would give an undefined rotation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from neurogeom.model.set_transformer import DeepSetsEncoder, SetTransformerEncoder
from neurogeom.shapes.registry import FAMILIES


def point_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-point input features: xyz plus log(N)/log(1024), so the encoder knows the cloud size."""
    n = mask.sum(-1, keepdim=True).to(x.dtype).clamp_min(1.0)
    logn = (torch.log(n) / math.log(1024.0)).unsqueeze(1).expand(-1, x.shape[1], -1)
    return torch.cat([x, logn], -1)


def _default_raw(f) -> torch.Tensor:
    """Raw vector of a unit, axis-aligned instance of the family (used as the head's output bias)."""
    g = torch.Generator().manual_seed(0)
    p = f.sample_params(1, g, "cpu")
    if "rot" in p:
        p["rot"] = torch.eye(3).unsqueeze(0)
    if "center" in p:
        p["center"] = torch.zeros(1, 3)
    raw = f.encode(p)[0]
    if f.name == "clusters":
        raw[:5] = 0.0                    # no prior over k
    if f.name == "polytope":
        raw[:6] = 0.0                    # no prior over subtype
    return raw


class ShapeFitter(nn.Module):
    def __init__(self, cfg_model: dict, amp_bf16: bool = False):
        super().__init__()
        self.amp_bf16 = amp_bf16
        d, hid = cfg_model["d_model"], cfg_model["head_hidden"]
        enc = SetTransformerEncoder if cfg_model.get("encoder", "set_transformer") == "set_transformer" else DeepSetsEncoder
        self.encoder = enc(in_dim=4, d_model=d, n_heads=cfg_model["n_heads"], n_inducing=cfg_model["n_inducing"],
                           n_isab=cfg_model["n_isab"])
        self.type_head = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, len(FAMILIES)))
        self.param_heads = nn.ModuleList()
        for f in FAMILIES:
            head = nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, f.n_raw))
            with torch.no_grad():
                head[-1].weight.mul_(0.1)                  # start close to the default shape
                head[-1].bias.copy_(_default_raw(f))
            self.param_heads.append(head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        # bf16 autocast only inside the encoder; heads, decoding and geometry stay in fp32 for precision.
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=self.amp_bf16 and x.is_cuda):
            z = self.encoder(point_features(x, mask), mask)
        z = z.float()
        return {"logits": self.type_head(z), "raw": [h(z) for h in self.param_heads], "embedding": z}
