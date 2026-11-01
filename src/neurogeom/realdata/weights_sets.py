"""Real datasets read directly from weight matrices: GPT-2 W_pos (R1), grokked modular addition W_E (R2),
Toy Models of Superposition W (R3), plus their controls.

Global context:
- R1: GPT-2 small learned positional embeddings W_pos (1024 x 768); rows 1..1023 lie on a helix in the
  top-3 PCs (Yedidia 2023, LessWrong qvWP3aBDBaqXvPNhS; row 0 is an outlier and is dropped as there).
  Control: the same matrix at Hugging Face's GPT-2 initialisation, N(0, 0.02^2) (GPT2Config.initializer_range;
  OpenAI's original code initialised wpe with stddev 0.01), -> blob.
- R2: a 1-layer transformer trained on (a + b) mod 113 until it groks, with the hyperparameters of the
  TransformerLens grokking demo / Nanda et al. 2023 (arXiv:2301.05217): d_model 128, 4 heads, d_mlp 512,
  no LayerNorm, 30% of pairs for training, full-batch AdamW lr 1e-3, weight decay 1.0, betas (0.9, 0.98);
  25,000 epochs instead of the paper's 40,000. Test accuracy (extraction log) was 0.075 at epoch 17,500, 0.936 at
  20,000 and 0.999 at 24,999, so training stops soon after grokking and the post-grokking cleanup of W_E may be
  less complete than in the paper's model.
  The token embeddings W_E[0:113] form circles per key frequency (Nanda et al.; Zhong et al. 2023).
  Controls: the embeddings at initialisation and at the memorisation checkpoint (before grokking).
- R3: Toy Models of Superposition (Elhage et al. 2022): x' = ReLU(W^T W x + b) with W in R^{3 x n},
  sparse uniform features; the columns of W (features with norm > 0.5) form polytopes in 3D.
  Control: dense features (sparsity 0), where only ~3 orthogonal features are represented.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurogeom.logging_utils import get_logger
from neurogeom.realdata.common import hf_env, save_cloud

log = get_logger(__name__)


# ------------------------------------------------------------------ R1: GPT-2 positional embeddings
def gpt2_wpos(cfg: dict) -> None:
    hf_env()
    from transformers import GPT2Model
    w = GPT2Model.from_pretrained("gpt2").wpe.weight.detach().numpy()
    save_cloud(cfg, "R1_gpt2_wpos", w[1:], np.arange(1, 1024),
               {"source": "gpt2 (small) wpe.weight rows 1..1023", "model": "gpt2", "layer": "embedding", "expected_coarse": "1d",
                "expected_fine": ["helix"], "prereg": True})
    rng = np.random.default_rng(0)
    save_cloud(cfg, "C3_gpt2_wpos_init", rng.normal(0, 0.02, size=(1023, 768)), np.arange(1, 1024),
               {"source": "HF GPT-2 initialisation N(0, 0.02^2) of W_pos", "control_for": "R1_gpt2_wpos", "expected_coarse": "3d",
                "expected_fine": ["blob"], "prereg": True})


# ------------------------------------------------------------------ R2: grokked modular addition
class OneLayerTransformer(nn.Module):
    """Nanda et al. 2023 architecture: embed + learned pos-embed, 1 attention layer, 1 ReLU MLP, unembed,
    no LayerNorm. Input [a, b, =]; the logits at '=' predict (a + b) mod p."""

    def __init__(self, p: int = 113, d: int = 128, heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(p + 1, d) / d ** 0.5)
        self.W_pos = nn.Parameter(torch.randn(3, d) / d ** 0.5)
        self.attn = nn.MultiheadAttention(d, heads, bias=False, batch_first=True)
        self.mlp_in, self.mlp_out = nn.Linear(d, d_mlp), nn.Linear(d_mlp, d)
        self.W_U = nn.Parameter(torch.randn(d, p) / d ** 0.5)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.W_E[tokens] + self.W_pos
        mask = torch.triu(torch.ones(3, 3, dtype=torch.bool, device=h.device), 1)
        h = h + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = h + self.mlp_out(F.relu(self.mlp_in(h)))
        return h[:, -1] @ self.W_U


def grokking(cfg: dict, p: int = 113, epochs: int = 25000, seed: int = 0) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(seed)
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    data = torch.stack([a.flatten(), b.flatten(), torch.full((p * p,), p)], 1)
    labels = (data[:, 0] + data[:, 1]) % p
    perm = torch.randperm(p * p, generator=g)
    n_train = int(0.3 * p * p)
    tr, te = perm[:n_train], perm[n_train:]
    torch.manual_seed(seed)
    model = OneLayerTransformer(p).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    xtr, ytr, xte, yte = data[tr].to(dev), labels[tr].to(dev), data[te].to(dev), labels[te].to(dev)
    snaps = {"init": model.W_E.detach()[:p].cpu().numpy().copy()}
    for ep in range(epochs):
        loss = F.cross_entropy(model(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 1000:
            snaps["memorised"] = model.W_E.detach()[:p].cpu().numpy().copy()
        if ep % 2500 == 0 or ep == epochs - 1:
            with torch.no_grad():
                acc_te = float((model(xte).argmax(-1) == yte).float().mean())
                acc_tr = float((model(xtr).argmax(-1) == ytr).float().mean())
            log.info("grokking epoch %d: train loss %.4f, train acc %.3f, test acc %.3f", ep, float(loss), acc_tr, acc_te)
    with torch.no_grad():
        acc_te = float((model(xte).argmax(-1) == yte).float().mean())
    w = model.W_E.detach()[:p].cpu().numpy()
    common = {"model": "1-layer transformer trained on (a+b) mod 113 (Nanda et al. 2023 hyperparameters)", "test_acc": acc_te,
              "epochs": epochs, "seed": seed}
    save_cloud(cfg, "R2_grokking_WE", w, np.arange(p), {**common, "source": "W_E[0:113] after training", "expected_coarse": "1d",
                                                        "expected_fine": ["arc", "ellipse"], "prereg": True})
    save_cloud(cfg, "C3_grokking_WE_init", snaps["init"], np.arange(p), {**common, "source": "W_E at initialisation",
                                                                        "control_for": "R2_grokking_WE", "expected_coarse": "3d", "expected_fine": ["blob"], "prereg": True})
    save_cloud(cfg, "C3_grokking_WE_memorised", snaps["memorised"], np.arange(p), {**common, "source": "W_E at epoch 1000 (memorisation, before grokking)",
                                                                                  "control_for": "R2_grokking_WE", "expected_coarse": None, "expected_fine": None, "prereg": False})


# ------------------------------------------------------------------ R3: toy models of superposition
def tms(cfg: dict, n_features: int, sparsity: float, m: int = 3, steps: int = 10000, instances: int = 8, seed: int = 0) -> np.ndarray:
    """Train `instances` ReLU-output toy models (Elhage et al. 2022, Sec. 2) and return the W (m x n) of
    the lowest-loss instance (instances escape bad local minima, as in the paper's notebook)."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    W = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(instances, m, n_features, device=dev)))
    bias = nn.Parameter(torch.zeros(instances, n_features, device=dev))
    opt = torch.optim.AdamW([W, bias], lr=1e-3)
    for step in range(steps):
        x = torch.rand(1024, instances, n_features, device=dev)
        x = x * (torch.rand_like(x) >= sparsity)                       # feature present with prob 1 - S
        h = torch.einsum("imn,bin->bim", W, x)
        out = F.relu(torch.einsum("imn,bim->bin", W, h) + bias)
        loss_i = ((out - x) ** 2).mean((0, 2))                         # uniform importance
        opt.zero_grad()
        loss_i.sum().backward()
        opt.step()
    best = int(loss_i.argmin())
    log.info("TMS n=%d S=%.2f: losses %s, best instance %d", n_features, sparsity, [round(float(v), 5) for v in loss_i], best)
    return W[best].detach().cpu().numpy()


def tms_sets(cfg: dict) -> None:
    for n, s, pre in ((6, 0.95, True), (8, 0.97, True), (20, 0.99, False)):
        w = tms(cfg, n, s)
        cols = w.T                                                     # one 3D point per feature
        keep = np.linalg.norm(cols, axis=1) > 0.5
        save_cloud(cfg, f"R3_tms_n{n}", cols[keep], np.arange(n)[keep],
                   {"model": f"toy model of superposition, n={n}, m=3, sparsity={s}", "source": "columns of W with norm > 0.5",
                    "n_represented": int(keep.sum()), "expected_coarse": "0d", "expected_fine": ["polytope", "clusters"], "prereg": pre})
    w = tms(cfg, 6, 0.0)
    cols = w.T
    keep = np.linalg.norm(cols, axis=1) > 0.5
    save_cloud(cfg, "C3_tms_dense", cols[keep], np.arange(6)[keep], {"model": "toy model, n=6, m=3, dense features (S=0)",
                                                                     "control_for": "R3_tms_n6", "n_represented": int(keep.sum()),
                                                                     "expected_coarse": None, "expected_fine": None, "prereg": False})
