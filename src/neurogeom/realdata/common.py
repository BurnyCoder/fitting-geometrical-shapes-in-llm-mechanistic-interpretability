"""Shared helpers for real-activation datasets: storage, logging of prompts, HF model loading.

Global context: every real dataset is saved as data_cache/real/<id>.npz with
  X       (N, D) float32 activation vectors (one point per row; the fitter later takes their top-3 PCA),
  labels  (N,)   the concept value of each point (weekday, year, character count, ...),
  meta    JSON   model, layer, token position, prompt template, expected shape (from the preregistration).
Layer convention: "layer L" = Hugging Face `output_hidden_states[L]`, i.e. the residual stream after
the first L transformer blocks (index 0 = embeddings). Models are truncated to L+1 blocks so index L is
never the final, norm-applied hidden state (in HF Llama/GPT-NeoX the last entry is post-final-norm).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from neurogeom.logging_utils import get_logger

log = get_logger(__name__)


def real_dir(cfg: dict) -> Path:
    d = Path(cfg["paths"]["data_cache"]) / "real"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_cloud(cfg: dict, did: str, x: np.ndarray, labels, meta: dict) -> Path:
    p = real_dir(cfg) / f"{did}.npz"
    np.savez_compressed(p, X=np.asarray(x, dtype=np.float32), labels=np.asarray(labels, dtype=object),
                        meta=json.dumps(meta))
    log.info("saved %s: X %s, meta %s", p.name, tuple(np.shape(x)), meta)
    return p


def load_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=True)
    return z["X"], z["labels"], json.loads(str(z["meta"]))


def log_prompts(name: str, prompts: list[str]) -> None:
    """Every prompt sent to an LLM is logged in full (user requirement); no text is generated here,
    only hidden states are read, so there are no LLM outputs to log besides activation statistics."""
    for i, p in enumerate(prompts):
        log.info("[%s prompt %d/%d] %r", name, i + 1, len(prompts), p)


def hf_env() -> None:
    from neurogeom.config import load_env
    load_env()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_truncated(model_id: str, layer: int, max_gpu_gib: float = 6.5, random_init: bool = False):
    """Causal LM truncated to layer+1 blocks in bf16; layers that do not fit in max_gpu_gib are
    offloaded to CPU by accelerate (big-model inference: weights are streamed to the GPU per forward)."""
    hf_env()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id)
    cfg.num_hidden_layers = layer + 1
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if random_init:  # control C3: same architecture, random weights (HF default initialisation)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, torch_dtype=torch.bfloat16, device_map="auto",
                                                     max_memory={0: f"{max_gpu_gib}GiB", "cpu": "24GiB"})
    model.eval()
    log.info("loaded %s truncated to %d blocks (random_init=%s)", model_id, layer + 1, random_init)
    return model, tok


@torch.no_grad()
def last_token_acts(model, tok, prompts: list[str], layer: int, batch: int = 32) -> np.ndarray:
    """hidden_states[layer] at the last (non-padding) token of every prompt, (N, D) float32."""
    out = []
    dev = next(model.parameters()).device if not hasattr(model, "hf_device_map") else "cuda"
    for s in range(0, len(prompts), batch):
        enc = tok(prompts[s:s + batch], return_tensors="pt", padding=True).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        last = enc["attention_mask"].sum(-1) - 1                         # right padding -> last real token
        out.append(hs[torch.arange(hs.shape[0], device=hs.device), last].float().cpu())
    acts = torch.cat(out).numpy()
    log.info("activations: %s, mean norm %.2f", acts.shape, float(np.linalg.norm(acts, axis=1).mean()))
    return acts


@torch.no_grad()
def token_acts(model, tok, prompts: list[str], targets: list[str], layer: int, batch: int = 32) -> np.ndarray:
    """hidden_states[layer] at the last token of the first occurrence of `targets[i]` in prompts[i]."""
    out = []
    dev = "cuda"
    for s in range(0, len(prompts), batch):
        ps, ts = prompts[s:s + batch], targets[s:s + batch]
        enc = tok(ps, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")
        enc = enc.to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        for j, (p, t) in enumerate(zip(ps, ts)):
            end = p.index(t) + len(t)                                    # character end of the target word
            ok = (offs[j, :, 1] <= end) & (offs[j, :, 1] > 0) & enc["attention_mask"][j].bool().cpu()
            pos = int(torch.nonzero(ok).max())                            # last token ending within the word
            out.append(hs[j, pos].float().cpu())
    acts = torch.stack(out).numpy()
    log.info("token activations: %s, mean norm %.2f", acts.shape, float(np.linalg.norm(acts, axis=1).mean()))
    return acts
