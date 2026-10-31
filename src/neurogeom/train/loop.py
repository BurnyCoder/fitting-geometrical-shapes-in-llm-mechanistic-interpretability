"""Training loop: fresh synthetic batches every step, AdamW + warmup/cosine, periodic validation.

Global context: phase `train` of run.py. Every step draws a new batch from the `train` seed stream
(no fixed training set, so no overfitting to particular samples and no leakage into the fixed
`val`/`test` streams). The point count N is shared within a step (drawn log-uniformly) so no padding
is needed during training; padding/masks are exercised at evaluation. Checkpoints and a JSON
training curve are written to checkpoints/<run>/ and results/<run>/train/.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from neurogeom.data.nuisance import log_uniform
from neurogeom.data.synth import generate_chunks, generate_mixed
from neurogeom.logging_utils import get_logger
from neurogeom.model.net import ShapeFitter
from neurogeom.seeding import make_generator, seed_everything, stream_seed
from neurogeom.shapes.registry import FAMILIES
from neurogeom.train.losses import compute_loss, to_floats

log = get_logger(__name__)


def lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 5% (Loshchilov & Hutter 2017, SGDR, arXiv:1608.03983)."""
    if step < warmup:
        return (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t))


def build_val_set(cfg: dict, device) -> list[list[dict]]:
    """Fixed validation set from the `val` stream: for each N-bin, one mixed batch of all families."""
    gen = make_generator(stream_seed(cfg["seed"], "val"), device)
    per_bin = max(len(FAMILIES), cfg["train"]["val_size"] // len(cfg["eval"]["n_bins"]))
    out = []
    for lo, hi in cfg["eval"]["n_bins"]:
        n = int(round(math.sqrt(lo * hi)))
        out.append(generate_mixed(per_bin, n, cfg["synth"], cfg["dense"], gen, device))
    return out


@torch.no_grad()
def validate(model, val_set, cfg) -> dict:
    model.eval()
    accs, cds = [], []
    for parts in val_set:
        x = torch.cat([p["x"] for p in parts])
        m = torch.cat([p["mask"] for p in parts])
        out = model(x, m)
        _, st = compute_loss(out, parts, cfg["loss"], cfg["dense"], aux_weight=0.0)
        st = to_floats(st)
        accs.append(st["acc"])
        cds.append(sum(v for k, v in st.items() if k.startswith("cd/")) / len(parts))
    model.train()
    return {"val_acc": sum(accs) / len(accs), "val_cd_teacher": sum(cds) / len(cds)}


def train(cfg: dict) -> dict:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    seed_everything(cfg["seed"])
    tc = cfg["train"]
    model = ShapeFitter(cfg["model"], amp_bf16=tc["amp_bf16"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %s encoder, %d parameters, device %s", cfg["model"]["encoder"], n_params, device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, tc["warmup_steps"], tc["steps"]))
    gen = make_generator(stream_seed(cfg["seed"], "train"), device)
    val_set = build_val_set(cfg, device)
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["paths"]["results"]) / "train"
    res_dir.mkdir(parents=True, exist_ok=True)
    curve, t0 = [], time.time()
    aux_steps = max(1, int(cfg["loss"]["aux_anneal_frac"] * tc["steps"]))
    last_step, queue = 0, []
    for step in range(tc["steps"]):
        if not queue:  # refill: several steps' data from one generator call (same N within a chunk)
            n = int(log_uniform(1, cfg["synth"]["n_min"], cfg["synth"]["n_max"], gen, device).round().clamp(cfg["synth"]["n_min"], cfg["synth"]["n_max"]))
            # token budget: cap batch x N so large clouds do not spill out of the 8 GB of VRAM
            b_eff = max(64, min(tc["batch_size"], tc["token_budget"] // n))
            queue = generate_chunks(b_eff, n, tc["gen_chunk"], cfg["synth"], cfg["dense"], gen, device)
        parts = queue.pop()
        x = torch.cat([p["x"] for p in parts])
        mask = torch.cat([p["mask"] for p in parts])
        out = model(x, mask)
        a = min(1.0, step / aux_steps)
        aux_w = cfg["loss"]["w_aux_start"] + a * (cfg["loss"]["w_aux_end"] - cfg["loss"]["w_aux_start"])
        loss, stats = compute_loss(out, parts, cfg["loss"], cfg["dense"], aux_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        if not torch.isfinite(loss):
            log.warning("step %d: non-finite loss, skipping update", step)
            continue
        opt.step()
        sched.step()
        last_step = step
        elapsed = time.time() - t0
        if step % tc["log_every"] == 0 or step == tc["steps"] - 1:
            stats = to_floats(stats)
            rec = {"step": step, "n": n, "lr": sched.get_last_lr()[0], "grad_norm": float(gn), "minutes": elapsed / 60, **stats}
            curve.append(rec)
            log.info("step %6d | N=%4d | loss %.4f | type %.3f acc %.3f | geo %.4f | gn %.2f | %.1f min",
                     step, n, stats["total"], stats["type"], stats["acc"], stats["geo"], float(gn), elapsed / 60)
        if (step + 1) % tc["val_every"] == 0 or step == tc["steps"] - 1:
            v = validate(model, val_set, cfg)
            curve.append({"step": step, "minutes": elapsed / 60, **v})
            log.info("VAL step %d | acc %.4f | teacher-forced chamfer %.4f", step, v["val_acc"], v["val_cd_teacher"])
        if (step + 1) % tc["ckpt_every"] == 0:
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, ckpt_dir / "last.pt")
        if elapsed / 60 > tc["max_minutes"]:
            log.info("wall-clock cap of %.0f min reached at step %d", tc["max_minutes"], step)
            break
    v = validate(model, val_set, cfg)
    curve.append({"step": last_step, "minutes": (time.time() - t0) / 60, **v})
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": last_step}, ckpt_dir / "final.pt")
    summary = {"steps_done": last_step + 1, "minutes": (time.time() - t0) / 60, "n_params": n_params, **v}
    with open(res_dir / "curve.json", "w") as f:
        json.dump(curve, f)
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("training done: %s", summary)
    return summary


def load_model(path: str | Path, device=None) -> tuple[ShapeFitter, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = ShapeFitter(cfg["model"], amp_bf16=cfg["train"]["amp_bf16"])
    model.load_state_dict(ck["model"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device).eval(), cfg
