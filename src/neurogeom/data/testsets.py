"""Fixed, cached evaluation sets: family x N-bin x noise-bin cells from a dedicated seed stream.

Global context: the preregistered test set has `test_per_cell` samples in every cell of
13 families x 4 point-count bins x 3 noise bins, drawn from the `test` stream (disjoint from training).
Samples are padded to the bin's maximum N with a mask, so batches mix point counts, which also
exercises the encoder's padding invariance. Ground truth is stored as dense samples (for Chamfer) and
as auxiliary targets (unsigned axis, log scale, discrete sub-choice) for parameter-error metrics.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neurogeom.data.nuisance import log_uniform
from neurogeom.data.synth import generate_family
from neurogeom.logging_utils import get_logger
from neurogeom.seeding import make_generator, stream_seed
from neurogeom.shapes.registry import FAMILIES

log = get_logger(__name__)
CHUNK = 25  # samples generated per call (each chunk has its own N)


def _pad(t: torch.Tensor, n: int, value=0.0) -> torch.Tensor:
    if t.shape[1] == n:
        return t
    pad = torch.full((t.shape[0], n - t.shape[1], *t.shape[2:]), value, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], 1)


def build_test_set(cfg: dict, stream: str = "test", per_cell: int | None = None, device="cuda") -> list[dict]:
    """Returns one padded batch dict per (N-bin, noise-bin) with all families mixed; cached on disk."""
    per_cell = per_cell or cfg["eval"]["test_per_cell"]
    cache = Path(cfg["paths"]["data_cache"]) / f"testset_{stream}_seed{cfg['seed']}_{per_cell}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    gen = make_generator(stream_seed(cfg["seed"], stream), device)
    cells = []
    for bi, (lo, hi) in enumerate(cfg["eval"]["n_bins"]):
        for si, (slo, shi) in enumerate(cfg["eval"]["noise_bins"]):
            parts = []
            for fi, f in enumerate(FAMILIES):
                left = per_cell
                while left > 0:
                    c = min(CHUNK, left)
                    left -= c
                    n = int(round(float(log_uniform(1, lo, hi, gen, device))))
                    sigma = log_uniform(c, slo, shi, gen, device)
                    d = generate_family(fi, c, n, cfg["synth"], cfg["dense"], gen, device, sigma=sigma)
                    axis, logscale = f.aux(d["gt"])
                    sel = d["gt"].get("k_idx", d["gt"].get("subtype"))
                    parts.append({
                        "x": _pad(d["x"], hi), "mask": _pad(d["mask"], hi, False), "family": d["family"],
                        "sigma": sigma, "n": torch.full((c,), n, device=device), "gt_dense": d["gt_dense"], "gt_w": d["gt_w"],
                        "gt_axis": axis if axis is not None else torch.zeros(c, 3, device=device),
                        "has_axis": torch.full((c,), axis is not None, device=device), "gt_logscale": logscale,
                        "gt_select": sel if sel is not None else torch.full((c,), -1, device=device),
                        "embed_dim": torch.full((c,), d["embed_dim"], device=device)})
            # dense GT sizes differ across families -> pad to the max with zero weight
            m = max(p["gt_dense"].shape[1] for p in parts)
            for p in parts:
                p["gt_dense"], p["gt_w"] = _pad(p["gt_dense"], m), _pad(p["gt_w"], m)
            cell = {k: torch.cat([p[k] for p in parts]).cpu() for k in parts[0]}
            cell.update({"n_bin": bi, "noise_bin": si})
            cells.append(cell)
            log.info("test cell N-bin %d noise-bin %d: %d samples", bi, si, cell["x"].shape[0])
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cells, cache)
    return cells


def iterate(cells: list[dict], batch: int, device, subset_per_cell: int | None = None):
    """Yield (cell, index tensor, batch dict on device). subset_per_cell keeps the first k per family."""
    for cell in cells:
        idx = torch.arange(cell["x"].shape[0])
        if subset_per_cell is not None:
            keep = torch.zeros_like(idx, dtype=torch.bool)
            for fi in range(len(FAMILIES)):
                pos = torch.nonzero(cell["family"] == fi).squeeze(-1)[:subset_per_cell]
                keep[pos] = True
            idx = idx[keep]
        for s in range(0, len(idx), batch):
            ii = idx[s:s + batch]
            yield cell, ii, {k: (v[ii].to(device) if torch.is_tensor(v) else v) for k, v in cell.items()}


def n_samples(cells: list[dict]) -> int:
    return sum(c["x"].shape[0] for c in cells)
