"""Training loss: family classification + shape-to-shape Chamfer + input residual + auxiliary terms.

Global context (user requirement): "in the loss evaluate how close the output shape is to the ground
truth synthetic shape". The main geometric term is therefore the symmetric Chamfer distance between
dense samples of the predicted shape (decoded from the ground-truth family's head) and dense samples
of the noise-free ground-truth shape, both in the normalised frame. Because Chamfer compares point
sets, it is invariant to every parameter symmetry (axis signs, phase, vertex order), so no
canonicalisation or Hungarian matching is needed in the loss. The residual term (input points ->
predicted shape) ties the fit to the observed points. Auxiliary axis/scale terms (annealed) help
escape Chamfer local minima such as a helix lying sideways (cf. SPFN's axis loss 1-|a.a*|,
Li et al. 2019; we use the smooth 1-(a.a*)^2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neurogeom.geometry.chamfer import chamfer, residual
from neurogeom.shapes.registry import FAMILIES


def select_of(p: dict) -> torch.Tensor | None:
    """Discrete sub-choice of point-set families (cluster count index / polytope subtype)."""
    if "k_idx" in p:
        return p["k_idx"]
    if "subtype" in p:
        return p["subtype"]
    return None


def param_space_loss(f, pred: dict, gt: dict) -> torch.Tensor:
    """Ablation A1: a parameter-space loss instead of Chamfer (center L2 + intrinsic differences).
    Lengths are compared in log space; point sets fall back to centroid Chamfer (their parameters ARE points)."""
    if not f.posed:
        a, wa = f.points(pred)
        b, wb = f.points(gt)
        return chamfer(a, wa, b, wb)
    loss = ((pred["center"] - gt["center"]) ** 2).sum(-1)
    for s in f.specs:
        if s.kind == "pos":
            loss = loss + (torch.log(pred[s.name] + 1e-6) - torch.log(gt[s.name] + 1e-6)) ** 2
        else:
            loss = loss + (pred[s.name] - gt[s.name]) ** 2
    return loss


def aux_loss(f, pred: dict, gt: dict) -> torch.Tensor:
    a_pred, l_pred = f.aux(pred)
    a_gt, l_gt = f.aux(gt)
    loss = (l_pred - l_gt).abs()
    if a_pred is not None:
        loss = loss + 1.0 - (a_pred * a_gt).sum(-1) ** 2      # unsigned axis: invariant to a -> -a
    return loss


def compute_loss(out: dict, parts: list[dict], cfg_loss: dict, cfg_dense: dict, aux_weight: float) -> tuple[torch.Tensor, dict]:
    """out: model output for the concatenated batch; parts: per-family generator outputs, in batch order."""
    fam = torch.cat([p["family"] for p in parts])
    l_type = F.cross_entropy(out["logits"], fam, label_smoothing=cfg_loss["label_smoothing"])
    # stats are kept as detached tensors (no GPU->CPU sync per step); callers convert when logging
    stats = {"type": l_type.detach(), "acc": (out["logits"].argmax(-1) == fam).float().mean()}
    geo_terms, start = [], 0
    for part in parts:
        b = part["x"].shape[0]
        sl = slice(start, start + b)
        start += b
        fi = int(part["family"][0])
        f = FAMILIES[fi]
        sel = select_of(part["gt"])
        pred = f.decode(out["raw"][fi][sl], sel)                 # teacher forcing on the discrete sub-choice
        pts, w = f.dense(pred, cfg_dense)
        if cfg_loss.get("param_only", False):
            l_shape = param_space_loss(f, pred, part["gt"])
        else:
            l_shape = chamfer(pts, w, part["gt_dense"], part["gt_w"])
        l_res = residual(part["x"], part["mask"], pts, w, cfg_loss["huber_delta"])
        l_aux = aux_loss(f, pred, part["gt"])
        term = cfg_loss["w_chamfer"] * l_shape + cfg_loss["w_residual"] * l_res + aux_weight * l_aux
        if sel is not None:
            term = term + cfg_loss["w_subtype"] * F.cross_entropy(pred["logits"], sel, reduction="none")
        geo_terms.append(term.mean())
        cd = l_shape if not cfg_loss.get("param_only", False) else chamfer(pts, w, part["gt_dense"], part["gt_w"])
        stats[f"cd/{f.name}"] = cd.detach().mean()
    l_geo = torch.stack(geo_terms).mean()                         # each family weighs equally
    stats["geo"] = l_geo.detach()
    total = cfg_loss["w_type"] * l_type + l_geo
    stats["total"] = total.detach()
    return total, stats


def to_floats(stats: dict) -> dict:
    return {k: float(v) for k, v in stats.items()}
