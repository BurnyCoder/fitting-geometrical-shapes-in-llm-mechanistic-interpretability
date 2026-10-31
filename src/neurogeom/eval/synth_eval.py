"""Phase eval-synth: method, ablated selection/refinement and baselines on the fixed synthetic test set.

Global context: produces the per-sample table results/<run>/synth/per_sample.csv (one row per sample
and system) plus probabilities for calibration analysis. Systems:
  method        network + refinement + calibrated classifier selection (preregistered method)
  A3_norefine   network output without test-time refinement
  A5_gaic       refined top-k candidates, selection by geometric AIC
  A5_hybrid     refined top-k candidates, geometric AIC among plausible classes
  B1_classical  classical fit of every family + geometric AIC          (subset)
  B2_features   hand-crafted features + gradient boosting, then B1 fit  (subset)
  B3a_null      always blob;  B3b_circle  always circle;  B3c_random  random shape   (subset)
Baselines run on the first `baseline_per_cell` samples per family of every cell (classical fitting of
13 families with multi-start is ~60x the cost of the method); all paired comparisons use that subset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurogeom.baselines import classical, features_gbm, trivial
from neurogeom.data.testsets import build_test_set, iterate, n_samples
from neurogeom.eval.metrics import shape_metrics, sample_meta
from neurogeom.eval.stats import fit_temperature
from neurogeom.infer.predict import fit_batch
from neurogeom.logging_utils import get_logger
from neurogeom.seeding import make_generator, stream_seed
from neurogeom.shapes.registry import FAMILIES, INDEX

log = get_logger(__name__)


def batch_size_for(cell: dict) -> int:
    return 512 if cell["x"].shape[1] <= 256 else 128


@torch.no_grad()
def calibrate_temperature(model, cfg, device) -> float:
    cells = build_test_set(cfg, stream="val", per_cell=cfg["eval"].get("val_per_cell", 10), device=device)
    logits, labels = [], []
    for cell, _, b in iterate(cells, 512, device):
        logits.append(model(b["x"], b["mask"])["logits"].cpu())
        labels.append(b["family"].cpu())
    t = fit_temperature(torch.cat(logits), torch.cat(labels))
    log.info("temperature fitted on %d validation samples: T = %.3f", sum(len(l) for l in labels), t)
    return t


def _alt_choice(fit, rule: str):
    """Re-select among the refined candidates of a Fit with another rule (A5 ablation)."""
    fams = [c[0] for c in fit.cand]
    aic = torch.tensor([c[2] for c in fit.cand])
    pc = torch.tensor([fit.probs[f] for f in fams])
    if rule == "gaic":
        s = int(aic.argmin())
    else:
        ok = pc >= 0.1 * pc.max()
        s = int(aic.masked_fill(~ok, float("inf")).argmin())
    return fams[s], fit.cand_raws[s]


def eval_method(model, cfg, cells, device, temperature: float, name: str = "method", variants: bool = True,
                subset_per_cell: int | None = None) -> tuple[list, dict]:
    rows, probs, labels = [], [], []
    t0 = time.time()
    done = 0
    for cell, ii, b in iterate(cells, 512, device, subset_per_cell=subset_per_cell):
        bs = batch_size_for(cell)
        for s in range(0, len(ii), bs):
            sl = slice(s, s + bs)
            sub = {k: (v[sl] if torch.is_tensor(v) else v) for k, v in b.items()}
            fits = fit_batch(model, sub["x"], sub["mask"], cfg["infer"], temperature)
            meta = sample_meta(cell, ii[sl], sub)
            systems = {name: ([f.family for f in fits], [f.raw for f in fits])}
            if variants:
                for rule in ("gaic", "hybrid"):
                    ch = [_alt_choice(f, rule) for f in fits]
                    systems[f"A5_{rule}"] = ([c[0] for c in ch], [c[1] for c in ch])
                nr = fit_batch(model, sub["x"], sub["mask"], dict(cfg["infer"], refine=False), temperature)
                systems["A3_norefine"] = ([f.family for f in nr], [f.raw for f in nr])
            for sys_name, (fam, raws) in systems.items():
                for m, r, f in zip(meta, shape_metrics(fam, raws, sub), fits):
                    rows.append({"system": sys_name, **m, **r, "prob_pred": float(f.probs[INDEX[r["family_pred"]]])
                                 if sys_name == name else np.nan})
            probs.append(np.stack([f.probs for f in fits]))
            labels.append(sub["family"].cpu().numpy())
            done += len(fits)
        log.info("%s: %d / %d samples (%.1f min)", name, done, n_samples(cells), (time.time() - t0) / 60)
    return rows, {"probs": np.concatenate(probs), "labels": np.concatenate(labels)}


def eval_baselines(cfg, cells, device, gbm, per_cell: int) -> list:
    rows = []
    back_w = cfg["infer"]["refine_back_weight"]
    gen = make_generator(stream_seed(cfg["seed"], "test", salt="baselines"), device)
    t0 = time.time()
    # Classical fitting is kernel-launch bound like everything else here: large batches cost about the same
    # wall time per batch as small ones (256 clouds for N <= 256, 64 for the largest bin to stay within 8 GB).
    for cell, ii, b in iterate(cells, 256, device, subset_per_cell=per_cell):
        bs = 256 if cell["x"].shape[1] <= 256 else 64
        for s0 in range(0, len(ii), bs):
            sl = slice(s0, s0 + bs)
            b_sub = {k: (v[sl] if torch.is_tensor(v) else v) for k, v in b.items()}
            rows += _baseline_batch(cfg, cell, ii[sl], b_sub, gbm, gen, back_w)
            log.info("baselines: cell N-bin %d noise-bin %d, %d clouds done (%.1f min)", cell["n_bin"], cell["noise_bin"],
                     len(ii[sl]), (time.time() - t0) / 60)
    return rows


def _baseline_batch(cfg, cell, ii, b, gbm, gen, back_w) -> list:
    """B1, B2 and B3 on one batch of test clouds; returns per-sample metric rows."""
    ec = cfg["eval"]
    meta = sample_meta(cell, ii, b)
    x, m = b["x"], b["mask"]
    # B1: classical fit of every family + geometric AIC
    cands, sel = classical.fit_classical(x, m, ec["classical_steps"], back_w, gen, restarts=ec["classical_restarts"])
    ci = sel["cand_index"].tolist()
    b1 = ([cands[c]["fi"] for c in ci], [cands[c]["raw"][i] for i, c in enumerate(ci)])
    amb = ambiguity(cands, b["family"])
    # B2: features + GBM, then the classical fit of the predicted family (reusing B1's per-family fits)
    pred = gbm.predict(features_gbm.features_batch(x, m))
    b2 = ([int(p) for p in pred], [best_cand_for(cands, int(p), i, m)["raw"] for i, p in enumerate(pred)])
    # B3: null (blob), always-circle, random shape
    n = len(ii)
    b3a = ([INDEX["blob"]] * n, [best_cand_for(cands, INDEX["blob"], i, m)["raw"] for i in range(n)])
    b3b = ([INDEX["arc"]] * n, [best_cand_for(cands, INDEX["arc"], i, m)["raw"] for i in range(n)])
    rnd = trivial.random_shapes(n, gen, x.device)
    b3c = ([fi for fi, _ in rnd], [_random_raw(fi, p) for fi, p in rnd])
    rows = []
    for name, (fam, raws) in {"B1_classical": b1, "B2_features": b2, "B3a_null": b3a, "B3b_circle": b3b, "B3c_random": b3c}.items():
        for k, (mm, r) in enumerate(zip(meta, shape_metrics(fam, raws, b))):
            rows.append({"system": name, **mm, **r, "prob_pred": np.nan, "ambiguous": bool(amb[k])})
    return rows


def best_cand_for(cands: list[dict], fi: int, i: int, mask) -> dict:
    """Among B1's candidates of family fi (clusters: one per k; polytope: one per subtype), the one with
    the lowest refinement objective for sample i."""
    cs = [c for c in cands if c["fi"] == fi]
    j = int(np.argmin([float(c["obj"][i]) for c in cs]))
    return {"raw": cs[j]["raw"][i]}


def ambiguity(cands: list[dict], fam_true: torch.Tensor) -> np.ndarray:
    """A sample is 'ambiguous' if a family with no more free parameters than the true one fits at least
    as well (objective <= 1.05 x the true family's best objective + 1e-3), excluding the blob."""
    obj = torch.stack([c["obj"] for c in cands], -1)
    p = torch.stack([c["p"] for c in cands], -1)
    fi = torch.tensor([c["fi"] for c in cands], device=obj.device)
    out = []
    for i in range(obj.shape[0]):
        t = int(fam_true[i])
        own = obj[i][fi == t].min()
        p_own = p[i][fi == t].min()
        other = (fi != t) & (fi != INDEX["blob"]) & (p[i] <= p_own) & (obj[i] <= 1.05 * own + 1e-3)
        out.append(bool(other.any()))
    return np.array(out)


def _random_raw(fi: int, p: dict) -> torch.Tensor:
    f = FAMILIES[fi]
    rms = f.rms_radius(p)                                    # rescale the random shape to the cloud's RMS radius 1
    dev = rms.device
    p = f.transform(p, torch.eye(3, device=dev).unsqueeze(0), torch.zeros(1, 3, device=dev), 1.0 / rms)
    return f.encode(p)[0]


def run_synth_eval(cfg: dict, baselines: bool | None = None) -> dict:
    from neurogeom.train.loop import load_model
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg["paths"]["results"]) / "synth"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(Path(cfg["paths"]["checkpoints"]) / "final.pt", device)
    temperature = calibrate_temperature(model, cfg, device)
    cells = build_test_set(cfg, device=device)
    log.info("test set: %d samples", n_samples(cells))
    ec = cfg["eval"]
    baselines = ec.get("baselines", True) if baselines is None else baselines
    rows, pr = eval_method(model, cfg, cells, device, temperature, variants=ec.get("variants", True),
                           subset_per_cell=ec.get("method_subset_per_cell"))
    np.savez_compressed(out_dir / "probs.npz", **pr)
    if baselines:
        gbm_path = Path(cfg["paths"]["data_cache"]) / f"gbm_seed{cfg['seed']}.pkl"
        if gbm_path.exists():
            import pickle
            gbm = pickle.load(open(gbm_path, "rb"))
        else:
            import pickle
            gbm = features_gbm.train_gbm(cfg, per_family=cfg["eval"].get("gbm_per_family", 1500), device=device)
            gbm_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(gbm, open(gbm_path, "wb"))
        rows += eval_baselines(cfg, cells, device, gbm, cfg["eval"]["baseline_per_cell"])
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    meta = {"temperature": temperature, "n_test": n_samples(cells), "checkpoint": str(Path(cfg["paths"]["checkpoints"]) / "final.pt")}
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    log.info("eval-synth done: %d rows -> %s", len(df), out_dir)
    return meta
