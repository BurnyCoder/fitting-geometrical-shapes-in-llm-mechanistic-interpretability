"""Real datasets from LLM forward passes (Llama-3.1-8B, Pythia-410m) and their random-init controls.

Global context (expected shapes are preregistered in reports/preregistration.md):
- R4 counting manifold: mean hidden state per "characters since the last newline" on text hard-wrapped
  to 150 columns, layer 5 (Gurnee et al. 2025, transformer-circuits.pub/2025/linebreaks; open-model
  replication corl-team/counting_manifolds: Llama-3.1-8B L5). Text: NeelNanda/pile-10k (FineWeb in the
  replication; documented deviation).
- R5a/b weekdays / months at the day/month token, layer 5, Engels et al. 2024 templates
  ("Let's do some days of the week math. {Two days} from {Monday} is", "Let's do some calendar math.
  {Four months} from {January} is"; repo JoshEngels/MultiDimensionalFeatures).
- R6 years, R7 colors, R8a temperature, R8b age: last token, layer 19, Bhalla et al. 2026 Table 1
  templates; value ranges are not given in the paper and are chosen here (documented in meta).
- R5c days with times ("It's {time} on {day}", Bhalla et al. Table 1; exploratory, not preregistered).
- R9 truth: geometry-of-truth `cities` statements (Marks & Tegmark 2024), last token, layer 12.
"""

from __future__ import annotations

import colorsys
import io
import textwrap
import urllib.request

import numpy as np
import torch

from neurogeom.logging_utils import get_logger
from neurogeom.realdata.common import free_cuda, hf_env, last_token_acts, load_truncated, log_prompts, save_cloud, token_acts

log = get_logger(__name__)
LLAMA = "meta-llama/Llama-3.1-8B"
NUM_WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


# ------------------------------------------------------------------ prompt sets
def calendar_prompts() -> dict:
    days = [(f"Let's do some days of the week math. {NUM_WORDS[k]} day{'s' if k != 1 else ''} from {d} is", d)
            for d in DAYS for k in range(1, 8)]
    months = [(f"Let's do some calendar math. {NUM_WORDS[k]} month{'s' if k != 1 else ''} from {m} is", m)
              for m in MONTHS for k in range(1, 13)]
    return {"days": days, "months": months}


def bhalla_prompts() -> dict:
    years = [(f"The date is year {y}", y) for y in range(1801, 2000)]                       # 199 years (n = 199 in Table 1)
    temps = [(f"Today it's {f} degrees Fahrenheit outside", f) for f in range(-20, 130)]       # 150 values (n = 150)
    ages = [(f"They are {a} years old.", a) for a in range(1, 100)]                            # 99 values (n = 99)
    colors = []
    for h in range(36):
        for s in (0.4, 0.55, 0.7, 0.85, 1.0):
            for lum in (0.25, 0.375, 0.5, 0.625, 0.75):
                r, g, b = colorsys.hls_to_rgb(h / 36, lum, s)
                hexc = f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"
                colors.append((f"The hex code {hexc} is for the color", f"{h * 10}|{s}|{lum}"))       # 900 colors (n ~ 900)
    times = [f"{h}:{m:02d} {ap}" for ap in ("AM", "PM") for h in range(1, 13) for m in (0, 30)]
    days = [(f"It's {t} on {d}", d) for d in DAYS for t in times]                               # 7 x 48 = 336
    return {"years": years, "temps": temps, "ages": ages, "colors": colors, "days_time": days}


def truth_prompts() -> list:
    url = "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv"
    import pandas as pd
    df = pd.read_csv(io.StringIO(urllib.request.urlopen(url, timeout=60).read().decode()))
    return list(zip(df["statement"].tolist(), df["label"].astype(int).tolist()))


def counting_texts(n_docs: int = 120, width: int = 150, max_chars: int = 3000) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    texts = []
    for row in ds:
        t = " ".join(row["text"].split())                     # remove existing newlines / whitespace runs
        if len(t) < 1500 or not t.isascii():
            continue
        texts.append(textwrap.fill(t[:max_chars], width=width))
        if len(texts) >= n_docs:
            break
    return texts


@torch.no_grad()
def counting_acts(model, tok, texts: list[str], layer: int, width: int = 150) -> tuple[np.ndarray, np.ndarray]:
    """Mean hidden state per character count since the last newline (1..width), over all tokens."""
    sums, cnts = None, np.zeros(width + 1)
    for t in texts:
        enc = tok(t, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=1024)
        offs = enc.pop("offset_mapping")[0]
        hs = model(**enc.to("cuda"), output_hidden_states=True).hidden_states[layer][0].float().cpu().numpy()
        if sums is None:
            sums = np.zeros((width + 1, hs.shape[1]))
        for i, (s, e) in enumerate(offs.tolist()):
            if e == 0 or "\n" in t[s:e]:
                continue
            c = e - (t.rfind("\n", 0, e) + 1)                     # characters since last newline, incl. this token
            if 1 <= c <= width:
                sums[c] += hs[i]
                cnts[c] += 1
    keep = cnts > 0
    log.info("counting: %d tokens, counts covered %d/%d", int(cnts.sum()), int(keep[1:].sum()), width)
    return sums[keep] / cnts[keep, None], np.arange(width + 1)[keep]


# ------------------------------------------------------------------ extraction per model/layer
def llama_layer5(cfg: dict, random_init: bool = False) -> None:
    tag = "C3" if random_init else "R"
    model, tok = load_truncated(LLAMA, 5, random_init=random_init)
    texts = counting_texts()
    log_prompts("counting (first 200 chars per document)", [t[:200] for t in texts])
    x, lab = counting_acts(model, tok, texts, 5)
    base = {"model": LLAMA + (" (random init)" if random_init else ""), "layer": 5, "random_init": random_init}
    save_cloud(cfg, f"{tag}4_llama_counting" if not random_init else "C3_llama_counting_init", x, lab,
               {**base, "source": "mean hidden state per char count since newline, pile-10k wrapped at 150",
                "expected_coarse": "3d" if random_init else "1d", "expected_fine": ["blob"] if random_init else ["helix"], "prereg": True})
    cal = calendar_prompts()
    for key, rid in (("days", "5a_llama_weekdays"), ("months", "5b_llama_months")):
        prompts, targets = [p for p, _ in cal[key]], [t for _, t in cal[key]]
        log_prompts(key, prompts)
        x = token_acts(model, tok, prompts, targets, 5)
        did = f"R{rid}" if not random_init else f"C3_{rid[3:]}_init"
        meta = {**base, "source": f"Engels et al. template, hidden state at the {key[:-1]} token", "template": prompts[0],
                "expected_coarse": "3d" if random_init else "1d", "expected_fine": ["blob"] if random_init else ["arc"], "prereg": True}
        save_cloud(cfg, did, x, targets, meta)
        if not random_init:  # class means: 7 / 12 points, as in the original circle plots
            u = sorted(set(targets), key=targets.index)
            means = np.stack([x[[t == v for t in targets]].mean(0) for v in u])
            save_cloud(cfg, did + "_means", means, u, {**meta, "source": meta["source"] + " (mean per class)"})
    del model
    free_cuda()


def llama_layer19(cfg: dict) -> None:
    model, tok = load_truncated(LLAMA, 19)
    sets = bhalla_prompts()
    spec = {"years": ("R6_llama_years", "1d", ["helix"], True), "colors": ("R7_llama_colors", "2d", ["paraboloid"], True),
            "temps": ("R8a_llama_temperature", "1d", ["line"], True), "ages": ("R8b_llama_age", "1d", ["line"], True),
            "days_time": ("R5c_llama_days_time", "1d", ["arc"], False)}
    for key, (did, coarse, fine, pre) in spec.items():
        prompts, labels = [p for p, _ in sets[key]], [l for _, l in sets[key]]
        log_prompts(key, prompts)
        x = last_token_acts(model, tok, prompts, 19, batch=32)
        save_cloud(cfg, did, x, labels, {"model": LLAMA, "layer": 19, "token": "last", "template": prompts[0],
                                         "source": "Bhalla et al. 2026 Table 1 template", "expected_coarse": coarse,
                                         "expected_fine": fine, "prereg": pre})
    del model
    free_cuda()


def llama_layer12_truth(cfg: dict) -> None:
    model, tok = load_truncated(LLAMA, 12)
    data = truth_prompts()
    prompts, labels = [p for p, _ in data], [l for _, l in data]
    log_prompts("truth cities", prompts)
    x = last_token_acts(model, tok, prompts, 12, batch=32)
    save_cloud(cfg, "R9_llama_truth_cities", x, labels, {"model": LLAMA, "layer": 12, "token": "last",
                                                        "source": "geometry-of-truth cities.csv (Marks & Tegmark 2024)",
                                                        "expected_coarse": "0d", "expected_fine": ["clusters"], "prereg": True})
    del model
    free_cuda()


def pythia_counting(cfg: dict) -> None:
    hf_env()
    model, tok = load_truncated("EleutherAI/pythia-410m", 6, max_gpu_gib=4)
    texts = counting_texts()
    x, lab = counting_acts(model, tok, texts, 6)
    save_cloud(cfg, "R4b_pythia_counting", x, lab, {"model": "EleutherAI/pythia-410m", "layer": 6,
                                                   "source": "mean hidden state per char count since newline, pile-10k wrapped at 150",
                                                   "expected_coarse": "1d", "expected_fine": ["helix"], "prereg": False})
    del model
    free_cuda()
