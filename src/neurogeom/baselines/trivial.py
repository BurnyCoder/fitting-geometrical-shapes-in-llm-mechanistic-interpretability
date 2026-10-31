"""Baseline B3 (trivial) and random controls.

Global context: floors that any useful fitter must beat.
- B3a "null": always predict the blob family (the chance-level majority on a balanced test set) and
  fit a Gaussian blob classically.
- B3b "always-circle": always fit a full circle/arc (the most commonly reported neural shape).
- B3c "random": a random family with random (unfitted) parameters at the cloud's scale.
"""

from __future__ import annotations

import torch

from neurogeom.baselines.classical import ARC, BLOB, fit_one_family
from neurogeom.shapes.registry import FAMILIES


def fit_null(x, mask, steps, back_w, gen=None) -> dict:
    return fit_one_family(BLOB, x, mask, steps, back_w, gen)


def fit_always_circle(x, mask, steps, back_w, gen=None) -> dict:
    return fit_one_family(ARC, x, mask, steps, back_w, gen)


def random_shapes(b: int, gen: torch.Generator, device) -> list[tuple[int, dict]]:
    """Per sample: (family index, params) of a random family drawn with the generator's ranges."""
    fams = torch.randint(0, len(FAMILIES), (b,), generator=gen, device=device)
    out = []
    for i in range(b):
        f = FAMILIES[int(fams[i])]
        p = f.sample_valid(1, gen, torch.zeros(1, device=device), device) if f.posed else f.sample_valid(1, gen, torch.zeros(1, device=device), device, k_max=6)
        out.append((int(fams[i]), p))
    return out
