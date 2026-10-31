"""Reproducibility helpers: global seeding and disjoint seed streams.

Global context: training data is generated on the fly, so train / validation / test sets must come
from disjoint random streams to rule out leakage. Each stream gets its own torch.Generator whose
seed is derived from (base_seed, stream_name) and can never coincide across streams.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

# Fixed offsets keep the streams disjoint: training uses seeds near 0, val near 10^9, test near 2*10^9.
_STREAM_OFFSETS = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000, "ood": 3_000_000_000, "real": 4_000_000_000}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stream_seed(base_seed: int, stream: str, salt: str = "") -> int:
    """Deterministic seed for a named stream; a salt (e.g. a family name) gives sub-streams."""
    h = int(hashlib.sha256(f"{stream}|{salt}".encode()).hexdigest()[:6], 16)  # < 2^24, stays inside the stream band
    return _STREAM_OFFSETS[stream] + base_seed * 100_003 % 900_000_000 + h % 1_000


def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """torch.Generator on the given device, seeded deterministically."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
