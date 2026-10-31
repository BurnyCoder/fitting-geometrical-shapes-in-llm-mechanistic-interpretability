"""Configuration loading: .env (secrets/paths) + YAML experiment configs.

Global context: every phase of run.py receives one merged config dict. configs/base.yaml holds the
defaults; configs/{smoke,main,...}.yaml override a subset (merged recursively), so an ablation is a
small YAML file instead of a code change.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repository root = two levels above this file (src/neurogeom/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_env() -> None:
    """Load .env from the repo root into os.environ (python-dotenv; existing env vars win)."""
    load_dotenv(REPO_ROOT / ".env", override=False)
    # HF_HOME must be set before transformers/huggingface_hub are imported to take effect.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (dicts merge, everything else replaces)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Return configs/base.yaml merged with the given YAML file (if any) and an override dict."""
    load_env()
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path is not None:
        path = Path(path)
        if not path.is_absolute() and not path.exists():
            path = CONFIG_DIR / path  # allow "smoke.yaml" shorthand
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
        cfg["config_name"] = path.stem
    else:
        cfg["config_name"] = "base"
    if overrides:
        cfg = deep_merge(cfg, overrides)
    # Resolve output directories relative to the repo root so phases can be run from anywhere.
    run_name = cfg.get("run_name") or cfg["config_name"]
    cfg["run_name"] = run_name
    cfg["paths"] = {
        "results": str(REPO_ROOT / "results" / run_name),
        "checkpoints": str(REPO_ROOT / "checkpoints" / run_name),
        "logs": str(REPO_ROOT / "logs"),
        "data_cache": str(REPO_ROOT / "data_cache"),
        "reports": str(REPO_ROOT / "reports"),
    }
    return cfg
