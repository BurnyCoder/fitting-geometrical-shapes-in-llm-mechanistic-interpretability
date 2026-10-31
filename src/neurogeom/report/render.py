"""Render the shape catalog (catalog.yaml) to docs/shape_catalog.md.

Global context: deliverable 1 ("find as many empirically found shapes in neural geometry as
possible"). The YAML is the single source of truth; this renderer produces the human-readable table
linked from README.md, so no catalog fact is typed twice.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from neurogeom.config import REPO_ROOT
from neurogeom.logging_utils import get_logger

log = get_logger(__name__)
CATALOG = Path(__file__).resolve().parents[1] / "catalog" / "catalog.yaml"


def load_catalog() -> dict:
    with open(CATALOG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


def render_catalog(out: Path | None = None) -> Path:
    cat = load_catalog()
    out = out or REPO_ROOT / "docs" / "shape_catalog.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Catalog of empirically found shapes in neural geometry",
        "",
        "This file is generated from `src/neurogeom/catalog/catalog.yaml` by `uv run python run.py catalog`; do not edit it by hand.",
        "",
        "Starting point for the review: [Goodfire, *Neural geometry*](https://www.goodfire.com/research/neural-geometry). The review was run on 2026-10-31.",
        "",
        "**Column key**",
        "",
        "- *Fitter family*: the class our network predicts for this shape.",
        "- *Top-3 PCA*: whether the source shows the shape in the top-3 principal components of raw activations, which is the input format of our fitter.",
        "- *Verified*: how we checked the entry.",
        "",
        "| Shape | Concept | Model / layer | Method | Top-3 PCA | Fitter family | Source | Verified |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in cat["entries"]:
        src = f"[{_cell(e['citation'])}]({e['url']})"
        lines.append(f"| {_cell(e['shape'])} | {_cell(e['concept'])} | {_cell(e['models'])} | {_cell(e['method'])} | "
                     f"{_cell(e['top3_pca'])} | `{e['vocab_class']}` | {src} | {_cell(e['verified'])} |")
    lines += ["", "## Known confounder", ""]
    for c in cat["confounders"]:
        urls = ", ".join(f"<{c[k]}>" for k in ("url", "url2", "url3", "url4") if k in c)
        lines.append(f"- **{c['shape']}**, which the fitter represents as `{c['vocab_class']}`. Any smoothly varying ordinal "
                     f"variable produces this curve in PCA, so an arch or helix seen in the top PCs is not evidence of a curved representation on its own. "
                     f"Sources: {c['citation']}: {urls}")
    families = sorted({e["vocab_class"] for e in cat["entries"]})
    lines += ["", f"Fitter families with at least one empirical anchor in this catalog: {', '.join(f'`{f}`' for f in families)}.", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s (%d entries)", out, len(cat["entries"]))
    return out
