"""Upload trained fitters to the Hugging Face Hub and add them to the project collection.

Global context: user requirement "upload models to Hugging Face into a collection; one collection for
this research". Each run's checkpoint (final.pt: state dict + full config) and its results summary are
uploaded with a model card generated from results files. Uses huggingface_hub's HfApi
(https://huggingface.co/docs/huggingface_hub/guides/upload, .../guides/collections).
"""

from __future__ import annotations

import json
from pathlib import Path

from neurogeom.config import REPO_ROOT, load_env
from neurogeom.logging_utils import get_logger

log = get_logger(__name__)
COLLECTION_TITLE = "Fitting geometric shapes to neural geometry"
GITHUB = "https://github.com/BurnyCoder/geometrical-shape-fitting-in-llm-mechanistic-interpretability"


PAGES = "https://burnycoder.github.io/geometrical-shape-fitting-in-llm-mechanistic-interpretability/"


def model_card(run: str, summary: dict, headline: str, temperature: float) -> str:
    """Model card: what the model does, the generated headline results (positive and negative), usage, links."""
    return f"""---
license: mit
tags: [interpretability, neural-geometry, point-cloud, shape-fitting, set-transformer]
---
# Neural-geometry shape fitter: run `{run}`

This is a Set Transformer trained **only on synthetic shapes**. Its input is a 3D point cloud: the top-3 PCA coordinates of a set of activations. It outputs one of 13 shape families and parameters that specify the shape uniquely. The 13 families are blob, clusters, polytope, line, arc/circle, ellipse, helix, PCA-ringing curve, disc, sphere, paraboloid, cone/frustum and torus.

- Code, preregistration and reports: {GITHUB}
- Paper: `paper.pdf` in this repository
- Interactive explorer of real and synthetic fits: {PAGES}

## Results (generated from the results files)

{headline}

Results describe the shape of the top-3 PCA projection, not the intrinsic manifold. See the paper for the failure analysis on real activations.

## Training summary

`{json.dumps(summary)}`

## Usage

```python
from neurogeom.train.loop import load_model
from neurogeom.infer.predict import fit_cloud, describe_fit
model, cfg = load_model("final.pt")
fit, frame, _ = fit_cloud(model, activations, cfg["infer"], temperature={temperature:.4f})   # activations: (N, D) numpy array
print(describe_fit(fit, frame))
```

Author: Libor "Burny" Burian.
"""


def upload_run(cfg: dict, repo_id: str) -> str:
    load_env()
    from huggingface_hub import HfApi
    api = HfApi()
    ck = Path(cfg["paths"]["checkpoints"]) / "final.pt"
    res = Path(cfg["paths"]["results"])
    summary = json.load(open(res / "train" / "summary.json"))
    headline = (REPO_ROOT / "reports" / "generated" / "headline.md").read_text(encoding="utf-8").strip()
    temperature = json.load(open(res / "synth" / "meta.json"))["temperature"]
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    card = res / "README_hf.md"
    card.write_text(model_card(cfg["run_name"], summary, headline, temperature), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(ck), path_in_repo="final.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id)
    for p in [res / "train" / "summary.json", res / "train" / "curve.json", res / "analysis.json", res / "synth" / "meta.json",
              res / "real" / "real_summary.csv", res / "real" / "real_results.json"]:
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"results/{p.name}", repo_id=repo_id)
    pdf = REPO_ROOT / "paper" / "main.pdf"
    if pdf.exists():
        api.upload_file(path_or_fileobj=str(pdf), path_in_repo="paper.pdf", repo_id=repo_id)
    log.info("uploaded %s -> https://huggingface.co/%s", cfg["run_name"], repo_id)
    return repo_id


def add_to_collection(repo_ids: list[str], owner: str = "BurnyCoder") -> str:
    load_env()
    from huggingface_hub import HfApi
    api = HfApi()
    existing = [c for c in api.list_collections(owner=owner, limit=100) if c.title == COLLECTION_TITLE]
    coll = existing[0] if existing else api.create_collection(title=COLLECTION_TITLE, namespace=owner, private=False,
                                                              description="Learned shape-type + parameter fitting for neural-geometry point clouds")
    for rid in repo_ids:
        api.add_collection_item(coll.slug, item_id=rid, item_type="model", exists_ok=True)
    log.info("collection: https://huggingface.co/collections/%s", coll.slug)
    return coll.slug
