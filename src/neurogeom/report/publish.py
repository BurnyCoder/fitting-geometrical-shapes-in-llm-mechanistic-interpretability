"""Upload trained fitters to the Hugging Face Hub and add them to the project collection.

Global context: user requirement "upload models to Hugging Face into a collection; one collection for
this research". Each run's checkpoint (final.pt: state dict + full config) and its results summary are
uploaded with a model card generated from results files. Uses huggingface_hub's HfApi
(https://huggingface.co/docs/huggingface_hub/guides/upload, .../guides/collections).
"""

from __future__ import annotations

import json
from pathlib import Path

from neurogeom.config import load_env
from neurogeom.logging_utils import get_logger

log = get_logger(__name__)
COLLECTION_TITLE = "Fitting geometric shapes to neural geometry"
GITHUB = "https://github.com/BurnyCoder/geometrical-shape-fitting-in-llm-mechanistic-interpretability"


def model_card(run: str, summary: dict, analysis: dict | None) -> str:
    acc = ""
    if analysis and "method" in analysis.get("systems", {}):
        m = analysis["systems"]["method"]
        acc = (f"\n## Synthetic test results (method)\n\n- fine family accuracy: {100 * m['fine_acc'][0]:.1f}% "
               f"(95% CI {100 * m['fine_acc'][1]:.1f}-{100 * m['fine_acc'][2]:.1f}), n = {m['n']}\n"
               f"- median Chamfer distance to ground-truth shape: {m['median_chamfer'][0]:.3f} (RMS-radius units)\n")
    return f"""---
license: mit
tags: [interpretability, neural-geometry, point-cloud, shape-fitting, set-transformer]
---
# Neural-geometry shape fitter: run `{run}`

This is a Set Transformer trained **only on synthetic shapes**. Its input is a 3D point cloud: the top-3 PCA coordinates of a set of activations. It outputs one of 13 shape families and parameters that specify the shape uniquely. The 13 families are blob, clusters, polytope, line, arc/circle, ellipse, helix, PCA-ringing curve, disc, sphere, paraboloid, cone/frustum and torus.

- Code, paper and reports: {GITHUB}
- Training summary: {json.dumps(summary)}
{acc}
## Usage

```python
from neurogeom.train.loop import load_model
from neurogeom.infer.predict import fit_cloud, describe_fit
model, cfg = load_model("final.pt")
fit, frame, _ = fit_cloud(model, activations, cfg["infer"])   # activations: (N, D) numpy array
print(describe_fit(fit, frame))
```

Author: Libor "Burny" Burian. The code was written with Claude Code.
"""


def upload_run(cfg: dict, repo_id: str, analysis_path: Path | None = None) -> str:
    load_env()
    from huggingface_hub import HfApi
    api = HfApi()
    ck = Path(cfg["paths"]["checkpoints"]) / "final.pt"
    res = Path(cfg["paths"]["results"])
    summary = json.load(open(res / "train" / "summary.json"))
    analysis = json.load(open(analysis_path)) if analysis_path and analysis_path.exists() else None
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    card = res / "README_hf.md"
    card.write_text(model_card(cfg["run_name"], summary, analysis), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(ck), path_in_repo="final.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id)
    for p in [res / "train" / "summary.json", res / "train" / "curve.json", res / "analysis.json"]:
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"results/{p.name}", repo_id=repo_id)
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
