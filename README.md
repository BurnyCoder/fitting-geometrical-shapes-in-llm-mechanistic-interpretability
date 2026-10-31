# Fitting geometric shapes to neural geometry

**Status: work in progress.** This README is filled in with real results as the experiments run.

Interpretability research keeps finding **low-dimensional shapes** in the activations of neural networks. Examples:
- circles for days and months ([Engels et al. 2024](https://arxiv.org/abs/2405.14860));
- helices for token position and character counts ([Yedidia 2023](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix); [Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html));
- a paraboloid for colors and lines for temperature and age ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119));
- clock circles in grokked modular addition ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)).

These shapes are usually spotted and fitted **by hand**.

This repo does two things:
1. **Catalogs** the shapes found empirically so far, with citations, in [docs/shape_catalog.md](docs/shape_catalog.md).
2. Trains a **network on synthetic shapes only**. It takes a 3D point cloud (the **top-3 PCA coordinates of activations**) and outputs:
   - the **shape family** (13 families, each with empirical support), and
   - **parameters that specify the shape uniquely**, after canonicalizing the shape's symmetries.

   The network is then evaluated on held-out synthetic data and on **real activations**: GPT-2, Llama-3.1-8B, a grokked modular-addition transformer, and toy models of superposition. It is compared against classical baselines, controls and ablations.

The hypotheses, metrics and falsification criteria were preregistered before any training: [reports/preregistration.md](reports/preregistration.md).

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run python run.py all --config configs/smoke.yaml
```

More detail will follow as the pipeline is completed.
