# Methodology and design choices

This document records each design decision and the reason for it. Results are in [../reports/summary.md](../reports/summary.md). The preregistered plan is in [../reports/preregistration.md](../reports/preregistration.md).

## 1. Problem definition
- **Input:** a 3D point cloud, given as the top-3 PCA coordinates of N activation vectors. N ranges from 5 to 1024.
- **Output:**
  - one of 13 shape families;
  - a parameter vector that specifies the shape **uniquely**;
  - the shape should pass as close as possible to the points.
- **Training data:** synthetic only.
- **Loss:** compares the predicted shape to the ground-truth shape.

## 2. Why these 13 families
The vocabulary comes from the literature catalog in [shape_catalog.md](shape_catalog.md).

Shapes kept, because they have empirical support and are visible in 3D:
- circle or arc, helix, line;
- sphere or cap, paraboloid, cone/frustum, torus, disc;
- clusters, polytope (six templates: the regular triangle, square, pentagon, tetrahedron and octahedron, and the square antiprism, which is uniform but not regular).

Three more families fill specific roles:
- **ellipse:** the image of a circle under the linear distortions that PCA and finite samples introduce;
- **blob:** the null class;
- **ringing:** the PCA horseshoe confounder.

Shapes left out, because they need four or more dimensions or are not well-defined as 3D primitives:
- tree;
- fractal in a simplex;
- Klein bottle;
- flat torus;
- nested shells.

Generalization families were chosen so that nearby cases are already covered:
- the frustum covers both cone and cylinder;
- the quadric sheet covers bowl, trough and saddle;
- the arc includes the full circle.

## 3. Uniqueness: canonical forms
Several parameter vectors can describe the same point set. Examples: flipping the sign of an axis, shifting the phase of a closed curve, reordering vertices. Each family defines `canonicalize` (in `src/neurogeom/shapes/*.py`), which picks one representative:

| Ambiguity | Rule |
|---|---|
| Sign of an axis | The largest-magnitude component of the axis is made positive (`dominant_positive`). |
| Continuous rotation about a symmetry axis (full circle, sphere, cone, torus) | e1 is set to the projection of the first PCA axis onto the plane orthogonal to the symmetry axis, or of the second PCA axis when the first is nearly parallel to the symmetry axis (\|e3·PC1\| > 0.9) (`frame_from_axis`). A closed sphere keeps the identity frame. |
| Point sets | Vertices or centroids are sorted by PC1, with approximate tie-breaks on PC2 and PC3 (sort key x·10⁶ + y·10³ + z). |
| Helix handedness | The signed pitch is stored relative to the canonical axis. Handedness is only meaningful for 3D inputs: for a projection from R^D the PCA frame may be mirrored, so the sign of the reported pitch carries no information there (all real helix fits came out left-handed). |

Tests (`tests/test_shapes.py`) check, for all 13 families, that `canonicalize` is idempotent and gives the same output under the symmetry generators listed in the tests.

**Training does not use canonical forms.** The Chamfer loss compares point sets, so it is invariant to all of these symmetries automatically. SPFN ([Li et al. 2019](https://arxiv.org/abs/1811.08988)) instead uses an axis loss on \|a·â\|, and ParSeNet ([Sharma et al. 2020](https://arxiv.org/abs/2003.12181)) a control-point loss minimised over the flips and swaps of each patch grid.

## 4. Nested families and label margins
Many families contain simpler families as limiting cases:
- an ellipse with a = b is a circle;
- a helix with pitch 0 is a circle;
- a ringing curve with a2 = a3 = 0 is a line;
- a polytope is a special case of clusters.

To keep labels well defined, each family samples ground-truth parameters only outside margins that grow with the noise level σ:

| Family | Margin |
|---|---|
| arc | sagitta ≥ max(0.1·chord, 3σ) |
| ellipse | b/a ≤ 1 − max(0.15, 3σ) |
| helix | height ≥ max(0.5r, 3σ) and r ≥ 0.1·height |
| ringing | a2 ≥ max(0.25·a1, 3σ) |
| paraboloid | κ1ρ ≥ 0.6 (≥ 1.5 for near-umbilic bowls) |
| clusters | layouts close to a polytope template are rejected |

Some residual ambiguity remains. B1's per-family fits measure it: a test sample counts as ambiguous when a family with no more parameters fits it equally well. Results are reported both with and without these samples.

## 5. Synthetic generator: nuisances and why

| Nuisance | Range | Why |
|---|---|---|
| N | log-uniform 5–1024 | Real clouds range from 7 (weekday means) to 1023 (GPT-2 positions). |
| Noise σ | log-uniform 0.003–0.15 of the RMS radius; 10% noise-free | Real activations range from nearly clean (W_pos) to very noisy (prompt variation). |
| Anisotropic noise | 30% of samples | Real noise is not isotropic. |
| Density | uniform / Beta / clumped, for curves and surfaces (not for blob, clusters or polytope) | Real concept values are not spread uniformly along the manifold. |
| Outliers | 0–10% of points, in 30% of samples | Tokenization and other quirks produce outliers. |
| R^D embedding | drawn once per family batch with probability 0.25 (so 25% of samples in expectation), D ∈ {16, 64, 256}, noise in all D dimensions, then PCA | Mimics the real pipeline. It exposes the fitter to projection effects, where a weak third shape axis can be overtaken by a noise direction, and to mirrored frames. |

This is domain randomization ([Tobin et al. 2017](https://arxiv.org/abs/1703.06907)). Data is generated on the fly from the `train` seed stream. Validation and test sets come from disjoint streams (`seeding.py`), so there is no leakage.

## 6. Model and training hyperparameters

| Choice | Value | Reason |
|---|---|---|
| Encoder | Set Transformer, 2 ISAB (m = 16) + PMA, d = 128, 4 heads | Permutation-invariant. Costs O(N·m), so it handles N = 5 and N = 1024 with one network (Lee et al. 2019). DeepSets is ablation A4. |
| Rotation output | 6D continuous representation | Continuous, unlike Euler angles or quaternions (Zhou et al. 2019). |
| Shape loss | Unsquared, weighted, symmetric Chamfer distance against the noise-free ground truth restricted to its true extent | Directly answers "how close is the output shape to the ground-truth shape". Unsquared, so values are in RMS-radius units. |
| Residual weight | 0.5 (Huber, δ = 0.05) | Ties the fit to the observed points without overriding the ground-truth shape term. |
| Sub-type cross-entropy | 0.1 × CE on the cluster count k and the polytope template | Lets each point-set head choose its discrete structure. |
| Classification | CE with label smoothing 0.05 | Softer targets for nested, partly ambiguous families. |
| Auxiliary axis/scale weight | 1.0 → 0.1 linearly over the first 30% of the planned steps | Avoids Chamfer local minima early on, such as a helix lying sideways. |
| Loss averaging | geometric terms averaged within each family, then over families | Every family weighs the same regardless of how many samples it has in a batch. |
| Optimizer | AdamW, lr 3e-4, weight decay 1e-4, warmup 2k steps, cosine schedule to 5% planned over 25k steps | Standard for transformer-style models. The 110-minute cap stopped `main_s0` before the end of the schedule. |
| Batch | max(64, min(512, 262144 // N)) clouds, i.e. a token budget of batch × N ≤ 262144, sliced per family | Steps are kernel-launch bound, so GPU time is only ~20% of each step and a larger batch is almost free. A batch of 512 at N = 1024 overflowed the 8 GB of VRAM. |
| Precision | bf16 autocast in the encoder; geometry in fp32 | Speed without losing geometric precision. |
| Budget | ≤ 110 min per seed | Fits an 8 GB laptop GPU. The first seed is published before the seed and ablation sweeps. |

## 7. Inference
1. Take the network's top-3 families.
2. Run 300 Adam steps (lr 1e-2, keeping the best iterate) per family on the two-sided objective (points → shape) + 0.25·(shape → points).
3. Select a family using the temperature-calibrated classifier.
4. Map the result back to PCA coordinates and canonicalize.

Why the objective is two-sided: the one-sided points → shape distance alone is minimized by degenerate shapes, such as huge spheres or full circles covering a half-arc. A test (`test_refine_does_not_inflate_half_circle`) guards against this.

## 8. Real-activation conventions
- **"Layer L"** is Hugging Face `output_hidden_states[L]`, the residual stream after L blocks. Models are truncated to L+1 blocks, so index L is never the norm-applied final state.
- **Precision and memory:** Llama-3.1-8B runs in bf16. Blocks that do not fit in 6.5 GiB of VRAM are offloaded to the CPU by accelerate. We do not quantize, because quantization could distort the geometry.
- **Prompts and value ranges:** listed in `src/neurogeom/realdata/llm_sets.py` and [../reports/deviations.md](../reports/deviations.md). Every prompt is logged with a timestamp.

## 9. Statistics
- Percentile bootstrap CIs (2000 replicates for every CI, including macro-F1, since 2026-11-01; before that macro-F1 used 500 and the paired macro-F1 difference 1000, see [deviations](../reports/deviations.md)).
- Paired bootstrap CIs for every comparison. p-values: H1 (macro-F1) from a normal approximation to the paired-bootstrap standard error; H2 and H5b from Wilcoxon signed-rank tests on per-cloud Chamfer distances. Holm correction across the preregistered comparisons that were run (H1, H2, H5b). Values below 10⁻¹⁶ are displayed as a floor.
- Exact (Clopper–Pearson) binomial CIs and a one-sided label-permutation null (10,000 permutations) for agreement on real data, using the subsample-majority coarse label.
- For real data, 200 subsamples of 80% each, without replacement (bootstrap with replacement would duplicate points in clouds as small as N = 7).
- 5-fold held-out residuals.
