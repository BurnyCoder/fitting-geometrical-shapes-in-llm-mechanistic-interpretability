# Preregistration: learned shape fitting for neural-geometry point clouds

- **Author:** Libor "Burny" Burian (implementation by Claude Code)
- **Date:** 2026-10-31
- **Status:** committed before any model was trained and before any real activation was extracted or looked at. Analyses not listed here are labelled *post-hoc* in every report.

## 1. Domain and question
Interpretability papers report low-dimensional shapes in activations, usually shown in the top principal components. Examples:
- circles for days and months ([Engels et al. 2024](https://arxiv.org/abs/2405.14860));
- helices for positions ([Yedidia 2023](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix)) and character counts ([Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html));
- a paraboloid for colors and lines for temperature and age ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)).

These shapes are identified and parameterized by hand.

**Question.** Can a network trained only on synthetic shapes take a 3D point cloud (the top-3 PCA coordinates of activations) and return:
- (a) the shape family, and
- (b) a parameter vector that specifies the shape uniquely,
- accurately enough to beat a classical fit-every-family-and-select baseline on synthetic data, and to recover the shapes reported in the literature on real activations?

## 2. Systems compared

**Method: the network plus refinement.**
- **Encoder:** a Set Transformer ([Lee et al. 2019](https://arxiv.org/abs/1810.00825)).
- **Outputs:** a classification head over 13 shape families and one parameter head per family.
- **Training:** on-the-fly synthetic shapes. The loss is cross-entropy on the family plus the Chamfer distance between the predicted shape and the noise-free ground-truth shape, plus a residual from the input points to the predicted shape.
- **Test-time refinement:** at inference, the top-3 families are each refined (300 Adam steps on a two-sided point–shape distance). The final family is chosen by calibrated classifier probability.

**Baselines.**
- **B1, classical.** Fit every family by multi-start optimization of the same two-sided objective, starting from PCA/heuristic initializations with no network. Then select with Kanatani's geometric AIC ([Kanatani 2004](https://iim.cs.tut.ac.jp/member/kanatani/papers/acmodel.pdf)).
- **B2, features + boosting.** Hand-crafted features (PCA eigenvalue ratios, ripser persistent homology H0/H1, local-PCA dimension, point count), a HistGradientBoosting classifier trained on synthetic data, then B1's fit for the predicted family.
- **B3, trivial.**
  - **B3a:** predict the majority class (uniform → chance).
  - **B3b:** always fit a full circle.
  - **B3c:** a random family with random parameters.

**Ablations** (all trained with the same budget):

| ID | Change |
|---|---|
| A1 | Parameter-space loss instead of Chamfer |
| A2 | No input-residual term |
| A3 | No test-time refinement |
| A4 | DeepSets encoder instead of Set Transformer |
| A5 | Selection rule: geometric AIC or hybrid instead of classifier |
| A6 | No R^D-embedding augmentation |

**The 13 shape families:**
- blob (null class)
- clusters (k = 2..6)
- regular polytope
- line segment
- circle/arc
- closed ellipse
- helix
- ringing curve, c + Σ_{k=1..3} a_k cos(kπs) e_k, the PCA-horseshoe confounder
- disc
- sphere/cap
- paraboloid sheet (bowl, saddle or trough)
- cone/frustum (includes cylinder)
- torus

**Coarse classes:**

| Coarse class | Families |
|---|---|
| 0D point sets | clusters, polytope |
| 1D curves | line, arc, ellipse, helix, ringing |
| 2D surfaces | disc, sphere, paraboloid, cone, torus |
| 3D | blob |

## 3. Data

**Synthetic.**
- The training data come from seed stream `train`.
- The validation and temperature-calibration set comes from stream `val`.
- The fixed test set comes from stream `test`: 250 samples per cell of family × N-bin × noise-bin. The N-bins are {5–12, 13–64, 65–256, 257–1024} and the noise-bins are σ ∈ {0.003–0.02, 0.02–0.06, 0.06–0.15}, in units of the shape's RMS radius.
- The OOD set comes from stream `ood`. It contains projections to 3D of higher-dimensional manifolds: a 4D flat torus, two-frequency Fourier circles, and the multi-period number helix with T = 2, 5, 10, 100.
- The streams are disjoint by construction (`src/neurogeom/seeding.py`).

**Real** (top-3 PCA after centering; never whitened). Expected labels come from the cited sources:

| ID | Source of points | Expected coarse | Expected fine | Literature basis |
|---|---|---|---|---|
| R1 | GPT-2 small positional-embedding matrix W_pos, rows 1–1023 | 1D | helix | Yedidia 2023 |
| R2 | Embedding W_E of a 1-layer transformer we train on a+b mod 113 (grokked) | 1D | arc (full circle) or ellipse | Nanda et al. 2023; Zhong et al. 2023 |
| R3 | Toy Models of Superposition with 3 hidden dimensions: columns of W with norm > 0.5 | 0D | polytope or clusters | Elhage et al. 2022 |
| R4 | Llama-3.1-8B layer 5, mean activation per character count since the last newline (counting task) | 1D | helix | Gurnee et al. 2025; corl-team replication |
| R5a | Llama-3.1-8B layer 5, weekday tokens in Engels' "days of the week math" prompts | 1D | arc | Engels et al. 2024 |
| R5b | Llama-3.1-8B layer 5, month tokens in Engels' "calendar math" prompts | 1D | arc | Engels et al. 2024 |
| R6 | Llama-3.1-8B layer 19, last token of "The date is year {Y}" | 1D | helix | Bhalla et al. 2026 (Table 1) |
| R7 | Llama-3.1-8B layer 19, last token of "The hex code {h} is for the color" | 2D | paraboloid | Bhalla et al. 2026 |
| R8a | Llama-3.1-8B layer 19, "Today it's {f} degrees Fahrenheit outside" | 1D | line | Bhalla et al. 2026 |
| R8b | Llama-3.1-8B layer 19, "They are {age} years old." | 1D | line | Bhalla et al. 2026 |
| R9 | Llama-3.1-8B, mid layer, geometry-of-truth `cities` true/false statements | 0D | clusters (k=2) | Marks & Tegmark 2024 |

Where Bhalla et al. do not give an exact template, ours is reconstructed from their Table 1 and documented in the extraction code. The weights come from `meta-llama/Llama-3.1-8B`, which we have access to. The ungated mirror `unsloth/Meta-Llama-3.1-8B` has identical safetensors sha256 hashes (verified on 2026-10-31).

**Controls:**

| ID | Construction | Expected label |
|---|---|---|
| C1 | Smooth Gaussian-process and random-walk trajectories over an ordinal index, in R^768 | ringing, not helix |
| C2 | For each real dataset, a Gaussian with matched mean and covariance and the same N | blob |
| C3 | The same prompts or matrices, from randomly initialized weights of the same architecture | blob |

## 4. Hypotheses, metrics and falsification criteria

**Primary synthetic metrics.** Both are computed on the in-distribution test set.
- **(i) Fine macro-F1** over the 13 families.
- **(ii) Median normalized Chamfer distance** between the predicted shape (selected family, refined) and the noise-free ground-truth shape. It is symmetric and unsquared, measured in units of the RMS radius of the input cloud.

**Primary real-data metric.** The number of datasets R1–R9 where the coarse label (majority over subsamples) matches the expected coarse label.

| Hypothesis | Prediction | Falsified if |
|---|---|---|
| H1 | Method fine macro-F1 > B1 fine macro-F1 | The 95% paired-bootstrap CI of (method − B1) includes values ≤ 0 |
| H2 | Method median Chamfer ≤ B1 median Chamfer | The 95% paired-bootstrap CI of the median difference (method − B1) lies entirely > 0 |
| H3 | Coarse-label agreement with the literature on the real datasets is ≥ 50% and exceeds a label-permutation null (one-sided p < 0.05) | Agreement < 50% |
| H4 | C2 is labelled blob in ≥ 80% of subsamples, and C1 is labelled ringing more often than helix | C2 blob rate < 80%, or helix rate ≥ ringing rate on C1 |
| H5 | The Chamfer loss (method) beats the parameter-space loss (A1) on median Chamfer, and refinement (method) beats no refinement (A3) on median Chamfer | The paired CI includes the opposite sign |

For R1 (helix) and R4 (helix), a helix claim additionally requires the calibrated p(helix)/p(ringing) ≥ 3 *and* a lower held-out residual for helix than for ringing. Otherwise we report "PCA-ringing confound not rejected".

## 5. Analysis plan
1. **Order of runs.** Train one seed of the method first and evaluate it, then publish. Then run 5 seeds of the method and 3 seeds per ablation.
2. **Synthetic CIs.** Hierarchical bootstrap: resample seeds, then test samples within each seed, 2000 replicates, percentile 95% CIs.
3. **Synthetic paired comparisons.**
   - Use identical test samples for every system.
   - Compute a paired bootstrap of the difference, plus a Wilcoxon signed-rank test on the per-sample Chamfer.
   - Apply Holm correction over the preregistered comparisons: H1, H2, and the two H5 comparisons.
4. **Calibration.** A temperature fitted on the `val` stream. Report ECE (15 bins) and reliability diagrams before and after.
5. **Per-dataset real-data procedure.**
   - Run 200 random 80% subsamples without replacement × all available seeds.
   - Report the label frequencies, the majority label, and parameter medians with IQR.
   - Compute a 5-fold held-out point residual for the method and for B1.
   - Report the PCA spectrum λ1..λ6 and flag λ3/λ4 < 1.5 ("top-3 projection may cut a higher-dimensional manifold").
6. **Real-data agreement.** Count agreements and give exact binomial 95% CIs. The permutation null shuffles the predicted coarse labels across datasets, 10,000 permutations.
7. **Reporting breakdowns.**
   - Every synthetic metric is broken down by N-bin and noise-bin.
   - The confusion matrix is reported at the fine and coarse levels.
   - "Acceptable-superset" accuracy counts a more general family that still fits as correct, for example ellipse for a circle.

## 6. Exclusions and deviations
- **Ambiguous synthetic samples.** No synthetic test sample is excluded. Samples where a simpler family also fits are tagged `ambiguous` and reported both with and without them.
- **Failed real extractions.** A real dataset whose extraction fails is reported as failed. It is never silently dropped.
- **Deviations.** Any deviation from this document, such as a changed hyperparameter or template, is logged in `reports/deviations.md` with its reason and date.

## 7. Claims language
- Results describe **the shape of the top-3 PCA projection**, not the intrinsic manifold.
- Agreement with the literature is a consistency check. It is not causal evidence.
