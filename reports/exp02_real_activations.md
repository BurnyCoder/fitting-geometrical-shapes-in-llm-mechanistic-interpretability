# Experiment 2: fitting real activation clouds

**Run:** `main_s0`, the first trained fitter. Commands, in order:
- `uv run python run.py extract-real --config main.yaml`
- `uv run python run.py eval-real --config main.yaml`
- `uv run python run.py report --config main.yaml`

Numbers between the markers are generated from `results/main_s0/real/` and must not be edited by hand.

## Question
Does a fitter trained only on synthetic shapes recover the shapes the literature reports for real activations (H3)? Does it avoid finding structure where there is none (H4: matched-covariance Gaussians should be classified as blob, ordinal processes as ringing rather than helix)?

## Datasets
The expected labels were fixed in the [preregistration](preregistration.md) before any activation was extracted.

| ID | What the points are | Model and layer (Hugging Face `hidden_states[L]`) | Expected |
|---|---|---|---|
| R1 | Rows 1–1023 of the positional-embedding matrix | GPT-2 small | helix ([Yedidia 2023](https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS)) |
| R2 | Token embeddings W_E of a 1-layer transformer trained here on (a+b) mod 113 until it grokked | trained from scratch with the architecture and optimizer of [Nanda et al. 2023](https://arxiv.org/abs/2301.05217), 25,000 instead of 40,000 epochs | circle or ellipse |
| R3 | Columns of W in toy models of superposition with 3 hidden dimensions and 6 or 8 features (both counted in H3) | trained here ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)) | polytope or clusters |
| R4 | Mean activation per "characters since the last newline" | Llama-3.1-8B, layer 5 | helix ([Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html); Llama replication by [Sinii & Balagansky 2026](https://t-tech-manifolds.hf.space/)) |
| R5a, R5b | Weekday and month tokens in Engels et al.'s arithmetic prompts (our weekday prompt says "days of the week math" where theirs says "day of the week math") | Llama-3.1-8B, layer 5 | circle ([Engels et al. 2024](https://arxiv.org/abs/2405.14860)) |
| R6 | Years, "The date is year {Y}", last token | Llama-3.1-8B, layer 19 | helix ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R7 | Colors, "The hex code {h} is for the color", last token | Llama-3.1-8B, layer 19 | paraboloid ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R8a, R8b | Temperature and age templates, last token | Llama-3.1-8B, layer 19 | line ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R9 | True and false city statements, last token | Llama-3.1-8B, layer 12 | clusters ([Marks & Tegmark 2024](https://arxiv.org/abs/2310.06824), who report the two classes separating in the top-2 PCs of LLaMA-2) |
| C1 | Gaussian processes and random walks over an ordinal index in R^768 | none | ringing |
| C2 | A Gaussian with the same mean and covariance as each R dataset | none | blob |
| C3 | The same inputs through randomly initialized weights (GPT-2 W_pos: Hugging Face's N(0, 0.02²) initialisation; OpenAI's original code used σ = 0.01 for W_pos) | random init | blob |

Exploratory, not preregistered: R4b (Pythia-410m counting, layer 6), R5c ("It's {time} on {day}", layer 19), R3 with n = 20, and class-mean versions of R5a/R5b (`*_means`; until 2026-11-01 the tables wrongly marked them "(prereg)", see [deviations.md](deviations.md)).

## Results

<!-- BEGIN:real_table -->
| Dataset | Model / layer | N x D | Top-3 var. | Expected (coarse / fine) | Fit (full cloud) | p | Subsample majority (freq.) | Held-out resid. method / B1 / helix / ringing | Matched-Gaussian -> blob | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| `C1_gp_0` (prereg) | RBF Gaussian process over an ordinal ind / - | 200 x 768 | 99% | 1d / ringing | **helix** | 0.72 | ringing (58%) | 0.075 / 0.025 / 0.075 / 0.025 | 100% | - |
| `C1_walk_0` (prereg) | Gaussian random walk, 200 steps in R^768 / - | 200 x 768 | 83% | 1d / ringing | **ringing** | 0.99 | ringing (100%) | 0.030 / 0.030 / 0.082 / 0.030 | 100% | - |
| `C3_gpt2_wpos_init` (prereg) | GPT-2 initialisation N(0, 0.02^2) of W_p / - | 1023 x 768 | 1% | 3d / blob | **blob** | 0.93 | blob (100%) | 0.235 / 0.553 / 0.326 / 0.504 | 100% | gap<1.5 |
| `C3_grokking_WE_init` (prereg) | 1-layer transformer trained on (a+b) mod / - | 113 x 128 | 9% | 3d / blob | **blob** | 0.95 | blob (100%) | 0.238 / 0.577 / 0.347 / 0.554 | 100% | gap<1.5 |
| `C3_grokking_WE_memorised` | 1-layer transformer trained on (a+b) mod / - | 113 x 128 | 15% | - / - | **blob** | 0.97 | blob (100%) | 0.236 / 0.501 / 0.326 / 0.542 | 100% | gap<1.5 |
| `C3_llama_counting_init` (prereg) | meta-llama/Llama-3.1-8B (random init) / 5 | 150 x 4096 | 14% | 3d / blob | **clusters** | 0.96 | clusters (98%) | 0.426 / 0.356 / 0.213 / 0.260 | 100% | gap<1.5, OOD |
| `C3_llama_months_init` (prereg) | meta-llama/Llama-3.1-8B (random init) / 5 | 144 x 4096 | 19% | 3d / blob | **clusters** | 0.93 | clusters (82%) | 0.395 / 0.224 / 0.228 / 0.393 | 100% | gap<1.5, OOD |
| `C3_llama_weekdays_init` (prereg) | meta-llama/Llama-3.1-8B (random init) / 5 | 49 x 4096 | 34% | 3d / blob | **polytope** | 0.72 | clusters (70%) | 0.389 / 0.123 / 0.125 / 0.175 | 100% | gap<1.5, OOD |
| `C3_tms_dense` | toy model, n=6, m=3, dense features (S=0 / - | 5 x 3 | 100% | - / - | **polytope** | 0.57 | polytope (100%) | 0.966 / 0.401 / 0.248 / 0.916 | 20% | - |
| `R1_gpt2_wpos` (prereg) | gpt2 / embedding | 1023 x 768 | 91% | 1d / helix | **sphere** | 0.65 | sphere (74%) | 0.069 / 0.068 / 0.130 / 0.383 | 100% | - |
| `R2_grokking_WE` (prereg) | 1-layer transformer trained on (a+b) mod / - | 113 x 128 | 32% | 1d / arc, ellipse | **helix** | 0.51 | helix (43%) | 0.185 / 0.299 / 0.191 / 0.494 | 100% | gap<1.5 |
| `R3_tms_n20` | toy model of superposition, n=20, m=3, s / - | 20 x 3 | 100% | 0d / polytope, clusters | **sphere** | 0.84 | sphere (78%) | 0.230 / 0.216 / 0.242 / 0.572 | 95% | - |
| `R3_tms_n6` (prereg) | toy model of superposition, n=6, m=3, sp / - | 6 x 3 | 100% | 0d / polytope, clusters | **polytope** | 1.00 | polytope (100%) | 0.447 / 0.002 / 0.215 / 1.129 | 25% | - |
| `R3_tms_n8` (prereg) | toy model of superposition, n=8, m=3, sp / - | 8 x 3 | 100% | 0d / polytope, clusters | **polytope** | 1.00 | sphere (56%) | 0.854 / 0.058 / 0.284 / 0.936 | 55% | - |
| `R4_llama_counting` (prereg) | meta-llama/Llama-3.1-8B / 5 | 150 x 4096 | 69% | 1d / helix | **ringing** | 0.74 | ringing (92%) | 0.094 / 0.257 / 0.133 / 0.094 | 100% | - |
| `R4b_pythia_counting` | EleutherAI/pythia-410m / 6 | 150 x 1024 | 75% | 1d / helix | **ringing** | 0.86 | ringing (100%) | 0.080 / 0.219 / 0.138 / 0.079 | 100% | - |
| `R5a_llama_weekdays` (prereg) | meta-llama/Llama-3.1-8B / 5 | 49 x 4096 | 66% | 1d / arc | **helix** | 0.83 | helix (58%) | 0.077 / 0.075 / 0.077 / 0.123 | 100% | gap<1.5 |
| `R5a_llama_weekdays_means` | meta-llama/Llama-3.1-8B / 5 | 7 x 4096 | 78% | 1d / arc | **sphere** | 0.40 | helix (74%) | 0.225 / 0.524 / 0.131 / 0.364 | 20% | gap<1.5 |
| `R5b_llama_months` (prereg) | meta-llama/Llama-3.1-8B / 5 | 144 x 4096 | 55% | 1d / arc | **helix** | 0.70 | helix (87%) | 0.078 / 0.086 / 0.078 / 0.312 | 100% | gap<1.5 |
| `R5b_llama_months_means` | meta-llama/Llama-3.1-8B / 5 | 12 x 4096 | 60% | 1d / arc | **helix** | 0.40 | helix (51%) | 0.230 / 0.215 / 0.136 / 0.492 | 60% | gap<1.5 |
| `R5c_llama_days_time` | meta-llama/Llama-3.1-8B / 19 | 336 x 4096 | 58% | 1d / arc | **ringing** | 0.83 | ringing (86%) | 0.163 / 0.106 / 0.109 / 0.163 | 100% | gap<1.5 |
| `R6_llama_years` (prereg) | meta-llama/Llama-3.1-8B / 19 | 199 x 4096 | 25% | 1d / helix | **cone** | 0.34 | paraboloid (62%) | 0.190 / 0.292 / 0.247 / 0.349 | 100% | gap<1.5 |
| `R7_llama_colors` (prereg) | meta-llama/Llama-3.1-8B / 19 | 900 x 4096 | 66% | 2d / paraboloid | **cone** | 0.79 | cone (83%) | 0.171 / 0.260 / 0.213 / 0.397 | 100% | gap<1.5 |
| `R8a_llama_temperature` (prereg) | meta-llama/Llama-3.1-8B / 19 | 150 x 4096 | 90% | 1d / line | **ringing** | 0.98 | ringing (100%) | 0.127 / 0.095 / 0.127 / 0.127 | 100% | - |
| `R8b_llama_age` (prereg) | meta-llama/Llama-3.1-8B / 19 | 99 x 4096 | 85% | 1d / line | **clusters** | 0.61 | clusters (96%) | 0.339 / 0.129 / 0.149 / 0.289 | 100% | OOD |
| `R9_llama_truth_cities` (prereg) | meta-llama/Llama-3.1-8B / 12 | 1496 x 4096 | 53% | 0d / clusters | **ringing** | 0.82 | ringing (100%) | 0.313 / 0.352 / 0.198 / 0.313 | 100% | OOD |
<!-- END:real_table -->

**Column notes:**
- *Fit* is the label for the full cloud.
- *Subsample majority* is the most frequent label across 200 random 80% subsamples, with the PCA recomputed on each subsample. H3 uses its coarse class.
- *Held-out residual* is the mean distance of held-out points to shapes fitted on the other 80% of points, in RMS units, using 5 folds.
- *gap<1.5* flags a λ3/λ4 ratio below 1.5. In that case the top-3 projection may cut through a higher-dimensional manifold.

### Preregistered hypotheses H3 and H4

<!-- BEGIN:hypotheses_table -->
| Hypothesis | Comparison | Estimate [95% CI] | p (Holm) | n | Verdict |
|---|---|---|---|---|---|
| H1 | macro-F1(method) - macro-F1(B1) | 0.3741 [0.3572, 0.3905] | < 1e-16 | 3120 | supported |
| H2 | median Chamfer(method) - median Chamfer(B1) | -0.0554 [-0.0645, -0.0462] | < 1e-16 | 3120 | supported |
| H5b_refinement | median Chamfer(method) - median Chamfer(no refinement) | -0.0033 [-0.0046, -0.0021] | < 1e-16 | 39000 | supported |
| H3 | Coarse-label agreement (subsample majority) with literature on preregistered real datasets | 7/12 = 58% [28, 85] (exact binomial) | perm. p = 0.125 | 12 | not supported |
| H4 | Controls: matched Gaussians -> blob (mean over 17 real datasets); ordinal processes -> ringing (not helix) | blob rate 86%; ringing 60% vs helix 40% | - | - | supported |
<!-- END:hypotheses_table -->

![real](../docs/figures/fig_real.png)
![controls](../docs/figures/fig_controls.png)

## Learnings

These notes interpret the tables above. Every number they rely on is in a generated table or in the generated headline of [summary.md](summary.md); none is typed here. Evidence type: **descriptive and correlational**. The labels describe the top-3 PCA projection and carry no causal claim about how the models use these representations. The explanations of failures come from inspecting the clouds after seeing the fits; they are post hoc and untested.

1. **The synthetic-to-real gap is large.**
   - On synthetic data the method beats every baseline (exp01). On real activations its coarse label (point set / curve / surface / blob; majority over subsamples) matches the literature on a little over half of the preregistered datasets.
   - That is not above the label-permutation null, so **H3 is not supported**.
   - H3 is not falsified either: agreement is not below 50%. This is a negative result for the fitter as a drop-in replacement for by-eye shape calls.
   - The coarse misses are GPT-2 positions (R1), the 8-feature toy model (R3, n = 8), years (R6), age (R8b) and true/false statements (R9). Points 2, 4, 7, 8 and 10 discuss them.
2. **Out-of-vocabulary curves get a wrong family, and refinement cannot fix that.**
   - GPT-2's positional embeddings (R1) form a 1D spiral whose radius changes along the curve, as the points in [fig_real](../docs/figures/fig_real.png) show.
   - No family covers this curve. *Helix* has a constant radius; *ringing* has only 3 harmonics.
   - The classifier picks *sphere*, a surface the spiral lies on. Refinement only adjusts parameters within the chosen family.
   - The held-out point residual cannot detect the error, because every held-out point lies on the fitted sphere.
   - **What would fix it:** a coverage check (the fraction of the fitted shape's area that lies near data), and a Fourier-curve family with more harmonics. Both are candidates for a later "more models" stage.
3. **More than 6 discrete groups have no point-set family (a fine-level failure only).**
   - Weekday (7) and month (12) activations form tight clusters, one per token.
   - *Clusters* allows at most 6 centroids, and *polytope* has only six fixed templates. The fitter therefore threads a helix through the clusters.
   - The coarse label "curve" matches the literature's circle, so these datasets count as agreements in H3, but the fine label is not meaningful.
4. **Low top-3 variance means the 3D view is a projection of something larger.**
   - Years (R6) and the grokked embeddings (R2) keep only a small share of variance in the top 3 PCs. λ3/λ4 is close to 1 (flag `gap<1.5`).
   - For R2 this is expected. The grokked model uses several Fourier frequencies, each a circle in its own plane ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217)), so the top-3 PCs mix circles. R2 is a coarse agreement (curve) but a fine failure (helix instead of circle).
   - Labels of flagged datasets should not be read as the shape of the representation.
5. **Smooth ordinal controls expose the helix/ringing confound, and the helix rule is necessary but not sufficient.**
   - Preregistered H4 passes on the pooled C1 controls, but only because the random walks are labelled *ringing* every time.
   - Post hoc, the smooth Gaussian-process controls are mostly labelled *helix* (see the H4 row and the headline numbers).
   - The preregistered two-part helix rule demands both p(helix)/p(ringing) ≥ 3 and a lower held-out residual than ringing. It rejects the helix claim for the fully evaluated GP control, where the held-out residual favours ringing.
   - Applied post hoc to every full-cloud helix fit with held-out folds, the same rule **accepts** helix for the grokked embeddings, weekdays, months and month means (the "Helix rule" headline line), all described as circles in the literature. The rule separates a helix from PCA ringing, not from a circle or from clusters threaded by a helix.
   - **Learning:** the classifier alone is not enough to claim a helix; the held-out check against ringing is necessary, and a check against circles and clusters is also needed.
6. **Randomly initialised networks are not structureless.**
   - The preregistered expectation for C3 (random-init Llama) was *blob*. The fitter instead reports point sets (*clusters* for months and counting, *polytope* for weekdays), and [fig_controls](../docs/figures/fig_controls.png) shows real clusters, one per input token.
   - A randomly initialised transformer still separates token identities, so the C3 expectation was wrong, not the fitter.
   - Random-init GPT-2 positional embeddings and the untrained grokking embeddings are blobs, as expected. They are i.i.d. random weight rows; no forward pass could separate token identities.
7. **Ordinal variables give ringing, not lines.**
   - Temperature (R8a) is labelled *ringing*: the arch-shaped curve PCA produces when the similarity between points decays smoothly with an ordinal variable ([Novembre & Stephens 2008](https://doi.org/10.1038/ng.139); [Antognini & Sohl-Dickstein 2018](https://arxiv.org/abs/1806.08805)).
   - That agrees at the coarse level with the literature's "line".
   - Age (R8b) is not one smooth curve: ordered by age, consecutive points jump apart around round decade values. It gets *clusters* and the OOD flag.
8. **The clusters family assumes clusters of equal, round spread.**
   - True and false city statements (R9) form two overlapping clouds of very different shape, one narrow and one broad.
   - That is out of distribution for the clusters family. The fit is a curve and the OOD flag is set.
9. **An extraction artifact, found by looking at outliers and fixed (post hoc, exploratory dataset only).**
   - In the first Pythia counting cloud (R4b), the count-1 mean lay 50× the median distance from the centre.
   - Pythia's tokenizer adds no BOS token, so each document's first token entered the count bin of its own length. The starting token carries massive activations ([Sun et al. 2024](https://arxiv.org/abs/2402.17762)) and acts as an attention sink ([Xiao et al. 2024](https://arxiv.org/abs/2309.17453)).
   - Sequence position 0 is now skipped for every model. For Llama position 0 is BOS, which was already excluded, so the preregistered Llama datasets are unchanged.
   - After the fix the Pythia cloud is a curve with the same family as Llama's counting cloud, instead of "clusters" made of one outlier plus everything else.
   - The superseded cloud is kept in `data_cache/real_superseded/`; see [deviations.md](deviations.md).
10. **Held-out residuals and subsample labels are unreliable for tiny clouds.**
    - With 6 or 8 toy-model vertices and 5 folds, each fit sees 4–7 points.
    - The network's fit to the remaining vertices does not reach the held-out vertex. The classical fit B1, which tries every polytope template, completes the polytope and gets a near-zero held-out residual.
    - At this N the held-out residual compares extrapolation ability, not fit quality.
    - The same applies to the subsample labels: the 8-feature toy model is a polytope on its full cloud, but most 80% subsamples (6 of 8 vertices) are labelled sphere, which makes it a coarse miss in H3.
11. **Engineering: free the LLM before the next phase.**
    - `del model` plus `torch.cuda.empty_cache()` did not release Llama after extraction. accelerate's dispatch hooks wrap `module.forward` in a partial that references the module, a reference cycle that only the garbage collector frees ([accelerate issue #2871](https://github.com/huggingface/accelerate/issues/2871)).
    - The following evaluation phase ran with 5.7 GB of GPU memory paged into system RAM. `gc.collect()` before `empty_cache()` fixes it.
