# Summary of all experiments

Every number in this file is generated from `results/` by `uv run python run.py report --config main.yaml`. Text between `<!-- BEGIN:... -->` and `<!-- END:... -->` markers is overwritten on every run.

Per-experiment reports:
- [exp01_synthetic.md](exp01_synthetic.md): training the fitter and evaluating it on held-out synthetic shapes against baselines.
- [exp02_real_activations.md](exp02_real_activations.md): fitting real activation clouds (GPT-2, Llama-3.1-8B, grokking, toy models) and controls.
- Seeds, trained ablations and hyperparameter sensitivity: postponed to later stages (`bash scripts/sweep.sh ablations|sensitivity`); see [deviations.md](deviations.md).
- [preregistration.md](preregistration.md) and [deviations.md](deviations.md).

## Plain-language overview
Researchers who study how neural networks work often collect the network's internal activations for many related inputs, for example every day of the week. They then flatten these activations to 3D with PCA and look at the picture. Sometimes the points form a circle, sometimes a spiral (helix), a line, a curved bowl, and so on. People normally decide what shape it is by eye.

We trained a small neural network to make that call automatically. Shown a 3D cloud of points, it says which of 13 shapes it is and gives the exact size, position and orientation of that shape. It never saw real activations during training, only computer-generated shapes with realistic noise. We then checked:
1. how well it does on new computer-generated shapes, compared with a careful classical method;
2. whether it finds the shapes other researchers reported in real models.

## Headline results (generated)

<!-- BEGIN:headline -->
- **Synthetic test set (39,000 clouds, 13 families):** the method picks the correct family in 87.9% [87.6, 88.2] of clouds; median Chamfer distance to the ground-truth shape 0.156 (RMS-radius units).
- **Same 3,120 clouds for every system:** method 87.4% [86.2, 88.5], B2 features + boosting 64.5% [62.8, 66.2], B1 classical fit + geometric AIC 52.2% [50.6, 54.0], trivial baselines 7.7-7.9%.
- **H1** (macro-F1 method - B1): 0.3741 [0.3572, 0.3905], Holm p < 1e-16: supported.
- **H2** (median Chamfer method - B1): -0.0554 [-0.0645, -0.0462], Holm p < 1e-16: supported.
- **H5b_refinement** (median Chamfer method - no refinement): -0.0033 [-0.0046, -0.0021], Holm p < 1e-16: supported.
- **Selection rule:** choosing the family by geometric AIC instead of the classifier drops accuracy to 64.6% [64.1, 65.1].
- **Calibration:** temperature T = 0.772 (fitted on the validation stream) lowers ECE from 0.044 to 0.022 and NLL from 0.397 to 0.373; the NLL-optimal T on the test set itself is 0.784.
- **H3, real activations:** the coarse shape class (point set / curve / surface / blob; majority label over random 80% subsamples) agrees with the literature on 7/12 preregistered datasets (exact 95% CI 28-85%), label-permutation p = 0.125: not supported. Post hoc, with the full-cloud label instead of the subsample majority: 8/12.
- **H4, controls:** matched-covariance Gaussians are labelled blob 86% of the time (mean over all 17 real datasets, exploratory ones included); ordinal-process controls ringing 60% vs helix 40%: supported. Post hoc: smooth Gaussian processes alone are labelled helix in 80%, random walks in 0%.
- **Helix rule (post hoc, every full-cloud helix fit with held-out folds):** the preregistered two-part rule rejects the helix claim for `C1_gp_0` and accepts it for `R2_grokking_WE`, `R5a_llama_weekdays`, `R5b_llama_months`, `R5b_llama_months_means`, all of which are expected to be non-helix shapes, so the rule is necessary, not sufficient.
<!-- END:headline -->

**For a layperson:**
- On made-up test shapes, the network is right far more often than the careful classical method.
- On real model activations, its rough shape class matches what researchers reported on a little over half of the datasets, which is not better than a shuffled-label baseline. Looking at the failures afterwards, it seems to fail when the real shape is not one of its 13 shapes, when the 3D picture is only a slice of something higher-dimensional, or when the cloud looks unlike its training shapes.
- It also tends to see spirals in smooth random curves. The extra check we planned catches those, but it still accepts spirals for clouds that researchers describe as circles, so a "spiral" verdict needs more than that check.

## Glossary
- **Activation / residual stream:** the vector of numbers a model computes for a token at a given layer.
- **PCA (principal component analysis):** rotates the data so that the first few axes ("top-3 PCs") capture the most variance. We look at the data in the top-3 PCs.
- **Shape family:** a kind of shape (circle, helix, ...). **Parameters:** the numbers that pin down one particular shape of that kind (center, radius, axis, ...). **Canonical / unique parameters:** one agreed-upon way to write those numbers when several descriptions give the same shape.
- **Chamfer distance:** a distance between two shapes. Each point of one shape is matched to its nearest point on the other, and the distances are averaged both ways. 0 means identical. Here it is measured in units of the cloud's RMS radius.
- **Residual:** the average distance from the observed points to the fitted shape.
- **Refinement:** after the network guesses, a few hundred gradient steps adjust the shape to fit the points better.
- **Set Transformer:** a neural network that reads an unordered set of points ([Lee et al. 2019](https://arxiv.org/abs/1810.00825)).
- **Geometric AIC (G-AIC):** a classical rule for choosing between shape families. It trades off fit quality against the number of parameters ([Kanatani 2004](https://iim.cs.tut.ac.jp/member/kanatani/papers/pamodel2.pdf)).
- **Ringing / horseshoe artifact:** when the similarity between points decays smoothly along a sequence (for example a random walk, or data whose covariance decays with distance), PCA produces arch- or spiral-shaped curves, even when nothing interesting is going on ([Novembre & Stephens 2008](https://doi.org/10.1038/ng.139); [Antognini & Sohl-Dickstein 2018](https://arxiv.org/abs/1806.08805); [Shinn 2023](https://doi.org/10.1073/pnas.2311420120)). A true helix has to beat this explanation.
- **Fine / coarse accuracy:** fine is the exact family. Coarse is only the dimension of the shape: point set (0D), curve (1D), surface (2D) or blob (3D).
- **Subsample majority:** the label given most often when the fitter is re-run on 200 random 80% subsets of a real cloud. H3 uses its coarse class.
- **Bootstrap CI:** an error bar computed by resampling the test samples (2000 resamples).
- **Preregistration:** writing down hypotheses and pass/fail criteria before seeing results.

## Models trained

<!-- BEGIN:models_table -->
| Run | Encoder | Training objective | Data / curriculum | Params | Steps | Minutes | Val. accuracy | Test fine accuracy | Test median Chamfer |
|---|---|---|---|---|---|---|---|---|---|
| main_s0 | Set Transformer | CE + Chamfer + residual + aux | full synthetic mix | 1,712,093 | 22643 | 110 | 90.2% | 87.9% | 0.156 |
<!-- END:models_table -->

## Main results on synthetic data

<!-- BEGIN:results_table -->
| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |
|---|---|---|---|---|---|---|---|---|
| Method (network + refinement) | 39000 | 87.9% [87.6, 88.2] | 0.879 [0.876, 0.882] | 92.7% | 88.4% | 0.156 [0.154, 0.158] | 0.105 | 1.1 |
| Method, baseline subset | 3120 | 87.4% [86.2, 88.5] | 0.873 [0.862, 0.884] | 92.3% | 87.9% | 0.154 [0.147, 0.161] | 0.106 | 1.0 |
| A3: no refinement | 39000 | 87.9% [87.6, 88.2] | 0.879 [0.876, 0.882] | 92.7% | 88.4% | 0.159 [0.157, 0.162] | 0.120 | 1.5 |
| A5: G-AIC selection | 39000 | 64.6% [64.1, 65.1] | 0.627 [0.623, 0.631] | 77.5% | 68.8% | 0.182 [0.179, 0.185] | 0.108 | 0.8 |
| A5: hybrid selection | 39000 | 85.6% [85.2, 85.9] | 0.858 [0.854, 0.861] | 91.1% | 86.6% | 0.157 [0.155, 0.159] | 0.103 | 1.1 |
| B1: classical fit + G-AIC | 3120 | 52.2% [50.6, 54.0] | 0.499 [0.484, 0.514] | 70.7% | 58.1% | 0.209 [0.200, 0.221] | 0.117 | 0.7 |
| B2: features + boosting | 3120 | 64.5% [62.8, 66.2] | 0.645 [0.628, 0.661] | 85.1% | 68.0% | 0.237 [0.221, 0.255] | 0.120 | 1.1 |
| B3a: always blob | 3120 | 7.7% [6.8, 8.7] | 0.011 [0.010, 0.012] | 7.7% | 7.7% | 0.543 [0.536, 0.550] | 0.120 | 11.7 |
| B3b: always arc (B1 arc fit) | 3120 | 7.7% [6.8, 8.6] | 0.011 [0.010, 0.012] | 38.5% | 7.7% | 0.545 [0.536, 0.555] | 0.276 | 0.5 |
| B3c: random shape | 3120 | 7.9% [7.1, 8.8] | 0.079 [0.070, 0.088] | 32.9% | 10.9% | 1.116 [1.100, 1.137] | 0.589 | 58.3 |
<!-- END:results_table -->

## Preregistered hypotheses

<!-- BEGIN:hypotheses_table -->
| Hypothesis | Comparison | Estimate [95% CI] | p (Holm) | n | Verdict |
|---|---|---|---|---|---|
| H1 | macro-F1(method) - macro-F1(B1) | 0.3741 [0.3572, 0.3905] | < 1e-16 | 3120 | supported |
| H2 | median Chamfer(method) - median Chamfer(B1) | -0.0554 [-0.0645, -0.0462] | < 1e-16 | 3120 | supported |
| H5b_refinement | median Chamfer(method) - median Chamfer(no refinement) | -0.0033 [-0.0046, -0.0021] | < 1e-16 | 39000 | supported |
| H3 | Coarse-label agreement (subsample majority) with literature on preregistered real datasets | 7/12 = 58% [28, 85] (exact binomial) | perm. p = 0.125 | 12 | not supported |
| H4 | Controls: matched Gaussians -> blob (mean over 17 real datasets); ordinal processes -> ringing (not helix) | blob rate 86%; ringing 60% vs helix 40% | - | - | supported |
<!-- END:hypotheses_table -->

## Real activations

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

## Learnings
See the "Learnings" section of each experiment report.
