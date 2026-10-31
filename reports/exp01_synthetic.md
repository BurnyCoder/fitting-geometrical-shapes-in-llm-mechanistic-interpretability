# Experiment 1: training the fitter and evaluating it on synthetic shapes

**Run:** `main_s0`
- Config: `configs/main.yaml` on top of `configs/base.yaml`.
- Commands, in order:
  - `uv run python run.py train --config main.yaml`
  - `uv run python run.py eval-synth --config main.yaml`
  - `uv run python run.py report --config main.yaml`
- Logs: `logs/*_main_s0_*.log`.

Numbers between the markers are generated from `results/main_s0/` and must not be edited by hand.

## Question
The network is trained only on synthetic shapes. Does it identify the shape family and recover the shape better than the strongest classical alternative? That alternative fits every family and picks one with geometric AIC. This is preregistered as H1 and H2. Does test-time refinement help (H5b)?

## Setup
- **Model:** Set Transformer (ISAB×2 with 16 inducing points, PMA, d = 128) with 13 per-family parameter heads. The parameter count is in the models table below.
- **Data:** on-the-fly synthetic shapes from the `train` seed stream, with the nuisances listed in [../docs/methodology.md](../docs/methodology.md).
- **Loss:** CE + Chamfer(predicted shape, noise-free ground-truth shape) + 0.5·residual + annealed auxiliary terms.
- **Optimization:**
  - AdamW, lr 3e-4, warmup then cosine schedule;
  - batch 512, token budget 262144;
  - wall-clock cap of 110 minutes on an RTX 5070 Laptop (8 GB).
- **Test set:** stream `test`, 250 clouds per cell of family × N-bin × noise-bin, 39,000 clouds in total.
- **Baselines:** run on a fixed subset of 20 clouds per family per cell (3,120 clouds); see [deviations.md](deviations.md).

## Models trained

<!-- BEGIN:models_table -->
| Run | Encoder | Training objective | Data / curriculum | Params | Steps | Minutes | Val. accuracy | Test fine accuracy | Test median Chamfer |
|---|---|---|---|---|---|---|---|---|---|
| main_s0 | Set Transformer | CE + Chamfer + residual + aux | full synthetic mix | 1,712,093 | 22643 | 110 | 90.2% | 87.9% | 0.156 |
<!-- END:models_table -->

## Results

<!-- BEGIN:results_table -->
| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |
|---|---|---|---|---|---|---|---|---|
| Method (network + refinement) | 39000 | 87.9% [87.6, 88.2] | 0.879 [0.876, 0.882] | 92.7% | 88.4% | 0.156 [0.154, 0.158] | 0.105 | 1.1 |
| Method, baseline subset | 3120 | 87.4% [86.2, 88.5] | 0.873 [0.862, 0.884] | 92.3% | 87.9% | 0.154 [0.147, 0.161] | 0.106 | 1.0 |
| A3: no refinement | 39000 | 87.9% [87.6, 88.2] | 0.879 [0.876, 0.882] | 92.7% | 88.4% | 0.159 [0.157, 0.162] | 0.120 | 1.5 |
| A5: G-AIC selection | 39000 | 64.6% [64.1, 65.1] | 0.627 [0.623, 0.632] | 77.5% | 68.8% | 0.182 [0.179, 0.185] | 0.108 | 0.8 |
| A5: hybrid selection | 39000 | 85.6% [85.2, 85.9] | 0.858 [0.854, 0.861] | 91.1% | 86.6% | 0.157 [0.155, 0.159] | 0.103 | 1.1 |
| B1: classical fit + G-AIC | 3120 | 52.2% [50.6, 54.0] | 0.499 [0.484, 0.515] | 70.7% | 58.1% | 0.209 [0.200, 0.221] | 0.117 | 0.7 |
| B2: features + boosting | 3120 | 64.5% [62.8, 66.2] | 0.645 [0.629, 0.662] | 85.1% | 68.0% | 0.237 [0.221, 0.255] | 0.120 | 1.1 |
| B3a: always blob | 3120 | 7.7% [6.8, 8.7] | 0.011 [0.010, 0.012] | 7.7% | 7.7% | 0.543 [0.536, 0.550] | 0.120 | 11.7 |
| B3b: always circle | 3120 | 7.7% [6.8, 8.6] | 0.011 [0.010, 0.012] | 38.5% | 7.7% | 0.545 [0.536, 0.555] | 0.276 | 0.5 |
| B3c: random shape | 3120 | 7.9% [7.1, 8.8] | 0.079 [0.070, 0.088] | 32.9% | 10.9% | 1.116 [1.100, 1.137] | 0.589 | 58.3 |
<!-- END:results_table -->

The table above restricted to test clouds that no simpler family fits equally well (the "unambiguous" subset):

<!-- BEGIN:unambiguous_table -->
| System | n | Fine accuracy [95% CI] | Macro-F1 [95% CI] | Coarse accuracy | Superset-OK accuracy | Median Chamfer to GT [95% CI] | Mean residual | Median axis error (deg) |
|---|---|---|---|---|---|---|---|---|
| Method (network + refinement) | 2060 | 94.0% [93.1, 95.0] | 0.922 [0.909, 0.935] | 96.4% | 94.3% | 0.119 [0.111, 0.128] | 0.091 | 0.7 |
| A3: no refinement | 2060 | 94.0% [93.1, 95.0] | 0.922 [0.909, 0.935] | 96.4% | 94.3% | 0.127 [0.119, 0.132] | 0.101 | 1.1 |
| A5: G-AIC selection | 2060 | 72.0% [70.3, 73.8] | 0.654 [0.639, 0.669] | 81.5% | 76.5% | 0.138 [0.134, 0.144] | 0.097 | 0.6 |
| A5: hybrid selection | 2060 | 92.6% [91.4, 93.7] | 0.902 [0.887, 0.916] | 94.8% | 93.1% | 0.119 [0.111, 0.128] | 0.090 | 0.7 |
| B1: classical fit + G-AIC | 2060 | 69.7% [67.6, 71.6] | 0.636 [0.615, 0.652] | 82.5% | 75.4% | 0.154 [0.148, 0.160] | 0.103 | 0.6 |
| B2: features + boosting | 2060 | 72.3% [70.2, 74.3] | 0.697 [0.674, 0.716] | 92.5% | 76.2% | 0.152 [0.147, 0.159] | 0.099 | 0.6 |
| B3a: always blob | 2060 | 4.5% [3.6, 5.3] | 0.007 [0.005, 0.008] | 4.5% | 4.5% | 0.523 [0.516, 0.529] | 0.119 | 10.2 |
| B3b: always circle | 2060 | 7.3% [6.4, 8.5] | 0.011 [0.009, 0.012] | 36.3% | 7.3% | 0.528 [0.516, 0.538] | 0.273 | 0.3 |
| B3c: random shape | 2060 | 8.3% [7.2, 9.6] | 0.079 [0.068, 0.091] | 34.1% | 11.6% | 1.114 [1.094, 1.138] | 0.582 | 58.6 |
<!-- END:unambiguous_table -->

### Preregistered hypotheses

<!-- BEGIN:hypotheses_table -->
| Hypothesis | Comparison | Estimate [95% CI] | p (Holm) | n | Verdict |
|---|---|---|---|---|---|
| H1 | macro-F1(method) - macro-F1(B1) | 0.3741 [0.3579, 0.3903] | < 1e-16 | 3120 | supported |
| H2 | median Chamfer(method) - median Chamfer(B1) | -0.0554 [-0.0645, -0.0462] | < 1e-16 | 3120 | supported |
| H5b_refinement | median Chamfer(method) - median Chamfer(no refinement) | -0.0033 [-0.0046, -0.0021] | < 1e-16 | 39000 | supported |
| H3 | Coarse-label agreement with literature on real datasets | 7/12 = 58% [28, 85] (exact binomial) | perm. p = 0.125 | 12 | not supported |
| H4 | Controls: matched Gaussians -> blob; ordinal processes -> ringing (not helix) | blob rate 86%; ringing 60% vs helix 40% | - | - | supported |
<!-- END:hypotheses_table -->

### Figures
- Training curves: ![training](../docs/figures/fig_training.png)
- Systems compared: ![systems](../docs/figures/fig_systems.png)
- Accuracy against N and against noise: ![by N](../docs/figures/fig_by_n.png) ![by noise](../docs/figures/fig_by_noise.png)
- Confusion matrix: ![confusion](../docs/figures/fig_confusion.png)
- Calibration: ![reliability](../docs/figures/fig_reliability.png)
- Example fits, method versus classical B1: ![gallery](../docs/figures/fig_gallery_synth.png)

## Learnings
These notes record what we tried, what went wrong, and how each problem was found and fixed. The timings were measured with `torch.profiler` and wall-clock timers during development; they are development observations, not results.

1. **Batched small linear algebra on the GPU was the bottleneck, not the model.**
   - The first generator took about 0.3 s per training step. The torch profiler showed `linalg_qr` accounting for 75% of CPU time with 68k kernel launches: batched QR on CUDA launches about one kernel per matrix.
   - Fix: random rotations from normalized 4D Gaussians, i.e. unit quaternions (Shoemake 1992); determinants via the triple product; tiny SVDs and eigendecompositions moved to the CPU.
   - FP64 `eigh` on the GPU was also slow, because consumer GPUs run FP64 at roughly 1/64 of FP32 throughput. It now runs in FP32 on the GPU, with FP64 kept for real data on the CPU.
   - Result: the generator became about 7× faster, at 0.04 s per step.
2. **Training is kernel-launch bound.**
   - The GPU was busy for only about 20% of each step. A batch of 512 costs about the same wall time as a batch of 256, so throughput doubled.
   - A batch of 512 with N = 1024 exceeded 8 GB, and Windows silently spilled into shared memory at 3.3 s per step. A token budget (batch × N ≤ 262144) prevents this.
3. **PCA of a 3D subspace of R^D can return a mirrored frame.**
   - In 3D we force det = +1. With D > 3, however, the map from the shape's 3D frame to the PCA frame can be a reflection.
   - Ground-truth parameters therefore need a mirror-aware transform: a y-mirror for most families, which flips the helix pitch sign, and a z-mirror for the ringing curve, which flips a3. A regression test checks reflections for every family.
4. **Flat shapes plus a 3rd PC that is pure noise make the frame map rank-deficient.**
   - For an arc embedded in R^64 with no noise, the third PC is an arbitrary null direction. The 3×3 map from the shape frame to the PCA frame then has singular values (1, 1, 0).
   - Using the mean singular value as the scale was wrong. The correct scale is exactly 1/RMS, because both maps are isometries on the shape's plane.
   - The Chamfer target never had this problem, because it is computed from exactly projected ground-truth samples.
5. **Test-design pitfalls caught by TDD.**
   - Comparing shapes point by point fails after a mirror, because the traversal direction flips. Shapes must be compared as point sets.
   - Symmetry tests need a sampling much finer than the grid's half-spacing. Otherwise a rotated phase grid looks like an error.

### Learnings from the results

The numbers behind these points are in the generated tables and figures above; none is typed here.

6. **The learned fitter beats the classical pipeline on every point-count bin (H1, H2 supported).**
   - On the same baseline-subset clouds, the method has higher macro-F1 and a lower median Chamfer distance to the ground-truth shape than B1, which fits every family and selects with geometric AIC. Both paired-bootstrap CIs exclude 0.
   - The accuracy gap is largest for the smallest clouds (5–12 points); see the accuracy-by-N figure.
   - The feature-based classifier B2 sits between the two.
7. **Model selection, not fitting, is what separates the systems.**
   - A5 keeps the network's refined candidates but picks among them by geometric AIC instead of the classifier, and accuracy drops close to B2's level.
   - With nested families and realistic noise, a residual-versus-complexity criterion often prefers the more flexible family. The classifier has learned the prior over families and noise that generated the data.
8. **Refinement helps the fit through the points, not family identification.**
   - Refinement never changes the selected family, so A3 has exactly the method's accuracy.
   - It lowers the mean point-to-shape residual clearly, but the median Chamfer distance to the ground truth only slightly. H5b is supported, with a small effect.
   - The network's direct parameter prediction is already close to the ground-truth shape; refinement mostly snaps the shape onto the noisy points.
9. **Hardest families.**
   - Cone, helix and paraboloid have the lowest recall; see the confusion matrix.
   - The largest confusions are helix ↔ cone (sparsely sampled helix turns lie on a cylinder- or cone-like surface) and paraboloid ↔ cone (both are curved sheets around an axis).
   - The fine-versus-coarse gap in the results table shows that most errors stay within the right dimension (curve, surface, point set).
10. **Calibration.**
    - The raw classifier is slightly underconfident, an expected effect of the 0.05 label smoothing.
    - A single temperature fitted on the `val` stream (Guo et al. 2017) roughly halves the expected calibration error. The temperature that would be optimal on the test set itself is almost the same.
11. **Engineering: autograd memory in refinement.**
    - The first full evaluation stalled at 21.5k of 39k clouds. Refining 3 candidate surfaces for 512 clouds of 256 points kept several (B, N, M) distance matrices for the backward pass. That needed more than 8 GB, and Windows paged GPU memory to system RAM.
    - Fix: find the nearest-point index without autograd, in chunks, and differentiate only the gathered pairs. This is the split pytorch3d's `knn_points` + `knn_gather` uses. Value and gradient are identical up to ties (unit test), and peak memory fell by more than 10×.
    - Reported numbers come from the restarted evaluation; see [deviations.md](deviations.md).
