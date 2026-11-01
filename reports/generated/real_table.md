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
