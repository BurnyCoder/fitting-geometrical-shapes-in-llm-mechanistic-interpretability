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
| R2 | Token embeddings W_E of a 1-layer transformer trained here on (a+b) mod 113 until it grokked | trained from scratch ([Nanda et al. 2023](https://arxiv.org/abs/2301.05217) hyperparameters) | circle or ellipse |
| R3 | Columns of W in toy models of superposition with 3 hidden dimensions | trained here ([Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html)) | polytope or clusters |
| R4 | Mean activation per "characters since the last newline" | Llama-3.1-8B, layer 5 | helix ([Gurnee et al. 2025](https://transformer-circuits.pub/2025/linebreaks/index.html)) |
| R5a, R5b | Weekday and month tokens in Engels et al.'s arithmetic prompts | Llama-3.1-8B, layer 5 | circle ([Engels et al. 2024](https://arxiv.org/abs/2405.14860)) |
| R6 | Years, "The date is year {Y}", last token | Llama-3.1-8B, layer 19 | helix ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R7 | Colors, "The hex code {h} is for the color", last token | Llama-3.1-8B, layer 19 | paraboloid ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R8a, R8b | Temperature and age templates, last token | Llama-3.1-8B, layer 19 | line ([Bhalla et al. 2026](https://arxiv.org/abs/2604.28119)) |
| R9 | True and false city statements, last token | Llama-3.1-8B, layer 12 | clusters ([Marks & Tegmark 2024](https://arxiv.org/abs/2310.06824)) |
| C1 | Gaussian processes and random walks over an ordinal index in R^768 | none | ringing |
| C2 | A Gaussian with the same mean and covariance as each R dataset | none | blob |
| C3 | The same inputs through randomly initialized weights | random init | blob |

Exploratory, not preregistered: R4b (Pythia-410m counting, layer 6), R5c ("It's {time} on {day}", layer 19), R3 with n = 20, and class-mean versions of R5a/R5b.

## Results

<!-- BEGIN:real_table -->
<!-- END:real_table -->

**Column notes:**
- *Fit* is the label for the full cloud.
- *Subsample majority* is the most frequent label across 200 random 80% subsamples, with the PCA recomputed on each subsample.
- *Held-out residual* is the mean distance of held-out points to shapes fitted on the other 80% of points, in RMS units, using 5 folds.
- *gap<1.5* flags a λ3/λ4 ratio below 1.5. In that case the top-3 projection may cut through a higher-dimensional manifold.

### Preregistered hypotheses H3 and H4

<!-- BEGIN:hypotheses_table -->
<!-- END:hypotheses_table -->

![real](../docs/figures/fig_real.png)
![controls](../docs/figures/fig_controls.png)

## Learnings

<!-- BEGIN:exp02_learnings -->
<!-- END:exp02_learnings -->
