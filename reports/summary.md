# Summary of all experiments

Every number in this file is generated from `results/` by `uv run python run.py report --config main.yaml`. Text between `<!-- BEGIN:... -->` and `<!-- END:... -->` markers is overwritten on every run.

Per-experiment reports:
- [exp01_synthetic.md](exp01_synthetic.md): training the fitter and evaluating it on held-out synthetic shapes against baselines.
- [exp02_real_activations.md](exp02_real_activations.md): fitting real activation clouds (GPT-2, Llama-3.1-8B, grokking, toy models) and controls.
- [exp03_seeds_ablations.md](exp03_seeds_ablations.md): seed sensitivity and ablations.
- [preregistration.md](preregistration.md) and [deviations.md](deviations.md).

## Plain-language overview
Researchers who study how neural networks work often collect the network's internal activations for many related inputs, for example every day of the week. They then flatten these activations to 3D with PCA and look at the picture. Sometimes the points form a circle, sometimes a spiral (helix), a line, a curved bowl, and so on. People normally decide what shape it is by eye.

We trained a small neural network to make that call automatically. Shown a 3D cloud of points, it says which of 13 shapes it is and gives the exact size, position and orientation of that shape. It never saw real activations during training, only computer-generated shapes with realistic noise. We then checked:
1. how well it does on new computer-generated shapes, compared with a careful classical method;
2. whether it finds the shapes other researchers reported in real models.

## Glossary
- **Activation / residual stream:** the vector of numbers a model computes for a token at a given layer.
- **PCA (principal component analysis):** rotates the data so that the first few axes ("top-3 PCs") capture the most variance. We look at the data in the top-3 PCs.
- **Shape family:** a kind of shape (circle, helix, ...). **Parameters:** the numbers that pin down one particular shape of that kind (center, radius, axis, ...). **Canonical / unique parameters:** one agreed-upon way to write those numbers when several descriptions give the same shape.
- **Chamfer distance:** a distance between two shapes. Each point of one shape is matched to its nearest point on the other, and the distances are averaged both ways. 0 means identical. Here it is measured in units of the cloud's RMS radius.
- **Residual:** the average distance from the observed points to the fitted shape.
- **Refinement:** after the network guesses, a few hundred gradient steps adjust the shape to fit the points better.
- **Set Transformer:** a neural network that reads an unordered set of points ([Lee et al. 2019](https://arxiv.org/abs/1810.00825)).
- **Geometric AIC (G-AIC):** a classical rule for choosing between shape families. It trades off fit quality against the number of parameters ([Kanatani 2004](https://iim.cs.tut.ac.jp/member/kanatani/papers/acmodel.pdf)).
- **Ringing / horseshoe artifact:** PCA of any smoothly changing sequence produces arch- or spiral-shaped curves, even when nothing interesting is going on ([Novembre & Stephens 2008](https://doi.org/10.1038/ng.139)). A true helix has to beat this explanation.
- **Fine / coarse accuracy:** fine is the exact family. Coarse is only the dimension of the shape: point set (0D), curve (1D), surface (2D) or blob (3D).
- **Bootstrap CI:** an error bar computed by resampling the test samples.
- **Preregistration:** writing down hypotheses and pass/fail criteria before seeing results.

## Models trained

<!-- BEGIN:models_table -->
<!-- END:models_table -->

## Main results on synthetic data

<!-- BEGIN:results_table -->
<!-- END:results_table -->

## Preregistered hypotheses

<!-- BEGIN:hypotheses_table -->
<!-- END:hypotheses_table -->

## Real activations

<!-- BEGIN:real_table -->
<!-- END:real_table -->

## Learnings
See the "Learnings" section of each experiment report.
