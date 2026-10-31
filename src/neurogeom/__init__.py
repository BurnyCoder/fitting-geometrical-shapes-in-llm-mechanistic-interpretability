"""neurogeom: learned shape-type + parameter fitting for 3D point clouds of neural activations.

Global context: interpretability papers keep finding low-dimensional shapes (circles, helices,
spheres, ...) in the top principal components of activations. This package trains a network on
synthetic shapes to map a 3D point cloud (top-3 PCA of activations) to a shape family and a unique
parameter vector, then evaluates it on held-out synthetic data and on real model activations.
See README.md and reports/preregistration.md.
"""

__version__ = "0.1.0"
