"""Registry of the 13 shape families, their coarse classes and 'acceptable superset' relations.

Global context: the classifier predicts an index into FAMILIES; evaluation reports fine (exact
family), coarse (0D point set / 1D curve / 2D surface / 3D blob) and acceptable-superset accuracy
(a more general family that still describes the ground truth counts as correct, e.g. an ellipse for
a full circle). The vocabulary was chosen in reports/preregistration.md from the shape catalog.
"""

from __future__ import annotations

from neurogeom.shapes.curves import Arc, Ellipse, Helix, Line, Ringing
from neurogeom.shapes.pointsets import Clusters, Polytope
from neurogeom.shapes.surfaces import Blob, Cone, Disc, Paraboloid, Sphere, Torus

FAMILIES = [Blob(), Clusters(), Polytope(), Line(), Arc(), Ellipse(), Helix(), Ringing(),
            Disc(), Sphere(), Paraboloid(), Cone(), Torus()]
NAMES = [f.name for f in FAMILIES]
INDEX = {n: i for i, n in enumerate(NAMES)}
BY_NAME = {f.name: f for f in FAMILIES}
COARSE_NAMES = ["0d", "1d", "2d", "3d"]
COARSE = [COARSE_NAMES.index(f.coarse) for f in FAMILIES]

# ground truth family -> more general families that can represent it exactly (limit cases inside their
# parameter space). Used for the "acceptable-superset" accuracy only; fine accuracy stays strict.
SUPERSETS = {
    "polytope": {"clusters"},
    "line": {"ringing"},          # a2 = a3 = 0 gives a segment
    "arc": {"ellipse", "helix"},  # full circle = ellipse with a = b; arc = helix with pitch 0
    "disc": {"paraboloid"},       # k1 = k2 = 0
}


def family(name_or_index) -> object:
    return FAMILIES[name_or_index] if isinstance(name_or_index, int) else BY_NAME[name_or_index]
