from distributions.bgd import MUD, BGD
from collections.abc import Sequence
from fractions import Fraction
import numpy as np


def mapping_to_bgd(mapping) -> BGD:
    """Convert a finite discrete distribution mapping to a one-dimensional BGD."""
    
    items = sorted(mapping.items(), key=lambda item: Fraction(item[0]))

    breakpoints = []
    masses = []

    first_point, first_mass = items[0]
    first_point = Fraction(first_point)

    # First Dirac interval [x0, x0].
    breakpoints.extend([first_point, first_point])
    masses.append(first_mass)

    for point, mass in items[1:]:
        point = Fraction(point)

        # Zero-mass gap from previous point to current point.
        breakpoints.append(point)
        masses.append(0)

        # Dirac interval [point, point].
        breakpoints.append(point)
        masses.append(mass)

    center = MUD([tuple(breakpoints)], np.array(masses, dtype=object))

    left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
    right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

    E = np.array([left, center, right], dtype=object)

    return BGD(E, alpha=[0], beta=[0])


def uniform_to_bgd(dist) -> BGD:
    """Convert ('Uniform', (a, b)) to a one-dimensional BGD."""
    
    if not isinstance(dist, tuple) or len(dist) != 2:
        raise ValueError("dist must be a tuple like ('Uniform', (a, b))")

    name, params = dist

    if name != "Uniform":
        raise ValueError(f"expected 'Uniform', got {name!r}")

    if not isinstance(params, tuple) or len(params) != 2:
        raise ValueError("Uniform params must be a tuple (a, b)")

    a, b = params
    a = Fraction(a)
    b = Fraction(b)

    if not a < b:
        raise ValueError("Uniform requires a < b")

    # Center block: Uniform[a,b] with total mass 1.
    # P stores mass, not density.
    center = MUD([(a, b)], [1])

    # Empty tail-start blocks.
    left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
    right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

    E = np.array([left, center, right], dtype=object)

    return BGD(E, alpha=[0], beta=[0])

def merge_bgds(bgds: Sequence[BGD]) -> BGD:
    """
    Merge a list of independent BGD components into one joint BGD.
    The dimension order follows the list order.
    """
    if len(bgds) == 0:
        raise ValueError("bgds must be non-empty")

    result = bgds[0]
    if not isinstance(result, BGD):
        raise TypeError("all elements of bgds must be BGD")

    for bgd in bgds[1:]:
        if not isinstance(bgd, BGD):
            raise TypeError("all elements of bgds must be BGD")
        result = result.independent_product(bgd)

    return result
