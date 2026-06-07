from distributions.bgd import MUD, BGD
from collections.abc import Sequence
from fractions import Fraction
import numpy as np
from numbers import Real
import math


def exponential_to_bgd(lam, mode, center_subdivision=None, block_subdivision=None) -> BGD:
    """Convert Exp(lam) to a one-dimensional BGD upper approximation.

    center_subdivision:
        Global breakpoints of the center block.
        If None, the center block is empty.

    block_subdivision:
        Local breakpoints of the right tail block.
        If None, use (0, 0.1, 0.2, ..., 1).
    """
    if not isinstance(lam, Real) or isinstance(lam, bool):
        raise ValueError("lam must be a positive numeric value")

    if lam <= 0:
        raise ValueError("lam must be positive")

    if mode == 'MUD':
        # ---------------- center block ----------------
        if center_subdivision is None:
            center = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
            center_right = Fraction(0)
        else:
            if len(center_subdivision) < 2:
                raise ValueError("center_subdivision must contain at least two points")
    
            center_points = tuple(Fraction(x) for x in center_subdivision)
    
            for left, right in zip(center_points, center_points[1:]):
                if not left < right:
                    raise ValueError("center_subdivision must be strictly increasing")
    
            if not center_points[0] <= 0:
                raise ValueError("center_subdivision must start with a non-positive value")
    
            if not center_points[-1] > 0:
                raise ValueError("center_subdivision must end with a positive value")
    
            center_masses = []
    
            for left, right in zip(center_points, center_points[1:]):
                if right <= 0:
                    mass = 0
                else:
                    start = max(left, Fraction(0))
                    upper_density = lam * math.exp(-float(lam) * float(start))
                    mass = upper_density * float(right - left)
    
                center_masses.append(mass)
    
            center = MUD([center_points], np.array(center_masses, dtype=object))
            center_right = center_points[-1]
    
        # ---------------- right tail block ----------------
        if block_subdivision is None:
            # local breakpoints: 0, 0.1, ..., 1
            block_points = tuple(Fraction(i, 10) for i in range(11))
        else:
            if len(block_subdivision) < 2:
                raise ValueError("block_subdivision must contain at least two points")
    
            block_points = tuple(Fraction(x) for x in block_subdivision)
            if block_points[0] != 0:
                raise ValueError("block_subdivision must start at 0")
            if not block_points[-1] > 0:
                raise ValueError("block_subdivision must end with a positive value")
    
            for left, right in zip(block_points, block_points[1:]):
                if not left < right:
                    raise ValueError("block_subdivision must be strictly increasing")
    
        block_length = block_points[-1]
    
        right_masses = []
    
        for local_left, local_right in zip(block_points, block_points[1:]):
            global_left = center_right + local_left
            global_right = center_right + local_right
    
            start = global_left
            upper_density = lam * math.exp(-float(lam) * float(start))
            mass = upper_density * float(global_right - global_left)
    
            right_masses.append(mass)
    
        right = MUD([block_points], np.array(right_masses, dtype=object))
    
        # ---------------- left tail block ----------------
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
    
        alpha = 0
        beta = math.exp(-float(lam) * float(block_length))
    
        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[alpha], beta=[beta])


def mapping_to_bgd(mapping, mode) -> BGD:
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

    if mode == 'MUD':
        center = MUD([tuple(breakpoints)], np.array(masses, dtype=object))

        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[0], beta=[0])


def uniform_to_bgd(dist, mode) -> BGD:
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

    if mode == 'MUD':
        # Center block: Uniform[a,b] with total mass 1.
        center = MUD([(a, b)], [1])

        # Empty tail-start blocks.
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

        E = np.array([left, center, right], dtype=object)
    else:
        pass

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
