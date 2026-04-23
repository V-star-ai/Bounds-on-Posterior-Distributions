import numpy as np
from fractions import Fraction

from distributions import EED


class Uniform:
    def __init__(self, lb: Fraction, ub: Fraction):
        self.lb = Fraction(lb)
        self.ub = Fraction(ub)
        
        # Check the validity of parameters.
        if self.lb >= self.ub:
            raise ValueError("requires lb < ub")

    def to_eed(self) -> EED:
        density = 1.0 / float(self.ub - self.lb)
        P = np.array([0, density, 0], dtype=float)
        return EED([[self.lb, self.ub]], P, [0.0], [0.0])
        
    def __str__(self) -> str:
        return f"Uniform({self.lb},{self.ub})"

    def __repr__(self) -> str:
        return f"Uniform({self.lb},{self.ub})"
