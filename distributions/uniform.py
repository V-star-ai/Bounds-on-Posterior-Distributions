import numpy as np
from fractions import Fraction

from distributions import EED


class Uniform:
    def __init__(self, lb: Fraction, ub: Fraction):
        self.lb = Fraction(lb)
        self.ub = Fraction(ub)

        if self.lb > self.ub:
            raise ValueError("requires lb <= ub")

    def to_eed(self) -> EED:
        if self.lb == self.ub:
            
            int_lb = int(self.lb)
            if self.lb != int_lb:
                raise ValueError("The support of a discrete distribution must be integers.")
                
            return EED([[int_lb - 1, int_lb, int_lb + 1]], [0, 1, 0], [0], [0], [True])
            
        else:
            density = 1.0 / float(self.ub - self.lb)
            P = np.array([0, density, 0], dtype=float)
            return EED([[self.lb, self.ub]], P, [0.0], [0.0])
        
    def __str__(self) -> str:
        return f"Uniform({self.lb},{self.ub})"

    def __repr__(self) -> str:
        return f"Uniform({self.lb},{self.ub})"
