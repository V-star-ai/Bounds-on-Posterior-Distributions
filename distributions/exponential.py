import math
import numpy as np

from distributions import EED


class Exponential:
    def __init__(self, lam: float):
        self.lam = lam

        if self.lam <= 0:
            raise ValueError("Exponential rate λ must be positive")

    def to_eed(self) -> EED:
        alpha = [math.exp(-self.lam)]
        P = np.array([0.0, self.lam, self.lam], dtype=float)
        return EED([[0]], P, alpha, [0.0], [False])
    
    def __str__(self) -> str:
        return f"Exponential({self.lam})"

    def __repr__(self) -> str:
        return f"Exponential({self.lam})"
