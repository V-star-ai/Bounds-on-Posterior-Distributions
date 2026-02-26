from typing import Tuple

from EED import EED
from abc import ABC, abstractmethod


class Adapter(ABC):
    @abstractmethod
    def build_leq(self, eed_constant: EED, S):
        raise NotImplementedError

    @abstractmethod
    def solve(self, eed_z3: EED, solver) -> EED:
        raise NotImplementedError

    @abstractmethod
    def max(self, a, b):
        raise NotImplementedError
