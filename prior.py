from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence, Dict, Union, Tuple
import math
import numpy as np
from numpy.typing import NDArray
from utils import to_simplest_fraction
from EED import EED

@dataclass
class Normal:
    mean: NDArray[np.float64]
    cov: NDArray[np.float64]
    
    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        self.cov = np.asarray(self.cov, dtype=float)
        
        # Allow scalar inputs for 1D cases
        if self.mean.ndim == 0:
            self.mean = self.mean[None]          # shape (1,)
        self.dim = len(self.mean)
        if self.cov.ndim == 0:
            self.cov = self.cov[None, None]      # shape (1,1)

        # precompute for eval_at
        self._inv_cov = np.linalg.inv(self.cov)
        self._det_cov = np.linalg.det(self.cov)

    def eval_at(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[None]
        diff = x - self.mean
        return float(math.exp(-0.5 * diff @ self._inv_cov @ diff) / math.sqrt((2 * math.pi) ** self.dim * self._det_cov))
    
    def to_eventual_exp(self, S: Sequence[Sequence[Fraction]] = None) -> EventualExp:
        if self.dim == 1:
            mean = self.mean[0]
            var = self.cov[0, 0]
            
            if S is not None:
                # If user provides S, require at least one breakpoints.
                if len(S[0]) == 0:
                    raise ValueError("S[0] must contain at least 1 breakpoints.")
                S0 = np.asarray(S[0], dtype=float)
            else:
                # Default partition:
                # 1) std_trunc  = truncate(std, 1 decimal)
                # 2) T          = max(std_trunc, 1.0)
                # 3) mean_trunc = truncate(mean, 1 decimal)
                # 4) interval   = [mean_trunc - T, mean_trunc + T]
                # 5) breakpoints every 0.1, inclusive endpoints

                # Convert to decimal-semantics Fractions first (avoids binary-float artifacts)
                mean_q = Fraction(str(mean))
                std = math.sqrt(var)
                std_q = Fraction(str(std))

                # Truncate to 1 decimal in Fraction land: trunc1(x) = floor(10*x)/10
                mean_trunc = (mean_q * 10) // 1
                mean_trunc = Fraction(mean_trunc, 10)

                std_trunc = (std_q * 10) // 1
                std_trunc = Fraction(std_trunc, 10)

                T = max(std_trunc, Fraction(1, 1))
                lb = mean_trunc - T
                ub = mean_trunc + T

                step = Fraction(1, 10)
                n = int((ub - lb) / step)                           
                S0 = [lb + i * step for i in range(n + 1)]          
                
            # Require the core interval to contain the mean, otherwise the bound can be very loose.
            l = S0[0]
            r = S0[-1]
            if not (l <= mean <= r):
                raise ValueError(
                    f"S[0] must contain the mean (got mean={mean}, interval=[{l}, {r}]). "
                    "Otherwise the eventual-exponential upper bound may be too loose."
                )
                
            P = np.zeros(max(3, len(S0)+1), dtype=float)

            # Endpoint pdf values
            f_l = self.eval_at(l)
            f_r = self.eval_at(r)

            # ---- Minimum-tail-mass exponential upper bound for 1D Normal----
            # given l <= mean <= r:
            # u = mean - l >= 0, d = (u + sqrt(u^2 + 4*var))/2
            # alpha = exp(-d/var), P0 = f(l) * exp(var/(2*d^2))
            # Symmetric for right tail with u = r - mean.

            uL = mean - l
            dL = 0.5 * (uL + math.sqrt(uL * uL + 4.0 * var))
            alpha = math.exp(-dL / var)
            P0 = f_l * math.exp(var / (2.0 * dL * dL))

            uR = r - mean
            dR = 0.5 * (uR + math.sqrt(uR * uR + 4.0 * var))
            beta = math.exp(-dR / var)
            Pm = f_r * math.exp(var / (2.0 * dR * dR))

            # Tail anchors
            P[0] = P0
            P[-1] = Pm
            
            if len(S0) <= 2:
                # Single core cell that contains the mean
                P[1] = self.eval_at(mean)
            else:
                # Interior segments: j=1..m-1 corresponds to [S0[j-1], S0[j])
                for j in range(1, len(S0)):
                    a = S0[j - 1]
                    b = S0[j]

                    # Max of unimodal normal pdf on [a, b]:
                    # - if mean in [a, b], max at mean
                    # - else max at the endpoint closer to mean
                    if a <= mean <= b:
                        x_star = mean
                    else:
                        x_star = a if abs(a - mean) <= abs(b - mean) else b

                    P[j] = self.eval_at(x_star)

            return EventualExp([S0], P, [alpha], [beta])

        else:
            raise NotImplementedError("Only dim==1 is supported for now.")
            
    def __str__(self) -> str:
        return f"mean: {self.mean}\ncov: {self.cov}"
        

    
@dataclass
class UniformBox:
    S: List[List[Fraction]]
    P: NDArray[np.float64]

    def __post_init__(self):
        # normalize
        self.S = [[to_simplest_fraction(bp) for bp in si] for si in self.S]
        self.P = np.asarray(self.P, dtype=float)
        self.dim = len(self.S)
        
        # forbid degeneracy into a discrete distribution
        for i, si in enumerate(self.S):
            if len(si) < 2:
                raise ValueError(f"S[{i}] must contain at least 2 breakpoints to avoid degeneracy; got {len(si)}")
            
        expected_spatial = tuple(len(si)-1 for si in self.S)
        if self.P.shape[:self.dim] != expected_spatial:
            raise ValueError(
                f"The first {self.dim} dimensions of P must have shape {expected_spatial}, "
                f"but got {self.P.shape[:self.dim]}."
            )
    
    def to_eventual_exp(self) -> EventualExp:
        dim = self.dim
        S_copy = [si.copy() for si in self.S]

        alpha = [0.0] * dim
        beta = [0.0] * dim

        full_shape = tuple(len(si) + 1 for si in S_copy)
        P_full = np.zeros(full_shape, dtype=float)

        interior_slices = tuple(slice(1, -1) for _ in range(dim))
        P_full[interior_slices] = self.P 

        return EventualExp(S=S_copy, P=P_full, alpha=alpha, beta=beta)

    def __str__(self) -> str:
        return f"S: {self.S}\nP: {self.P}"
    
            
        
        
@dataclass
class EventualExp:
    S: List[List[Fraction]]
    P: NDArray[np.float64]
    alpha: List[float]
    beta: List[float]

    def __post_init__(self):
        # normalize
        self.S = [[to_simplest_fraction(bp) for bp in si] for si in self.S]
        self.P = np.asarray(self.P, dtype=float)
        self.alpha = [float(al) for al in self.alpha]
        self.beta = [float(be) for be in self.beta]
        self.dim = len(self.S)
        
        # if either the decay rate or the boundary value is zero, set both to zero
        for i in range(self.dim):
            slicer = [slice(None)] * self.dim

            # left tail
            slicer[i] = 0
            if self.alpha[i] == 0 or np.all(self.P[tuple(slicer)] == 0):
                self.alpha[i] = 0.0
                self.P[tuple(slicer)] = 0.0

            # right tail
            slicer[i] = -1
            if self.beta[i] == 0 or np.all(self.P[tuple(slicer)] == 0):
                self.beta[i] = 0.0
                self.P[tuple(slicer)] = 0.0

        if len(self.alpha) != self.dim or len(self.beta) != self.dim:
            raise ValueError(f"alpha and beta must have length equal to the dimension = {self.dim}")

        for i, si in enumerate(self.S):
            if len(si) < 1:
                raise ValueError(f"S[{i}] must be one-dimensional with at least one breakpoint")

            if len(si) >= 2 and not all(si[j+1] > si[j] for j in range(len(si)-1)):
                raise ValueError(f"S[{i}] breakpoints must be strictly increasing")
                
        expected_spatial = tuple(max(3, len(si) + 1) for si in self.S)
        if self.P.shape[:self.dim] != expected_spatial:
            raise ValueError(
                f"The first {self.dim} dimensions of P must have shape {expected_spatial}, "
                f"but got {self.P.shape[:self.dim]}."
            )
            
        # forbid degeneracy into a discrete distribution
        if (all(len(si) == 1 for si in self.S) and not any(self.alpha) and not any(self.beta)):
            raise ValueError("The prior must not be discrete")
            
    def scale_integerize_to_eed(self, k: Sequence[int]) -> EED:
        """
        Scale + integerize with user-provided integer sequence k.

        Transform: x_new_i = k[i] * x_i
        - Breakpoints: S'_i[j] = int(k[i] * S_i[j])   (caller guarantees integerization)
        - Density: P' = P / (k[0] * k[1] * ... * k[dim-1])
        - Tails: alpha'_i = alpha_i ** (1/k[i]), beta'_i  = beta_i  ** (1/k[i])
        """
        # Validate k: must be a length-dim sequence of integers >= 1
        if len(k) != self.dim:
            raise ValueError(f"k must have length {self.dim}, got {len(k)}")

        for i, ki in enumerate(k):
            if not isinstance(ki, int):
                raise TypeError(f"k[{i}] must be an int, got {type(ki)}")
            if ki < 1:
                raise ValueError(f"k[{i}] must be >= 1, got {ki}")

        # 1) scaled integer breakpoints (deep copy)
        S_new: List[List[int]] = []
        for i in range(self.dim):
            S_new.append([int(k[i] * bp) for bp in self.S[i]])

        # 2) Jacobian scaling for densities (deep copy)
        prod = 1
        for ki in k:
            prod *= ki
        P_new = (self.P / prod).copy()

        # 3) decay-rate rescaling
        alpha_new = self.alpha.copy()
        beta_new  = self.beta.copy()
        for i in range(self.dim):
            if alpha_new[i] != 0.0:
                alpha_new[i] = alpha_new[i] ** (1.0 / k[i])
            if beta_new[i] != 0.0:
                beta_new[i] = beta_new[i] ** (1.0 / k[i])

        return EED(S=S_new, P=P_new, alpha=alpha_new, beta=beta_new)
    
    # ---------- merge/product API ----------
    
    @classmethod
    def product(cls, *parts: EventualExp) -> EventualExp:
        """
        Dimension-wise product (independent join):
          - concatenate S/alpha/beta
          - P becomes the outer product via broadcasting multiplication
        """
        parts = [p for p in parts if p is not None]
        if len(parts) == 0:
            return None

        S: List[List[Fraction]] = []
        alpha: List[float] = []
        beta: List[float] = []
        P = np.array(1.0)

        for p in parts:
            if not isinstance(p, cls):
                raise TypeError(f"Expected {cls.__name__}, got {type(p)}")

            S.extend([*si] for si in p.S)
            alpha.extend(p.alpha)
            beta.extend(p.beta)

            # append dimensions by broadcasting multiplication
            if P.ndim == 0:
                P = P * p.P
            else:
                P = P.reshape(P.shape + (1,) * p.P.ndim) * p.P

        return cls(S=S, P=P, alpha=alpha, beta=beta)

    def __matmul__(self, other: EventualExp) -> EventualExp:
        if not isinstance(other, EventualExp):
            return NotImplemented
        return EventualExp.product(self, other)

    def __rmatmul__(self, other: EventualExp) -> EventualExp:
        if not isinstance(other, EventualExp):
            return NotImplemented
        return EventualExp.product(other, self)
    
    def __str__(self) -> str:
        return f"S: {self.S}\nP: {self.P}\nalpha: {self.alpha}\nbeta: {self.beta}"


def merge_eventual_exp(exps: Sequence[EventualExp]) -> EventualExp:
    """
    Merge multiple mutually independent EventualExp.
    """
    if len(exps) == 0:
        raise ValueError("exps must be non-empty")
    if len(exps) == 1:
        return exps[0]

    # concatenate dimensions in the given order
    S_new = [list(si) for e in exps for si in e.S]   # copy to avoid aliasing
    alpha_new = [a for e in exps for a in e.alpha]
    beta_new  = [b for e in exps for b in e.beta]

    # tensor product of core block constants via broadcasting multiplication
    P_new = np.asarray(exps[0].P, dtype=float).copy()
    for e in exps[1:]:
        P_e = np.asarray(e.P, dtype=float)
        if P_new.ndim == 0:
            P_new = P_new * P_e
        else:
            # append dimensions of P_e to the end of P_new
            P_new = P_new.reshape(P_new.shape + (1,) * P_e.ndim) * P_e

    return EventualExp(S=S_new, P=P_new, alpha=alpha_new, beta=beta_new)


def merge_prior(prior: Dict[Tuple[str, ...], Union[UniformBox, EventualExp, Normal]]) -> Tuple[Tuple[str, ...], EventualExp]:
    """
    Merge a prior dict into a var tuple and a single joint EventualExp.

    The variable order is determined by the iteration order of the dictionary.
    Keys (tuples of variable names) are concatenated in order, and the
    corresponding distributions are first converted to EventualExp (if needed)
    and then merged in the same order.
    """
    vars_merged: list[str] = []
    exps: list[Union[UniformBox, EventualExp, Normal]] = []

    # Convert UniformBox and Normal to EventualExp
    for vars_tuple, dist in prior.items():
        vars_merged.extend(vars_tuple)
        if isinstance(dist, UniformBox):
            dist = dist.to_eventual_exp()
        elif isinstance(dist, Normal):
            dist = dist.to_eventual_exp()
        exps.append(dist)

    merged_vars = tuple(vars_merged)
    merged_exp = merge_eventual_exp(exps)

    return merged_vars, merged_exp
