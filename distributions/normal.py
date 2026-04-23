from fractions import Fraction
from typing import Any, Sequence
import math
import numpy as np
from distributions import EED



class Normal:
    def __init__(self, mean: Sequence[float], cov: Sequence[Sequence[float]]):
        
        # Check the validity of cov and prevent the normal from degenerating into a discrete distribution
        for row in cov:
            for value in row:
                if value <= 0:
                    raise ValueError("Invalid covariance matrix")

        # Normalize inputs
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        
        # Allow scalar inputs for 1D cases
        if self.mean.ndim == 0:
            self.mean = self.mean[None]          # shape (1,)
        if self.cov.ndim == 0:
            self.cov = self.cov[None, None]      # shape (1,1)
            
        self.dim = len(self.mean)

        # shape checks
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be a scalar or a 1D sequence, got ndim={self.mean.ndim}")
        if self.cov.shape != (self.dim, self.dim):
            raise ValueError(f"cov must have shape ({self.dim}, {self.dim}), got {self.cov.shape}")

        # precompute for eval_at
        self._inv_cov = np.linalg.inv(self.cov)
        self._det_cov = np.linalg.det(self.cov)

    def eval_at(self, x) -> float:
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[None]
        diff = x - self.mean
        return float(math.exp(-0.5 * diff @ self._inv_cov @ diff) / math.sqrt((2 * math.pi) ** self.dim * self._det_cov))
    
    def to_eed(self, S: Sequence[Sequence[Fraction]] = None) -> EED:
        if self.dim == 1:
            mean = self.mean[0]
            var = self.cov[0, 0]
            
            if S is not None:
                # If user provides S, require at least one breakpoints.
                if len(S[0]) == 0:
                    raise ValueError("S[0] must contain at least 1 breakpoint.")
                S0 = np.asarray(S[0], dtype=float)
            else:
                # Default partition:
                # 1) scaled_std_trunc = truncate(1.2 * std, 1 decimal)
                # 2) T                = max(scaled_std_trunc, 1.0)
                # 3) mean_trunc       = truncate(mean, 1 decimal)
                # 4) interval         = [mean_trunc - T, mean_trunc + T]
                # 5) breakpoints every 0.1, inclusive endpoints

                # Convert to decimal-semantics Fractions first (avoids binary-float artifacts)
                mean_q = Fraction(str(mean))
                std = math.sqrt(var)
                scaled_std_q = Fraction(str(1.2 * std))

                # Truncate to 1 decimal in Fraction land: trunc1(x) = floor(10*x)/10
                mean_trunc = (mean_q * 10) // 1
                mean_trunc = Fraction(mean_trunc, 10)

                scaled_std_trunc = (scaled_std_q * 10) // 1
                scaled_std_trunc = Fraction(scaled_std_trunc, 10)

                T = max(scaled_std_trunc, Fraction(1, 1))
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
                
            P = np.zeros(len(S0)+1, dtype=float)

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

            return EED([S0], P, [alpha], [beta])

        else:
            raise NotImplementedError("Only dim==1 is supported for now.")
            
    def __str__(self) -> str:
        return f"Normal({self.mean},{self.cov})"
    
    def __repr__(self) -> str:
        return f"Normal(mean={self.mean}, cov={self.cov})"
