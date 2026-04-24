from typing import Sequence, Any
import numpy as np
from fractions import Fraction
from distributions import EED, Normal, Exponential, Uniform


def merge_eed(eeds: Sequence[EED]) -> EED:
    """
    Merge multiple mutually independent EEDs.

    The merged EED concatenates dimensions in the given order:
      - S / alpha / beta / discrete_dims are concatenated
      - P is the tensor product via broadcasting multiplication
    """
    if len(eeds) == 0:
        return None
    if len(eeds) == 1:
        return eeds[0]

    # concatenate dimensions in the given order
    S_new = [list(si) for e in eeds for si in e.S]   # copy to avoid aliasing
    alpha_new = [a for e in eeds for a in e.alpha]
    beta_new = [b for e in eeds for b in e.beta]
    discrete_dims_new = [d for e in eeds for d in e.discrete_mask]

    # tensor product of block values via broadcasting multiplication
    P_new = np.asarray(eeds[0].P).copy()
    for e in eeds[1:]:
        P_e = np.asarray(e.P)
        if P_new.ndim == 0:
            P_new = P_new * P_e
        else:
            # append dimensions of P_e to the end of P_new
            P_new = P_new.reshape(P_new.shape + (1,) * P_e.ndim) * P_e

    return EED(
        S=S_new,
        P=P_new,
        alpha=alpha_new,
        beta=beta_new,
        discrete_dims=discrete_dims_new,
    )


def merge_prior(prior: dict[tuple[str, ...], Any], conted_vars: set[str]) -> tuple[tuple[str, ...], EED]:
    """
    Merge a prior dict into a var tuple and a single joint EED.

    The variable order is determined by the iteration order of the dictionary.
    Keys (tuples of variable names) are concatenated in order, and the
    corresponding distributions are first converted to EED (if needed)
    and then merged in the same order.
    """
    
    vars_merged: list[str] = []
    eeds: list[EED] = []

    for vars_tuple, dist in prior.items():
        vars_merged.extend(vars_tuple)
        
        # Convert distribution to EED
        if isinstance(dist, EED):
            pass
        elif isinstance(dist, Normal):
            dist = dist.to_eed()
        elif isinstance(dist, (Uniform, Exponential)):
            dist = dist.to_eed()
        elif isinstance(dist, Dict):
            # Infer dimension
            n = len(next(iter(dist)))

            # Build S from the exact per-dimension support values.
            S = [sorted({pt[i] for pt in dist}) for i in range(n)]
            
            P = np.zeros(tuple(len(si) for si in S), dtype=float)
            # Fill probability table
            for pt, prob in dist.items():
                idx = tuple(S[i].index(pt[i]) for i in range(n))
                P[idx] = prob

            dist = EED(S, P, [0] * n, [0] * n, [True] * n)

        else:
            #If it is not one of the above categories, it means number.
            num = Fraction(dist)

            if var_tuple[0] in conted_vars:
                dist_obj = EED([[num]], [0, 1, 0], [0], [0], [False])
            else:
                dist_obj = EED([[num - 1, num, num + 1]], [0, 1, 0], [0], [0], [True])

        eeds.append(dist)

    merged_vars = tuple(vars_merged)
    merged_eed = merge_eed(eeds)

    return merged_vars, merged_eed
