from typing import Sequence, Dict, Tuple, Union
import numpy as np
from distributions import EED, Normal, Exponential, Uniform


def merge_eed(eeds: Sequence[EED]) -> EED:
    """
    Merge multiple mutually independent EEDs.

    The merged EED concatenates dimensions in the given order:
      - S / alpha / beta are concatenated
      - P is the tensor product via broadcasting multiplication
      - continuous_dims are shifted and merged accordingly
    """
    if len(eeds) == 0:
        return None
    if len(eeds) == 1:
        return eeds[0]

    # concatenate dimensions in the given order
    S_new = [list(si) for e in eeds for si in e.S]   # copy to avoid aliasing
    alpha_new = [a for e in eeds for a in e.alpha]
    beta_new = [b for e in eeds for b in e.beta]

    # merge continuous dimension indices with offsets
    continuous_dims_new = set()
    offset = 0
    for e in eeds:
        continuous_dims_new.update(offset + i for i in e.continuous_dims)
        offset += e.n

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
        continuous_dims=continuous_dims_new,
    )


def merge_prior(prior: Dict[Tuple[str, ...], Union[Normal, Uniform, Exponential, EED]]) -> Tuple[Tuple[str, ...], EED]:
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
        if isinstance(dist, Normal):
            dist = dist.to_eed()
        elif isinstance(dist, (Uniform, Exponential)):
            dist = dist.to_eed()
            
        eeds.append(dist)

    merged_vars = tuple(vars_merged)
    merged_eed = merge_eed(eeds)

    return merged_vars, merged_eed
