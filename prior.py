from __future__ import annotations
from typing import Dict, Sequence, Tuple, Union
import numpy as np


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
