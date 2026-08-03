from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from numbers import Integral

import numpy as np

from distributions.bgd import BGD
from distributions.mud import iter_indices
from distributions.polynomial_mud import PolynomialMUD
from semantics.constraints import ConstraintContext
from semantics.polynomial import ParameterPolynomial, StatePolynomial


def symbolic_polynomial_bgd_template(
    shape: BGD,
    total_degree: int | Sequence[int],
    context: ConstraintContext,
    *,
    name_prefix: str = "template",
) -> BGD:
    """Create a polynomial BGD template and register all validity constraints.

    An integer selects the legacy total-degree basis. A sequence gives one
    maximum exponent per state dimension and selects the tensor-product basis.
    """

    if not isinstance(shape, BGD):
        raise TypeError("shape must be a BGD")
    degree_spec = _normalize_degree_spec(total_degree, shape.ndim)
    if not isinstance(context, ConstraintContext):
        raise TypeError("context must be a ConstraintContext")
    if not isinstance(name_prefix, str) or not name_prefix.strip():
        raise ValueError("name_prefix must be a non-empty string")
    if name_prefix != name_prefix.strip():
        raise ValueError("name_prefix must not have surrounding whitespace")

    canonical_shape = shape.standardize(skip_static_zero=False)
    E = np.empty(canonical_shape.E.shape, dtype=object)
    for edge_index in iter_indices(canonical_shape.E.shape):
        shape_mud = canonical_shape.E[edge_index]
        payloads = np.empty(shape_mud.shape, dtype=object)
        for cell_index in iter_indices(shape_mud.shape):
            intervals = shape_mud._intervals_for_index(cell_index)
            active_dims = tuple(
                dim
                for dim, (left, right) in enumerate(intervals)
                if left < right
            )
            terms = {}
            if isinstance(degree_spec, int):
                exponents_iter = _total_degree_exponents(
                    canonical_shape.ndim,
                    active_dims,
                    degree_spec,
                )
            else:
                exponents_iter = _axis_degree_exponents(
                    canonical_shape.ndim,
                    active_dims,
                    degree_spec,
                )
            for exponents in exponents_iter:
                variable = context.declare(
                    _coefficient_name(
                        name_prefix,
                        edge_index,
                        cell_index,
                        exponents,
                    )
                )
                terms[exponents] = ParameterPolynomial.variable(variable)
            payloads[cell_index] = StatePolynomial(
                canonical_shape.ndim,
                terms,
            )
        E[edge_index] = PolynomialMUD(shape_mud.S, payloads)

    alpha = tuple(
        ParameterPolynomial.variable(
            context.declare(f"{name_prefix}_alpha_{dim}")
        )
        for dim in range(canonical_shape.ndim)
    )
    beta = tuple(
        ParameterPolynomial.variable(
            context.declare(f"{name_prefix}_beta_{dim}")
        )
        for dim in range(canonical_shape.ndim)
    )
    template = BGD(E, alpha, beta)
    for constraint in template.nonnegative_constraints():
        context.add(constraint)
    return template


def _normalize_degree_spec(
    degree: int | Sequence[int],
    ndim: int,
) -> int | tuple[int, ...]:
    if isinstance(degree, Integral) and not isinstance(degree, bool):
        degree = int(degree)
        if degree < 0:
            raise ValueError("total_degree must be nonnegative")
        return degree
    if isinstance(degree, (str, bytes)) or not isinstance(degree, Sequence):
        raise TypeError(
            "total_degree must be an integer or a per-dimension sequence"
        )
    if len(degree) != ndim:
        raise ValueError(
            "per-dimension degree sequence length must match shape dimensions"
        )
    normalized = []
    for value in degree:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError("per-dimension degrees must be integers")
        value = int(value)
        if value < 0:
            raise ValueError("per-dimension degrees must be nonnegative")
        normalized.append(value)
    return tuple(normalized)


def _total_degree_exponents(
    ndim: int,
    active_dims: tuple[int, ...],
    total_degree: int,
):
    for active_exponents in product(
        range(total_degree + 1),
        repeat=len(active_dims),
    ):
        if sum(active_exponents) > total_degree:
            continue
        exponents = [0] * ndim
        for dim, exponent in zip(active_dims, active_exponents):
            exponents[dim] = exponent
        yield tuple(exponents)


def _axis_degree_exponents(
    ndim: int,
    active_dims: tuple[int, ...],
    degrees: tuple[int, ...],
):
    for active_exponents in product(
        *(range(degrees[dim] + 1) for dim in active_dims),
    ):
        exponents = [0] * ndim
        for dim, exponent in zip(active_dims, active_exponents):
            exponents[dim] = exponent
        yield tuple(exponents)


def _coefficient_name(
    prefix: str,
    edge_index: tuple[int, ...],
    cell_index: tuple[int, ...],
    exponents: tuple[int, ...],
) -> str:
    edge = "_".join(str(value) for value in edge_index)
    cell = "_".join(str(value) for value in cell_index)
    powers = "_".join(str(value) for value in exponents)
    return f"{prefix}_E_{edge}_cell_{cell}_coef_{powers}"
