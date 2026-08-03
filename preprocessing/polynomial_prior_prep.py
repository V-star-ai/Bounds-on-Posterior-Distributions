from __future__ import annotations

from fractions import Fraction
from typing import Mapping

import numpy as np

from distributions import BGD, PolynomialMUD
from semantics.polynomial import StatePolynomial


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _empty_polynomial_mud() -> PolynomialMUD:
    return PolynomialMUD(
        [(Fraction(0),)],
        np.empty((0,), dtype=object),
    )


def _finite_support_bgd(center: PolynomialMUD) -> BGD:
    edges = np.empty((3,), dtype=object)
    edges[0] = _empty_polynomial_mud()
    edges[1] = center
    edges[2] = _empty_polynomial_mud()
    return BGD(edges, alpha=[Fraction(0)], beta=[Fraction(0)])


def empty_polynomial_bgd() -> BGD:
    return _finite_support_bgd(_empty_polynomial_mud())


def uniform_to_polynomial_bgd(left, right) -> BGD:
    left = _as_fraction(left)
    right = _as_fraction(right)
    if left >= right:
        raise ValueError("Uniform requires left < right")

    density = StatePolynomial.constant(1, 1 / (right - left))
    return _finite_support_bgd(PolynomialMUD([[left, right]], [density]))


def mapping_to_polynomial_bgd(mapping: Mapping) -> BGD:
    if not isinstance(mapping, Mapping):
        raise TypeError("mapping must be a mapping")
    if not mapping:
        return empty_polynomial_bgd()

    items = sorted(
        (
            (_as_fraction(point), _as_fraction(mass))
            for point, mass in mapping.items()
        ),
        key=lambda item: item[0],
    )
    if any(mass < 0 for _, mass in items):
        raise ValueError("mapping masses must be nonnegative")

    breakpoints = [items[0][0], items[0][0]]
    payloads = [StatePolynomial.constant(1, items[0][1])]
    for point, mass in items[1:]:
        if point == breakpoints[-1]:
            raise ValueError("mapping points must be distinct")
        breakpoints.extend((point, point))
        payloads.extend(
            (
                StatePolynomial.zero(1),
                StatePolynomial.constant(1, mass),
            )
        )
    return _finite_support_bgd(PolynomialMUD([breakpoints], payloads))


def num_to_polynomial_bgd(value) -> BGD:
    point = _as_fraction(value)
    mass = StatePolynomial.constant(1, 1)
    return _finite_support_bgd(PolynomialMUD([[point, point]], [mass]))


def distribution_to_polynomial_bgd(dist_spec) -> BGD:
    dist_name, params = dist_spec
    if dist_name == "Uniform":
        left, right = params
        return uniform_to_polynomial_bgd(left, right)
    if dist_name == "Mapping":
        return mapping_to_polynomial_bgd(params)
    if dist_name == "Num":
        return num_to_polynomial_bgd(params)
    if dist_name in ("Normal", "Exponential"):
        raise ValueError(
            f"{dist_name} has no exact finite piecewise-polynomial representation"
        )
    raise ValueError(f"unsupported polynomial prior distribution: {dist_name!r}")


def prior_to_polynomial_bgd(prior: dict) -> tuple[BGD, tuple[str, ...]]:
    if not prior:
        return empty_polynomial_bgd(), ("x",)

    components = []
    variable_order = []
    for variables, dist_spec in prior.items():
        component = distribution_to_polynomial_bgd(dist_spec)
        if len(variables) != component.ndim:
            raise ValueError(
                f"variables {variables} has length {len(variables)}, "
                f"but converted BGD has dimension {component.ndim}"
            )
        variable_order.extend(variables)
        components.append(component)

    result = components[0]
    for component in components[1:]:
        result = result.independent_product(component)
    return result, tuple(variable_order)


__all__ = (
    "distribution_to_polynomial_bgd",
    "empty_polynomial_bgd",
    "mapping_to_polynomial_bgd",
    "num_to_polynomial_bgd",
    "prior_to_polynomial_bgd",
    "uniform_to_polynomial_bgd",
)
