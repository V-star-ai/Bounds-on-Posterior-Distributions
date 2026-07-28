from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Sequence

import numpy as np

from distributions.mud import (
    Breakpoints,
    CellOps,
    GridMUD,
    Interval,
    MassMUD,
    _as_fraction,
    _normalize_breakpoints,
    _object_array,
    _shape_from_breakpoints,
    _uniform_convolution_breakpoints,
    iter_indices,
)
from semantics.constraints import (
    DomainPolynomialConstraint,
    ParameterConstraint,
    Relation,
    UnitBoxDomain,
)
from semantics.polynomial import ParameterPolynomial, StatePolynomial


@dataclass(frozen=True)
class PolynomialCell:
    """A density polynomial in normalized local cell coordinates."""

    polynomial: StatePolynomial

    def __post_init__(self) -> None:
        if not isinstance(self.polynomial, StatePolynomial):
            raise TypeError("polynomial must be a StatePolynomial")


@dataclass(frozen=True)
class PolynomialCellOps(CellOps):
    ndim: int

    def __post_init__(self) -> None:
        if not isinstance(self.ndim, int) or isinstance(self.ndim, bool):
            raise TypeError("ndim must be an integer")
        if self.ndim < 0:
            raise ValueError("ndim must be nonnegative")

    def zero(self) -> PolynomialCell:
        return PolynomialCell(StatePolynomial.zero(self.ndim))

    def add(
        self,
        left: PolynomialCell,
        right: PolynomialCell,
    ) -> PolynomialCell:
        self._validate_cell(left)
        self._validate_cell(right)
        return PolynomialCell(left.polynomial + right.polynomial)

    def scale(self, value: PolynomialCell, factor) -> PolynomialCell:
        self._validate_cell(value)
        return PolynomialCell(
            value.polynomial * ParameterPolynomial.coerce(factor)
        )

    def product(
        self,
        left: PolynomialCell,
        right: PolynomialCell,
    ) -> PolynomialCell:
        if not isinstance(left, PolynomialCell) or not isinstance(
            right, PolynomialCell
        ):
            raise TypeError("polynomial cell product requires PolynomialCell values")
        return PolynomialCell(
            left.polynomial.independent_product(right.polynomial)
        )

    def restrict(
        self,
        value: PolynomialCell,
        source_intervals: tuple[Interval, ...],
        target_intervals: tuple[Interval, ...],
    ) -> PolynomialCell:
        self._validate_cell(value)
        self._validate_intervals(source_intervals, "source_intervals")
        self._validate_intervals(target_intervals, "target_intervals")

        polynomial = value.polynomial
        for dim, (source, target) in enumerate(
            zip(source_intervals, target_intervals)
        ):
            source_left, source_right = source
            target_left, target_right = target
            if target_left < source_left or target_right > source_right:
                raise ValueError("target interval must be contained in source interval")

            source_length = source_right - source_left
            target_length = target_right - target_left
            if source_length == 0:
                if target != source:
                    return self.zero()
                if polynomial.depends_on(dim):
                    raise ValueError(
                        f"polynomial depends on Dirac dimension {dim}"
                    )
                continue
            if target_length == 0:
                return self.zero()

            offset = (target_left - source_left) / source_length
            scale = target_length / source_length
            if offset != 0 or scale != 1:
                polynomial = polynomial.affine_substitute(dim, offset, scale)
        return PolynomialCell(polynomial)

    def mass(
        self,
        value: PolynomialCell,
        intervals: tuple[Interval, ...],
    ) -> ParameterPolynomial:
        self._validate_cell(value)
        self._validate_intervals(intervals, "intervals")

        volume = Fraction(1)
        active_dims = []
        for dim, (left, right) in enumerate(intervals):
            length = right - left
            if length > 0:
                volume *= length
                active_dims.append(dim)
            elif value.polynomial.depends_on(dim):
                raise ValueError(f"polynomial depends on Dirac dimension {dim}")

        result = ParameterPolynomial.zero()
        for exponents, coefficient in value.polynomial.terms.items():
            integral = coefficient
            for dim in active_dims:
                integral /= exponents[dim] + 1
            result += integral * volume
        return result

    def marginalize_dim(
        self,
        value: PolynomialCell,
        intervals: tuple[Interval, ...],
        dim: int,
    ) -> PolynomialCell:
        self._validate_cell(value)
        self._validate_intervals(intervals, "intervals")
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")

        left, right = intervals[dim]
        length = right - left
        if length == 0 and value.polynomial.depends_on(dim):
            raise ValueError(f"polynomial depends on Dirac dimension {dim}")
        factor = length if length > 0 else Fraction(1)
        return PolynomialCell(
            value.polynomial.integrate_unit(dim, remove=True) * factor
        )

    def permute_dims(
        self,
        value: PolynomialCell,
        order: Sequence[int],
    ) -> PolynomialCell:
        self._validate_cell(value)
        return PolynomialCell(value.polynomial.permute_dims(order))

    def convolve_uniform_dim(
        self,
        value: PolynomialCell,
        source_intervals: tuple[Interval, ...],
        dim: int,
        noise_left: Fraction,
        noise_right: Fraction,
        target_interval: Interval,
    ) -> PolynomialCell:
        self._validate_cell(value)
        self._validate_intervals(source_intervals, "source_intervals")
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        noise_left = _as_fraction(noise_left)
        noise_right = _as_fraction(noise_right)
        if noise_left >= noise_right:
            raise ValueError("requires noise_left < noise_right")

        target_left = _as_fraction(target_interval[0])
        target_right = _as_fraction(target_interval[1])
        if target_left >= target_right:
            raise ValueError("target_interval must have positive length")

        source_left, source_right = source_intervals[dim]
        breakpoints = _uniform_convolution_breakpoints(
            source_left,
            source_right,
            noise_left,
            noise_right,
        )
        if any(
            target_left < point < target_right
            for point in breakpoints
        ):
            raise ValueError(
                "target_interval must not cross a convolution breakpoint"
            )

        support_left = source_left + noise_left
        support_right = source_right + noise_right
        if target_right <= support_left or target_left >= support_right:
            return self.zero()

        noise_length = noise_right - noise_left
        if source_left == source_right:
            if value.polynomial.depends_on(dim):
                raise ValueError(
                    f"polynomial depends on Dirac dimension {dim}"
                )
            return PolynomialCell(value.polynomial / noise_length)

        source_length = source_right - source_left
        target_length = target_right - target_left
        midpoint = (target_left + target_right) / 2
        antiderivative = value.polynomial.antiderivative(dim)

        if midpoint - noise_left >= source_right:
            upper = antiderivative.affine_substitute(dim, 1, 0)
        else:
            upper = antiderivative.affine_substitute(
                dim,
                (target_left - noise_left - source_left) / source_length,
                target_length / source_length,
            )

        if midpoint - noise_right <= source_left:
            lower = antiderivative.affine_substitute(dim, 0, 0)
        else:
            lower = antiderivative.affine_substitute(
                dim,
                (target_left - noise_right - source_left) / source_length,
                target_length / source_length,
            )

        return PolynomialCell(
            (upper - lower) * source_length / noise_length
        )

    def evaluate(
        self,
        value: PolynomialCell,
        intervals: tuple[Interval, ...],
        point: Sequence,
    ) -> ParameterPolynomial:
        self._validate_cell(value)
        self._validate_intervals(intervals, "intervals")
        if len(point) != self.ndim:
            raise ValueError(f"point must contain {self.ndim} coordinates")

        local_point = []
        for dim, ((left, right), coordinate) in enumerate(zip(intervals, point)):
            coordinate = _as_fraction(coordinate)
            if coordinate < left or coordinate > right:
                raise ValueError(f"point coordinate {dim} is outside its cell interval")
            if left == right:
                if value.polynomial.depends_on(dim):
                    raise ValueError(
                        f"polynomial depends on Dirac dimension {dim}"
                    )
                local_point.append(Fraction(0))
            else:
                local_point.append((coordinate - left) / (right - left))
        return value.polynomial.evaluate_state(local_point)

    def is_static_zero(self, value: PolynomialCell) -> bool:
        self._validate_cell(value)
        return value.polynomial.is_zero

    def parameter_constraint(
        self,
        left,
        relation: str,
        right,
        *,
        name: str,
        constraint_factory=None,
    ) -> ParameterConstraint:
        if constraint_factory is not None:
            raise TypeError(
                "constraint_factory is not supported for polynomial constraints"
            )

        left = ParameterPolynomial.coerce(left)
        right = ParameterPolynomial.coerce(right)
        if relation == "<=":
            polynomial = right - left
            semantic_relation = Relation.GE
        elif relation == "<":
            polynomial = right - left
            semantic_relation = Relation.GT
        elif relation == "==":
            polynomial = left - right
            semantic_relation = Relation.EQ
        elif relation == ">=":
            polynomial = left - right
            semantic_relation = Relation.GE
        elif relation == ">":
            polynomial = left - right
            semantic_relation = Relation.GT
        else:
            raise ValueError("unsupported parameter constraint relation")
        return ParameterConstraint(polynomial, semantic_relation)

    def nonnegative_constraint(
        self,
        value: PolynomialCell,
        intervals: tuple[Interval, ...],
        *,
        name: str = "",
    ) -> DomainPolynomialConstraint:
        self._validate_cell(value)
        domain = self._domain_for_intervals(value, intervals)
        return DomainPolynomialConstraint(
            value.polynomial,
            Relation.GE,
            domain,
        )

    def le_constraint(
        self,
        left: PolynomialCell,
        right: PolynomialCell,
        intervals: tuple[Interval, ...],
        *,
        name: str = "",
        constraint_factory=None,
    ) -> DomainPolynomialConstraint:
        if constraint_factory is not None:
            raise TypeError(
                "constraint_factory is not supported for polynomial constraints"
            )
        self._validate_cell(left)
        self._validate_cell(right)
        domain = self._domain_for_intervals(left, intervals)
        self._domain_for_intervals(right, intervals)
        return DomainPolynomialConstraint(
            right.polynomial - left.polynomial,
            Relation.GE,
            domain,
        )

    def _domain_for_intervals(
        self,
        value: PolynomialCell,
        intervals: tuple[Interval, ...],
    ) -> UnitBoxDomain:
        self._validate_intervals(intervals, "intervals")
        active_dims = []
        for dim, (left, right) in enumerate(intervals):
            if left < right:
                active_dims.append(dim)
            elif value.polynomial.depends_on(dim):
                raise ValueError(f"polynomial depends on Dirac dimension {dim}")
        return UnitBoxDomain(self.ndim, active_dims)

    def _validate_cell(self, value: PolynomialCell) -> None:
        if not isinstance(value, PolynomialCell):
            raise TypeError("cell payload must be a PolynomialCell")
        if value.polynomial.ndim != self.ndim:
            raise ValueError(
                f"cell polynomial must contain {self.ndim} state dimensions"
            )

    def _validate_intervals(
        self,
        intervals: tuple[Interval, ...],
        name: str,
    ) -> None:
        if len(intervals) != self.ndim:
            raise ValueError(f"{name} must contain {self.ndim} intervals")
        if any(left > right for left, right in intervals):
            raise ValueError(f"{name} must contain ordered intervals")


@dataclass(frozen=True, init=False)
class PolynomialMUD(GridMUD):
    """Grid distribution with a multivariate polynomial density in each cell."""

    cell_ops: ClassVar[PolynomialCellOps] = PolynomialCellOps(0)
    _cell_ops: PolynomialCellOps

    def __init__(self, S: Sequence[Sequence], P):
        breakpoints = _normalize_breakpoints(S)
        ndim = len(breakpoints)
        payloads = _object_array(P, _shape_from_breakpoints(breakpoints), "P")
        normalized = np.empty(payloads.shape, dtype=object)
        for index in iter_indices(payloads.shape):
            value = payloads[index]
            if isinstance(value, StatePolynomial):
                value = PolynomialCell(value)
            if not isinstance(value, PolynomialCell):
                raise TypeError(
                    f"P{index} must be a PolynomialCell or StatePolynomial"
                )
            normalized[index] = value

        object.__setattr__(self, "_cell_ops", PolynomialCellOps(ndim))
        GridMUD.__init__(self, breakpoints, normalized)
        self._validate_payload_dimensions()

    @classmethod
    def from_mass_mud(cls, mud: MassMUD) -> "PolynomialMUD":
        if not isinstance(mud, MassMUD):
            raise TypeError("mud must be a MassMUD")

        payloads = np.empty(mud.shape, dtype=object)
        for index in iter_indices(mud.shape):
            intervals = mud._intervals_for_index(index)
            volume = Fraction(1)
            for left, right in intervals:
                length = right - left
                if length > 0:
                    volume *= length
            density = ParameterPolynomial.coerce(mud.P[index]) / volume
            payloads[index] = PolynomialCell(
                StatePolynomial.constant(mud.ndim, density)
            )
        return cls(mud.S, payloads)

    @classmethod
    def empty_like_restrict(
        cls,
        S: Sequence[Sequence],
        dim: int,
        point,
    ) -> "PolynomialMUD":
        breakpoints = _normalize_breakpoints(S)
        if dim < 0 or dim >= len(breakpoints):
            raise ValueError("dim out of range")

        empty_S = []
        for current_dim, sequence in enumerate(breakpoints):
            if current_dim == dim:
                empty_S.append((_as_fraction(point),))
            elif len(sequence) == 1 or sequence[0] == sequence[-1]:
                empty_S.append((sequence[0],))
            else:
                empty_S.append((sequence[0], sequence[-1]))

        shape = _shape_from_breakpoints(tuple(empty_S))
        empty_P = np.empty(shape, dtype=object)
        empty_P.fill(PolynomialCell(StatePolynomial.zero(len(breakpoints))))
        return cls(tuple(empty_S), empty_P)

    def marginalize(self, dim: int) -> "PolynomialMUD":
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if self.ndim == 1:
            raise ValueError("cannot marginalize the only MUD dimension")

        result_S = self.S[:dim] + self.S[dim + 1 :]
        result_shape = self.shape[:dim] + self.shape[dim + 1 :]
        result_ops = PolynomialCellOps(self.ndim - 1)
        result_P = np.empty(result_shape, dtype=object)
        result_P.fill(result_ops.zero())

        for index in iter_indices(self.shape):
            result_index = index[:dim] + index[dim + 1 :]
            contribution = self.ops.marginalize_dim(
                self.P[index],
                self._intervals_for_index(index),
                dim,
            )
            result_P[result_index] = result_ops.add(
                result_P[result_index],
                contribution,
            )
        return PolynomialMUD(result_S, result_P)

    def convolve_uniform(self, dim: int, low, high) -> "PolynomialMUD":
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")

        noise_left = _as_fraction(low)
        noise_right = _as_fraction(high)
        if noise_left >= noise_right:
            raise ValueError("requires low < high")
        if self.is_empty:
            return self._empty_like_restrict(
                dim,
                self.S[dim][0] + noise_left,
            )

        result_points = set()
        for interval_index in range(self.shape[dim]):
            result_points.update(
                _uniform_convolution_breakpoints(
                    self.S[dim][interval_index],
                    self.S[dim][interval_index + 1],
                    noise_left,
                    noise_right,
                )
            )

        result_S = list(self.S)
        result_S[dim] = tuple(sorted(result_points))
        result_S = tuple(result_S)
        result_shape = _shape_from_breakpoints(result_S)
        result_P = np.empty(result_shape, dtype=object)
        result_P.fill(self.ops.zero())

        for source_index in iter_indices(self.shape):
            source_intervals = self._intervals_for_index(source_index)
            value = self.P[source_index]
            for target_dim_index in range(result_shape[dim]):
                target_interval = (
                    result_S[dim][target_dim_index],
                    result_S[dim][target_dim_index + 1],
                )
                contribution = self.ops.convolve_uniform_dim(
                    value,
                    source_intervals,
                    dim,
                    noise_left,
                    noise_right,
                    target_interval,
                )
                if self.ops.is_static_zero(contribution):
                    continue

                target_index = list(source_index)
                target_index[dim] = target_dim_index
                target_index = tuple(target_index)
                result_P[target_index] = self.ops.add(
                    result_P[target_index],
                    contribution,
                )

        return PolynomialMUD(result_S, result_P)

    def permute_dims(self, order: Sequence[int]) -> "PolynomialMUD":
        order = self._validate_permutation(order, self.ndim)
        transposed = np.transpose(self.P, axes=order)
        result_P = np.empty(transposed.shape, dtype=object)
        for index in iter_indices(transposed.shape):
            result_P[index] = self.ops.permute_dims(transposed[index], order)
        result_S = tuple(self.S[dim] for dim in order)
        return PolynomialMUD(result_S, result_P)

    def _new(self, S: Sequence[Sequence], P):
        return PolynomialMUD(S, P)

    def _validate_payload_dimensions(self) -> None:
        for index in iter_indices(self.shape):
            cell = self.P[index]
            self.ops._validate_cell(cell)
            intervals = self._intervals_for_index(index)
            for dim, (left, right) in enumerate(intervals):
                if left == right and cell.polynomial.depends_on(dim):
                    raise ValueError(
                        f"P{index} polynomial depends on Dirac dimension {dim}"
                    )
