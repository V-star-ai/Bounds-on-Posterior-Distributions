from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from numbers import Real
from typing import Iterable, Mapping, Sequence

import numpy as np


Breakpoints = tuple[tuple[Fraction, ...], ...]
Direction = tuple[int, ...]
Index = tuple[int, ...]
Interval = tuple[Fraction, Fraction]


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    return Fraction(value)


def _normalize_breakpoints(S: Sequence[Sequence]) -> Breakpoints:
    if not isinstance(S, Sequence) or len(S) == 0:
        raise ValueError("S must be a non-empty sequence of breakpoint sequences")

    normalized = []
    for dim, breakpoints in enumerate(S):
        if not isinstance(breakpoints, Sequence) or len(breakpoints) < 1:
            raise ValueError(f"S[{dim}] must contain at least one breakpoint")

        seq = tuple(_as_fraction(x) for x in breakpoints)
        for left, right in zip(seq, seq[1:]):
            if left > right:
                raise ValueError(f"S[{dim}] must be non-decreasing")
        normalized.append(seq)

    return tuple(normalized)


def _shape_from_breakpoints(S: Breakpoints) -> tuple[int, ...]:
    return tuple(len(s) - 1 for s in S)


def _object_array(values, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.shape != shape:
        raise ValueError(f"{name} shape must be {shape}, got {array.shape}")
    return array.copy()


def _direction_to_index(direction: Direction) -> Index:
    return tuple(d + 1 for d in direction)


def _index_to_direction(index: Index) -> Direction:
    return tuple(i - 1 for i in index)


def _is_static_real(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_static_zero(value) -> bool:
    return _is_static_real(value) and value == 0


def _default_decay_max(left, right, name: str):
    if not _is_static_real(left) or not _is_static_real(right):
        raise ValueError(f"{name} max requires statically comparable numeric decays")
    return max(left, right)


def _array_is_static_zero(array: np.ndarray) -> bool:
    return all(_is_static_zero(value) for value in array.flat)


def _validate_decay(value, name: str) -> None:
    if _is_static_real(value) and not (0 <= value < 1):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")


def _iter_indices(shape: tuple[int, ...]) -> Iterable[Index]:
    return product(*(range(n) for n in shape))


@dataclass(frozen=True)
class BGDBlock:
    index: Index
    direction: Direction
    distribution: "MUD"
    translation: tuple[Fraction, ...]
    decay_factor: object


def object_sum(values: Iterable, start=0):
    result = start
    for value in values:
        result = result + value
    return result


def object_product(values: Iterable, start=1):
    result = start
    for value in values:
        result = result * value
    return result


def _object_power(value, exponent: int):
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    if exponent == 0:
        return 1
    if exponent == 1:
        return value
    return value**exponent


def fraction_lcm(*values) -> Fraction:
    if not values:
        raise ValueError("fraction_lcm requires at least one value")

    fractions = tuple(_as_fraction(value) for value in values)
    for value in fractions:
        if value <= 0:
            raise ValueError("fraction_lcm values must be positive")

    numerator_lcm = 1
    denominator_gcd = fractions[0].denominator
    for value in fractions:
        numerator_lcm = _int_lcm(numerator_lcm, abs(value.numerator))
        denominator_gcd = gcd(denominator_gcd, value.denominator)
    return Fraction(numerator_lcm, denominator_gcd)


def _int_lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)


def scale_object_array(array: np.ndarray, factor) -> np.ndarray:
    scaled = np.empty(array.shape, dtype=object)
    for index in iter_indices(array.shape):
        scaled[index] = array[index] * factor
    return scaled


def iter_indices(shape: Sequence[int]) -> Iterable[Index]:
    return _iter_indices(tuple(shape))


def is_dirac_interval(left, right) -> bool:
    return _as_fraction(left) == _as_fraction(right)


def interval_length(left, right) -> Fraction:
    left = _as_fraction(left)
    right = _as_fraction(right)
    if left > right:
        raise ValueError("requires left <= right")
    return right - left


def interval_intersection(left_a, right_a, left_b, right_b) -> Interval | None:
    left_a = _as_fraction(left_a)
    right_a = _as_fraction(right_a)
    left_b = _as_fraction(left_b)
    right_b = _as_fraction(right_b)
    if left_a > right_a or left_b > right_b:
        raise ValueError("requires left <= right")

    left = max(left_a, left_b)
    right = min(right_a, right_b)
    if left > right:
        return None
    return left, right


def point_in_interval(point, left, right) -> bool:
    point = _as_fraction(point)
    left = _as_fraction(left)
    right = _as_fraction(right)
    if left > right:
        raise ValueError("requires left <= right")
    return left <= point <= right


def merge_breakpoints(
    *sequences: Sequence, preserve_dirac: bool = True
) -> tuple[Fraction, ...]:
    points = {_as_fraction(point) for sequence in sequences for point in sequence}
    if len(points) < 1:
        raise ValueError("merged breakpoints must contain at least one point")

    dirac_points = set()
    if preserve_dirac:
        for sequence in sequences:
            normalized = tuple(_as_fraction(point) for point in sequence)
            for left, right in zip(normalized, normalized[1:]):
                if left == right:
                    dirac_points.add(left)

    result = []
    for point in sorted(points):
        result.append(point)
        if point in dirac_points:
            result.append(point)
    return tuple(result)


def _support_is_covered(source: Breakpoints, target: Breakpoints) -> bool:
    for source_dim, target_dim in zip(source, target):
        if len(source_dim) == 1:
            if not point_in_interval(source_dim[0], target_dim[0], target_dim[-1]):
                return False
            continue

        if target_dim[0] > source_dim[0] or target_dim[-1] < source_dim[-1]:
            return False

        source_dirac_points = {
            left for left, right in zip(source_dim, source_dim[1:]) if left == right
        }
        target_dirac_points = {
            left for left, right in zip(target_dim, target_dim[1:]) if left == right
        }
        if not source_dirac_points.issubset(target_dirac_points):
            return False

    return True


def _find_dirac_target_index(point: Fraction, target: tuple[Fraction, ...]) -> int | None:
    for index, (left, right) in enumerate(zip(target, target[1:])):
        if left == point and right == point:
            return index
    return None


def _target_interval_owns_point(
    point: Fraction, target: tuple[Fraction, ...], target_index: int
) -> bool:
    dirac_index = _find_dirac_target_index(point, target)
    return target_index == dirac_index


def _interval_overlap_ratio(
    source_left: Fraction,
    source_right: Fraction,
    target: tuple[Fraction, ...],
    target_index: int,
) -> Fraction:
    target_left = target[target_index]
    target_right = target[target_index + 1]

    if source_left == source_right:
        if _target_interval_owns_point(source_left, target, target_index):
            return Fraction(1)
        return Fraction(0)

    if target_left == target_right:
        return Fraction(0)

    intersection = interval_intersection(
        source_left, source_right, target_left, target_right
    )
    if intersection is None:
        return Fraction(0)

    left, right = intersection
    return (right - left) / (source_right - source_left)


def _boundary_slice_mud(mud: "MUD", dim: int, side: str) -> "MUD | None":
    if mud.is_empty:
        return None

    if side == "left":
        interval_index = 0
        if mud.S[dim][0] != mud.S[dim][1]:
            return None
        point = mud.S[dim][0]
    elif side == "right":
        interval_index = mud.shape[dim] - 1
        if mud.S[dim][-2] != mud.S[dim][-1]:
            return None
        point = mud.S[dim][-1]
    else:
        raise ValueError("side must be 'left' or 'right'")

    slicer = [slice(None)] * mud.ndim
    slicer[dim] = interval_index
    boundary_values = np.asarray(mud.P[tuple(slicer)], dtype=object).copy()
    boundary_P = np.expand_dims(boundary_values, axis=dim)

    boundary_S = list(mud.S)
    boundary_S[dim] = (point, point)
    return MUD(tuple(boundary_S), boundary_P)


def _remove_boundary_slice(mud: "MUD", dim: int, side: str) -> "MUD":
    if mud.is_empty:
        return mud.copy()

    if side == "left":
        interval_index = 0
        if mud.S[dim][0] != mud.S[dim][1]:
            return mud.copy()
        breakpoint_index = 0
    elif side == "right":
        interval_index = mud.shape[dim] - 1
        if mud.S[dim][-2] != mud.S[dim][-1]:
            return mud.copy()
        breakpoint_index = len(mud.S[dim]) - 1
    else:
        raise ValueError("side must be 'left' or 'right'")

    result_S = list(mud.S)
    result_S[dim] = (
        mud.S[dim][:breakpoint_index] + mud.S[dim][breakpoint_index + 1 :]
    )
    result_P = np.delete(mud.P, interval_index, axis=dim)
    return MUD(tuple(result_S), result_P)


def _move_slice_to_point(mud: "MUD", dim: int, point) -> "MUD":
    point = _as_fraction(point)
    moved_S = list(mud.S)
    moved_S[dim] = (point, point)
    return MUD(tuple(moved_S), mud.P.copy())


def _shift_mud_dim(mud: "MUD", dim: int, offset) -> "MUD":
    offset = _as_fraction(offset)
    shifted_S = list(mud.S)
    shifted_S[dim] = tuple(point + offset for point in mud.S[dim])
    return MUD(tuple(shifted_S), mud.P.copy())


def _zero_mud(S: Sequence[Sequence]) -> "MUD":
    breakpoints = _normalize_breakpoints(S)
    shape = _shape_from_breakpoints(breakpoints)
    P = np.empty(shape, dtype=object)
    P.fill(0)
    return MUD(breakpoints, P)


def _align_mud_dim_to_extent(
    mud: "MUD", dim: int, left: Fraction, right: Fraction
) -> "MUD":
    if left > right:
        raise ValueError("requires left <= right")

    target_dim = merge_breakpoints((left, right), mud.S[dim], preserve_dirac=True)
    target_S = list(mud.S)
    target_S[dim] = target_dim

    if mud.is_empty:
        return _zero_mud(tuple(target_S))

    if mud.S[dim][0] < left or mud.S[dim][-1] > right:
        raise ValueError("mud support is outside the requested extent")

    return mud.align(tuple(target_S))


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _point_satisfies(point: Fraction, op: str, threshold: Fraction) -> bool:
    if op == ">":
        return point > threshold
    if op == ">=":
        return point >= threshold
    if op == "<":
        return point < threshold
    if op == "<=":
        return point <= threshold
    raise ValueError("op must be one of >, >=, <, <=")


def _restrict_interval(left: Fraction, right: Fraction, op: str, threshold: Fraction):
    if left == right:
        if _point_satisfies(left, op, threshold):
            return left, right, Fraction(1)
        return None

    if op in (">", ">="):
        new_left = max(left, threshold)
        new_right = right
    elif op in ("<", "<="):
        new_left = left
        new_right = min(right, threshold)
    else:
        raise ValueError("op must be one of >, >=, <, <=")

    if new_left >= new_right:
        return None
    return new_left, new_right, (new_right - new_left) / (right - left)


@dataclass(frozen=True)
class MUD:
    """Mixture Uniform Distribution.

    P stores block masses, not densities. Values are kept as Python objects so
    solver variables can participate in later symbolic arithmetic.
    """

    S: Breakpoints
    P: np.ndarray

    def __init__(self, S: Sequence[Sequence], P):
        breakpoints = _normalize_breakpoints(S)
        masses = _object_array(P, _shape_from_breakpoints(breakpoints), "P")

        object.__setattr__(self, "S", breakpoints)
        object.__setattr__(self, "P", masses)

    @property
    def ndim(self) -> int:
        return len(self.S)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.P.shape

    @property
    def block_lengths(self) -> tuple[Fraction, ...]:
        return tuple(s[-1] - s[0] for s in self.S)

    @property
    def is_empty(self) -> bool:
        return any(length == 0 for length in self.shape)

    @classmethod
    def empty_like_restrict(cls, S: Sequence[Sequence], dim: int, point) -> MUD:
        breakpoints = _normalize_breakpoints(S)
        if dim < 0 or dim >= len(breakpoints):
            raise ValueError("dim out of range")

        empty_S = []
        for current_dim, seq in enumerate(breakpoints):
            if current_dim == dim:
                empty_S.append((_as_fraction(point),))
            else:
                empty_S.append((seq[0], seq[-1]))

        shape = _shape_from_breakpoints(tuple(empty_S))
        empty_P = np.empty(shape, dtype=object)
        empty_P.fill(0)
        return MUD(tuple(empty_S), empty_P)

    def mass(self):
        return object_sum(self.P.flat)

    def copy(self) -> MUD:
        return MUD(self.S, self.P.copy())

    def scale(self, factor) -> MUD:
        return MUD(self.S, scale_object_array(self.P, factor))

    def __mul__(self, factor) -> MUD:
        return self.scale(factor)

    def __rmul__(self, factor) -> MUD:
        return self.scale(factor)

    def add(self, other: MUD) -> MUD:
        if not isinstance(other, MUD):
            raise TypeError("other must be a MUD")
        if other.ndim != self.ndim:
            raise ValueError(f"other.ndim must be {self.ndim}")

        target = tuple(
            merge_breakpoints(self.S[dim], other.S[dim], preserve_dirac=True)
            for dim in range(self.ndim)
        )
        left = self.align(target)
        right = other.align(target)

        result_P = np.empty(left.shape, dtype=object)
        for index in iter_indices(left.shape):
            result_P[index] = left.P[index] + right.P[index]
        return MUD(target, result_P)

    def __add__(self, other: MUD) -> MUD:
        return self.add(other)

    def independent_product(self, other: MUD) -> MUD:
        if not isinstance(other, MUD):
            raise TypeError("other must be a MUD")

        left_shape = self.shape + (1,) * other.ndim
        right_shape = (1,) * self.ndim + other.shape
        result_P = self.P.reshape(left_shape) * other.P.reshape(right_shape)
        return MUD(self.S + other.S, result_P)

    def restrict(self, dim: int, op: str, c) -> MUD:
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        threshold = _as_fraction(c)

        if self.is_empty:
            return MUD.empty_like_restrict(self.S, dim, threshold)

        restricted = []
        for interval_index in range(self.shape[dim]):
            left = self.S[dim][interval_index]
            right = self.S[dim][interval_index + 1]
            result = _restrict_interval(left, right, op, threshold)
            if result is not None:
                restricted.append((interval_index, *result))

        if not restricted:
            return MUD.empty_like_restrict(self.S, dim, threshold)

        pieces = []
        for interval_index, new_left, new_right, ratio in restricted:
            slicer = [slice(None)] * self.ndim
            slicer[dim] = interval_index
            piece_P = np.expand_dims(
                np.asarray(self.P[tuple(slicer)], dtype=object).copy(), axis=dim
            )
            piece_P = scale_object_array(piece_P, ratio)

            piece_S = list(self.S)
            piece_S[dim] = (new_left, new_right)
            pieces.append(MUD(tuple(piece_S), piece_P))

        result = pieces[0]
        for piece in pieces[1:]:
            result = result + piece
        return result

    def align(self, target_S: Sequence[Sequence]) -> MUD:
        target = _normalize_breakpoints(target_S)
        if len(target) != self.ndim:
            raise ValueError(f"target_S must contain {self.ndim} dimensions")
        if not _support_is_covered(self.S, target):
            raise ValueError("target_S must cover the source support")

        target_shape = _shape_from_breakpoints(target)
        target_P = np.empty(target_shape, dtype=object)
        target_P.fill(0)

        for source_index in iter_indices(self.shape):
            source_mass = self.P[source_index]
            per_dim_ratios = []
            for dim, interval_index in enumerate(source_index):
                source_left = self.S[dim][interval_index]
                source_right = self.S[dim][interval_index + 1]
                ratios = [
                    _interval_overlap_ratio(source_left, source_right, target[dim], j)
                    for j in range(target_shape[dim])
                ]
                per_dim_ratios.append(ratios)

            for target_index in iter_indices(target_shape):
                ratio = object_product(
                    per_dim_ratios[dim][target_index[dim]]
                    for dim in range(self.ndim)
                )
                if ratio == 0:
                    continue
                target_P[target_index] = target_P[target_index] + source_mass * ratio

        return MUD(target, target_P)


@dataclass(frozen=True)
class BGD:
    """Block Geometric Distribution."""

    E: np.ndarray
    alpha: tuple[object, ...]
    beta: tuple[object, ...]
    center_lefts: tuple[Fraction, ...]
    center_rights: tuple[Fraction, ...]
    center_lengths: tuple[Fraction, ...]
    left_lengths: tuple[Fraction, ...]
    right_lengths: tuple[Fraction, ...]

    def __init__(self, E, alpha: Sequence, beta: Sequence):
        edge_tensor = self._normalize_E(E)
        ndim = edge_tensor.ndim
        center = edge_tensor[(1,) * ndim]

        if len(alpha) != ndim:
            raise ValueError(f"alpha must contain {ndim} values")
        if len(beta) != ndim:
            raise ValueError(f"beta must contain {ndim} values")

        alpha_tuple = tuple(alpha)
        beta_tuple = tuple(beta)
        for dim, value in enumerate(alpha_tuple):
            _validate_decay(value, f"alpha[{dim}]")
        for dim, value in enumerate(beta_tuple):
            _validate_decay(value, f"beta[{dim}]")

        center_lengths = center.block_lengths
        center_lefts = tuple(s[0] for s in center.S)
        center_rights = tuple(s[-1] for s in center.S)
        left_lengths, right_lengths = self._validate_edges(edge_tensor, center_lengths)

        object.__setattr__(self, "E", edge_tensor)
        object.__setattr__(self, "alpha", alpha_tuple)
        object.__setattr__(self, "beta", beta_tuple)
        object.__setattr__(self, "center_lefts", center_lefts)
        object.__setattr__(self, "center_rights", center_rights)
        object.__setattr__(self, "center_lengths", center_lengths)
        object.__setattr__(self, "left_lengths", left_lengths)
        object.__setattr__(self, "right_lengths", right_lengths)

    @property
    def ndim(self) -> int:
        return self.E.ndim

    @property
    def C(self) -> MUD:
        return self.E[(1,) * self.ndim]

    @staticmethod
    def direction_to_index(direction: Direction) -> Index:
        if any(d not in (-1, 0, 1) for d in direction):
            raise ValueError("direction entries must be -1, 0, or 1")
        return _direction_to_index(direction)

    @staticmethod
    def index_to_direction(index: Index) -> Direction:
        if any(i not in (0, 1, 2) for i in index):
            raise ValueError("index entries must be 0, 1, or 2")
        return _index_to_direction(index)

    def direction(self, k: Sequence[int]) -> Direction:
        self._validate_block_coordinate(k)
        return tuple(-1 if value < 0 else 1 if value > 0 else 0 for value in k)

    def decay_factor(self, k: Sequence[int]):
        self._validate_block_coordinate(k)
        factors = []
        for dim, value in enumerate(k):
            if value < 0:
                factors.append(self.alpha[dim] ** (-value - 1))
            elif value > 0:
                factors.append(self.beta[dim] ** (value - 1))
        return object_product(factors)

    def local_lengths(self, direction: Direction) -> tuple[Fraction, ...]:
        self._validate_direction(direction)
        lengths = []
        for dim, value in enumerate(direction):
            if value < 0:
                lengths.append(self.left_lengths[dim])
            elif value > 0:
                lengths.append(self.right_lengths[dim])
            else:
                lengths.append(self.center_lengths[dim])
        return tuple(lengths)

    def translation(self, k: Sequence[int]) -> tuple[Fraction, ...]:
        self._validate_block_coordinate(k)
        translation = []
        for dim, value in enumerate(k):
            if value < 0:
                translation.append(self.center_lefts[dim] + value * self.left_lengths[dim])
            elif value > 0:
                translation.append(
                    self.center_rights[dim] + (value - 1) * self.right_lengths[dim]
                )
            else:
                translation.append(self.center_lefts[dim])
        return tuple(translation)

    def block_at(self, k: Sequence[int]) -> BGDBlock:
        direction = self.direction(k)
        index = self.direction_to_index(direction)
        return BGDBlock(
            index=index,
            direction=direction,
            distribution=self.E[index],
            translation=self.translation(k),
            decay_factor=self.decay_factor(k),
        )

    def mass(self):
        center_index = (1,) * self.ndim
        total = self.E[center_index].mass()
        for index in iter_indices(self.E.shape):
            if index == center_index:
                continue
            direction = self.index_to_direction(index)
            block_mass = self.E[index].mass()
            if _is_static_zero(block_mass):
                continue
            total = total + block_mass * self._tail_factor(direction)
        return total

    def standardize(self) -> BGD:
        standardized = np.empty(self.E.shape, dtype=object)
        for index in iter_indices(self.E.shape):
            standardized[index] = self.E[index].copy()

        for _ in range(self.ndim):
            for index in list(iter_indices(standardized.shape)):
                direction = self.index_to_direction(index)
                for dim, value in enumerate(direction):
                    if value < 0:
                        self._standardize_negative_boundary(standardized, index, dim)
                    elif value > 0:
                        self._standardize_positive_boundary(standardized, index, dim)

        return BGD(standardized, self.alpha, self.beta)

    def restrict(self, dim: int, op: str, c) -> BGD:
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if op not in (">", ">=", "<", "<="):
            raise ValueError("op must be one of >, >=, <, <=")

        threshold = _as_fraction(c)
        if op in (">", ">="):
            restricted = self._restrict_greater(dim, op, threshold)
        else:
            restricted = self._restrict_less(dim, op, threshold)
        return BGD(restricted, self.alpha, self.beta).standardize()

    def align_center_domain(self, lefts: Sequence, rights: Sequence) -> BGD:
        """Exactly re-express this BGD using a larger center rectangle."""
        if len(lefts) != self.ndim:
            raise ValueError(f"lefts must contain {self.ndim} values")
        if len(rights) != self.ndim:
            raise ValueError(f"rights must contain {self.ndim} values")

        target_lefts = tuple(_as_fraction(value) for value in lefts)
        target_rights = tuple(_as_fraction(value) for value in rights)
        for dim, (left, right) in enumerate(zip(target_lefts, target_rights)):
            if left > right:
                raise ValueError(f"target center interval {dim} must satisfy left <= right")
            if left > self.center_lefts[dim] or right < self.center_rights[dim]:
                raise ValueError("target center must contain the current center")

        result = self
        for dim in range(self.ndim):
            if target_lefts[dim] < result.center_lefts[dim]:
                result = result._align_center_left(dim, target_lefts[dim])
            if target_rights[dim] > result.center_rights[dim]:
                result = result._align_center_right(dim, target_rights[dim])

        return result.standardize()

    def align_edge_periods(
        self, left_lengths: Sequence, right_lengths: Sequence
    ) -> BGD:
        """Exactly re-express edge periods using integer-multiple lengths."""
        if len(left_lengths) != self.ndim:
            raise ValueError(f"left_lengths must contain {self.ndim} values")
        if len(right_lengths) != self.ndim:
            raise ValueError(f"right_lengths must contain {self.ndim} values")

        target_lefts = tuple(_as_fraction(value) for value in left_lengths)
        target_rights = tuple(_as_fraction(value) for value in right_lengths)
        left_multipliers = tuple(
            self._integer_period_multiple(
                target_lefts[dim], self.left_lengths[dim], f"left_lengths[{dim}]"
            )
            for dim in range(self.ndim)
        )
        right_multipliers = tuple(
            self._integer_period_multiple(
                target_rights[dim], self.right_lengths[dim], f"right_lengths[{dim}]"
            )
            for dim in range(self.ndim)
        )

        expanded = np.empty(self.E.shape, dtype=object)
        center_index = (1,) * self.ndim
        for index in iter_indices(self.E.shape):
            if index == center_index:
                expanded[index] = self.E[index].copy()
                continue

            direction = self.index_to_direction(index)
            per_dim_options = []
            for dim, value in enumerate(direction):
                if value < 0:
                    multiplier = left_multipliers[dim]
                    length = self.left_lengths[dim]
                    per_dim_options.append(
                        [
                            (dim, offset * length, multiplier - offset - 1)
                            for offset in range(multiplier)
                        ]
                    )
                elif value > 0:
                    multiplier = right_multipliers[dim]
                    length = self.right_lengths[dim]
                    per_dim_options.append(
                        [
                            (dim, offset * length, offset)
                            for offset in range(multiplier)
                        ]
                    )
                else:
                    per_dim_options.append([(dim, Fraction(0), 0)])

            pieces = []
            for option_tuple in product(*per_dim_options):
                piece = self.E[index].copy()
                factors = []
                for dim, offset, exponent in option_tuple:
                    if offset != 0:
                        piece = _shift_mud_dim(piece, dim, offset)
                    if direction[dim] < 0:
                        factors.append(_object_power(self.alpha[dim], exponent))
                    elif direction[dim] > 0:
                        factors.append(_object_power(self.beta[dim], exponent))
                pieces.append(piece.scale(object_product(factors)))

            expanded[index] = pieces[0]
            for piece in pieces[1:]:
                expanded[index] = expanded[index] + piece

        alpha = tuple(
            _object_power(self.alpha[dim], left_multipliers[dim])
            for dim in range(self.ndim)
        )
        beta = tuple(
            _object_power(self.beta[dim], right_multipliers[dim])
            for dim in range(self.ndim)
        )
        return BGD(expanded, alpha, beta).standardize()

    def relax_decay(self, alpha: Sequence, beta: Sequence, *, validate: bool = True) -> BGD:
        """Return an upper BGD by increasing geometric decay rates."""
        if len(alpha) != self.ndim:
            raise ValueError(f"alpha must contain {self.ndim} values")
        if len(beta) != self.ndim:
            raise ValueError(f"beta must contain {self.ndim} values")

        alpha_tuple = tuple(alpha)
        beta_tuple = tuple(beta)
        for dim, value in enumerate(alpha_tuple):
            _validate_decay(value, f"alpha[{dim}]")
            if validate:
                self._validate_decay_relaxation(
                    self.alpha[dim], value, f"alpha[{dim}]"
                )
        for dim, value in enumerate(beta_tuple):
            _validate_decay(value, f"beta[{dim}]")
            if validate:
                self._validate_decay_relaxation(self.beta[dim], value, f"beta[{dim}]")

        return BGD(self._copy_E(), alpha_tuple, beta_tuple)

    def align_frame(
        self,
        lefts: Sequence,
        rights: Sequence,
        left_lengths: Sequence,
        right_lengths: Sequence,
    ) -> BGD:
        """Exactly re-express this BGD in the requested center and period frame."""
        return self.align_center_domain(lefts, rights).align_edge_periods(
            left_lengths, right_lengths
        )

    def add(self, other: BGD, *, max_fn=None) -> BGD:
        if not isinstance(other, BGD):
            raise TypeError("other must be a BGD")
        if other.ndim != self.ndim:
            raise ValueError(f"other.ndim must be {self.ndim}")
        if max_fn is None:
            max_fn = _default_decay_max
            validate_relaxation = True
        else:
            validate_relaxation = False

        center_lefts = tuple(
            min(self.center_lefts[dim], other.center_lefts[dim])
            for dim in range(self.ndim)
        )
        center_rights = tuple(
            max(self.center_rights[dim], other.center_rights[dim])
            for dim in range(self.ndim)
        )
        left_lengths = tuple(
            fraction_lcm(self.left_lengths[dim], other.left_lengths[dim])
            for dim in range(self.ndim)
        )
        right_lengths = tuple(
            fraction_lcm(self.right_lengths[dim], other.right_lengths[dim])
            for dim in range(self.ndim)
        )

        left = self.align_frame(center_lefts, center_rights, left_lengths, right_lengths)
        right = other.align_frame(
            center_lefts, center_rights, left_lengths, right_lengths
        )

        alpha = tuple(
            max_fn(left.alpha[dim], right.alpha[dim], f"alpha[{dim}]")
            for dim in range(self.ndim)
        )
        beta = tuple(
            max_fn(left.beta[dim], right.beta[dim], f"beta[{dim}]")
            for dim in range(self.ndim)
        )
        left = left.relax_decay(alpha, beta, validate=validate_relaxation)
        right = right.relax_decay(alpha, beta, validate=validate_relaxation)

        result_E = np.empty(left.E.shape, dtype=object)
        for index in iter_indices(left.E.shape):
            result_E[index] = left.E[index] + right.E[index]

        return BGD(result_E, alpha, beta).standardize()

    def __add__(self, other: BGD) -> BGD:
        return self.add(other)

    def independent_product(self, other: BGD) -> BGD:
        if not isinstance(other, BGD):
            raise TypeError("other must be a BGD")

        shape = (3,) * (self.ndim + other.ndim)
        result_E = np.empty(shape, dtype=object)
        self_center = (1,) * self.ndim
        other_center = (1,) * other.ndim
        for left_index in iter_indices(self.E.shape):
            for right_index in iter_indices(other.E.shape):
                is_joint_center = (
                    left_index == self_center and right_index == other_center
                )
                left_mud = self._product_component_mud(
                    left_index, use_global_center=is_joint_center
                )
                right_mud = other._product_component_mud(
                    right_index, use_global_center=is_joint_center
                )
                result_E[left_index + right_index] = left_mud.independent_product(
                    right_mud
                )

        return BGD(result_E, self.alpha + other.alpha, self.beta + other.beta).standardize()

    def _tail_factor(self, direction: Direction):
        self._validate_direction(direction)
        factors = []
        for dim, value in enumerate(direction):
            if value < 0:
                factors.append(1 / (1 - self.alpha[dim]))
            elif value > 0:
                factors.append(1 / (1 - self.beta[dim]))
        if not factors:
            return Fraction(1)

        result = factors[0]
        for factor in factors[1:]:
            result = result * factor
        return result

    def _align_center_left(self, dim: int, left: Fraction) -> BGD:
        center_side = self._restrict_with_left_prefix(dim, ">=", left)
        tail_side = self._restrict_with_left_phase(dim, "<", left)

        result = np.empty(self.E.shape, dtype=object)
        for index in iter_indices(self.E.shape):
            direction = self.index_to_direction(index)
            if direction[dim] < 0:
                result[index] = tail_side[index]
            else:
                result[index] = center_side[index]

        result = self._force_frame_dim(
            result,
            dim,
            left,
            self.center_rights[dim],
            self.left_lengths[dim],
            self.right_lengths[dim],
        )
        return BGD(result, self.alpha, self.beta)

    def _align_center_right(self, dim: int, right: Fraction) -> BGD:
        center_side = self._restrict_with_right_prefix(dim, "<=", right)
        tail_side = self._restrict_with_right_phase(dim, ">", right)

        result = np.empty(self.E.shape, dtype=object)
        for index in iter_indices(self.E.shape):
            direction = self.index_to_direction(index)
            if direction[dim] > 0:
                result[index] = tail_side[index]
            else:
                result[index] = center_side[index]

        result = self._force_frame_dim(
            result,
            dim,
            self.center_lefts[dim],
            right,
            self.left_lengths[dim],
            self.right_lengths[dim],
        )
        return BGD(result, self.alpha, self.beta)

    def _force_frame_dim(
        self,
        E: np.ndarray,
        dim: int,
        center_left: Fraction,
        center_right: Fraction,
        left_length: Fraction,
        right_length: Fraction,
    ) -> np.ndarray:
        result = np.empty(E.shape, dtype=object)
        center_index = (1,) * self.ndim
        center_length = center_right - center_left

        for index in iter_indices(E.shape):
            direction = self.index_to_direction(index)
            if direction[dim] < 0:
                left, right = Fraction(0), left_length
            elif direction[dim] > 0:
                left, right = Fraction(0), right_length
            elif index == center_index:
                left, right = center_left, center_right
            else:
                left, right = Fraction(0), center_length
            result[index] = _align_mud_dim_to_extent(E[index], dim, left, right)

        return result

    def _copy_E(self) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        for index in iter_indices(self.E.shape):
            result[index] = self.E[index].copy()
        return result

    def _product_component_mud(
        self, index: Index, *, use_global_center: bool
    ) -> MUD:
        center_index = (1,) * self.ndim
        mud = self.E[index].copy()
        if index != center_index or use_global_center:
            return mud

        for dim, left in enumerate(self.center_lefts):
            if left != 0:
                mud = _shift_mud_dim(mud, dim, -left)
        return mud

    @staticmethod
    def _validate_decay_relaxation(current, target, name: str) -> None:
        if current is target:
            return
        if not _is_static_real(current) or not _is_static_real(target):
            raise ValueError(f"{name} relaxation requires statically comparable decays")
        if target < current:
            raise ValueError(f"{name} must be greater than or equal to current decay")

    @staticmethod
    def _integer_period_multiple(target: Fraction, current: Fraction, name: str) -> int:
        if current <= 0:
            raise ValueError("current edge period length must be positive")
        if target <= 0:
            raise ValueError(f"{name} must be positive")
        ratio = target / current
        if ratio.denominator != 1 or ratio.numerator < 1:
            raise ValueError(f"{name} must be an integer multiple of the current length")
        return ratio.numerator

    def _restrict_greater(self, dim: int, op: str, threshold: Fraction) -> np.ndarray:
        left = self.center_lefts[dim]
        right = self.center_rights[dim]
        if threshold < left:
            return self._restrict_with_left_prefix(dim, op, threshold)
        if threshold <= right:
            return self._restrict_inside_center(dim, op, threshold, keep_right=True)
        return self._restrict_with_right_phase(dim, op, threshold)

    def _restrict_less(self, dim: int, op: str, threshold: Fraction) -> np.ndarray:
        left = self.center_lefts[dim]
        right = self.center_rights[dim]
        if threshold > right:
            return self._restrict_with_right_prefix(dim, op, threshold)
        if threshold >= left:
            return self._restrict_inside_center(dim, op, threshold, keep_right=False)
        return self._restrict_with_left_phase(dim, op, threshold)

    def _restrict_inside_center(
        self, dim: int, op: str, threshold: Fraction, keep_right: bool
    ) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        old_left = self.center_lefts[dim]
        local_threshold = threshold - old_left

        for index in iter_indices(self.E.shape):
            direction = self.index_to_direction(index)
            is_center_direction = direction[dim] == 0
            is_true_center = index == (1,) * self.ndim

            if keep_right:
                if direction[dim] < 0:
                    result[index] = self._empty_for_index(index, dim, threshold)
                elif direction[dim] > 0:
                    result[index] = self.E[index].copy()
                elif is_true_center:
                    result[index] = self.E[index].restrict(dim, op, threshold)
                else:
                    result[index] = _shift_mud_dim(
                        self.E[index].restrict(dim, op, local_threshold),
                        dim,
                        old_left - threshold,
                    )
            else:
                if direction[dim] < 0:
                    result[index] = self.E[index].copy()
                elif direction[dim] > 0:
                    result[index] = self._empty_for_index(index, dim, threshold)
                elif is_true_center:
                    result[index] = self.E[index].restrict(dim, op, threshold)
                else:
                    result[index] = self.E[index].restrict(dim, op, local_threshold)

        return result

    def _restrict_with_left_prefix(
        self, dim: int, op: str, threshold: Fraction
    ) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        old_left = self.center_lefts[dim]
        old_right = self.center_rights[dim]
        length = self.left_lengths[dim]
        block_count = _ceil_fraction((old_left - threshold) / length)

        for center_index in self._line_center_indices(dim):
            left_index = self._replace_index(center_index, dim, 0)
            right_index = self._replace_index(center_index, dim, 2)
            is_true_center = center_index == (1,) * self.ndim
            center = self.E[center_index].copy()
            if not is_true_center:
                center = _shift_mud_dim(center, dim, old_left - threshold)

            for block_number in range(1, block_count + 1):
                block_start = old_left - block_number * length
                local_threshold = threshold - block_start
                piece = self.E[left_index].restrict(dim, op, local_threshold)
                if piece.is_empty:
                    continue
                piece = piece.scale(self.alpha[dim] ** (block_number - 1))
                shift = block_start if is_true_center else block_start - threshold
                center = center + _shift_mud_dim(piece, dim, shift)

            result[left_index] = self._empty_for_index(left_index, dim, threshold)
            result[center_index] = center
            result[right_index] = self.E[right_index].copy()

        return result

    def _restrict_with_right_prefix(
        self, dim: int, op: str, threshold: Fraction
    ) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        old_right = self.center_rights[dim]
        length = self.right_lengths[dim]
        block_count = _ceil_fraction((threshold - old_right) / length)

        for center_index in self._line_center_indices(dim):
            left_index = self._replace_index(center_index, dim, 0)
            right_index = self._replace_index(center_index, dim, 2)
            is_true_center = center_index == (1,) * self.ndim
            center = self.E[center_index].copy()

            for block_number in range(block_count):
                block_start = old_right + block_number * length
                local_threshold = threshold - block_start
                piece = self.E[right_index].restrict(dim, op, local_threshold)
                if piece.is_empty:
                    continue
                piece = piece.scale(self.beta[dim] ** block_number)
                shift = block_start if is_true_center else block_start - self.center_lefts[dim]
                center = center + _shift_mud_dim(piece, dim, shift)

            result[left_index] = self.E[left_index].copy()
            result[center_index] = center
            result[right_index] = self._empty_for_index(right_index, dim, threshold)

        return result

    def _restrict_with_right_phase(
        self, dim: int, op: str, threshold: Fraction
    ) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        old_right = self.center_rights[dim]
        length = self.right_lengths[dim]
        distance = threshold - old_right
        quotient = distance // length
        phase = distance - quotient * length

        for center_index in self._line_center_indices(dim):
            left_index = self._replace_index(center_index, dim, 0)
            right_index = self._replace_index(center_index, dim, 2)
            right_mud = self.E[right_index]

            current = right_mud.restrict(dim, op, phase)
            current = _shift_mud_dim(current, dim, -phase).scale(
                self.beta[dim] ** quotient
            )

            next_prefix = right_mud.restrict(dim, "<=", phase)
            next_prefix = _shift_mud_dim(next_prefix, dim, length - phase).scale(
                self.beta[dim] ** (quotient + 1)
            )

            result[left_index] = self._empty_for_index(left_index, dim, threshold)
            result[center_index] = self._empty_for_index(center_index, dim, threshold)
            result[right_index] = current + next_prefix

        return result

    def _restrict_with_left_phase(
        self, dim: int, op: str, threshold: Fraction
    ) -> np.ndarray:
        result = np.empty(self.E.shape, dtype=object)
        old_left = self.center_lefts[dim]
        length = self.left_lengths[dim]
        distance = old_left - threshold
        quotient = distance // length
        remainder = distance - quotient * length
        if remainder == 0:
            block_number = quotient
            phase = Fraction(0)
        else:
            block_number = quotient + 1
            phase = length - remainder

        for center_index in self._line_center_indices(dim):
            left_index = self._replace_index(center_index, dim, 0)
            right_index = self._replace_index(center_index, dim, 2)
            left_mud = self.E[left_index]

            suffix = left_mud.restrict(dim, ">=", phase)
            suffix = _shift_mud_dim(suffix, dim, -phase).scale(
                self.alpha[dim] ** block_number
            )

            prefix = left_mud.restrict(dim, op, phase)
            prefix = _shift_mud_dim(prefix, dim, length - phase).scale(
                self.alpha[dim] ** (block_number - 1)
            )

            result[left_index] = suffix + prefix
            result[center_index] = self._empty_for_index(center_index, dim, threshold)
            result[right_index] = self._empty_for_index(right_index, dim, threshold)

        return result

    def _line_center_indices(self, dim: int) -> Iterable[Index]:
        other_ranges = [range(3) if axis != dim else (1,) for axis in range(self.ndim)]
        return product(*other_ranges)

    @staticmethod
    def _replace_index(index: Index, dim: int, value: int) -> Index:
        result = list(index)
        result[dim] = value
        return tuple(result)

    def _empty_for_index(self, index: Index, dim: int, threshold: Fraction) -> MUD:
        point = threshold if index == (1,) * self.ndim else Fraction(0)
        return MUD.empty_like_restrict(self.E[index].S, dim, point)

    def _standardize_negative_boundary(
        self, E: np.ndarray, index: Index, dim: int
    ) -> None:
        boundary = _boundary_slice_mud(E[index], dim, "right")
        if boundary is None or _array_is_static_zero(boundary.P):
            return

        E[index] = _remove_boundary_slice(E[index], dim, "right")

        target_direction = list(self.index_to_direction(index))
        target_direction[dim] = 0
        target_index = self.direction_to_index(tuple(target_direction))
        target_point = (
            self.center_lefts[dim]
            if target_index == (1,) * self.ndim
            else Fraction(0)
        )

        E[target_index] = E[target_index] + _move_slice_to_point(
            boundary, dim, target_point
        )
        E[index] = E[index] + _move_slice_to_point(boundary, dim, 0).scale(
            self.alpha[dim]
        )

    def _standardize_positive_boundary(
        self, E: np.ndarray, index: Index, dim: int
    ) -> None:
        boundary = _boundary_slice_mud(E[index], dim, "left")
        if boundary is None or _array_is_static_zero(boundary.P):
            return

        E[index] = _remove_boundary_slice(E[index], dim, "left")

        target_direction = list(self.index_to_direction(index))
        target_direction[dim] = 0
        target_index = self.direction_to_index(tuple(target_direction))
        target_point = (
            self.center_rights[dim]
            if target_index == (1,) * self.ndim
            else self.center_lengths[dim]
        )

        E[target_index] = E[target_index] + _move_slice_to_point(
            boundary, dim, target_point
        )
        E[index] = E[index] + _move_slice_to_point(
            boundary, dim, self.right_lengths[dim]
        ).scale(self.beta[dim])

    @classmethod
    def _normalize_E(cls, E) -> np.ndarray:
        if isinstance(E, Mapping):
            if not E:
                raise ValueError("E mapping must not be empty")
            key_lengths = {len(tuple(key)) for key in E}
            if len(key_lengths) != 1:
                raise ValueError("E mapping keys must have a consistent dimension")
            ndim = key_lengths.pop()
            shape = (3,) * ndim
            tensor = np.empty(shape, dtype=object)
            tensor.fill(None)
            for key, value in E.items():
                index = tuple(key)
                if all(i in (-1, 0, 1) for i in index):
                    index = _direction_to_index(index)
                if any(i not in (0, 1, 2) for i in index):
                    raise ValueError("E indices must be 0, 1, 2 or directions -1, 0, 1")
                tensor[index] = value
        else:
            tensor = np.asarray(E, dtype=object)
            if tensor.ndim == 0:
                raise ValueError("E must be a tensor with shape (3,) * ndim")
            ndim = tensor.ndim
            shape = (3,) * ndim
            if tensor.shape != shape:
                raise ValueError(f"E shape must be {shape}, got {tensor.shape}")
            tensor = tensor.copy()

        for index in _iter_indices(shape):
            if not isinstance(tensor[index], MUD):
                raise TypeError(f"E{index} must be a MUD")
            if tensor[index].ndim != ndim:
                raise ValueError(f"E{index}.ndim must be {ndim}")

        return tensor

    def _validate_block_coordinate(self, k: Sequence[int]) -> None:
        if len(k) != self.ndim:
            raise ValueError(f"k must contain {self.ndim} coordinates")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in k):
            raise TypeError("k coordinates must be integers")

    def _validate_direction(self, direction: Direction) -> None:
        if len(direction) != self.ndim:
            raise ValueError(f"direction must contain {self.ndim} values")
        if any(value not in (-1, 0, 1) for value in direction):
            raise ValueError("direction entries must be -1, 0, or 1")

    @staticmethod
    def _validate_edges(
        E: np.ndarray, center_lengths: tuple[Fraction, ...]
    ) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
        ndim = len(center_lengths)
        left_lengths: list[Fraction | None] = [None] * ndim
        right_lengths: list[Fraction | None] = [None] * ndim

        for index in _iter_indices(E.shape):
            if index == (1,) * ndim:
                continue

            mud = E[index]
            direction = _index_to_direction(index)
            for dim, breakpoints in enumerate(mud.S):
                if breakpoints[0] != 0:
                    raise ValueError(f"E{index}.S[{dim}][0] must be 0")

                endpoint = breakpoints[-1]
                if direction[dim] == 0:
                    if endpoint != center_lengths[dim]:
                        raise ValueError(
                            f"E{index}.S[{dim}][-1] must equal center length "
                            f"{center_lengths[dim]}"
                        )
                elif direction[dim] == -1:
                    if left_lengths[dim] is None:
                        left_lengths[dim] = endpoint
                    elif endpoint != left_lengths[dim]:
                        raise ValueError(
                            f"E{index}.S[{dim}][-1] must equal left edge length "
                            f"{left_lengths[dim]}"
                        )
                else:
                    if right_lengths[dim] is None:
                        right_lengths[dim] = endpoint
                    elif endpoint != right_lengths[dim]:
                        raise ValueError(
                            f"E{index}.S[{dim}][-1] must equal right edge length "
                            f"{right_lengths[dim]}"
                        )

        if any(length is None for length in left_lengths):
            raise ValueError("left edge lengths could not be inferred")
        if any(length is None for length in right_lengths):
            raise ValueError("right edge lengths could not be inferred")

        return (
            tuple(length for length in left_lengths if length is not None),
            tuple(length for length in right_lengths if length is not None),
        )
