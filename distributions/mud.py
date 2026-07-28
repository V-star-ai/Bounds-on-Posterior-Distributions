from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from numbers import Real
from typing import ClassVar, Iterable, Sequence

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


def _array_is_static_zero(array: np.ndarray) -> bool:
    return all(_is_static_zero(value) for value in array.flat)


def _iter_indices(shape: tuple[int, ...]) -> Iterable[Index]:
    return product(*(range(n) for n in shape))


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

        target_counts = Counter(target_dim)
        for point, count in Counter(source_dim).items():
            if target_counts[point] < count:
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


def _shift_grid_dim(mud: GridMUD, dim: int, offset):
    offset = _as_fraction(offset)
    shifted_S = list(mud.S)
    shifted_S[dim] = tuple(point + offset for point in mud.S[dim])
    return mud._new(tuple(shifted_S), mud.P.copy())


def _zero_mud(S: Sequence[Sequence]) -> "MUD":
    breakpoints = _normalize_breakpoints(S)
    shape = _shape_from_breakpoints(breakpoints)
    P = np.empty(shape, dtype=object)
    P.fill(0)
    return MUD(breakpoints, P)


def _zero_affine_mud(S: Sequence[Sequence], affine_dim: int) -> "AffineMUD":
    breakpoints = _normalize_breakpoints(S)
    shape = _shape_from_breakpoints(breakpoints)
    P = np.empty(shape, dtype=object)
    P.fill(AffineCell(0, 0, sloped=False))
    return AffineMUD(breakpoints, P, affine_dim)


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


def _uniform_convolution_breakpoints(
    source_left: Fraction,
    source_right: Fraction,
    noise_left: Fraction,
    noise_right: Fraction,
) -> tuple[Fraction, ...]:
    return tuple(
        sorted(
            {
                source_left + noise_left,
                source_left + noise_right,
                source_right + noise_left,
                source_right + noise_right,
            }
        )
    )


def _uniform_convolution_density_at(
    mass,
    source_left: Fraction,
    source_right: Fraction,
    noise_left: Fraction,
    noise_right: Fraction,
    point: Fraction,
):
    noise_length = noise_right - noise_left
    if source_left == source_right:
        if source_left + noise_left <= point <= source_left + noise_right:
            return mass / noise_length
        return 0

    overlap_left = max(source_left, point - noise_right)
    overlap_right = min(source_right, point - noise_left)
    if overlap_left >= overlap_right:
        return 0

    source_length = source_right - source_left
    return mass * (overlap_right - overlap_left) / (source_length * noise_length)


def _uniform_convolution_overlap_width(
    source_left: Fraction,
    source_right: Fraction,
    noise_left: Fraction,
    noise_right: Fraction,
    point: Fraction,
) -> Fraction:
    overlap_left = max(source_left, point - noise_right)
    overlap_right = min(source_right, point - noise_left)
    if overlap_left >= overlap_right:
        return Fraction(0)
    return overlap_right - overlap_left


def _uniform_convolution_cell(
    mass,
    source_left: Fraction,
    source_right: Fraction,
    noise_left: Fraction,
    noise_right: Fraction,
    target_left: Fraction,
    target_right: Fraction,
) -> AffineCell:
    if _is_static_zero(mass):
        return AffineCell(0, 0, sloped=False)

    noise_length = noise_right - noise_left
    if source_left == source_right:
        support_left = source_left + noise_left
        support_right = source_left + noise_right
        if target_left >= support_left and target_right <= support_right:
            value = mass / noise_length
            return AffineCell(value, value, sloped=False)
        return AffineCell(0, 0, sloped=False)

    left_value = _uniform_convolution_density_at(
        mass, source_left, source_right, noise_left, noise_right, target_left
    )
    right_value = _uniform_convolution_density_at(
        mass, source_left, source_right, noise_left, noise_right, target_right
    )
    left_width = _uniform_convolution_overlap_width(
        source_left, source_right, noise_left, noise_right, target_left
    )
    right_width = _uniform_convolution_overlap_width(
        source_left, source_right, noise_left, noise_right, target_right
    )
    return AffineCell(
        left_value,
        right_value,
        sloped=left_width != right_width,
    )


@dataclass(frozen=True)
class CellOps:
    """Operations for values stored in a grid cell."""

    def zero(self):
        return 0

    def add(self, left, right):
        return left + right

    def scale(self, value, factor):
        return value * factor

    def product(self, left, right):
        return left * right

    def restrict(
        self,
        value,
        source_intervals: tuple[Interval, ...],
        target_intervals: tuple[Interval, ...],
    ):
        raise NotImplementedError

    def mass(self, value, intervals: tuple[Interval, ...]):
        raise NotImplementedError


@dataclass(frozen=True)
class MassCellOps(CellOps):
    """Cell operations where each payload is total cell mass."""

    def restrict(
        self,
        value,
        source_intervals: tuple[Interval, ...],
        target_intervals: tuple[Interval, ...],
    ):
        ratio = object_product(
            _interval_subinterval_ratio(source, target)
            for source, target in zip(source_intervals, target_intervals)
        )
        return self.scale(value, ratio)

    def mass(self, value, intervals: tuple[Interval, ...]):
        return value


@dataclass(frozen=True)
class AffineCell:
    """Endpoint values for a linear density along one distinguished dimension."""

    left: object
    right: object
    sloped: bool = False


@dataclass(frozen=True)
class AffineCellOps(CellOps):
    """Cell operations for payloads that are affine in one dimension."""

    affine_dim: int

    def zero(self):
        return AffineCell(0, 0, sloped=False)

    def add(self, left: AffineCell, right: AffineCell):
        return AffineCell(
            left.left + right.left,
            left.right + right.right,
            sloped=left.sloped or right.sloped,
        )

    def scale(self, value: AffineCell, factor):
        sloped = value.sloped
        if _is_static_zero(factor):
            sloped = False
        return AffineCell(value.left * factor, value.right * factor, sloped=sloped)

    def product(self, left, right):
        raise NotImplementedError("AffineMUD independent_product is not defined")

    def restrict(
        self,
        value: AffineCell,
        source_intervals: tuple[Interval, ...],
        target_intervals: tuple[Interval, ...],
    ):
        result = value
        for dim, (source, target) in enumerate(
            zip(source_intervals, target_intervals)
        ):
            if dim == self.affine_dim:
                result = self._restrict_affine_dim(result, source, target)
            else:
                result = self.scale(
                    result, _interval_subinterval_ratio(source, target)
                )
        return result

    def mass(self, value: AffineCell, intervals: tuple[Interval, ...]):
        left, right = intervals[self.affine_dim]
        length = right - left
        if length == 0:
            return 0
        return (value.left + value.right) * length / 2

    def _restrict_affine_dim(
        self, value: AffineCell, source: Interval, target: Interval
    ) -> AffineCell:
        source_left, source_right = source
        target_left, target_right = target
        source_length = source_right - source_left
        target_length = target_right - target_left
        if source_length == 0 or target_length == 0:
            return self.zero()

        slope = (value.right - value.left) / source_length
        return AffineCell(
            value.left + slope * (target_left - source_left),
            value.left + slope * (target_right - source_left),
            sloped=value.sloped,
        )


def _interval_subinterval_ratio(source: Interval, target: Interval) -> Fraction:
    source_left, source_right = source
    target_left, target_right = target

    if source_left == source_right:
        if target_left == source_left and target_right == source_right:
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


@dataclass(frozen=True)
class GridMUD:
    """Grid distribution with pluggable cell payload operations.

    Subclasses define how cell payloads are interpreted. The mass version stores
    total block masses and recovers the original MUD semantics.
    """

    cell_ops: ClassVar[CellOps] = CellOps()

    S: Breakpoints
    P: np.ndarray

    def __init__(self, S: Sequence[Sequence], P):
        breakpoints = _normalize_breakpoints(S)
        payloads = _object_array(P, _shape_from_breakpoints(breakpoints), "P")

        object.__setattr__(self, "S", breakpoints)
        object.__setattr__(self, "P", payloads)

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

    @property
    def ops(self) -> CellOps:
        return getattr(self, "_cell_ops", self.cell_ops)

    @classmethod
    def empty_like_restrict(cls, S: Sequence[Sequence], dim: int, point):
        breakpoints = _normalize_breakpoints(S)
        if dim < 0 or dim >= len(breakpoints):
            raise ValueError("dim out of range")

        empty_S = []
        for current_dim, seq in enumerate(breakpoints):
            if current_dim == dim:
                empty_S.append((_as_fraction(point),))
            elif len(seq) == 1 or seq[0] == seq[-1]:
                empty_S.append((seq[0],))
            else:
                empty_S.append((seq[0], seq[-1]))

        shape = _shape_from_breakpoints(tuple(empty_S))
        empty_P = np.empty(shape, dtype=object)
        empty_P.fill(cls.cell_ops.zero())
        return cls(tuple(empty_S), empty_P)

    def _empty_like_restrict(self, dim: int, point):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")

        empty_S = []
        for current_dim, seq in enumerate(self.S):
            if current_dim == dim:
                empty_S.append((_as_fraction(point),))
            elif len(seq) == 1 or seq[0] == seq[-1]:
                empty_S.append((seq[0],))
            else:
                empty_S.append((seq[0], seq[-1]))

        shape = _shape_from_breakpoints(tuple(empty_S))
        empty_P = np.empty(shape, dtype=object)
        empty_P.fill(self.ops.zero())
        return self._new(tuple(empty_S), empty_P)

    def mass(self):
        return object_sum(
            self.ops.mass(self.P[index], self._intervals_for_index(index))
            for index in iter_indices(self.shape)
        )

    def copy(self):
        return self._new(self.S, self.P.copy())

    def shift(self, dim: int, offset):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        return _shift_grid_dim(self, dim, offset)

    def scale(self, factor):
        return self._new(self.S, self._scale_payload_array(self.P, factor))

    def __mul__(self, factor):
        return self.scale(factor)

    def __rmul__(self, factor):
        return self.scale(factor)

    def add(self, other):
        self._validate_compatible_grid(other)

        target = tuple(
            merge_breakpoints(self.S[dim], other.S[dim], preserve_dirac=True)
            for dim in range(self.ndim)
        )
        left = self.align(target)
        right = other.align(target)

        result_P = np.empty(left.shape, dtype=object)
        for index in iter_indices(left.shape):
            result_P[index] = self.ops.add(left.P[index], right.P[index])
        return self._new(target, result_P)

    def __add__(self, other):
        if not isinstance(other, GridMUD):
            if self.ndim != 1:
                raise ValueError("constant addition is only defined for one-dimensional MUD")
            return self.shift(0, other)
        return self.add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def independent_product(self, other):
        self._validate_compatible_cell_type(other)

        left_shape = self.shape + (1,) * other.ndim
        right_shape = (1,) * self.ndim + other.shape
        left_payloads = self.P.reshape(left_shape)
        right_payloads = other.P.reshape(right_shape)

        result_shape = self.shape + other.shape
        result_P = np.empty(result_shape, dtype=object)
        for index in iter_indices(result_shape):
            left_index = index[: self.ndim]
            right_index = index[self.ndim :]
            result_P[index] = self.ops.product(
                left_payloads[left_index + (0,) * other.ndim],
                right_payloads[(0,) * self.ndim + right_index],
            )
        return self._new(self.S + other.S, result_P)

    def marginalize(self, dim: int):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if self.ndim == 1:
            raise ValueError("cannot marginalize the only MUD dimension")

        result_S = self.S[:dim] + self.S[dim + 1 :]
        result_shape = self.shape[:dim] + self.shape[dim + 1 :]
        result_P = np.empty(result_shape, dtype=object)
        result_P.fill(self.ops.zero())

        for index in iter_indices(self.shape):
            result_index = index[:dim] + index[dim + 1 :]
            result_P[result_index] = self.ops.add(
                result_P[result_index], self.P[index]
            )

        return self._new(result_S, result_P)

    def permute_dims(self, order: Sequence[int]):
        order = self._validate_permutation(order, self.ndim)
        result_S = tuple(self.S[dim] for dim in order)
        result_P = np.transpose(self.P, axes=order).copy()
        return self._new(result_S, result_P)

    def restrict(self, dim: int, op: str, c):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        threshold = _as_fraction(c)

        if self.is_empty:
            return self._empty_like_restrict(dim, threshold)

        restricted = []
        for interval_index in range(self.shape[dim]):
            left = self.S[dim][interval_index]
            right = self.S[dim][interval_index + 1]
            result = _restrict_interval(left, right, op, threshold)
            if result is not None:
                restricted.append((interval_index, *result))

        if not restricted:
            return self._empty_like_restrict(dim, threshold)

        pieces = []
        for interval_index, new_left, new_right, ratio in restricted:
            piece_S = list(self.S)
            piece_S[dim] = (new_left, new_right)
            piece_shape = _shape_from_breakpoints(tuple(piece_S))
            piece_P = np.empty(piece_shape, dtype=object)
            piece_P.fill(self.ops.zero())

            for piece_index in iter_indices(piece_shape):
                source_index = list(piece_index)
                source_index[dim] = interval_index
                source_index = tuple(source_index)
                source_intervals = self._intervals_for_index(source_index)
                target_intervals = tuple(
                    (
                        piece_S[current_dim][piece_index[current_dim]],
                        piece_S[current_dim][piece_index[current_dim] + 1],
                    )
                    for current_dim in range(self.ndim)
                )
                piece_P[piece_index] = self.ops.restrict(
                    self.P[source_index], source_intervals, target_intervals
                )
            pieces.append(self._new(tuple(piece_S), piece_P))

        result = pieces[0]
        for piece in pieces[1:]:
            result = result + piece
        return result

    def align(self, target_S: Sequence[Sequence]):
        target = _normalize_breakpoints(target_S)
        if len(target) != self.ndim:
            raise ValueError(f"target_S must contain {self.ndim} dimensions")
        if not _support_is_covered(self.S, target):
            raise ValueError("target_S must cover the source support")
        if target == self.S:
            return self.copy()

        target_shape = _shape_from_breakpoints(target)
        target_P = np.empty(target_shape, dtype=object)
        target_P.fill(self.ops.zero())
        overlaps_by_dim = []
        for dim in range(self.ndim):
            dim_overlaps = []
            for interval_index in range(self.shape[dim]):
                source_left = self.S[dim][interval_index]
                source_right = self.S[dim][interval_index + 1]
                overlaps = []
                for target_index in range(target_shape[dim]):
                    ratio = _interval_overlap_ratio(
                        source_left,
                        source_right,
                        target[dim],
                        target_index,
                    )
                    if ratio != 0:
                        overlaps.append((target_index, ratio))
                dim_overlaps.append(tuple(overlaps))
            overlaps_by_dim.append(tuple(dim_overlaps))

        for source_index in iter_indices(self.shape):
            source_payload = self.P[source_index]
            source_intervals = self._intervals_for_index(source_index)
            per_dim_ratios = []
            has_overlap = True
            for dim, interval_index in enumerate(source_index):
                overlaps = overlaps_by_dim[dim][interval_index]
                if not overlaps:
                    has_overlap = False
                    break
                per_dim_ratios.append(overlaps)
            if not has_overlap:
                continue

            for overlap_tuple in product(*per_dim_ratios):
                target_index = tuple(item[0] for item in overlap_tuple)
                ratio = object_product(item[1] for item in overlap_tuple)
                target_intervals = tuple(
                    (
                        target[dim][target_index[dim]],
                        target[dim][target_index[dim] + 1],
                    )
                    for dim in range(self.ndim)
                )
                target_P[target_index] = self.ops.add(
                    target_P[target_index],
                    self.ops.restrict(
                        source_payload, source_intervals, target_intervals
                    ),
                )

        return self._new(target, target_P)

    def _new(self, S: Sequence[Sequence], P):
        return type(self)(S, P)

    def _validate_compatible_grid(self, other) -> None:
        self._validate_compatible_cell_type(other)
        if other.ndim != self.ndim:
            raise ValueError(f"other.ndim must be {self.ndim}")

    def _validate_compatible_cell_type(self, other) -> None:
        if not isinstance(other, GridMUD):
            raise TypeError("other must be a MUD")
        if type(other) is not type(self):
            raise TypeError("other must use the same MUD cell type")

    def _scale_payload_array(self, array: np.ndarray, factor) -> np.ndarray:
        scaled = np.empty(array.shape, dtype=object)
        for index in iter_indices(array.shape):
            scaled[index] = self.ops.scale(array[index], factor)
        return scaled

    def _intervals_for_index(self, index: Index) -> tuple[Interval, ...]:
        return tuple(
            (self.S[dim][interval_index], self.S[dim][interval_index + 1])
            for dim, interval_index in enumerate(index)
        )

    @staticmethod
    def _validate_permutation(order: Sequence[int], ndim: int) -> tuple[int, ...]:
        order_tuple = tuple(order)
        if len(order_tuple) != ndim:
            raise ValueError(f"order must contain {ndim} dimensions")
        if set(order_tuple) != set(range(ndim)):
            raise ValueError("order must be a permutation of dimensions")
        return order_tuple


@dataclass(frozen=True, init=False)
class MassMUD(GridMUD):
    """Mixture Uniform Distribution.

    P stores block masses, not densities. Values are kept as Python objects so
    solver variables can participate in later symbolic arithmetic.
    """

    cell_ops: ClassVar[MassCellOps] = MassCellOps()

    def convolve_uniform(self, dim: int, low, high) -> "AffineMUD":
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")

        noise_left = _as_fraction(low)
        noise_right = _as_fraction(high)
        if noise_left >= noise_right:
            raise ValueError("requires low < high")

        if self.is_empty:
            return self._empty_like_convolve_uniform(dim, noise_left)

        affine_points = set()
        for interval_index in range(self.shape[dim]):
            source_left = self.S[dim][interval_index]
            source_right = self.S[dim][interval_index + 1]
            affine_points.update(
                _uniform_convolution_breakpoints(
                    source_left, source_right, noise_left, noise_right
                )
            )

        result_S = list(self.S)
        result_S[dim] = tuple(sorted(affine_points))
        result_S = tuple(result_S)
        result_shape = _shape_from_breakpoints(result_S)
        result_P = np.empty(result_shape, dtype=object)
        result_P.fill(AffineCell(0, 0))

        for source_index in iter_indices(self.shape):
            source_left = self.S[dim][source_index[dim]]
            source_right = self.S[dim][source_index[dim] + 1]
            mass = self.P[source_index]

            for affine_index in range(result_shape[dim]):
                left = result_S[dim][affine_index]
                right = result_S[dim][affine_index + 1]
                contribution = _uniform_convolution_cell(
                    mass, source_left, source_right, noise_left, noise_right, left, right
                )
                if _is_static_zero(contribution.left) and _is_static_zero(
                    contribution.right
                ):
                    continue

                result_index = list(source_index)
                result_index[dim] = affine_index
                result_index = tuple(result_index)
                old = result_P[result_index]
                result_P[result_index] = AffineCell(
                    old.left + contribution.left,
                    old.right + contribution.right,
                    sloped=old.sloped or contribution.sloped,
                )

        return AffineMUD(result_S, result_P, dim)

    def convolve_uniform_upper(
        self,
        dim: int,
        low,
        high,
        *,
        max_fn=None,
        bound_factory=None,
        max_interval=None,
    ):
        return self.convolve_uniform(dim, low, high).to_mass_mud_upper(
            max_fn=max_fn, bound_factory=bound_factory, max_interval=max_interval
        )

    def _empty_like_convolve_uniform(
        self, dim: int, noise_left: Fraction
    ) -> "AffineMUD":
        empty_S = []
        for current_dim, seq in enumerate(self.S):
            if current_dim == dim:
                empty_S.append((seq[0] + noise_left,))
            elif len(seq) == 1:
                empty_S.append(seq)
            else:
                empty_S.append((seq[0], seq[-1]))

        shape = _shape_from_breakpoints(tuple(empty_S))
        empty_P = np.empty(shape, dtype=object)
        empty_P.fill(AffineCell(0, 0))
        return AffineMUD(tuple(empty_S), empty_P, dim)


MUD = MassMUD


@dataclass(frozen=True, init=False)
class AffineMUD(GridMUD):
    """Grid distribution whose payloads are linear densities in one dimension."""

    _affine_dim: int
    _cell_ops: AffineCellOps

    def __init__(self, S: Sequence[Sequence], P, affine_dim: int):
        breakpoints = _normalize_breakpoints(S)
        if affine_dim < 0 or affine_dim >= len(breakpoints):
            raise ValueError("affine_dim out of range")

        object.__setattr__(self, "_affine_dim", affine_dim)
        object.__setattr__(self, "_cell_ops", AffineCellOps(affine_dim))
        GridMUD.__init__(self, breakpoints, P)

    @property
    def affine_dim(self) -> int:
        return self._affine_dim

    def permute_dims(self, order: Sequence[int]):
        order = self._validate_permutation(order, self.ndim)
        result_S = tuple(self.S[dim] for dim in order)
        result_P = np.transpose(self.P, axes=order).copy()
        return AffineMUD(result_S, result_P, order.index(self.affine_dim))

    def marginalize(self, dim: int):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if dim == self.affine_dim:
            raise ValueError("cannot marginalize the affine dimension")
        if self.ndim == 1:
            raise ValueError("cannot marginalize the only MUD dimension")

        result_S = self.S[:dim] + self.S[dim + 1 :]
        result_shape = self.shape[:dim] + self.shape[dim + 1 :]
        result_P = np.empty(result_shape, dtype=object)
        result_P.fill(self.ops.zero())

        for index in iter_indices(self.shape):
            result_index = index[:dim] + index[dim + 1 :]
            result_P[result_index] = self.ops.add(
                result_P[result_index], self.P[index]
            )

        affine_dim = (
            self.affine_dim - 1 if dim < self.affine_dim else self.affine_dim
        )
        return AffineMUD(result_S, result_P, affine_dim)

    def refine_affine(self, max_interval) -> "AffineMUD":
        max_interval = _as_fraction(max_interval)
        if max_interval <= 0:
            raise ValueError("max_interval must be positive")
        if self.is_empty:
            return self.copy()

        refined_dim = [self.S[self.affine_dim][0]]
        for interval_index, (left, right) in enumerate(
            zip(self.S[self.affine_dim], self.S[self.affine_dim][1:])
        ):
            length = right - left
            if (
                length > max_interval
                and self._affine_interval_has_sloped_cell(interval_index)
            ):
                pieces = _ceil_fraction(length / max_interval)
                for step in range(1, pieces + 1):
                    refined_dim.append(left + length * Fraction(step, pieces))
            else:
                refined_dim.append(right)

        target_S = list(self.S)
        target_S[self.affine_dim] = tuple(refined_dim)
        return self.align(tuple(target_S))

    def to_mass_mud_upper(self, *, max_fn=None, bound_factory=None, max_interval=None):
        if max_fn is not None and bound_factory is not None:
            raise ValueError("provide either max_fn or bound_factory, not both")
        if max_interval is not None:
            return self.refine_affine(max_interval).to_mass_mud_upper(
                max_fn=max_fn, bound_factory=bound_factory
            )
        if max_fn is None and bound_factory is None:
            max_fn = _default_endpoint_max

        result_P = np.empty(self.shape, dtype=object)
        constraints = []
        for index in iter_indices(self.shape):
            cell = self.P[index]
            intervals = self._intervals_for_index(index)
            length = intervals[self.affine_dim][1] - intervals[self.affine_dim][0]
            if bound_factory is None:
                upper = max_fn(cell.left, cell.right, f"cell{index}")
            else:
                upper, new_constraints = bound_factory(
                    f"cell{index}", cell.left, cell.right
                )
                constraints.extend(new_constraints)
            result_P[index] = upper * length

        result = MassMUD(self.S, result_P)
        if bound_factory is None:
            return result
        return result, constraints

    def _new(self, S: Sequence[Sequence], P):
        return AffineMUD(S, P, self.affine_dim)

    def _affine_interval_has_sloped_cell(self, interval_index: int) -> bool:
        for index in iter_indices(self.shape):
            if index[self.affine_dim] == interval_index and self.P[index].sloped:
                return True
        return False


def _default_endpoint_max(left, right, name: str):
    if not _is_static_real(left) or not _is_static_real(right):
        raise ValueError(f"{name} max requires statically comparable endpoint values")
    return max(left, right)
