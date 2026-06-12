from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from distributions.mud import (
    AffineCell,
    AffineCellOps,
    AffineMUD,
    Breakpoints,
    CellOps,
    Direction,
    GridMUD,
    Index,
    Interval,
    MUD,
    MassCellOps,
    MassMUD,
    _align_mud_dim_to_extent,
    _array_is_static_zero,
    _as_fraction,
    _boundary_slice_mud,
    _ceil_fraction,
    _direction_to_index,
    _index_to_direction,
    _is_static_real,
    _is_static_zero,
    _iter_indices,
    _move_slice_to_point,
    _object_power,
    _remove_boundary_slice,
    _shift_grid_dim,
    _shift_mud_dim,
    _zero_affine_mud,
    fraction_lcm,
    interval_intersection,
    interval_length,
    is_dirac_interval,
    iter_indices,
    merge_breakpoints,
    object_product,
    object_sum,
    point_in_interval,
    scale_object_array,
)


def _default_decay_max(left, right, name: str):
    if not _is_static_real(left) or not _is_static_real(right):
        raise ValueError(f"{name} max requires statically comparable numeric decays")
    return max(left, right)


def _validate_decay(value, name: str) -> None:
    if _is_static_real(value) and not (0 <= value < 1):
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1")


@dataclass(frozen=True)
class BGDBlock:
    index: Index
    direction: Direction
    distribution: "MUD"
    translation: tuple[Fraction, ...]
    decay_factor: object


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

    def scale(self, factor) -> BGD:
        result_E = np.empty(self.E.shape, dtype=object)
        for index in iter_indices(self.E.shape):
            result_E[index] = self.E[index].scale(factor)
        return BGD(result_E, self.alpha, self.beta)

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

    def marginalize(self, dim: int):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if self.ndim == 1:
            return self.mass()

        result_shape = (3,) * (self.ndim - 1)
        result_E = np.empty(result_shape, dtype=object)
        result_center = (1,) * (self.ndim - 1)
        old_center = (1,) * self.ndim
        remaining_dims = tuple(axis for axis in range(self.ndim) if axis != dim)

        for result_index in iter_indices(result_shape):
            pieces = []
            for removed_index in range(3):
                old_index = result_index[:dim] + (removed_index,) + result_index[dim:]
                piece = self.E[old_index].marginalize(dim)
                if result_index == result_center and old_index != old_center:
                    for new_axis, old_axis in enumerate(remaining_dims):
                        piece = _shift_mud_dim(
                            piece, new_axis, self.center_lefts[old_axis]
                        )
                piece = piece.scale(self._marginalize_removed_axis_factor(dim, removed_index))
                pieces.append(piece)

            result = pieces[0]
            for piece in pieces[1:]:
                result = result + piece
            result_E[result_index] = result

        alpha = self.alpha[:dim] + self.alpha[dim + 1 :]
        beta = self.beta[:dim] + self.beta[dim + 1 :]
        return BGD(result_E, alpha, beta).standardize()

    def permute_dims(self, order: Sequence[int]) -> BGD:
        order = MUD._validate_permutation(order, self.ndim)
        transposed_E = np.transpose(self.E, axes=order)
        result_E = np.empty(transposed_E.shape, dtype=object)
        for index in iter_indices(result_E.shape):
            result_E[index] = transposed_E[index].permute_dims(order)

        alpha = tuple(self.alpha[dim] for dim in order)
        beta = tuple(self.beta[dim] for dim in order)
        return BGD(result_E, alpha, beta)

    def replace_dim(self, dim: int, new: BGD) -> BGD:
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if not isinstance(new, BGD):
            raise TypeError("new must be a BGD")
        if new.ndim != 1:
            raise ValueError("new must be a one-dimensional BGD")

        if self.ndim == 1:
            return new.scale(self.mass()).standardize()

        rest = self.marginalize(dim)
        joint = rest.independent_product(new)
        order = tuple(range(dim)) + (self.ndim - 1,) + tuple(range(dim, self.ndim - 1))
        return joint.permute_dims(order).standardize()

    def convolve_uniform(
        self,
        dim: int,
        low,
        high,
        *,
        max_fn=None,
        bound_factory=None,
        max_interval=None,
    ):
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        if max_fn is not None and bound_factory is not None:
            raise ValueError("provide either max_fn or bound_factory, not both")

        noise_left = _as_fraction(low)
        noise_right = _as_fraction(high)
        if noise_left >= noise_right:
            raise ValueError("requires low < high")
        if self.left_lengths[dim] <= 0 or self.right_lengths[dim] <= 0:
            raise ValueError("edge period lengths must be positive")

        result_E = np.empty(self.E.shape, dtype=object)
        constraints = []
        for center_index in self._line_center_indices(dim):
            for target_axis in range(3):
                target_index = self._replace_index(center_index, dim, target_axis)
                block_bound_factory = None
                if bound_factory is not None:
                    block_bound_factory = self._prefixed_bound_factory(
                        bound_factory, f"E{target_index}"
                    )
                result = self._convolve_uniform_target_block(
                    center_index,
                    target_axis,
                    dim,
                    noise_left,
                    noise_right,
                    max_fn=max_fn,
                    bound_factory=block_bound_factory,
                    max_interval=max_interval,
                )
                if bound_factory is None:
                    result_E[target_index] = result
                else:
                    result_E[target_index], block_constraints = result
                    constraints.extend(block_constraints)

        result = BGD(result_E, self.alpha, self.beta).standardize()
        if bound_factory is None:
            return result
        return result, constraints

    def _convolve_uniform_target_block(
        self,
        center_index: Index,
        target_axis: int,
        dim: int,
        noise_left: Fraction,
        noise_right: Fraction,
        *,
        max_fn,
        bound_factory,
        max_interval,
    ):
        target_index = self._replace_index(center_index, dim, target_axis)
        target_left, target_right = self._convolve_uniform_target_interval(
            dim, target_axis, noise_left, noise_right
        )
        target_S = self._convolve_uniform_target_S(
            target_index, dim, target_left, target_right
        )
        accumulated = _zero_affine_mud(target_S, dim)

        for source_k in self._convolve_uniform_source_blocks(
            dim, target_axis, noise_left, noise_right, target_left, target_right
        ):
            source_axis = 0 if source_k < 0 else 2 if source_k > 0 else 1
            source_index = self._replace_index(center_index, dim, source_axis)
            source_left, source_right = self._block_interval_for_dim(dim, source_k)
            source = self.E[source_index].scale(
                self._dim_decay_factor_for_block(dim, source_k)
            )

            if source_index != (1,) * self.ndim:
                source = _shift_mud_dim(source, dim, source_left)

            piece = source.convolve_uniform(dim, noise_left, noise_right)
            piece = piece.restrict(dim, ">=", target_left)
            piece = piece.restrict(dim, "<=", target_right)
            if piece.is_empty:
                continue

            if target_index != (1,) * self.ndim:
                piece = _shift_grid_dim(piece, dim, -target_left)
            accumulated = accumulated + piece

        return accumulated.to_mass_mud_upper(
            max_fn=max_fn, bound_factory=bound_factory, max_interval=max_interval
        )

    def _convolve_uniform_target_S(
        self, index: Index, dim: int, target_left: Fraction, target_right: Fraction
    ) -> Breakpoints:
        target_S = list(self.E[index].S)
        if index == (1,) * self.ndim:
            target_S[dim] = (target_left, target_right)
        else:
            target_S[dim] = (Fraction(0), target_right - target_left)
        return tuple(target_S)

    def _convolve_uniform_target_interval(
        self,
        dim: int,
        target_axis: int,
        noise_left: Fraction,
        noise_right: Fraction,
    ) -> Interval:
        center_left = self.center_lefts[dim] + noise_left
        center_right = self.center_rights[dim] + noise_right
        if target_axis == 0:
            return center_left - self.left_lengths[dim], center_left
        if target_axis == 1:
            return center_left, center_right
        if target_axis == 2:
            return center_right, center_right + self.right_lengths[dim]
        raise ValueError("target_axis must be 0, 1, or 2")

    def _convolve_uniform_source_blocks(
        self,
        dim: int,
        target_axis: int,
        noise_left: Fraction,
        noise_right: Fraction,
        target_left: Fraction,
        target_right: Fraction,
    ) -> Iterable[int]:
        if target_axis in (0, 1):
            block = -1
            while True:
                source_left, source_right = self._block_interval_for_dim(dim, block)
                if source_right + noise_right <= target_left:
                    break
                if source_left + noise_left < target_right:
                    yield block
                block -= 1

        if target_axis == 1:
            yield 0

        if target_axis in (1, 2):
            block = 1
            while True:
                source_left, source_right = self._block_interval_for_dim(dim, block)
                if source_left + noise_left >= target_right:
                    break
                if source_right + noise_right > target_left:
                    yield block
                block += 1

    def _block_interval_for_dim(self, dim: int, block: int) -> Interval:
        if block < 0:
            left = self.center_lefts[dim] + block * self.left_lengths[dim]
            return left, left + self.left_lengths[dim]
        if block > 0:
            left = self.center_rights[dim] + (block - 1) * self.right_lengths[dim]
            return left, left + self.right_lengths[dim]
        return self.center_lefts[dim], self.center_rights[dim]

    def _dim_decay_factor_for_block(self, dim: int, block: int):
        if block < 0:
            return _object_power(self.alpha[dim], -block - 1)
        if block > 0:
            return _object_power(self.beta[dim], block - 1)
        return Fraction(1)

    @staticmethod
    def _prefixed_bound_factory(bound_factory, prefix: str):
        def wrapped(name, left, right):
            return bound_factory(f"{prefix}_{name}", left, right)

        return wrapped

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

    def _marginalize_removed_axis_factor(self, dim: int, removed_index: int):
        if removed_index == 0:
            return 1 / (1 - self.alpha[dim])
        if removed_index == 1:
            return Fraction(1)
        if removed_index == 2:
            return 1 / (1 - self.beta[dim])
        raise ValueError("removed_index must be 0, 1, or 2")

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
