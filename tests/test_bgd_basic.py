from fractions import Fraction
import unittest

import numpy as np

from distributions.bgd import (
    AffineCell,
    AffineMUD,
    BGD,
    MUD,
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


class Symbol:
    def __init__(self, expr):
        self.expr = expr

    def __str__(self):
        return self.expr

    def __repr__(self):
        return self.expr

    def __add__(self, other):
        return Symbol(f"({self.expr}+{other})")

    def __radd__(self, other):
        return Symbol(f"({other}+{self.expr})")

    def __mul__(self, other):
        return Symbol(f"({self.expr}*{other})")

    def __rmul__(self, other):
        return Symbol(f"({other}*{self.expr})")

    def __sub__(self, other):
        return Symbol(f"({self.expr}-{other})")

    def __rsub__(self, other):
        return Symbol(f"({other}-{self.expr})")

    def __truediv__(self, other):
        return Symbol(f"({self.expr}/{other})")

    def __rtruediv__(self, other):
        return Symbol(f"({other}/{self.expr})")

    def __pow__(self, exponent):
        return Symbol(f"({self.expr}**{exponent})")


def make_mud(direction):
    S = []
    for d in direction:
        if d == -1:
            right = Fraction(2)
        elif d == 0:
            right = Fraction(1)
        else:
            right = Fraction(3)
        S.append((Fraction(0), right))
    return MUD(S, np.ones((1,) * len(direction), dtype=object))


def make_e():
    E = np.empty((3, 3), dtype=object)
    E.fill(None)
    for i in range(3):
        for j in range(3):
            if (i, j) == (1, 1):
                E[i, j] = MUD([[0, 1], [0, 1]], [[1]])
            else:
                E[i, j] = make_mud((i - 1, j - 1))
    return E


def make_zero_mud(direction):
    S = []
    for d in direction:
        if d == -1:
            right = Fraction(2)
        elif d == 0:
            right = Fraction(1)
        else:
            right = Fraction(3)
        S.append((Fraction(0), right))
    return MUD(S, np.zeros((1,) * len(direction), dtype=object))


def make_zero_e():
    E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            E[i, j] = make_zero_mud((i - 1, j - 1))
    return E


def boundary_mass(mud, dim, side):
    if side == "left":
        if mud.S[dim][0] != mud.S[dim][1]:
            return 0
        interval_index = 0
    else:
        if mud.S[dim][-2] != mud.S[dim][-1]:
            return 0
        interval_index = mud.shape[dim] - 1

    slicer = [slice(None)] * mud.ndim
    slicer[dim] = interval_index
    return object_sum(mud.P[tuple(slicer)].flat)


def make_1d_bgd(left_mass=0, center_mass=0, right_mass=0, alpha=Fraction(1, 2), beta=Fraction(1, 2)):
    E = np.empty((3,), dtype=object)
    E[0] = MUD([[0, 1]], [left_mass])
    E[1] = MUD([[0, 1]], [center_mass])
    E[2] = MUD([[0, 1]], [right_mass])
    return BGD(E, [alpha], [beta])


class TestBGDBasic(unittest.TestCase):
    def test_mud_keeps_fraction_breakpoints_and_object_masses(self):
        token = Symbol("x")
        mud = MUD([[0, Fraction(1, 2), 1]], [token, 3])

        self.assertEqual(mud.S, ((Fraction(0), Fraction(1, 2), Fraction(1)),))
        self.assertEqual(mud.P.dtype, object)
        self.assertIs(mud.P[0], token)
        self.assertEqual(mud.shape, (2,))

    def test_mud_rejects_wrong_mass_shape(self):
        with self.assertRaisesRegex(ValueError, "P shape"):
            MUD([[0, 1], [0, 1]], [1])

    def test_mud_mass_uses_object_sum(self):
        token = Symbol("x")
        mass = MUD([[0, 1, 2]], [token, 3]).mass()

        self.assertIsInstance(mass, Symbol)
        self.assertEqual(mass.expr, "((0+x)+3)")

    def test_mud_align_splits_continuous_interval_by_length(self):
        aligned = MUD([[0, 2]], [10]).align([[0, 1, 2]])

        self.assertEqual(aligned.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(aligned.P.tolist(), [Fraction(5), Fraction(5)])

    def test_mud_align_requires_target_to_preserve_source_dirac(self):
        with self.assertRaisesRegex(ValueError, "target_S must cover"):
            MUD([[1, 1]], [10]).align([[0, 1, 2]])

    def test_mud_align_assigns_dirac_to_explicit_dirac_target(self):
        aligned = MUD([[1, 1]], [10]).align([[0, 1, 1, 2]])

        self.assertEqual(aligned.P.tolist(), [0, Fraction(10), 0])

    def test_mud_align_handles_mixed_continuous_and_dirac_dimensions(self):
        aligned = MUD([[0, 2], [1, 1]], [[12]]).align(
            [[0, 1, 2], [0, 1, 1, 2]]
        )

        self.assertEqual(
            aligned.P.tolist(),
            [[0, Fraction(6), 0], [0, Fraction(6), 0]],
        )

    def test_mud_align_requires_target_support_by_default(self):
        with self.assertRaisesRegex(ValueError, "target_S must cover"):
            MUD([[0, 2]], [10]).align([[0, 1]])

    def test_mud_align_requires_target_to_include_source_breakpoints(self):
        with self.assertRaisesRegex(ValueError, "target_S must cover"):
            MUD([[0, 1, 2]], [4, 6]).align([[0, 2]])

    def test_mud_align_keeps_object_mass(self):
        token = Symbol("x")
        aligned = MUD([[0, 2]], [token]).align([[0, 1, 2]])

        self.assertEqual(aligned.P[0].expr, "(0+(x*1/2))")
        self.assertEqual(aligned.P[1].expr, "(0+(x*1/2))")

    def test_mud_add_aligns_continuous_grids(self):
        result = MUD([[0, 2]], [10]) + MUD([[0, 1, 2]], [1, 2])

        self.assertEqual(result.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.P.tolist(), [Fraction(6), Fraction(7)])

    def test_mud_add_preserves_dirac_breakpoints(self):
        result = MUD([[1, 1]], [3]) + MUD([[0, 1, 2]], [5, 7])

        self.assertEqual(
            result.S,
            ((Fraction(0), Fraction(1), Fraction(1), Fraction(2)),),
        )
        self.assertEqual(result.P.tolist(), [Fraction(5), Fraction(3), Fraction(7)])

    def test_mud_add_keeps_object_expression(self):
        token = Symbol("x")
        result = MUD([[0, 1]], [token]) + MUD([[0, 1]], [2])

        self.assertEqual(result.P[0].expr, "((0+(x*1))+2)")

    def test_mud_add_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "other.ndim"):
            MUD([[0, 1]], [1]) + MUD([[0, 1], [0, 1]], [[1]])

    def test_mud_independent_product_concatenates_dimensions_and_multiplies_masses(self):
        left = MUD([[0, 1, 1]], [2, 3])
        right = MUD([[10, 12]], [5])

        result = left.independent_product(right)

        self.assertEqual(
            result.S,
            (
                (Fraction(0), Fraction(1), Fraction(1)),
                (Fraction(10), Fraction(12)),
            ),
        )
        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.P.tolist(), [[Fraction(10)], [Fraction(15)]])
        self.assertEqual(result.mass(), left.mass() * right.mass())

    def test_mud_independent_product_keeps_object_expressions(self):
        token = Symbol("x")

        result = MUD([[0, 1]], [token]).independent_product(MUD([[0, 1]], [3]))

        self.assertEqual(result.P[0, 0].expr, "(x*3)")

    def test_mud_marginalize_sums_axis_and_preserves_dirac_breakpoints(self):
        mud = MUD(
            [[0, 1, 1], [10, 11, 12]],
            [[2, 3], [5, 7]],
        )

        result = mud.marginalize(0)

        self.assertEqual(result.S, ((Fraction(10), Fraction(11), Fraction(12)),))
        self.assertEqual(result.P.tolist(), [Fraction(7), Fraction(10)])
        self.assertEqual(result.mass(), mud.mass())

    def test_mud_marginalize_rejects_last_dimension(self):
        with self.assertRaisesRegex(ValueError, "only MUD dimension"):
            MUD([[0, 1]], [1]).marginalize(0)

    def test_mud_permute_dims_reorders_breakpoints_and_tensor(self):
        mud = MUD(
            [[0, 1, 2], [10, 11, 11], [20, 21]],
            np.array([[[2], [3]], [[5], [7]]], dtype=object),
        )

        result = mud.permute_dims([1, 0, 2])

        self.assertEqual(
            result.S,
            (
                (Fraction(10), Fraction(11), Fraction(11)),
                (Fraction(0), Fraction(1), Fraction(2)),
                (Fraction(20), Fraction(21)),
            ),
        )
        self.assertEqual(result.shape, (2, 2, 1))
        self.assertEqual(result.P.tolist(), [[[2], [5]], [[3], [7]]])

    def test_mud_allows_empty_dimension(self):
        mud = MUD([[1], [0, 2]], np.empty((0, 1), dtype=object))

        self.assertTrue(mud.is_empty)
        self.assertEqual(mud.shape, (0, 1))
        self.assertEqual(mud.mass(), 0)

    def test_mud_restrict_continuous_interval(self):
        restricted = MUD([[0, 2]], [10]).restrict(0, ">=", 1)

        self.assertEqual(restricted.S, ((Fraction(1), Fraction(2)),))
        self.assertEqual(restricted.P.tolist(), [Fraction(5)])

    def test_mud_restrict_strict_and_nonstrict_are_same_for_continuous(self):
        gt = MUD([[0, 2]], [10]).restrict(0, ">", 1)
        ge = MUD([[0, 2]], [10]).restrict(0, ">=", 1)
        lt = MUD([[0, 2]], [10]).restrict(0, "<", 1)
        le = MUD([[0, 2]], [10]).restrict(0, "<=", 1)

        self.assertEqual(gt.S, ge.S)
        self.assertEqual(gt.P.tolist(), ge.P.tolist())
        self.assertEqual(lt.S, le.S)
        self.assertEqual(lt.P.tolist(), le.P.tolist())

    def test_mud_restrict_dirac_respects_strictness(self):
        self.assertEqual(MUD([[1, 1]], [10]).restrict(0, ">=", 1).P.tolist(), [10])
        self.assertEqual(MUD([[1, 1]], [10]).restrict(0, "<=", 1).P.tolist(), [10])
        self.assertTrue(MUD([[1, 1]], [10]).restrict(0, ">", 1).is_empty)
        self.assertTrue(MUD([[1, 1]], [10]).restrict(0, "<", 1).is_empty)

    def test_mud_restrict_empty_uses_canonical_empty_shape(self):
        mud = MUD([[0, 1], [0, Fraction(1, 2), 1]], [[3, 4]])
        restricted = mud.restrict(0, ">", 2)

        self.assertTrue(restricted.is_empty)
        self.assertEqual(restricted.S, ((Fraction(2),), (Fraction(0), Fraction(1))))
        self.assertEqual(restricted.shape, (0, 1))

    def test_mud_restrict_multidimensional_single_axis(self):
        restricted = MUD([[0, 2], [0, 1, 2]], [[10, 20]]).restrict(0, "<=", 1)

        self.assertEqual(
            restricted.S,
            ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1), Fraction(2))),
        )
        self.assertEqual(restricted.P.tolist(), [[Fraction(5), Fraction(10)]])

    def test_mud_restrict_rejects_bad_inputs(self):
        mud = MUD([[0, 1]], [1])

        with self.assertRaisesRegex(ValueError, "dim out of range"):
            mud.restrict(1, ">=", 0)
        with self.assertRaisesRegex(ValueError, "op must be"):
            mud.restrict(0, "==", 0)

    def test_mud_convolve_uniform_continuous_interval_returns_affine_triangle(self):
        result = MUD([[0, 1]], [1]).convolve_uniform(0, 0, 1)

        self.assertIsInstance(result, AffineMUD)
        self.assertEqual(result.affine_dim, 0)
        self.assertEqual(result.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.P.tolist(), [AffineCell(0, 1), AffineCell(1, 0)])
        self.assertEqual(result.mass(), Fraction(1))

    def test_mud_convolve_uniform_continuous_interval_returns_affine_trapezoid(self):
        result = MUD([[0, 2]], [10]).convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.S,
            ((Fraction(0), Fraction(1), Fraction(2), Fraction(3)),),
        )
        self.assertEqual(
            result.P.tolist(),
            [AffineCell(0, 5), AffineCell(5, 5), AffineCell(5, 0)],
        )
        self.assertEqual(result.mass(), Fraction(10))

    def test_mud_convolve_uniform_dirac_interval_returns_uniform_density(self):
        result = MUD([[1, 1]], [6]).convolve_uniform(0, 0, 2)

        self.assertEqual(result.S, ((Fraction(1), Fraction(3)),))
        self.assertEqual(result.P.tolist(), [AffineCell(3, 3)])
        self.assertEqual(result.mass(), Fraction(6))

    def test_mud_convolve_uniform_dirac_does_not_leak_to_adjacent_cell(self):
        result = MUD([[0, 0, 1]], [2, 3]).convolve_uniform(0, 0, 1)

        self.assertEqual(result.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.P.tolist(), [AffineCell(2, 5), AffineCell(3, 0)])
        self.assertEqual(result.mass(), Fraction(5))

    def test_mud_convolve_uniform_multidimensional_single_axis(self):
        result = MUD([[0, 1], [10, 12]], [[4]]).convolve_uniform(0, 0, 1)

        self.assertEqual(
            result.S,
            (
                (Fraction(0), Fraction(1), Fraction(2)),
                (Fraction(10), Fraction(12)),
            ),
        )
        self.assertEqual(result.P.tolist(), [[AffineCell(0, 4)], [AffineCell(4, 0)]])
        self.assertEqual(result.mass(), Fraction(4))

    def test_mud_convolve_uniform_keeps_object_mass_expression(self):
        token = Symbol("m")

        result = MUD([[0, 0]], [token]).convolve_uniform(0, 0, 2)

        self.assertEqual(str(result.P[0].left), "(0+(m/2))")
        self.assertEqual(str(result.P[0].right), "(0+(m/2))")

    def test_mud_convolve_uniform_empty_uses_canonical_empty_shape(self):
        mud = MUD([[1], [0, 2]], np.empty((0, 1), dtype=object))

        result = mud.convolve_uniform(0, 0, 1)

        self.assertTrue(result.is_empty)
        self.assertEqual(result.affine_dim, 0)
        self.assertEqual(result.S, ((Fraction(1),), (Fraction(0), Fraction(2))))
        self.assertEqual(result.shape, (0, 1))

    def test_mud_convolve_uniform_rejects_bad_inputs(self):
        mud = MUD([[0, 1]], [1])

        with self.assertRaisesRegex(ValueError, "dim out of range"):
            mud.convolve_uniform(1, 0, 1)
        with self.assertRaisesRegex(ValueError, "low < high"):
            mud.convolve_uniform(0, 1, 1)

    def test_mud_convolve_uniform_upper_wraps_affine_upper(self):
        result = MUD([[0, 1]], [1]).convolve_uniform_upper(0, 0, 1)

        self.assertIsInstance(result, MUD)
        self.assertEqual(result.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.P.tolist(), [Fraction(1), Fraction(1)])

    def test_affine_mud_mass_integrates_endpoint_values(self):
        mud = AffineMUD([[0, 2]], [AffineCell(1, 3)], affine_dim=0)

        self.assertEqual(mud.mass(), Fraction(4))

    def test_affine_mud_align_splits_affine_dimension_by_interpolation(self):
        aligned = AffineMUD([[0, 2]], [AffineCell(1, 3)], affine_dim=0).align(
            [[0, 1, 2]]
        )

        self.assertEqual(aligned.S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(aligned.P.tolist(), [AffineCell(1, 2), AffineCell(2, 3)])
        self.assertEqual(aligned.mass(), Fraction(4))

    def test_affine_mud_restrict_affine_dimension_interpolates(self):
        restricted = AffineMUD(
            [[0, 2]], [AffineCell(4, 8)], affine_dim=0
        ).restrict(0, ">=", 1)

        self.assertEqual(restricted.S, ((Fraction(1), Fraction(2)),))
        self.assertEqual(restricted.P.tolist(), [AffineCell(6, 8)])
        self.assertEqual(restricted.mass(), Fraction(7))

    def test_affine_mud_restrict_non_affine_dimension_scales_endpoints(self):
        restricted = AffineMUD(
            [[0, 2], [0, 4]], [[AffineCell(4, 8)]], affine_dim=0
        ).restrict(1, "<=", 1)

        self.assertEqual(
            restricted.S,
            ((Fraction(0), Fraction(2)), (Fraction(0), Fraction(1))),
        )
        self.assertEqual(restricted.P.tolist(), [[AffineCell(1, 2)]])
        self.assertEqual(restricted.mass(), Fraction(3))

    def test_affine_mud_to_mass_mud_upper_uses_endpoint_max(self):
        upper = AffineMUD(
            [[0, 2]], [AffineCell(1, 3)], affine_dim=0
        ).to_mass_mud_upper()

        self.assertIsInstance(upper, MUD)
        self.assertEqual(upper.P.tolist(), [Fraction(6)])

    def test_affine_mud_to_mass_mud_upper_accepts_bound_factory(self):
        def bound_factory(name, left, right):
            upper = Symbol(f"h_{name}")
            return upper, [(upper, ">=", left), (upper, ">=", right)]

        upper, constraints = AffineMUD(
            [[0, 2]], [AffineCell(Symbol("l"), Symbol("r"))], affine_dim=0
        ).to_mass_mud_upper(bound_factory=bound_factory)

        self.assertEqual(upper.P[0].expr, "(h_cell(0,)*2)")
        self.assertEqual(str(constraints[0][0]), "h_cell(0,)")
        self.assertEqual(constraints[0][1], ">=")
        self.assertEqual(str(constraints[0][2]), "l")
        self.assertEqual(str(constraints[1][2]), "r")

    def test_affine_mud_marginalize_non_affine_dim_adjusts_affine_dim(self):
        mud = AffineMUD(
            [[0, 1, 2], [0, 2]],
            [[AffineCell(1, 2)], [AffineCell(3, 4)]],
            affine_dim=1,
        )

        result = mud.marginalize(0)

        self.assertIsInstance(result, AffineMUD)
        self.assertEqual(result.affine_dim, 0)
        self.assertEqual(result.S, ((Fraction(0), Fraction(2)),))
        self.assertEqual(result.P.tolist(), [AffineCell(4, 6)])
        self.assertEqual(result.mass(), mud.mass())

    def test_affine_mud_rejects_marginalizing_affine_dim(self):
        with self.assertRaisesRegex(ValueError, "affine dimension"):
            AffineMUD(
                [[0, 1], [0, 1]], [[AffineCell(1, 1)]], affine_dim=0
            ).marginalize(0)

    def test_bgd_validates_core_structure(self):
        bgd = BGD(
            make_e(),
            [0, Fraction(1, 2)],
            [0.25, 0],
        )

        self.assertEqual(bgd.ndim, 2)
        self.assertEqual(bgd.center_lefts, (Fraction(0), Fraction(0)))
        self.assertEqual(bgd.center_rights, (Fraction(1), Fraction(1)))
        self.assertEqual(bgd.center_lengths, (Fraction(1), Fraction(1)))
        self.assertEqual(bgd.left_lengths, (Fraction(2), Fraction(2)))
        self.assertEqual(bgd.right_lengths, (Fraction(3), Fraction(3)))
        self.assertIs(bgd.C, bgd.E[1, 1])
        self.assertEqual(BGD.direction_to_index((-1, 0, 1)), (0, 1, 2))
        self.assertEqual(BGD.index_to_direction((0, 1, 2)), (-1, 0, 1))

    def test_bgd_allows_nonzero_center_origin(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 2]], [1])
        E[1] = MUD([[10, 12]], [1])
        E[2] = MUD([[0, 3]], [1])
        bgd = BGD(E, [0], [0])

        self.assertEqual(bgd.center_lefts, (Fraction(10),))
        self.assertEqual(bgd.center_rights, (Fraction(12),))
        self.assertEqual(bgd.translation((0,)), (Fraction(10),))
        self.assertEqual(bgd.translation((-1,)), (Fraction(8),))
        self.assertEqual(bgd.translation((1,)), (Fraction(12),))

    def test_bgd_rejects_noncenter_edge_origin(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[1, 3]], [1])
        E[1] = MUD([[10, 12]], [1])
        E[2] = MUD([[0, 3]], [1])

        with self.assertRaisesRegex(ValueError, r"E\(0,\).S\[0\]\[0\]"):
            BGD(E, [0], [0])

    def test_bgd_rejects_inconsistent_edge_lengths(self):
        E = make_e()
        E[2, 1] = MUD([[0, 4], [0, 1]], [[1]])

        with self.assertRaisesRegex(ValueError, "right edge length"):
            BGD(E, [0, 0], [0, 0])

    def test_bgd_core_block_queries(self):
        bgd = BGD(make_e(), [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)])

        self.assertEqual(bgd.direction((-3, 0)), (-1, 0))
        self.assertEqual(bgd.direction((0, 2)), (0, 1))
        self.assertEqual(bgd.local_lengths((-1, 0)), (Fraction(2), Fraction(1)))
        self.assertEqual(bgd.local_lengths((0, 1)), (Fraction(1), Fraction(3)))
        self.assertEqual(bgd.translation((-3, 0)), (Fraction(-6), Fraction(0)))
        self.assertEqual(bgd.translation((0, 2)), (Fraction(0), Fraction(4)))
        self.assertEqual(bgd.decay_factor((-3, 2)), Fraction(1, 28))

        center = bgd.block_at((0, 0))
        edge = bgd.block_at((-3, 2))

        self.assertEqual(center.index, (1, 1))
        self.assertEqual(center.direction, (0, 0))
        self.assertIs(center.distribution, bgd.C)
        self.assertEqual(center.translation, (Fraction(0), Fraction(0)))
        self.assertEqual(center.decay_factor, 1)

        self.assertEqual(edge.index, (0, 2))
        self.assertEqual(edge.direction, (-1, 1))
        self.assertIs(edge.distribution, bgd.E[0, 2])
        self.assertEqual(edge.translation, (Fraction(-6), Fraction(4)))
        self.assertEqual(edge.decay_factor, Fraction(1, 28))

    def test_bgd_block_queries_validate_inputs(self):
        bgd = BGD(make_e(), [0, 0], [0, 0])

        with self.assertRaisesRegex(ValueError, "k must contain"):
            bgd.direction((0,))
        with self.assertRaisesRegex(TypeError, "integers"):
            bgd.direction((0, 1.5))
        with self.assertRaisesRegex(ValueError, "direction must contain"):
            bgd.local_lengths((0,))
        with self.assertRaisesRegex(ValueError, "direction entries"):
            bgd.local_lengths((0, 2))

    def test_bgd_mass_uses_geometric_tail_factors_1d(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 2]], [2])
        E[1] = MUD([[0, 1]], [3])
        E[2] = MUD([[0, 3]], [5])
        bgd = BGD(E, [Fraction(1, 2)], [Fraction(1, 4)])

        self.assertEqual(bgd.mass(), Fraction(41, 3))

    def test_bgd_mass_uses_direction_tail_product_2d(self):
        E = make_e()
        E[1, 1] = MUD([[0, 1], [0, 1]], [[2]])
        E[0, 1] = MUD([[0, 2], [0, 1]], [[3]])
        E[2, 2] = MUD([[0, 3], [0, 3]], [[5]])
        zero = MUD([[0, 2], [0, 2]], [[0]])
        E[0, 0] = zero
        E[0, 2] = MUD([[0, 2], [0, 3]], [[0]])
        E[1, 0] = MUD([[0, 1], [0, 2]], [[0]])
        E[1, 2] = MUD([[0, 1], [0, 3]], [[0]])
        E[2, 0] = MUD([[0, 3], [0, 2]], [[0]])
        E[2, 1] = MUD([[0, 3], [0, 1]], [[0]])
        bgd = BGD(E, [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 4), Fraction(1, 5)])

        self.assertEqual(bgd.mass(), Fraction(49, 3))

    def test_bgd_mass_keeps_object_decay_expression(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 2]], [2])
        E[1] = MUD([[0, 1]], [3])
        E[2] = MUD([[0, 3]], [0])
        alpha = Symbol("a")
        bgd = BGD(E, [alpha], [0])
        mass = bgd.mass()

        self.assertIsInstance(mass, Symbol)
        self.assertEqual(mass.expr, "(3+(2*(1/(1-a))))")

    def test_bgd_standardize_1d_moves_boundary_dirac_mass(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 0, 2, 2]], [5, 0, 7])
        E[1] = MUD([[0, 1, 1]], [3, 4])
        E[2] = MUD([[0, 0, 3, 3]], [11, 0, 13])
        bgd = BGD(E, [Fraction(1, 2)], [Fraction(1, 3)])

        before_mass = bgd.mass()
        standardized = bgd.standardize()

        self.assertIsNot(standardized, bgd)
        self.assertEqual(bgd.E[0].P.tolist(), [5, 0, 7])
        self.assertEqual(standardized.mass(), before_mass)
        self.assertEqual(
            standardized.E[0].S,
            ((Fraction(0), Fraction(0), Fraction(2)),),
        )
        self.assertEqual(standardized.E[0].P.tolist(), [Fraction(17, 2), 0])
        self.assertEqual(standardized.E[1].S, ((Fraction(0), Fraction(0), Fraction(1), Fraction(1)),))
        self.assertEqual(standardized.E[1].P.tolist(), [Fraction(7), Fraction(3), Fraction(15)])
        self.assertEqual(
            standardized.E[2].S,
            ((Fraction(0), Fraction(3), Fraction(3)),),
        )
        self.assertEqual(standardized.E[2].P.tolist(), [0, Fraction(50, 3)])

    def test_bgd_standardize_uses_global_center_endpoints(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 2, 2]], [0, 7])
        E[1] = MUD([[10, 12]], [3])
        E[2] = MUD([[0, 0, 3]], [11, 0])
        bgd = BGD(E, [Fraction(1, 2)], [Fraction(1, 3)])

        standardized = bgd.standardize()

        self.assertEqual(
            standardized.E[1].S,
            ((Fraction(10), Fraction(10), Fraction(12), Fraction(12)),),
        )
        self.assertEqual(standardized.E[1].P.tolist(), [Fraction(7), Fraction(3), Fraction(11)])
        self.assertEqual(standardized.mass(), bgd.mass())

    def test_bgd_restrict_inside_center_greater(self):
        bgd = make_1d_bgd(left_mass=7, center_mass=10, right_mass=3)
        restricted = bgd.restrict(0, ">=", Fraction(1, 2))

        self.assertTrue(restricted.E[0].is_empty)
        self.assertEqual(restricted.E[1].S, ((Fraction(1, 2), Fraction(1)),))
        self.assertEqual(restricted.E[1].P.tolist(), [Fraction(5)])
        self.assertEqual(restricted.E[2].P.tolist(), [3])

    def test_bgd_restrict_right_phase_greater(self):
        bgd = make_1d_bgd(right_mass=10, beta=Fraction(1, 2))
        restricted = bgd.restrict(0, ">", Fraction(3, 2))

        self.assertTrue(restricted.E[0].is_empty)
        self.assertTrue(restricted.E[1].is_empty)
        self.assertEqual(restricted.E[1].S, ((Fraction(3, 2),),))
        self.assertEqual(restricted.E[2].S, ((Fraction(0), Fraction(1, 2), Fraction(1)),))
        self.assertEqual(restricted.E[2].P.tolist(), [Fraction(5), Fraction(5, 2)])
        self.assertEqual(restricted.mass(), Fraction(15))

    def test_bgd_restrict_right_prefix_less(self):
        bgd = make_1d_bgd(center_mass=3, right_mass=10, beta=Fraction(1, 2))
        restricted = bgd.restrict(0, "<", Fraction(3, 2))

        self.assertEqual(restricted.E[1].S, ((Fraction(0), Fraction(1), Fraction(3, 2)),))
        self.assertEqual(restricted.E[1].P.tolist(), [Fraction(3), Fraction(5)])
        self.assertTrue(restricted.E[2].is_empty)
        self.assertEqual(restricted.mass(), Fraction(8))

    def test_bgd_restrict_left_phase_less(self):
        bgd = make_1d_bgd(left_mass=10, alpha=Fraction(1, 2))
        restricted = bgd.restrict(0, "<", Fraction(-1, 2))

        self.assertEqual(restricted.E[0].S, ((Fraction(0), Fraction(1, 2), Fraction(1)),))
        self.assertEqual(restricted.E[0].P.tolist(), [Fraction(5, 2), Fraction(5)])
        self.assertTrue(restricted.E[1].is_empty)
        self.assertTrue(restricted.E[2].is_empty)
        self.assertEqual(restricted.mass(), Fraction(15))

    def test_bgd_restrict_left_prefix_greater(self):
        bgd = make_1d_bgd(left_mass=10, center_mass=3, alpha=Fraction(1, 2))
        restricted = bgd.restrict(0, ">", Fraction(-1, 2))

        self.assertTrue(restricted.E[0].is_empty)
        self.assertEqual(restricted.E[1].S, ((Fraction(-1, 2), Fraction(0), Fraction(1)),))
        self.assertEqual(restricted.E[1].P.tolist(), [Fraction(5), Fraction(3)])
        self.assertEqual(restricted.mass(), Fraction(8))

    def test_bgd_align_center_domain_expands_left_and_right_exactly(self):
        bgd = make_1d_bgd(
            left_mass=8,
            center_mass=3,
            right_mass=10,
            alpha=Fraction(1, 2),
            beta=Fraction(1, 3),
        )

        aligned = bgd.align_center_domain([Fraction(-1, 2)], [Fraction(3, 2)])

        self.assertEqual(aligned.center_lefts, (Fraction(-1, 2),))
        self.assertEqual(aligned.center_rights, (Fraction(3, 2),))
        self.assertEqual(aligned.left_lengths, bgd.left_lengths)
        self.assertEqual(aligned.right_lengths, bgd.right_lengths)
        self.assertEqual(aligned.alpha, bgd.alpha)
        self.assertEqual(aligned.beta, bgd.beta)
        self.assertEqual(aligned.mass(), bgd.mass())
        self.assertEqual(
            aligned.E[1].S,
            ((Fraction(-1, 2), Fraction(0), Fraction(1), Fraction(3, 2)),),
        )
        self.assertEqual(aligned.E[1].P.tolist(), [Fraction(4), Fraction(3), Fraction(5)])

    def test_bgd_align_center_domain_rejects_shrinking_center(self):
        bgd = make_1d_bgd(center_mass=3)

        with self.assertRaisesRegex(ValueError, "target center must contain"):
            bgd.align_center_domain([Fraction(1, 2)], [1])

    def test_bgd_align_edge_periods_expands_periods_and_decays(self):
        bgd = make_1d_bgd(
            left_mass=8,
            center_mass=3,
            right_mass=10,
            alpha=Fraction(1, 2),
            beta=Fraction(1, 3),
        )

        aligned = bgd.align_edge_periods([2], [3])

        self.assertEqual(aligned.center_lefts, bgd.center_lefts)
        self.assertEqual(aligned.center_rights, bgd.center_rights)
        self.assertEqual(aligned.left_lengths, (Fraction(2),))
        self.assertEqual(aligned.right_lengths, (Fraction(3),))
        self.assertEqual(aligned.alpha, (Fraction(1, 4),))
        self.assertEqual(aligned.beta, (Fraction(1, 27),))
        self.assertEqual(aligned.mass(), bgd.mass())
        self.assertEqual(aligned.E[0].S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(aligned.E[0].P.tolist(), [Fraction(4), Fraction(8)])
        self.assertEqual(
            aligned.E[2].S,
            ((Fraction(0), Fraction(1), Fraction(2), Fraction(3)),),
        )
        self.assertEqual(
            aligned.E[2].P.tolist(),
            [Fraction(10), Fraction(10, 3), Fraction(10, 9)],
        )

    def test_bgd_align_edge_periods_rejects_non_integer_multiple(self):
        bgd = make_1d_bgd()

        with self.assertRaisesRegex(ValueError, "integer multiple"):
            bgd.align_edge_periods([Fraction(3, 2)], [1])

    def test_bgd_relax_decay_increases_tail_mass_as_upper_bound(self):
        bgd = make_1d_bgd(
            left_mass=8,
            center_mass=3,
            right_mass=10,
            alpha=Fraction(1, 3),
            beta=Fraction(1, 4),
        )

        relaxed = bgd.relax_decay([Fraction(1, 2)], [Fraction(1, 2)])

        self.assertEqual(relaxed.alpha, (Fraction(1, 2),))
        self.assertEqual(relaxed.beta, (Fraction(1, 2),))
        self.assertEqual(relaxed.E[0].P.tolist(), bgd.E[0].P.tolist())
        self.assertGreater(relaxed.mass(), bgd.mass())

    def test_bgd_relax_decay_rejects_smaller_decay(self):
        bgd = make_1d_bgd(alpha=Fraction(1, 2), beta=Fraction(1, 2))

        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            bgd.relax_decay([Fraction(1, 3)], [Fraction(1, 2)])

    def test_bgd_add_aligns_center_periods_and_sums_with_common_decay(self):
        left = make_1d_bgd(
            left_mass=8,
            center_mass=3,
            right_mass=9,
            alpha=Fraction(1, 2),
            beta=Fraction(1, 3),
        )
        right = BGD(
            [
                MUD([[0, 2]], [5]),
                MUD([[Fraction(1, 2), 1]], [7]),
                MUD([[0, 3]], [11]),
            ],
            [Fraction(1, 4)],
            [Fraction(1, 27)],
        )

        result = left + right

        self.assertEqual(result.center_lefts, (Fraction(0),))
        self.assertEqual(result.center_rights, (Fraction(1),))
        self.assertEqual(result.left_lengths, (Fraction(2),))
        self.assertEqual(result.right_lengths, (Fraction(3),))
        self.assertEqual(result.alpha, (Fraction(1, 4),))
        self.assertEqual(result.beta, (Fraction(1, 27),))
        self.assertEqual(result.mass(), left.mass() + right.mass())
        self.assertEqual(
            result.E[1].S,
            ((Fraction(0), Fraction(1, 2), Fraction(1)),),
        )
        self.assertEqual(result.E[1].P.tolist(), [Fraction(11, 4), Fraction(17, 2)])

    def test_bgd_add_uses_max_decay_as_upper_bound_when_decays_differ(self):
        left = make_1d_bgd(left_mass=2, alpha=Fraction(1, 2), beta=0)
        right = make_1d_bgd(left_mass=3, alpha=Fraction(1, 3), beta=0)

        result = left + right

        self.assertEqual(result.alpha, (Fraction(1, 2),))
        self.assertEqual(result.E[0].P.tolist(), [Fraction(5)])
        self.assertGreater(result.mass(), left.mass() + right.mass())

    def test_bgd_add_rejects_symbolic_decay_max(self):
        left = make_1d_bgd(alpha=Symbol("a"), beta=0)
        right = make_1d_bgd(alpha=Fraction(1, 2), beta=0)

        with self.assertRaisesRegex(ValueError, "max requires"):
            left + right

    def test_bgd_add_accepts_custom_symbolic_max_function(self):
        left = make_1d_bgd(left_mass=2, alpha=Symbol("a"), beta=0)
        right = make_1d_bgd(left_mass=3, alpha=Fraction(1, 2), beta=0)

        def symbolic_max(first, second, name):
            return Symbol(f"max_{name}({first},{second})")

        result = left.add(right, max_fn=symbolic_max)

        self.assertIsInstance(result.alpha[0], Symbol)
        self.assertEqual(result.alpha[0].expr, "max_alpha[0](a,1/2)")
        self.assertEqual(result.E[0].P.tolist(), [Fraction(5)])

    def test_bgd_convolve_uniform_center_only_updates_center_frame(self):
        bgd = make_1d_bgd(center_mass=1, alpha=0, beta=0)

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.center_lefts, (Fraction(0),))
        self.assertEqual(result.center_rights, (Fraction(2),))
        self.assertEqual(result.left_lengths, bgd.left_lengths)
        self.assertEqual(result.right_lengths, bgd.right_lengths)
        self.assertEqual(result.E[1].S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.E[1].P.tolist(), [Fraction(1), Fraction(1)])
        self.assertEqual(result.E[0].P.tolist(), [Fraction(0)])
        self.assertEqual(result.E[2].P.tolist(), [Fraction(0)])

    def test_bgd_convolve_uniform_center_receives_left_tail_contribution(self):
        bgd = make_1d_bgd(left_mass=2, center_mass=0, right_mass=0, alpha=0, beta=0)

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.center_lefts, (Fraction(0),))
        self.assertEqual(result.center_rights, (Fraction(2),))
        self.assertEqual(result.E[0].S, ((Fraction(0), Fraction(1)),))
        self.assertEqual(result.E[0].P.tolist(), [Fraction(2)])
        self.assertEqual(result.E[1].S, ((Fraction(0), Fraction(1), Fraction(2)),))
        self.assertEqual(result.E[1].P.tolist(), [Fraction(2), Fraction(0)])
        self.assertEqual(result.E[2].P.tolist(), [Fraction(0)])

    def test_bgd_convolve_uniform_edges_do_not_receive_old_center_contribution(self):
        bgd = make_1d_bgd(left_mass=0, center_mass=3, right_mass=0, alpha=0, beta=0)

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.E[0].P.tolist(), [Fraction(0)])
        self.assertEqual(result.E[2].P.tolist(), [Fraction(0)])
        self.assertEqual(result.E[1].P.tolist(), [Fraction(3), Fraction(3)])

    def test_bgd_convolve_uniform_dirac_center_becomes_uniform_mass(self):
        E = np.empty((3,), dtype=object)
        E[0] = MUD([[0, 1]], [0])
        E[1] = MUD([[0, 0]], [4])
        E[2] = MUD([[0, 1]], [0])
        bgd = BGD(E, [0], [0])

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.center_lefts, (Fraction(0),))
        self.assertEqual(result.center_rights, (Fraction(1),))
        self.assertEqual(result.E[1].S, ((Fraction(0), Fraction(1)),))
        self.assertEqual(result.E[1].P.tolist(), [Fraction(4)])

    def test_bgd_convolve_uniform_accepts_bound_factory(self):
        bgd = make_1d_bgd(center_mass=1, alpha=0, beta=0)

        def bound_factory(name, left, right):
            upper = Symbol(f"h_{name}")
            return upper, [(upper, ">=", left), (upper, ">=", right)]

        result, constraints = bgd.convolve_uniform(
            0, 0, 1, bound_factory=bound_factory
        )

        self.assertIsInstance(result, BGD)
        self.assertGreater(len(constraints), 0)
        self.assertTrue(str(constraints[0][0]).startswith("h_E"))

    def test_bgd_convolve_uniform_multidimensional_single_axis(self):
        E = make_zero_e()
        E[1, 1] = MUD([[0, 1], [0, 1]], [[1]])
        bgd = BGD(E, [0, 0], [0, 0])

        result = bgd.convolve_uniform(0, 0, 1)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.center_lefts, (Fraction(0), Fraction(0)))
        self.assertEqual(result.center_rights, (Fraction(2), Fraction(1)))
        self.assertEqual(
            result.E[1, 1].S,
            ((Fraction(0), Fraction(1), Fraction(2)), (Fraction(0), Fraction(1))),
        )
        self.assertEqual(result.E[1, 1].P.tolist(), [[Fraction(1)], [Fraction(1)]])

    def test_bgd_independent_product_combines_1d_bgds_into_2d_bgd(self):
        left = BGD(
            [
                MUD([[0, 1]], [2]),
                MUD([[0, 1, 1]], [3, 5]),
                MUD([[0, 2]], [7]),
            ],
            [Fraction(1, 2)],
            [Fraction(1, 3)],
        )
        right = BGD(
            [
                MUD([[0, 3]], [11]),
                MUD([[10, 12]], [13]),
                MUD([[0, 4]], [17]),
            ],
            [Fraction(1, 5)],
            [Fraction(1, 7)],
        )

        result = left.independent_product(right)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.alpha, (Fraction(1, 2), Fraction(1, 5)))
        self.assertEqual(result.beta, (Fraction(1, 3), Fraction(1, 7)))
        self.assertEqual(result.center_lefts, (Fraction(0), Fraction(10)))
        self.assertEqual(result.center_rights, (Fraction(1), Fraction(12)))
        self.assertEqual(result.left_lengths, (Fraction(1), Fraction(3)))
        self.assertEqual(result.right_lengths, (Fraction(2), Fraction(4)))
        self.assertEqual(result.mass(), left.mass() * right.mass())
        self.assertEqual(
            result.E[1, 1].S,
            (
                (Fraction(0), Fraction(1), Fraction(1)),
                (Fraction(10), Fraction(12)),
            ),
        )
        self.assertEqual(result.E[1, 1].P.tolist(), [[Fraction(39)], [Fraction(65)]])
        self.assertEqual(result.E[0, 2].P.tolist(), [[Fraction(34)]])

    def test_bgd_independent_product_combines_dimensions_in_order(self):
        left = make_1d_bgd(left_mass=2, center_mass=3, right_mass=5)
        right = BGD(make_e(), [Fraction(1, 3), Fraction(1, 4)], [Fraction(1, 5), Fraction(1, 6)])

        result = left.independent_product(right)

        self.assertEqual(result.ndim, 3)
        self.assertEqual(
            result.alpha,
            (Fraction(1, 2), Fraction(1, 3), Fraction(1, 4)),
        )
        self.assertEqual(
            result.beta,
            (Fraction(1, 2), Fraction(1, 5), Fraction(1, 6)),
        )
        self.assertEqual(result.E[1, 1, 1].P.tolist(), [[[Fraction(3)]]])
        self.assertEqual(result.mass(), left.mass() * right.mass())

    def test_bgd_independent_product_rejects_non_bgd(self):
        with self.assertRaisesRegex(TypeError, "other must be a BGD"):
            make_1d_bgd().independent_product(MUD([[0, 1]], [1]))

    def test_bgd_scale_multiplies_all_blocks(self):
        bgd = make_1d_bgd(left_mass=2, center_mass=3, right_mass=5)

        result = bgd.scale(7)

        self.assertEqual(result.E[0].P.tolist(), [Fraction(14)])
        self.assertEqual(result.E[1].P.tolist(), [Fraction(21)])
        self.assertEqual(result.E[2].P.tolist(), [Fraction(35)])
        self.assertEqual(result.mass(), bgd.mass() * 7)

    def test_bgd_marginalize_removes_dimension_and_preserves_mass(self):
        E = make_e()
        E[1, 1] = MUD([[0, 1], [10, 12]], [[3]])
        E[0, 1] = MUD([[0, 2], [0, 2]], [[5]])
        E[2, 1] = MUD([[0, 3], [0, 2]], [[7]])
        bgd = BGD(E, [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)])

        result = bgd.marginalize(0)

        self.assertIsInstance(result, BGD)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.alpha, (Fraction(1, 3),))
        self.assertEqual(result.beta, (Fraction(1, 7),))
        self.assertEqual(result.center_lefts, (Fraction(10),))
        self.assertEqual(result.center_rights, (Fraction(12),))
        self.assertEqual(result.mass(), bgd.mass())
        self.assertEqual(result.E[1].S, ((Fraction(10), Fraction(12)),))

    def test_bgd_marginalize_1d_returns_scalar_mass(self):
        bgd = make_1d_bgd(left_mass=2, center_mass=3, right_mass=5)

        self.assertEqual(bgd.marginalize(0), bgd.mass())

    def test_bgd_permute_dims_reorders_frame_and_blocks(self):
        bgd = BGD(make_e(), [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)])

        result = bgd.permute_dims([1, 0])

        self.assertEqual(result.alpha, (Fraction(1, 3), Fraction(1, 2)))
        self.assertEqual(result.beta, (Fraction(1, 7), Fraction(1, 5)))
        self.assertEqual(result.left_lengths, (Fraction(2), Fraction(2)))
        self.assertEqual(result.right_lengths, (Fraction(3), Fraction(3)))
        self.assertEqual(result.E[2, 0].P.tolist(), bgd.E[0, 2].P.tolist())

    def test_bgd_replace_dim_replaces_with_independent_1d_bgd(self):
        original = BGD(make_e(), [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)])
        new_dim = make_1d_bgd(
            left_mass=11,
            center_mass=13,
            right_mass=17,
            alpha=Fraction(1, 4),
            beta=Fraction(1, 6),
        )

        result = original.replace_dim(0, new_dim)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.alpha, (Fraction(1, 4), Fraction(1, 3)))
        self.assertEqual(result.beta, (Fraction(1, 6), Fraction(1, 7)))
        self.assertEqual(result.mass(), original.mass() * new_dim.mass())
        self.assertEqual(result.E[1, 1].S[0], new_dim.E[1].S[0])

    def test_bgd_replace_dim_1d_scales_new_distribution_by_old_mass(self):
        original = make_1d_bgd(left_mass=2, center_mass=3, right_mass=5)
        new_dim = make_1d_bgd(left_mass=7, center_mass=11, right_mass=13)

        result = original.replace_dim(0, new_dim)

        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.mass(), original.mass() * new_dim.mass())
        self.assertEqual(result.E[1].P.tolist(), [new_dim.E[1].P[0] * original.mass()])

    def test_bgd_replace_dim_rejects_bad_inputs(self):
        bgd = make_1d_bgd()

        with self.assertRaisesRegex(ValueError, "dim out of range"):
            bgd.replace_dim(1, make_1d_bgd())
        with self.assertRaisesRegex(TypeError, "new must be a BGD"):
            bgd.replace_dim(0, MUD([[0, 1]], [1]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            bgd.replace_dim(0, BGD(make_e(), [0, 0], [0, 0]))

    def test_bgd_restrict_rejects_bad_inputs(self):
        bgd = make_1d_bgd()

        with self.assertRaisesRegex(ValueError, "dim out of range"):
            bgd.restrict(1, ">=", 0)
        with self.assertRaisesRegex(ValueError, "op must be"):
            bgd.restrict(0, "==", 0)

    def test_bgd_standardize_high_dimensional_near_center_boundaries(self):
        E = make_zero_e()
        corner_P = np.zeros((2, 2), dtype=object)
        corner_P[1, 1] = 12
        E[0, 0] = MUD([[0, 2, 2], [0, 2, 2]], corner_P)
        bgd = BGD(E, [Fraction(1, 2), Fraction(1, 3)], [Fraction(1, 5), Fraction(1, 7)])

        before_mass = bgd.mass()
        standardized = bgd.standardize()

        self.assertEqual(standardized.mass(), before_mass)
        for index in iter_indices(standardized.E.shape):
            direction = BGD.index_to_direction(index)
            mud = standardized.E[index]
            for dim, value in enumerate(direction):
                if value < 0:
                    self.assertEqual(boundary_mass(mud, dim, "right"), 0)
                elif value > 0:
                    self.assertEqual(boundary_mass(mud, dim, "left"), 0)


class TestBGDUtilities(unittest.TestCase):
    def test_interval_helpers_use_fractions(self):
        self.assertTrue(is_dirac_interval(Fraction(1, 2), Fraction(1, 2)))
        self.assertFalse(is_dirac_interval(0, 1))
        self.assertEqual(interval_length(0, Fraction(3, 2)), Fraction(3, 2))
        self.assertEqual(
            interval_intersection(0, 2, Fraction(1, 2), 3),
            (Fraction(1, 2), Fraction(2)),
        )
        self.assertIsNone(interval_intersection(0, 1, 2, 3))
        self.assertTrue(point_in_interval(Fraction(1, 2), 0, 1))
        self.assertFalse(point_in_interval(Fraction(3, 2), 0, 1))

    def test_merge_breakpoints_preserves_dirac_by_default(self):
        self.assertEqual(
            merge_breakpoints([0, 1, 1, 2], [0, 2, 2, 3]),
            (
                Fraction(0),
                Fraction(1),
                Fraction(1),
                Fraction(2),
                Fraction(2),
                Fraction(3),
            ),
        )

    def test_merge_breakpoints_can_sort_and_deduplicate(self):
        self.assertEqual(
            merge_breakpoints(
                [1, 0, Fraction(1, 2)],
                [Fraction(1, 2), 2],
                preserve_dirac=False,
            ),
            (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2)),
        )

    def test_fraction_lcm(self):
        self.assertEqual(fraction_lcm(Fraction(1, 2), Fraction(1, 3)), Fraction(1))
        self.assertEqual(fraction_lcm(Fraction(2, 3), Fraction(4, 9)), Fraction(4, 3))

        with self.assertRaisesRegex(ValueError, "positive"):
            fraction_lcm(Fraction(0), Fraction(1, 2))

    def test_iter_indices(self):
        self.assertEqual(
            list(iter_indices((2, 3))),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        )

    def test_object_arithmetic_helpers(self):
        token = Symbol("x")

        summed = object_sum([token, 2])
        product_value = object_product([token, 3])
        scaled = scale_object_array(np.array([token, 2], dtype=object), 5)

        self.assertEqual(summed.expr, "((0+x)+2)")
        self.assertEqual(product_value.expr, "((1*x)*3)")
        self.assertEqual(scaled[0].expr, "(x*5)")
        self.assertEqual(scaled[1], 10)


if __name__ == "__main__":
    unittest.main()
