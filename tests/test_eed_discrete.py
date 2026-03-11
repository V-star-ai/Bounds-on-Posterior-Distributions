from fractions import Fraction
import numpy as np
from jedi.plugins import pytest

from distributions import EED


def _make_mixed_eed():
    # dim0 continuous, dim1 discrete
    S0 = [Fraction(0), Fraction(1)]
    S1 = [0, 2, 3]
    P = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=float,
    )
    return EED([S0, S1], P, [0.2, 0.0], [0.3, 0.0], [False, True])


def test_init_shape_and_discrete_validation():
    eed = _make_mixed_eed()
    assert eed.P.shape == (3, 3)
    assert eed.discrete_mask == [False, True]

    # discrete S must be integers
    try:
        EED([[Fraction(0), Fraction(1)]], [0, 1, 0], [0.0], [0.0], [True])
        assert False, "expected ValueError for non-integer discrete S"
    except ValueError:
        pass


def test_align_to_mixed_dims():
    eed = _make_mixed_eed()
    # add a discrete point 4 and a continuous breakpoint 1/2
    S_new = [[Fraction(0), Fraction(1, 2), Fraction(1)], [0, 2, 3, 4]]
    g = eed.align_to(S_new)
    assert g.P.shape == (4, 4)

    # new discrete point should be zero column
    assert np.allclose(g.P[:, 3], 0.0)


def test_add_and_leq_with_discrete_dims():
    eed1 = _make_mixed_eed()
    eed2 = _make_mixed_eed().times_constant(2.0)

    eed_sum = EED.add(eed1, eed2)
    assert np.allclose(eed_sum.P, eed1.P + eed2.P)

    # eed1 <= eed2 should hold (since eed2 = 2 * eed1)
    assert eed1 <= eed2


def test_restrict_on_discrete_axis():
    eed = _make_mixed_eed()
    r = eed.restrict_ge(axis=1, a=2)
    # points with value 0 should be zeroed
    assert np.allclose(r.P[:, 0], 0.0)
    # points with value 2 and 3 remain
    assert np.allclose(r.P[:, 1:], eed.P[:, 1:])

    r2 = eed.restrict_lt(axis=1, b=3)
    # point with value 3 should be zeroed
    assert np.allclose(r2.P[:, 2], 0.0)


def test_restrict_on_continuous_axis():
    eed = _make_mixed_eed()
    r = eed.restrict_ge(axis=0, a=Fraction(1))
    # after restricting x >= 1, left ring should be 0
    assert np.allclose(r.P[0, :], 0.0)
    # alpha on axis 0 should be 0 (left side invalid)
    assert r.alpha[0] == 0

    r2 = eed.restrict_lt(axis=0, b=Fraction(1))
    # after restricting x < 1, right ring should be 0
    assert np.allclose(r2.P[-1, :], 0.0)
    assert r2.beta[0] == 0


def test_restrict_interval_discrete():
    eed = _make_mixed_eed()
    r = eed.restrict_interval(axis=1, intervals=[(1, 3)])
    # keep only discrete points in [1,3): only value 2 remains
    assert np.allclose(r.P[:, 0], 0.0)
    assert np.allclose(r.P[:, 2], 0.0)
    assert np.allclose(r.P[:, 1], eed.P[:, 1])

if __name__ == "__main__":
    test_restrict_interval_discrete()
    test_init_shape_and_discrete_validation()
    test_add_and_leq_with_discrete_dims()
    test_align_to_mixed_dims()
    test_restrict_on_discrete_axis()
    test_restrict_on_continuous_axis()