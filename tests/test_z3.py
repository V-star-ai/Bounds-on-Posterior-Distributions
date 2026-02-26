import numpy as np
import z3

from EED import EED
from Adapter.Z3Adapter import Z3Adapter


def z3max(a, b):
    return z3.If(a >= b, a, b)


def test_eed_z3_basic():
    # 1D: S must be known integer breakpoints
    S1 = [0, 2, 5]
    p0, p1, p2, p3 = z3.Reals("p0 p1 p2 p3")
    P1 = np.array([p0, p1, p2, p3], dtype=object)
    a1, b1 = z3.Reals("a1 b1")
    eed1 = EED([S1], P1, [a1], [b1])

    # Another 1D: constant P and alpha/beta
    S2 = [1, 3, 5]
    P2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=object)
    eed2 = EED([S2], P2, [0.3], [0.4])

    s = z3.Solver()

    # alpha/beta bounds
    s.add(a1 >= 0, a1 < 1, b1 >= 0, b1 < 1)

    # merge_breakpoints / align_to path used by add & leq
    _ = EED.merge_breakpoints(np.array(S1), np.array(S2), interval=-1)

    # add with z3-compatible max
    _ = EED.add(eed1, eed2, interval=-1, max_function=z3max)

    # build constraints: eed1 <= eed2
    c1 = EED.leq(
        eed1,
        eed2,
        return_list=False,
        and_function=z3.And,
        true=z3.BoolVal(True),
        false=z3.BoolVal(False),
    )
    s.add(c1)

    # restrict_ge / restrict_lt
    _ = eed1.restrict_ge(axis=0, a=1)
    _ = eed1.restrict_lt(axis=0, b=4)

    # simple feasible bounds on P
    s.add(p0 <= 10, p1 <= 10, p2 <= 10, p3 <= 10)

    assert s.check() == z3.sat


def test_eed_z3_const_leq():
    # constant EED <= z3 EED
    S = [0, 3, 7]
    P_const = np.array([1.0, 1.0, 2.0, 2.0], dtype=object)
    eed_const = EED([S], P_const, [0.2], [0.3])

    q0, q1, q2, q3 = z3.Reals("q0 q1 q2 q3")
    P_z3 = np.array([q0, q1, q2, q3], dtype=object)
    a2, b2 = z3.Reals("a2 b2")
    eed_z3 = EED([S], P_z3, [a2], [b2])

    s = z3.Solver()
    s.add(a2 >= 0, a2 < 1, b2 >= 0, b2 < 1)

    c = EED.leq(
        eed_const,
        eed_z3,
        return_list=False,
        and_function=z3.And,
        true=z3.BoolVal(True),
        false=z3.BoolVal(False),
    )
    s.add(c)

    # make it feasible: lower-bound z3 P
    s.add(q0 >= 1, q1 >= 1, q2 >= 2, q3 >= 2)

    assert s.check() == z3.sat


def test_z3_adapter_build_and_solve():
    adapter = Z3Adapter()

    S = [0, 2, 4]
    P_const = np.array([1.0, 2.0, 3.0, 4.0], dtype=object)
    eed_const = EED([S], P_const, [0.2], [0.3])

    eed_z3, solver = adapter.build_leq(eed_const, [S], name_prefix="t")

    # add simple feasibility constraints
    for v in eed_z3.P.ravel():
        solver.add(v >= 0)

    eed_sol = adapter.solve(eed_z3, solver)

    assert isinstance(eed_sol, EED)
    assert eed_sol.P.shape == eed_const.P.shape
    assert len(eed_sol.alpha) == eed_const.n
    assert len(eed_sol.beta) == eed_const.n


if __name__ == '__main__':
    test_eed_z3_basic()
    test_eed_z3_const_leq()
    test_z3_adapter_build_and_solve()
    print("All tests passed")
