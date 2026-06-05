from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import BGD, MUD


def q(n, d=1):
    return Fraction(n, d)


def make_1d_bgd(left, center, right, *, alpha=q(1, 2), beta=q(1, 3)):
    E = np.empty((3,), dtype=object)
    E[0] = left
    E[1] = center
    E[2] = right
    return BGD(E, [alpha], [beta])


def fmt_seq(seq):
    return "(" + ", ".join(str(x) for x in seq) + ")"


def fmt_list(values):
    return "[" + ", ".join(str(x) for x in values) + "]"


def describe_1d(label, bgd):
    print(f"\n== {label} ==")
    print(f"center=[{bgd.center_lefts[0]}, {bgd.center_rights[0]}]")
    print(f"periods: left={bgd.left_lengths[0]}, right={bgd.right_lengths[0]}")
    print(f"decay: alpha={bgd.alpha[0]}, beta={bgd.beta[0]}")
    print(f"total mass={bgd.mass()}")
    for index, name in [(0, "E0 left"), (1, "E1 center"), (2, "E2 right")]:
        mud = bgd.E[(index,)]
        print(
            f"{name}: S={fmt_seq(mud.S[0])}, "
            f"P={fmt_list(mud.P.tolist())}, mass={mud.mass()}"
        )


def describe_2d(label, bgd):
    print(f"\n== {label} ==")
    print(
        "center="
        f"x[{bgd.center_lefts[0]}, {bgd.center_rights[0]}], "
        f"y[{bgd.center_lefts[1]}, {bgd.center_rights[1]}]"
    )
    print(f"left_lengths={bgd.left_lengths}, right_lengths={bgd.right_lengths}")
    print(f"alpha={bgd.alpha}, beta={bgd.beta}")
    print(f"total mass={bgd.mass()}")
    print("block masses by E index:")
    for row in range(3):
        print(f"  y-index {row}: {[str(bgd.E[(col, row)].mass()) for col in range(3)]}")
    for index in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        mud = bgd.E[index]
        print(
            f"  E{index}: Sx={fmt_seq(mud.S[0])}, Sy={fmt_seq(mud.S[1])}, "
            f"P={mud.P.tolist()}"
        )


def compare(label, before, after):
    print(f"\n-- {label} mass check --")
    print(f"before.mass() = {before.mass()}")
    print(f"after.mass()  = {after.mass()}")


def example_1d_both_near_center_boundaries():
    bgd = make_1d_bgd(
        left=MUD([[0, 1, 1]], [4, 10]),
        center=MUD([[0, 1]], [3]),
        right=MUD([[0, 0, 1]], [12, 5]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    standardized = bgd.standardize()

    describe_1d("1A. before: left right-boundary Dirac, right left-boundary Dirac", bgd)
    describe_1d("1B. after standardize()", standardized)
    compare("1. both near-center boundary Diracs", bgd, standardized)
    print(
        "直观解释: E0 的右端 Dirac 移到中心左端 0，并以 alpha 倍加到 E0 左端；"
        "E2 的左端 Dirac 移到中心右端 1，并以 beta 倍加到 E2 右端。"
    )


def example_1d_edge_has_both_boundary_diracs():
    bgd = make_1d_bgd(
        left=MUD([[0, 0, 1, 1]], [7, 4, 10]),
        center=MUD([[0, 1]], [2]),
        right=MUD([[0, 0, 1, 1]], [12, 5, 9]),
        alpha=q(1, 2),
        beta=q(1, 4),
    )
    standardized = bgd.standardize()

    describe_1d("2A. before: edge blocks have both endpoint Diracs", bgd)
    describe_1d("2B. after standardize()", standardized)
    compare("2. endpoint Diracs inside edge blocks", bgd, standardized)
    print(
        "直观解释: 负方向只处理靠近中心的右端 Dirac；正方向只处理靠近中心的左端 Dirac。"
        "远离中心一侧的 Dirac 仍留在边缘块里，并会收到衰减后加回的质量。"
    )


def example_nonzero_center_global_endpoints():
    bgd = make_1d_bgd(
        left=MUD([[0, 2, 2]], [6, 8]),
        center=MUD([[10, 12]], [5]),
        right=MUD([[0, 0, 3]], [9, 7]),
        alpha=q(1, 3),
        beta=q(1, 5),
    )
    standardized = bgd.standardize()

    describe_1d("3A. before: nonzero center [10,12]", bgd)
    describe_1d("3B. after standardize()", standardized)
    compare("3. nonzero center endpoints", bgd, standardized)
    print(
        "直观解释: 移入中心的点质量会落在全局端点 10 和 12，"
        "不是局部坐标 0 和中心长度 2。"
    )


def zero_mud_2d(direction):
    S = []
    for d in direction:
        S.append((0, 1))
    return MUD(S, np.array([[0]], dtype=object))


def make_2d_corner_bgd():
    E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            direction = (i - 1, j - 1)
            if (i, j) == (1, 1):
                E[i, j] = MUD([[0, 1], [0, 1]], np.array([[0]], dtype=object))
            else:
                E[i, j] = zero_mud_2d(direction)

    P = np.zeros((2, 2), dtype=object)
    P[1, 1] = 24
    E[0, 0] = MUD([[0, 1, 1], [0, 1, 1]], P)
    return BGD(E, [q(1, 2), q(1, 3)], [q(1, 5), q(1, 7)])


def example_2d_corner_dirac():
    bgd = make_2d_corner_bgd()
    standardized = bgd.standardize()

    describe_2d("4A. before: E(0,0) has Dirac at near-center corner", bgd)
    describe_2d("4B. after standardize()", standardized)
    compare("4. 2D corner boundary Dirac", bgd, standardized)
    print(
        "直观解释: E(0,0) 的右上角 Dirac 同时靠近两个中心边界。"
        "standardize 会经过两个维度把质量分配到 E(1,0)、E(0,1)、中心块和自身远端边界，"
        "并乘上对应 alpha 衰减。"
    )


def example_already_standardized():
    bgd = make_1d_bgd(
        left=MUD([[0, 0, 1]], [5, 4]),
        center=MUD([[0, 0, 1, 1]], [7, 2, 11]),
        right=MUD([[0, 1, 1]], [6, 3]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    standardized = bgd.standardize()

    describe_1d("5A. before: already standardized boundary convention", bgd)
    describe_1d("5B. after standardize()", standardized)
    compare("5. already standardized", bgd, standardized)
    print(
        "直观解释: 左边缘靠近中心的右端没有 Dirac，右边缘靠近中心的左端没有 Dirac，"
        "所以标准化不会改变表示。"
    )


def main():
    example_1d_both_near_center_boundaries()
    example_1d_edge_has_both_boundary_diracs()
    example_nonzero_center_global_endpoints()
    example_2d_corner_dirac()
    example_already_standardized()


if __name__ == "__main__":
    main()
