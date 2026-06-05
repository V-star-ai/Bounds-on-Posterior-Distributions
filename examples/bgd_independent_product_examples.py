from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import BGD, MUD


def q(n, d=1):
    return Fraction(n, d)


def empty_1d_mud(point=0):
    return MUD([[point]], np.empty((0,), dtype=object))


def make_bgd(left, center, right, *, alpha=q(1, 2), beta=q(1, 3)):
    E = np.empty((3,), dtype=object)
    E[0] = left
    E[1] = center
    E[2] = right
    return BGD(E, [alpha], [beta])


def fmt_seq(seq):
    return "(" + ", ".join(str(x) for x in seq) + ")"


def fmt_breakpoints(S):
    return "[" + ", ".join(fmt_seq(seq) for seq in S) + "]"


def fmt_p(P):
    return str(P.tolist())


def describe_mud(label, mud):
    print(f"\n== {label} ==")
    print(f"S={fmt_breakpoints(mud.S)}")
    print(f"shape={mud.shape}, is_empty={mud.is_empty}")
    print(f"P={fmt_p(mud.P)}")
    print(f"mass={mud.mass()}")


def describe_1d_bgd(label, bgd):
    print(f"\n== {label} ==")
    print(f"center=[{bgd.center_lefts[0]}, {bgd.center_rights[0]}]")
    print(f"periods: left={bgd.left_lengths[0]}, right={bgd.right_lengths[0]}")
    print(f"decay: alpha={bgd.alpha[0]}, beta={bgd.beta[0]}")
    print(f"mass={bgd.mass()}")
    for index, name in [(0, "E0 left"), (1, "E1 center"), (2, "E2 right")]:
        mud = bgd.E[(index,)]
        print(
            f"{name}: S={fmt_breakpoints(mud.S)}, shape={mud.shape}, "
            f"is_empty={mud.is_empty}, P={fmt_p(mud.P)}, mass={mud.mass()}"
        )


def describe_2d_bgd(label, bgd, detail_indices=((1, 1), (0, 1), (1, 0), (2, 1), (1, 2))):
    print(f"\n== {label} ==")
    print(
        f"center=x[{bgd.center_lefts[0]}, {bgd.center_rights[0]}], "
        f"y[{bgd.center_lefts[1]}, {bgd.center_rights[1]}]"
    )
    print(f"left_lengths={bgd.left_lengths}, right_lengths={bgd.right_lengths}")
    print(f"alpha={bgd.alpha}, beta={bgd.beta}")
    print(f"mass={bgd.mass()}")
    print("block mass matrix by E index:")
    for y_index in range(3):
        row = [str(bgd.E[(x_index, y_index)].mass()) for x_index in range(3)]
        print(f"  y-index {y_index}: {row}")
    print("selected block details:")
    for index in detail_indices:
        mud = bgd.E[index]
        print(
            f"  E{index}: S={fmt_breakpoints(mud.S)}, shape={mud.shape}, "
            f"is_empty={mud.is_empty}, P={fmt_p(mud.P)}, mass={mud.mass()}"
        )


def compare_product(label, left, right, result):
    print(f"\n-- {label} mass check --")
    print(f"left.mass() * right.mass() = {left.mass() * right.mass()}")
    print(f"result.mass()              = {result.mass()}")


def example_mud_dirac_product():
    x = MUD([[1, 1]], [5])
    y = MUD([[0, 2]], [7])
    result = x.independent_product(y)

    describe_mud("1A. MUD x: Dirac at x=1", x)
    describe_mud("1B. MUD y: continuous interval [0,2]", y)
    describe_mud("1C. x.independent_product(y)", result)
    print("直观解释: 结果是二维块 [1,1] x [0,2]，质量为 5*7=35。")


def example_bgd_dirac_and_nonzero_center():
    x = make_bgd(
        left=MUD([[0, 1]], [2]),
        center=MUD([[0, q(1, 2), q(1, 2), 1]], [3, 11, 5]),
        right=MUD([[0, 2]], [7]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    y = make_bgd(
        left=MUD([[0, 3]], [13]),
        center=MUD([[10, 12]], [17]),
        right=MUD([[0, 4]], [19]),
        alpha=q(1, 5),
        beta=q(1, 7),
    )
    result = x.independent_product(y)

    describe_1d_bgd("2A. x BGD: center has Dirac at x=1/2", x)
    describe_1d_bgd("2B. y BGD: nonzero center [10,12]", y)
    describe_2d_bgd("2C. x independent_product y", result)
    compare_product("2. BGD Dirac and nonzero center", x, y, result)
    print(
        "直观解释: 联合中心 E(1,1) 保留 y 的全局坐标 [10,12]；"
        "但非中心块 E(0,1) 中的 y-center 会转成局部坐标 [0,2]。"
    )


def example_empty_center():
    x = make_bgd(
        left=MUD([[0, 1]], [4]),
        center=empty_1d_mud(0),
        right=MUD([[0, 2]], [6]),
        alpha=q(1, 2),
        beta=q(1, 4),
    )
    y = make_bgd(
        left=MUD([[0, 1]], [2]),
        center=MUD([[0, 1]], [5]),
        right=MUD([[0, 1]], [3]),
        alpha=q(1, 3),
        beta=q(1, 5),
    )
    result = x.independent_product(y)

    describe_1d_bgd("3A. x BGD: center C is empty", x)
    describe_1d_bgd("3B. y BGD: ordinary", y)
    describe_2d_bgd("3C. product with empty x-center", result)
    compare_product("3. empty center C", x, y, result)
    print(
        "直观解释: 联合中心 E(1,1) 的第一个维度 shape 为 0，因此中心块为空；"
        "但 x 的左右尾部仍然和 y 的各方向块正常组合。"
    )


def example_empty_side():
    x = make_bgd(
        left=empty_1d_mud(0),
        center=MUD([[0, 1]], [7]),
        right=MUD([[0, 1]], [9]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    y = make_bgd(
        left=MUD([[0, 2]], [5]),
        center=MUD([[10, 11]], [6]),
        right=MUD([[0, 2]], [8]),
        alpha=q(1, 4),
        beta=q(1, 5),
    )
    result = x.independent_product(y)

    describe_1d_bgd("4A. x BGD: left side E0 is empty", x)
    describe_1d_bgd("4B. y BGD: nonzero center [10,11]", y)
    describe_2d_bgd("4C. product with empty x-left side", result)
    compare_product("4. empty side E0", x, y, result)
    print(
        "直观解释: 所有 x 方向为左侧的联合块 E(0,*) 都为空；"
        "其他方向仍正常，且 y-center 在非中心块中使用局部坐标 [0,1]。"
    )


def example_dirac_side_and_empty_other_side():
    x = make_bgd(
        left=MUD([[0, 0, 1]], [3, 4]),
        center=MUD([[0, 1]], [5]),
        right=MUD([[0, 1, 1]], [6, 7]),
        alpha=q(1, 2),
        beta=q(1, 3),
    ).standardize()
    y = make_bgd(
        left=MUD([[0, 1]], [11]),
        center=MUD([[0, 0]], [13]),
        right=empty_1d_mud(0),
        alpha=q(1, 5),
        beta=q(1, 7),
    )
    result = x.independent_product(y)

    describe_1d_bgd("5A. x BGD: side blocks contain Dirac after standardize", x)
    describe_1d_bgd("5B. y BGD: center is Dirac, right side is empty", y)
    describe_2d_bgd("5C. product with side Dirac and empty right side", result)
    compare_product("5. Dirac side and empty side", x, y, result)
    print(
        "直观解释: y 的中心是 Dirac [0,0]，所以联合中心和所有 y-center方向块"
        "在第二维都是 Dirac；y 的右侧为空，因此所有 E(*,2) 块为空。"
    )


def main():
    example_mud_dirac_product()
    example_bgd_dirac_and_nonzero_center()
    example_empty_center()
    example_empty_side()
    example_dirac_side_and_empty_other_side()


if __name__ == "__main__":
    main()
