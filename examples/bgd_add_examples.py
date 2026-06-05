from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import BGD, MUD


def q(n, d=1):
    return Fraction(n, d)


def make_bgd(left, center, right, *, alpha=Fraction(1, 2), beta=Fraction(1, 3)):
    E = np.empty((3,), dtype=object)
    E[0] = left
    E[1] = center
    E[2] = right
    return BGD(E, [alpha], [beta])


def zero_edge(length):
    return MUD([[0, length]], [0])


def fmt_seq(seq):
    return "(" + ", ".join(str(x) for x in seq) + ")"


def fmt_list(values):
    return "[" + ", ".join(str(x) for x in values) + "]"


def describe(label, bgd):
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


def describe_2d(label, bgd, detail_indices=((1, 1), (0, 1), (2, 1), (1, 2))):
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
        values = [str(bgd.E[(col, row)].mass()) for col in range(3)]
        print(f"  y-index {row}: {values}")
    print("selected block details:")
    for index in detail_indices:
        mud = bgd.E[index]
        print(
            f"  E{index}: Sx={fmt_seq(mud.S[0])}, Sy={fmt_seq(mud.S[1])}, "
            f"P={mud.P.tolist()}"
        )


def compare_sum(label, left, right, result):
    print(f"\n-- {label} mass check --")
    print(f"left.mass() + right.mass() = {left.mass() + right.mass()}")
    print(f"result.mass()              = {result.mass()}")


def example_same_frame_center_dirac():
    left = make_bgd(
        left=MUD([[0, 1]], [2]),
        center=MUD([[0, q(1, 2), q(1, 2), 1]], [4, 7, 6]),
        right=MUD([[0, 1]], [3]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    right = make_bgd(
        left=MUD([[0, 1]], [5]),
        center=MUD([[0, q(1, 2), q(1, 2), 1]], [1, 3, 2]),
        right=MUD([[0, 1]], [4]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    result = left + right

    describe("1A. same frame left operand, center has Dirac at 1/2", left)
    describe("1B. same frame right operand, center has Dirac at 1/2", right)
    describe("1C. result = 1A + 1B", result)
    compare_sum("1. same frame, exact add", left, right, result)
    print("直观解释: 两个中心块都有 [1/2,1/2] Dirac，结果中该点质量直接相加为 7+3=10。")


def example_boundary_dirac_standardized_by_add():
    left = make_bgd(
        left=zero_edge(1),
        center=MUD([[0, 1]], [1]),
        right=MUD([[0, 0, 1]], [9, 6]),
        alpha=q(1, 2),
        beta=q(1, 2),
    )
    right = make_bgd(
        left=zero_edge(1),
        center=MUD([[0, 1, 1]], [2, 5]),
        right=zero_edge(1),
        alpha=q(1, 2),
        beta=q(1, 2),
    )
    result = left + right

    describe("2A. right edge has a left-boundary Dirac at local 0", left)
    describe("2B. center has a right-boundary Dirac at global 1", right)
    describe("2C. result = 2A + 2B, then standardize()", result)
    compare_sum("2. boundary Dirac, exact add after standardize", left, right, result)
    print(
        "直观解释: E2 的 local 0 Dirac 在全局上落到中心右端点 1，"
        "add 后的 standardize 把它并入中心 [1,1]；同时把 beta 倍的尾部边界质量放到 E2 的右端。"
    )


def example_frame_alignment_with_dirac():
    left = make_bgd(
        left=MUD([[0, 1]], [4]),
        center=MUD([[0, q(1, 2), q(1, 2), 1]], [3, 5, 7]),
        right=MUD([[0, 1]], [6]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    right = make_bgd(
        left=MUD([[0, 1, 1, 2]], [8, 11, 10]),
        center=MUD([[q(1, 2), 1, 1, q(3, 2)]], [2, 13, 9]),
        right=MUD([[0, q(3, 2), q(3, 2), 3]], [12, 17, 15]),
        alpha=q(1, 4),
        beta=q(1, 27),
    )
    result = left + right

    describe("3A. center [0,1], periods 1/1, center Dirac at 1/2", left)
    describe("3B. center [1/2,3/2], periods 2/3, several Dirac intervals", right)
    describe("3C. result = 3A + 3B, after center/period alignment", result)
    compare_sum("3. frame alignment with matching expanded decays, exact add", left, right, result)
    print(
        "直观解释: 共同中心变成 [0,3/2]，共同周期是 left=2/right=3。"
        "第一个 BGD 会先把周期扩大，第二个 BGD 会把左侧靠近中心的半个周期并入中心。"
    )


def example_decay_max_upper_bound_with_dirac():
    left = make_bgd(
        left=MUD([[0, q(1, 2), q(1, 2), 1]], [4, 5, 6]),
        center=MUD([[0, 1]], [1]),
        right=zero_edge(1),
        alpha=q(1, 2),
        beta=0,
    )
    right = make_bgd(
        left=MUD([[0, q(1, 2), q(1, 2), 1]], [7, 11, 13]),
        center=MUD([[0, 1]], [2]),
        right=zero_edge(1),
        alpha=q(1, 3),
        beta=0,
    )
    result = left + right

    describe("4A. alpha=1/2, left edge has internal Dirac", left)
    describe("4B. alpha=1/3, left edge has internal Dirac", right)
    describe("4C. result uses alpha=max(1/2,1/3)=1/2", result)
    compare_sum("4. different decays, result is an upper bound", left, right, result)
    print(
        "直观解释: 两个框架已经一致，但 alpha 不同。结果取更慢衰减 1/2，"
        "所以右操作数的左尾被放大，result.mass() 大于真实和。"
    )


def simple_2d_mud(direction, mass, *, center_x=(0, 1), center_y=(0, 1)):
    S = []
    for dim, d in enumerate(direction):
        if d == -1:
            right = 1 if dim == 0 else 2
            S.append((0, right))
        elif d == 0:
            center = center_x if dim == 0 else center_y
            S.append((0, center[1] - center[0]))
        else:
            right = 2 if dim == 0 else 1
            S.append((0, right))
    return MUD(S, np.array([[mass]], dtype=object))


def make_simple_2d_bgd(*, center_x=(0, 1), center_y=(0, 1), alpha=(q(1, 2), q(1, 3)), beta=(q(1, 5), q(1, 7))):
    E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            direction = (i - 1, j - 1)
            mass = 10 * i + j + 1
            if (i, j) == (1, 1):
                E[i, j] = MUD(
                    [
                        (center_x[0], q(1, 2), q(1, 2), center_x[1])
                        if center_x == (0, 1)
                        else (center_x[0], center_x[1]),
                        (center_y[0], center_y[1]),
                    ],
                    np.array([[4], [9], [6]], dtype=object)
                    if center_x == (0, 1)
                    else np.array([[19]], dtype=object),
                )
            else:
                E[i, j] = simple_2d_mud(
                    direction, mass, center_x=center_x, center_y=center_y
                )
    return BGD(E, alpha, beta)


def example_2d_add_with_dirac_and_alignment():
    left = make_simple_2d_bgd(
        center_x=(0, 1),
        center_y=(0, 1),
        alpha=(q(1, 2), q(1, 3)),
        beta=(q(1, 5), q(1, 7)),
    )

    right_E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            direction = (i - 1, j - 1)
            right_E[i, j] = simple_2d_mud(
                direction,
                3 * (10 * i + j + 1),
                center_x=(q(1, 2), q(3, 2)),
                center_y=(0, 1),
            )
    right_E[1, 1] = MUD(
        [[q(1, 2), 1, 1, q(3, 2)], [0, q(1, 2), q(1, 2), 1]],
        np.array(
            [
                [2, 5, 3],
                [7, 11, 13],
                [17, 19, 23],
            ],
            dtype=object,
        ),
    )
    right_E[2, 1] = MUD(
        [[0, 0, 2], [0, 1]],
        np.array([[29], [31]], dtype=object),
    )
    right = BGD(
        right_E,
        [q(1, 4), q(1, 3)],
        [q(1, 25), q(1, 7)],
    )

    result = left + right

    describe_2d("5A. 2D left: center x[0,1], center has vertical Dirac line x=1/2", left)
    describe_2d("5B. 2D right: center x[1/2,3/2], center has grid Dirac, right edge has boundary Dirac", right)
    describe_2d("5C. 2D result = 5A + 5B", result)
    compare_sum("5. 2D add with center alignment, period lcm, and Dirac", left, right, result)
    print(
        "直观解释: 共同中心变成 x[0,3/2], y[0,1]；"
        "第一维左右周期分别对齐到 left=1/right=2。"
        "中心块里能看到 x=1/2 和 x=1 的 Dirac 断点，"
        "右侧边缘 local x=0 的 Dirac 会在 standardize 后并入靠近中心的块。"
    )


def main():
    example_same_frame_center_dirac()
    example_boundary_dirac_standardized_by_add()
    example_frame_alignment_with_dirac()
    example_decay_max_upper_bound_with_dirac()
    example_2d_add_with_dirac_and_alignment()


if __name__ == "__main__":
    main()
