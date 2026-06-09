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


def make_1d_bgd(left, center, right, *, alpha=q(1, 2), beta=q(1, 3)):
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


def describe_1d(label, bgd):
    print(f"\n== {label} ==")
    print(f"center=[{bgd.center_lefts[0]}, {bgd.center_rights[0]}]")
    print(f"left_lengths={bgd.left_lengths}, right_lengths={bgd.right_lengths}")
    print(f"alpha={bgd.alpha}, beta={bgd.beta}")
    print(f"mass={bgd.mass()}")
    for index, name in [(0, "E0 left"), (1, "E1 center"), (2, "E2 right")]:
        mud = bgd.E[(index,)]
        print(
            f"{name}: S={fmt_breakpoints(mud.S)}, shape={mud.shape}, "
            f"is_empty={mud.is_empty}, P={fmt_p(mud.P)}, mass={mud.mass()}"
        )


def describe_2d(label, bgd, detail_indices=((1, 1), (0, 1), (1, 0), (2, 1), (1, 2))):
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


def compare_replace(label, original, new_dim, result):
    print(f"\n-- {label} mass check --")
    print(f"original.mass() * new_dim.mass() = {original.mass() * new_dim.mass()}")
    print(f"result.mass()                    = {result.mass()}")


def make_2d_original():
    E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            x_direction = i - 1
            y_direction = j - 1
            x_end = 2 if x_direction < 0 else 1 if x_direction == 0 else 3
            y_end = 4 if y_direction < 0 else 2 if y_direction == 0 else 5
            E[i, j] = MUD([[0, x_end], [0, y_end]], [[10 * i + j + 1]])

    E[1, 1] = MUD(
        [[0, q(1, 2), q(1, 2), 1], [10, 11, 11, 12]],
        np.array(
            [
                [2, 3, 5],
                [7, 11, 13],
                [17, 19, 23],
            ],
            dtype=object,
        ),
    )
    E[0, 1] = MUD([[0, 0, 1, 2], [0, 2]], [[29], [31], [37]])
    E[2, 1] = MUD([[0, 1, 3, 3], [0, 2]], [[41], [43], [47]])
    return BGD(
        E,
        [q(1, 2), q(1, 3)],
        [q(1, 5), q(1, 7)],
    )


def make_new_x():
    return make_1d_bgd(
        left=MUD([[0, 1]], [5]),
        center=MUD([[100, 101, 101, 102]], [7, 11, 13]),
        right=MUD([[0, 2]], [17]),
        alpha=q(1, 4),
        beta=q(1, 6),
    )


def make_new_y_with_dirac_and_empty_side():
    return make_1d_bgd(
        left=MUD([[0, 0, 1]], [3, 5]),
        center=MUD([[20, 20]], [7]),
        right=empty_1d_mud(0),
        alpha=q(1, 8),
        beta=q(1, 9),
    )


def example_replace_first_dim():
    original = make_2d_original()
    new_x = make_new_x()
    marginalized = original.marginalize(0)
    result = original.replace_dim(0, new_x)

    describe_2d("1A. original 2D BGD", original)
    describe_1d("1B. original.marginalize(0), dependency on old x removed", marginalized)
    describe_1d("1C. new x BGD, center has Dirac at 101", new_x)
    describe_2d("1D. original.replace_dim(0, new_x)", result)
    compare_replace("1. replace first dimension", original, new_x, result)
    print(
        "直观解释: 第0维被替换成 new_x，所以结果中心的 x 坐标变成 [100,102]；"
        "第1维保留 original 边缘化后的 y 坐标 [10,12]。"
    )


def example_replace_second_dim_with_dirac_and_empty_side():
    original = make_2d_original()
    new_y = make_new_y_with_dirac_and_empty_side()
    marginalized = original.marginalize(1)
    result = original.replace_dim(1, new_y)

    describe_2d("2A. original 2D BGD", original)
    describe_1d("2B. original.marginalize(1), dependency on old y removed", marginalized)
    describe_1d("2C. new y BGD: center is Dirac, right side is empty", new_y)
    describe_2d("2D. original.replace_dim(1, new_y)", result)
    compare_replace("2. replace second dimension", original, new_y, result)
    print(
        "直观解释: 第1维被替换成 new_y，所以结果中心的 y 是 Dirac [20,20]；"
        "new_y 的右侧为空，因此所有 y-index=2 的联合块为空。"
    )


def example_replace_only_dimension():
    original = make_1d_bgd(
        left=MUD([[0, 1]], [2]),
        center=MUD([[0, q(1, 2), q(1, 2), 1]], [3, 5, 7]),
        right=MUD([[0, 1]], [11]),
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    new_dim = make_1d_bgd(
        left=MUD([[0, 2]], [13]),
        center=MUD([[10, 11]], [17]),
        right=MUD([[0, 3]], [19]),
        alpha=q(1, 5),
        beta=q(1, 7),
    )
    result = original.replace_dim(0, new_dim)

    describe_1d("3A. original 1D BGD", original)
    describe_1d("3B. new 1D BGD", new_dim)
    describe_1d("3C. original.replace_dim(0, new_dim)", result)
    compare_replace("3. replace only dimension", original, new_dim, result)
    print(
        "直观解释: 原分布被完全边缘化成标量 mass(original)，"
        "结果就是 new_dim 的所有质量整体乘上这个标量。"
    )


def example_replace_with_empty_original_center():
    E = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            x_direction = i - 1
            y_direction = j - 1
            x_end = 1 if x_direction < 0 else 0 if x_direction == 0 else 2
            y_end = 1 if y_direction < 0 else 2 if y_direction == 0 else 3
            E[i, j] = MUD([[0, x_end], [0, y_end]], [[i + j + 1]])
    E[1, 1] = MUD([[0], [10, 12]], np.empty((0, 1), dtype=object))
    original = BGD(E, [q(1, 2), q(1, 3)], [q(1, 4), q(1, 5)])
    new_x = make_new_x()
    result = original.replace_dim(0, new_x)

    describe_2d("4A. original: center is empty along x", original)
    describe_1d("4B. new x BGD", new_x)
    describe_2d("4C. replace_dim(0, new_x)", result)
    compare_replace("4. replace dimension when original center is empty", original, new_x, result)
    print(
        "直观解释: 原中心为空不会阻止替换；边缘尾部边缘化后仍贡献质量，"
        "再与 new_x 独立组合。"
    )


def main():
    example_replace_first_dim()
    example_replace_second_dim_with_dirac_and_empty_side()
    example_replace_only_dimension()
    example_replace_with_empty_original_center()


if __name__ == "__main__":
    main()
