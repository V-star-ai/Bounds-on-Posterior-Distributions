from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import BGD, MUD, fraction_lcm


def q(n, d=1):
    return Fraction(n, d)


def make_1d_bgd(
    left_mass,
    center_mass,
    right_mass,
    *,
    center=(0, 1),
    left_length=1,
    right_length=1,
    alpha=Fraction(1, 2),
    beta=Fraction(1, 3),
):
    E = np.empty((3,), dtype=object)
    E[0] = MUD([[0, left_length]], [left_mass])
    E[1] = MUD([[center[0], center[1]]], [center_mass])
    E[2] = MUD([[0, right_length]], [right_mass])
    return BGD(E, [alpha], [beta])


def format_seq(seq):
    return "(" + ", ".join(str(x) for x in seq) + ")"


def format_list(values):
    return "[" + ", ".join(str(x) for x in values) + "]"


def describe(label, bgd):
    print(f"\n== {label} ==")
    print(f"center = [{bgd.center_lefts[0]}, {bgd.center_rights[0]}]")
    print(f"periods: left={bgd.left_lengths[0]}, right={bgd.right_lengths[0]}")
    print(f"decay: alpha={bgd.alpha[0]}, beta={bgd.beta[0]}")
    print(f"total mass = {bgd.mass()}")
    for index, name in [(0, "E0 left"), (1, "E1 center"), (2, "E2 right")]:
        mud = bgd.E[(index,)]
        print(
            f"{name}: S={format_seq(mud.S[0])}, "
            f"P={format_list(mud.P.tolist())}, mass={mud.mass()}"
        )


def example_align_center_domain():
    base = make_1d_bgd(
        left_mass=8,
        center_mass=3,
        right_mass=10,
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    aligned = base.align_center_domain([q(-1, 2)], [q(3, 2)])

    describe("1. base: center [0, 1], periods both 1", base)
    describe("1. align_center_domain([-1/2], [3/2])", aligned)
    print(
        "直观解释: 左尾靠近中心的半个周期质量 8/2=4 进入中心左侧；"
        "右尾靠近中心的半个周期质量 10/2=5 进入中心右侧。"
    )


def example_align_edge_periods():
    base = make_1d_bgd(
        left_mass=8,
        center_mass=3,
        right_mass=10,
        alpha=q(1, 2),
        beta=q(1, 3),
    )
    aligned = base.align_edge_periods([2], [3])

    describe("2. base: edge periods left=1, right=1", base)
    describe("2. align_edge_periods([2], [3])", aligned)
    print(
        "直观解释: 新左周期长度为 2，由两个旧左周期拼接，"
        "质量从远到近是 8*alpha=4 和 8；"
        "新右周期长度为 3，由三个旧右周期拼接，"
        "质量从近到远是 10, 10*beta=10/3, 10*beta^2=10/9。"
    )


def example_shifted_center():
    base = make_1d_bgd(
        left_mass=12,
        center_mass=6,
        right_mass=9,
        center=(10, 12),
        left_length=2,
        right_length=3,
        alpha=q(1, 3),
        beta=q(1, 4),
    )
    aligned = base.align_center_domain([9], [14])

    describe("3. base: nonzero center [10, 12]", base)
    describe("3. align_center_domain([9], [14])", aligned)
    print(
        "直观解释: 中心坐标使用全局坐标。新中心 [9, 14] 吃掉左侧 [9,10] "
        "和右侧 [12,14]，剩余左右无限尾部按新边界重构。"
    )


def example_fraction_lcm():
    left_a = q(1, 2)
    left_b = q(1, 3)
    right_a = q(2, 3)
    right_b = q(4, 9)
    print("\n== 4. fraction_lcm for edge period alignment ==")
    print(f"lcm_Q({left_a}, {left_b}) = {fraction_lcm(left_a, left_b)}")
    print(f"lcm_Q({right_a}, {right_b}) = {fraction_lcm(right_a, right_b)}")
    print("直观解释: 目标周期必须是两个原周期的正整数倍。")


def main():
    example_align_center_domain()
    example_align_edge_periods()
    example_shifted_center()
    example_fraction_lcm()


if __name__ == "__main__":
    main()
