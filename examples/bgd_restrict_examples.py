from fractions import Fraction
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distributions import BGD, MUD


def f(value):
    return Fraction(value)


def make_bgd(left, center, right, alpha=Fraction(1, 2), beta=Fraction(1, 2)):
    E = np.empty((3,), dtype=object)
    E[0] = left
    E[1] = center
    E[2] = right
    return BGD(E, [alpha], [beta])


def describe_mud(mud):
    return {
        "S": tuple(tuple(str(x) for x in seq) for seq in mud.S),
        "P": [str(x) for x in mud.P.tolist()],
        "mass": str(mud.mass()),
    }


def describe_bgd(label, bgd):
    print(f"\n== {label} ==")
    print(f"center_lefts={bgd.center_lefts}, center_rights={bgd.center_rights}")
    print(f"left_lengths={bgd.left_lengths}, right_lengths={bgd.right_lengths}")
    print(f"mass={bgd.mass()}")
    for index in [(0,), (1,), (2,)]:
        block = bgd.block_at((index[0] - 1,))
        print(
            f"E{index}: direction={block.direction}, "
            f"translation={block.translation}, decay={block.decay_factor}, "
            f"{describe_mud(bgd.E[index])}"
        )


def main():
    base = make_bgd(
        left=MUD([[0, 1]], [10]),
        center=MUD([[0, 1]], [3]),
        right=MUD([[0, 1]], [10]),
        alpha=Fraction(1, 2),
        beta=Fraction(1, 2),
    )
    describe_bgd("base", base)
    describe_bgd("restrict x >= 1/2 (inside center)", base.restrict(0, ">=", f(1) / 2))
    describe_bgd("restrict x > 3/2 (right phase)", base.restrict(0, ">", f(3) / 2))
    describe_bgd("restrict x < 3/2 (right prefix into center)", base.restrict(0, "<", f(3) / 2))
    describe_bgd("restrict x < -1/2 (left phase)", base.restrict(0, "<", -f(1) / 2))
    describe_bgd("restrict x > -1/2 (left prefix into center)", base.restrict(0, ">", -f(1) / 2))

    shifted_center = make_bgd(
        left=MUD([[0, 2]], [4]),
        center=MUD([[10, 12]], [6]),
        right=MUD([[0, 3]], [9]),
        alpha=Fraction(1, 3),
        beta=Fraction(1, 4),
    )
    describe_bgd("nonzero center base", shifted_center)
    describe_bgd(
        "nonzero center restrict x >= 11",
        shifted_center.restrict(0, ">=", 11),
    )


if __name__ == "__main__":
    main()
