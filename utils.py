from numbers import Real
from fractions import Fraction


def to_simplest_fraction(x: str | Real) -> Fraction:
    """
    Convert x to Fraction under decimal semantics.
    """
    if isinstance(x, Fraction):
        return x

    if isinstance(x, int):
        return Fraction(x, 1)

    if isinstance(x, str):
        return Fraction(x.strip())

    if isinstance(x, float):
        return Fraction(str(x))

    # accept other real-like scalars (e.g., numpy float64)
    if isinstance(x, Real):
        return Fraction(str(float(x)))

    raise TypeError(f"Unsupported type: {type(x)}")
