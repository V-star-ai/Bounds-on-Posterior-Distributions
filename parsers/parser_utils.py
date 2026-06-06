from fractions import Fraction


def parse_number(s: str, forced_type: str | None = None) -> int | float | Fraction:
    """Parse a numeric string into int / float / Fraction, optionally forced."""
    
    if forced_type is not None:
        forced_type = forced_type.lower()
        if forced_type == "int":
            return int(s)
        if forced_type == "float":
            return float(s)
        if forced_type == "fraction":
            return Fraction(s)
        raise ValueError(f"Unknown forced type: {forced_type}")

    if "/" in s:
        return Fraction(s)
    if "." in s:
        return float(s)
    return int(s)
