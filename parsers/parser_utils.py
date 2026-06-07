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


def split_top_level(s: str, sep: str = ",") -> list[str]:
    """Split at top-level separators, ignoring separators inside [], (), and {}."""

    parts = []
    buf = []
    d_sq = d_rd = d_cu = 0

    for ch in s:
        if ch == "[":
            d_sq += 1
            buf.append(ch)
        elif ch == "]":
            d_sq -= 1
            if d_sq < 0:
                raise ValueError("Unbalanced ']'")
            buf.append(ch)
        elif ch == "(":
            d_rd += 1
            buf.append(ch)
        elif ch == ")":
            d_rd -= 1
            if d_rd < 0:
                raise ValueError("Unbalanced ')'")
            buf.append(ch)
        elif ch == "{":
            d_cu += 1
            buf.append(ch)
        elif ch == "}":
            d_cu -= 1
            if d_cu < 0:
                raise ValueError("Unbalanced '}'")
            buf.append(ch)
        elif ch == sep and d_sq == 0 and d_rd == 0 and d_cu == 0:
            part = "".join(buf)
            if not part:
                raise ValueError(f"Empty item encountered when splitting by '{sep}'")
            parts.append(part)
            buf = []
        else:
            buf.append(ch)

    if d_sq != 0 or d_rd != 0 or d_cu != 0:
        raise ValueError("Unbalanced brackets/braces/parentheses")

    tail = "".join(buf)
    if tail:
        parts.append(tail)
    elif s:
        raise ValueError(f"Trailing '{sep}' or empty item encountered")

    return parts


def parse_object_sequence_string(s: str, type_map: dict[int, str] | None = None):
    """
    Parse a top-level comma-separated string into Python objects.

    type_map:
        {i: type} forces numeric parsing for the i-th top-level item.
        Negative indices are allowed, e.g. -1 means the last item.
        type ∈ {'int', 'float', 'fraction'}.

    Example:
        s = '[[1.2,1,2,3],[3.4,5.5,1]],(1,2)'
        type_map = {0: 'fraction'}

        Result:
        [
            [[Fraction(6,5), Fraction(1,1), Fraction(2,1), Fraction(3,1)],
             [Fraction(17,5), Fraction(11,2), Fraction(1,1)]],
            (1, 2)
        ]
    """
    
    s = "".join(s.split())

    if s == "":
        return []

    parts = split_top_level(s)
    n = len(parts)

    type_map = {} if type_map is None else type_map
    norm_type_map = {}
    for k, v in type_map.items():
        k = k + n if k < 0 else k
        if not (0 <= k < n):
            raise IndexError(f"type_map index {k} out of range for {n} top-level item(s)")
        norm_type_map[k] = v

    def parse_obj(x: str, forced_type: str | None) -> list:
        if not x:
            raise ValueError("Empty item encountered")

        if x[0] == "[" and x[-1] == "]":
            inner = x[1:-1]
            return [] if inner == "" else [parse_obj(p, forced_type) for p in split_top_level(inner)]

        if x[0] == "(" and x[-1] == ")":
            inner = x[1:-1]
            return () if inner == "" else tuple(parse_obj(p, forced_type) for p in split_top_level(inner))

        if x[0] == "{" and x[-1] == "}":
            inner = x[1:-1]
            return set() if inner == "" else {parse_obj(p, forced_type) for p in split_top_level(inner)}

        return parse_number(x, forced_type)

    return [parse_obj(part, norm_type_map.get(i)) for i, part in enumerate(parts)]
