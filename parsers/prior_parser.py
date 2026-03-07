from fractions import Fraction
import re

def parse_number(s: str, forced_type: str | None = None):
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

    def parse_obj(x: str, forced_type: str | None):
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


def parse_mapping_string(s: str):
    """
    Parse a mapping string into a Python dict.
    Keys may be numbers or tuples of numbers. Values must be numbers.

    Example: {1:2,0:1,(1,2):3/4}' -> {1: 2, 0: 1, (1, 2): Fraction(3, 4)}
    """
    s = "".join(s.split())
    if s == "":
        return {}

    if s[0] != "{" or s[-1] != "}":
        raise ValueError("Mapping string must be enclosed in {...}")

    inner = s[1:-1]
    if inner == "":
        return {}

    def parse_key(x: str):
        if not x:
            raise ValueError("Empty key encountered")

        if x[0] == "(" and x[-1] == ")":
            body = x[1:-1]
            return () if body == "" else tuple(parse_number(p) for p in split_top_level(body))

        return parse_number(x)

    result = {}
    for item in split_top_level(inner):
        parts = split_top_level(item, sep=":")
        if len(parts) != 2:
            raise ValueError(f"Each mapping item must contain exactly one top-level ':', got {item}")

        k_str, v_str = parts
        
        k = parse_key(k_str)
        k_check = (k,) if not isinstance(k, tuple) else k
        if not all(isinstance(x, int) or (isinstance(x, float) and x.is_integer()) or (isinstance(x, Fraction) and x.denominator == 1) for x in k_check):
            raise ValueError("The support of a discrete distribution must consist of integers.")
        k = tuple(int(x) for x in k_check)

        v = parse_number(v_str)
        if v < 0:
            raise ValueError("Probability values must be nonnegative.")
        elif v > 0:
            # Drop entries with zero probability.
            result[k] = v

    if not result:
        raise ValueError("An empty mapping cannot represent a distribution.")

    return result


def parse_prior_line(line_no_ws: str) -> Tuple[Tuple[str, ...], Union[Normal, EED]]:
    """
    Parse one prior assignment line，Assumes no whitespace, e.g.
      "x=Normal(0,1)"
      "x=Uniform(0,1)"
      "x=Exponential(1)"
      "x={0:0.2,1:0.5,3:0.3}"
      "x,y=EED([[0,1],[0,1/2]],[[0.2,0.1,0.3],[0.1,0.4,0.1]],[0.1,0.2],[0.3,0.4],{0,1})"

    Returns: (vars_tuple, dist_instance)
    """
    # local registry (edit here when adding new distributions)
    DIST_NAMES = ("Normal", "Uniform", "Exponential", "EED")
    
    # split into LHS and RHS around '='
    lhs, rhs = line_no_ws.split('=', 1)
    if not lhs:
        raise ValueError("Missing variable(s) on the left-hand side.")
    
    # ensure no more than one distribution name occurs
    hits = [name for name in DIST_NAMES if name in rhs]
    if len(hits) > 1:
        raise ValueError("A line must not contain more than one distribution name.")
    dist_name= hits[0] if hits else None

    rhs.replace(dist_name, "", 1) if dist_name else rhs
        
    vars_tuple = tuple(v for v in lhs.split(",") if v)
    if not vars_tuple:
        raise ValueError("No variables found on the left-hand side.")

    # rhs: "(...)" required
    if not (rhs.startswith("(") and rhs.endswith(")")):
        raise ValueError(f"Expected '{dist_name}(...)'.")
    args_str = rhs[1:-1]
    
    if dist_name == 'EED':
        args = parse_object_sequence_string(args_str, {0:'fraction', -1:'int'})
        dist_obj = EED(*args)
        
    elif dist_name == 'Normal':
        args = parse_object_sequence_string(args_str)
        dist_obj = Normal(*args)
        
    elif dist_name == 'Uniform':
        args = parse_object_sequence_string(args_str, {0:'fraction', 1:'fraction'})
        
        if args[0] == args[1]:
            dist_obj = EED([[args[0]]], [0,1,0], [0], [0], set())
            
        elif args[0] > args[1]:
            raise ValueError("Uniform(a, b) requires a < b")
            
        else:
            density = 1.0 / float(args[1] - args[0])
            P = np.array([0, density, 0], dtype=float)
            dist_obj = EED([args], P, [0.0], [0.0], {0})
            
    elif dist_name == 'Exponential':
        args = parse_object_sequence_string(args_str)
        
        if len(args) != 1:
            raise ValueError("Exponential distribution expects exactly 1 parameter (λ)")
        lam = args[0]
        
        if lam <= 0:
            raise ValueError("Exponential rate λ must be positive")
        alpha = [math.exp(-lam)]
        P = np.array([0, lam, lam], dtype=float)
        dist_obj = EED([[0]], P, alpha, [0.0], {0})
        
    else:
        mapping = parse_mapping_string(args_str)

        # Infer dimension
        n = len(next(iter(mapping)))

        # Build S from per-dimension integer ranges, extended by one layer on both sides
        mins = [min(pt[i] for pt in mapping) for i in range(n)]
        maxs = [max(pt[i] for pt in mapping) for i in range(n)]

        S = [list(range(mins[i] - 1, maxs[i] + 2)) for i in range(n)]
        P = np.zeros(tuple(len(si) for si in S), dtype=float)

        # Fill probability table
        for pt, prob in mapping.items():
            idx = tuple(pt[i] - mins[i] + 1 for i in range(n))
            P[idx] = prob

        dist_obj = EED(S, P, [0] * n, [0] * n, set())
        
    return vars_tuple, dist_obj

def parse_prior(prior):
    """Parse the prior section into a dict mapping vars_tuple to a distribution instance."""
    prior_items = re.split(r"[\n;]+", prior)
    prior_dict = {}
    for item in prior_items:
        item = "".join(item.split())
        if not item:
            continue
        vars_tuple, dist_obj = parse_prior_line(item)
        prior_dict[vars_tuple] = dist_obj
    return prior_dict

