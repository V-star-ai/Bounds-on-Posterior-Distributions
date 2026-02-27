from typing import List, Tuple, Optional

Interval = Tuple[Optional[int], Optional[int]]


def const_int_value(expr):
    from probably.pgcl.ast.expressions import NatLitExpr, RealLitExpr

    if isinstance(expr, NatLitExpr):
        return int(expr.value)
    if isinstance(expr, RealLitExpr):
        fr = expr.to_fraction()
        if fr.denominator == 1:
            return int(fr.numerator)
    raise ValueError("If condition constant must be an integer literal")


def interval_union(a: List[Interval], b: List[Interval]) -> List[Interval]:
    if not a:
        return list(b)
    if not b:
        return list(a)

    def key_lo(it):
        lo, _ = it
        return -float("inf") if lo is None else lo

    merged = []
    for lo, hi in sorted(list(a) + list(b), key=key_lo):
        if not merged:
            merged.append([lo, hi])
            continue
        mlo, mhi = merged[-1]
        if mhi is None:
            continue
        if lo is None or lo <= mhi:
            if mhi is None or hi is None:
                merged[-1][1] = None
            else:
                merged[-1][1] = max(mhi, hi)
        else:
            merged.append([lo, hi])
    return [tuple(x) for x in merged]


def interval_intersect(a: List[Interval], b: List[Interval]) -> List[Interval]:
    if not a or not b:
        return []

    def hi_val(hi):
        return float("inf") if hi is None else hi

    def lo_val(lo):
        return -float("inf") if lo is None else lo

    res = []
    for lo1, hi1 in a:
        for lo2, hi2 in b:
            lo = max(lo_val(lo1), lo_val(lo2))
            hi = min(hi_val(hi1), hi_val(hi2))
            if lo >= hi:
                continue
            out_lo = None if lo == -float("inf") else int(lo)
            out_hi = None if hi == float("inf") else int(hi)
            res.append((out_lo, out_hi))
    return interval_union(res, [])


def interval_complement(intervals: List[Interval]) -> List[Interval]:
    """
    Complement of a union of intervals on the number line.
    Input intervals are treated as [lo, hi) with None for -inf/+inf.
    """
    if not intervals:
        return [(None, None)]

    norm = interval_union(intervals, [])
    res: List[Interval] = []

    # left gap
    first_lo, _ = norm[0]
    if first_lo is not None:
        res.append((None, first_lo))

    # middle gaps
    for i in range(len(norm) - 1):
        _, hi = norm[i]
        lo2, _ = norm[i + 1]
        if hi is None:
            return res
        if lo2 is None:
            continue
        if hi < lo2:
            res.append((hi, lo2))

    # right gap
    _, last_hi = norm[-1]
    if last_hi is not None:
        res.append((last_hi, None))

    return res
