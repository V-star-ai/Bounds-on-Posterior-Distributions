from fractions import Fraction
from typing import List, Optional, Tuple

Interval = Tuple[Optional[Fraction], bool, Optional[Fraction], bool]


def const_int_value(expr):
    from probably.pgcl.ast.expressions import NatLitExpr, RealLitExpr

    if isinstance(expr, NatLitExpr):
        return Fraction(expr.value)
    if isinstance(expr, RealLitExpr):
        return expr.to_fraction()
    raise ValueError("If condition constant must be a fraction literal")


def interval_is_empty(interval: Interval) -> bool:
    lo, lo_closed, hi, hi_closed = interval
    if lo is None or hi is None:
        return False
    if lo < hi:
        return False
    if lo > hi:
        return True
    return not (lo_closed and hi_closed)


def _interval_lo_key(interval: Interval):
    lo, lo_closed, _hi, _hi_closed = interval
    return (float("-inf") if lo is None else lo, 0 if lo_closed else 1)


def _interval_intersection(left: Interval, right: Interval) -> Interval | None:
    lo1, lo1_closed, hi1, hi1_closed = left
    lo2, lo2_closed, hi2, hi2_closed = right

    if lo1 is None:
        lo, lo_closed = lo2, lo2_closed
    elif lo2 is None:
        lo, lo_closed = lo1, lo1_closed
    elif lo1 > lo2:
        lo, lo_closed = lo1, lo1_closed
    elif lo2 > lo1:
        lo, lo_closed = lo2, lo2_closed
    else:
        lo, lo_closed = lo1, lo1_closed and lo2_closed

    if hi1 is None:
        hi, hi_closed = hi2, hi2_closed
    elif hi2 is None:
        hi, hi_closed = hi1, hi1_closed
    elif hi1 < hi2:
        hi, hi_closed = hi1, hi1_closed
    elif hi2 < hi1:
        hi, hi_closed = hi2, hi2_closed
    else:
        hi, hi_closed = hi1, hi1_closed and hi2_closed

    result = (lo, lo_closed, hi, hi_closed)
    if interval_is_empty(result):
        return None
    return result


def interval_intersect(
    left_intervals: List[Interval], right_intervals: List[Interval]
) -> List[Interval]:
    result = []
    for left in left_intervals:
        for right in right_intervals:
            intersection = _interval_intersection(left, right)
            if intersection is not None:
                result.append(intersection)
    return interval_union(result, [])


def _intervals_touch_or_overlap(left: Interval, right: Interval) -> bool:
    _lo1, _lo1_closed, hi1, hi1_closed = left
    lo2, lo2_closed, _hi2, _hi2_closed = right
    if hi1 is None or lo2 is None:
        return True
    if hi1 > lo2:
        return True
    if hi1 < lo2:
        return False
    return hi1_closed or lo2_closed


def _merge_intervals(left: Interval, right: Interval) -> Interval:
    lo1, lo1_closed, hi1, hi1_closed = left
    _lo2, _lo2_closed, hi2, hi2_closed = right

    if hi1 is None or hi2 is None:
        return lo1, lo1_closed, None, False
    if hi1 > hi2:
        return left
    if hi2 > hi1:
        return lo1, lo1_closed, hi2, hi2_closed
    return lo1, lo1_closed, hi1, hi1_closed or hi2_closed


def interval_union(
    left_intervals: List[Interval], right_intervals: List[Interval]
) -> List[Interval]:
    intervals = [
        interval
        for interval in list(left_intervals) + list(right_intervals)
        if not interval_is_empty(interval)
    ]
    if not intervals:
        return []

    intervals.sort(key=_interval_lo_key)
    merged = [intervals[0]]
    for interval in intervals[1:]:
        if _intervals_touch_or_overlap(merged[-1], interval):
            merged[-1] = _merge_intervals(merged[-1], interval)
        else:
            merged.append(interval)
    return merged


def interval_complement(intervals: List[Interval]) -> List[Interval]:
    normalized = interval_union(intervals, [])
    if not normalized:
        return [(None, False, None, False)]

    result = []
    first_lo, first_lo_closed, _first_hi, _first_hi_closed = normalized[0]
    if first_lo is not None:
        result.append((None, False, first_lo, not first_lo_closed))

    for left, right in zip(normalized, normalized[1:]):
        _lo1, _lo1_closed, hi1, hi1_closed = left
        lo2, lo2_closed, _hi2, _hi2_closed = right
        gap = (hi1, not hi1_closed, lo2, not lo2_closed)
        if not interval_is_empty(gap):
            result.append(gap)

    _last_lo, _last_lo_closed, last_hi, last_hi_closed = normalized[-1]
    if last_hi is not None:
        result.append((last_hi, not last_hi_closed, None, False))

    return result
