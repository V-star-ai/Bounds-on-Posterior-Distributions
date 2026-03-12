import math
import numpy as np
from typing import Sequence, List, Any, Optional, Iterable
from numpy.typing import ArrayLike, NDArray
from fractions import Fraction

def _pow_int_array(base: Any, exps: np.ndarray, one: Any = 1) -> np.ndarray:
    """
    逐元素计算 base ** exps（exps 是非负整数数组）。
    - 对数值 base：尽量走 numpy 快路径
    - 对 object base：用 Python int 指数，适配 SMT/符号对象的 __pow__
    并把 exponent==0 的位置强制替换成 one，避免出现 (a**0) 这种表达式。
    """
    exps = np.asarray(exps, dtype=int)

    try:
        # base 不是普通数值时，用 Python int 指数（避免 numpy.int64 让某些符号对象不适配）
        use_obj_exp = not isinstance(base, (int, float, complex, np.number))
        out = np.power(base, exps.astype(object) if use_obj_exp else exps)
    except Exception:
        # 兜底：纯 Python 计算（慢一些，但最兼容）
        out = np.array([base ** int(k) for k in exps.flat], dtype=object).reshape(exps.shape)

    if np.any(exps == 0):
        out = np.asarray(out).copy()
        if out.shape != exps.shape:  # 防御：某些对象可能返回标量
            out = np.full(exps.shape, out, dtype=object)
        out[exps == 0] = one

    return out


def _is_int_value(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) or (isinstance(x, Fraction) and x.denominator == 1)


def _pow_array(base: Any, exps: Sequence[Any], one: Any = 1) -> np.ndarray:
    """
    Power with integer or fractional exponents.
    - If all exponents are integer-valued, use _pow_int_array.
    - Otherwise, require numeric base and use Python power.
    """
    if all(_is_int_value(e) for e in exps):
        exps_int = np.array([int(e) for e in exps], dtype=int)
        return _pow_int_array(base, exps_int, one=one)
    if not isinstance(base, (int, float, complex, np.number)):
        raise TypeError("Non-integer exponents require numeric base.")
    return np.array([base ** float(e) for e in exps], dtype=object)


def _sorted_unique(seq: Sequence[Any]) -> List[Any]:
    return sorted(set(seq))


def _is_plain_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, complex, Fraction, np.number))


class EED:
    """
    EED(S, P, alpha, beta, discrete_dims)

    S:
      - Continuous dimension: strictly increasing breakpoints (Fractions allowed).
      - Discrete dimension: strictly increasing integer support points.

    P:
      - Continuous dimension: length = len(S_i) + 1 (with boundary rings).
      - Discrete dimension: length = len(S_i) (point masses).

    alpha/beta:
      - Only meaningful for continuous dimensions (0 <= < 1).

    discrete_dims:
      - Boolean list of length n, True means discrete dimension.
    """
    DefaultInterval : int = -1

    def __init__(
        self,
        S: Sequence[Sequence[Fraction]],
        P: Any,
        alpha: Sequence[Any],
        beta: Sequence[Any],
        discrete_dims: Sequence[bool] = None,
    ):
        self.S: list[np.ndarray] = [np.asarray(si, dtype=object) for si in S]
        self.P: np.ndarray = np.asarray(P)
        self.alpha = list(alpha)
        self.beta = list(beta)
        self.n = len(self.S)
        if discrete_dims is None:
            self.discrete_mask = [False] * self.n
        else:
            if len(discrete_dims) != self.n:
                raise ValueError(f"discrete_dims length must be {self.n}")
            self.discrete_mask = [bool(x) for x in discrete_dims]
        
        # shape checks
        for i, si in enumerate(self.S):
            if si.ndim != 1 or len(si) < 1:
                raise ValueError(f"S[{i}] must contain at least one breakpoint")
            if len(si) >= 2 and not np.all(si[1:] > si[:-1]):
                raise ValueError(f"S[{i}] must be strictly increasing")
            if self.discrete_mask[i]:
                if not all(_is_int_value(v) for v in si):
                    raise ValueError(f"S[{i}] must be integers for discrete dimension")

        actual_spatial = self.P.shape[: self.n]
        for i, si in enumerate(self.S):
            if self.discrete_mask[i]:
                expected = len(si)
                if actual_spatial[i] != expected:
                    raise ValueError(
                        f"P.shape[{i}] must be {expected} for discrete dim, got {actual_spatial[i]}"
                    )
            else:
                expected = len(si) + 1
                if actual_spatial[i] != expected:
                    raise ValueError(
                        f"P.shape[{i}] must be {expected} for continuous dim, got {actual_spatial[i]}"
                    )
            
        if len(self.alpha) != self.n or len(self.beta) != self.n:
            raise ValueError(f"alpha and beta must have length {self.n}")
            
    def align_to(self, S_new, *,
                 exp_approx: str = "max",
                 check_subset: bool = True) -> "EED":
        """
        exp_approx:
          - "max": 指数区域/扩展区域在 cell 内取最大（上近似）
          - "min": 指数区域/扩展区域在 cell 内取最小（下近似）

        ring_min（只影响 exp_approx="min" 时的无穷 ring）:
          - "boundary": ring 取边界面/角点对应的值（紧下界，推荐）
          - "infimum" : ring 取无穷远处的下确界（通常为 0，较松）

        one/zero:
          - 适配 object/SMT：指数为0时返回 one；ring_min="infimum" 时用 zero
        """
        exp_approx = exp_approx.lower()
        if exp_approx not in ("max", "min"):
            raise ValueError("exp_approx 只能是 'max' 或 'min'")

        S_new = [np.asarray(si, dtype=object) for si in S_new]
        if len(S_new) != self.n:
            raise ValueError(f"S_new 维度数应为 {self.n}，但得到 {len(S_new)}")

        idx_list = []
        factor_list = []

        for axis, (so, sn) in enumerate(zip(self.S, S_new)):
            if len(sn) < 1:
                raise ValueError(f"S_new[{axis}] 至少要 1 个断点")
            if len(sn) >= 2 and not np.all(sn[1:] > sn[:-1]):
                raise ValueError(f"S_new[{axis}] 必须严格递增")
            if self.discrete_mask[axis]:
                if not all(_is_int_value(v) for v in sn):
                    raise ValueError(f"S_new[{axis}] must be integers for discrete dimension")

            if check_subset and (not np.isin(so, sn).all()):
                missing = so[~np.isin(so, sn)]
                raise ValueError(f"第 {axis} 维 S_old 不完全包含于 S_new，缺少断点: {missing}")

            if self.discrete_mask[axis]:
                # discrete axis: map point masses; new points -> 0
                idx = []
                so_list = list(so.tolist())
                index_map = {v: i for i, v in enumerate(so_list)}
                for v in sn.tolist():
                    idx.append(index_map.get(v, -1))
                idx_list.append(np.array(idx, dtype=int))
                factor_list.append(None)
            else:
                b = so[0]
                e = so[-1]
                m = len(so)  # 旧断点数；旧该轴 P 长度 m+1；右圈索引 m

                starts = sn[:-1]
                ends = sn[1:]

                # --- interior cells -> old index（带 ring）
                k_mid = np.empty(len(starts), dtype=int)
                for i, (st, en) in enumerate(zip(starts, ends)):
                    if en <= b:
                        k_mid[i] = 0
                    elif st >= e:
                        k_mid[i] = m
                    else:
                        j = int(np.searchsorted(so, st, side="right") - 1)  # 0..m-2
                        k_mid[i] = 1 + j  # 1..m-1

                k = np.concatenate(([0], k_mid, [m]))
                idx_list.append(k)

                # --- 指数近似（左用 alpha，右用 beta）
                left_exp_mid = []
                right_exp_mid = []
                if exp_approx == "max":
                    for st, en in zip(starts, ends):
                        left_exp_mid.append(max(b - en, 0))
                        right_exp_mid.append(max(st - e, 0))
                else:
                    for st, en in zip(starts, ends):
                        left_exp_mid.append(max(b - st, 0))
                        right_exp_mid.append(max(en - e, 0))

                f_mid = _pow_array(self.alpha[axis], left_exp_mid) * _pow_array(self.beta[axis], right_exp_mid)

                f0 = _pow_array(self.alpha[axis], [max(b - sn[0], 0)])[0]
                flast = _pow_array(self.beta[axis], [max(sn[-1] - e, 0)])[0]

                f = np.concatenate(([f0], f_mid, [flast]))
                factor_list.append(f)

        # --- 拉伸旧 P 到新网格
        P_new = self.P
        for axis in range(self.n):
            if self.discrete_mask[axis]:
                idx = idx_list[axis]
                idx_clip = np.where(idx >= 0, idx, 0)
                P_new = np.take(P_new, idx_clip, axis=axis)
                # zero out newly introduced points
                missing = np.where(idx < 0)[0]
                for mi in missing:
                    slicer = [slice(None)] * P_new.ndim
                    slicer[axis] = mi
                    P_new[tuple(slicer)] = 0
            else:
                P_new = np.take(P_new, idx_list[axis], axis=axis)

        # --- 乘上连续维度衰减因子
        extra = P_new.ndim - self.n
        for axis, f in enumerate(factor_list):
            if f is None:
                continue
            shape = [1] * self.n
            shape[axis] = len(f)
            f_rs = f.reshape(shape)
            if extra > 0:
                f_rs = f_rs[(...,) + (None,) * extra]
            P_new = P_new * f_rs

        return EED(S_new, P_new, self.alpha, self.beta, self.discrete_mask)

    @staticmethod
    def merge_breakpoints(
            S1: ArrayLike,
            S2: ArrayLike,
            interval: int,
    ) -> NDArray[np.int64]:
        """
        合并两个已排序的一维 int 数组，并在两端补点使得：
          对每个 Si:
            [gmin, Si[0]] 与 [Si[-1], gmax] 内的相邻点最大间隔 <= interval
        interval = -1 时不补点。

        返回
        ----
        result : 升序 int64 ndarray
        """
        S1 = np.asarray(S1, dtype=object)
        S2 = np.asarray(S2, dtype=object)

        if S1.ndim != 1 or S2.ndim != 1:
            raise ValueError("S1 和 S2 必须是一维数组")
        if S1.size == 0 or S2.size == 0:
            raise ValueError("S1 和 S2 不能为空（若需要支持空数组我也可以改）")

        # interval = -1 => 只合并排序
        if interval == -1:
            merged = _sorted_unique(list(S1) + list(S2))
            return np.asarray(merged, dtype=object)

        if interval <= 0:
            raise ValueError("interval 必须为 -1 或者正整数")
        if not all(_is_int_value(v) for v in list(S1) + list(S2)):
            raise ValueError("interval-based merge only supports integer breakpoints")

        # 合并骨架：用去重的骨架计算 gaps（重复点不影响 gap）
        U = np.array(_sorted_unique(list(S1) + list(S2)), dtype=object)
        if U.size <= 1:
            merged = U
            return merged

        gmin, gmax = U[0], U[-1]
        s1_l, s1_r = S1[0], S1[-1]
        s2_l, s2_r = S2[0], S2[-1]

        left = np.asarray(U[:-1], dtype=int)
        right = np.asarray(U[1:], dtype=int)
        gaps = right - left

        # 需要约束的 gaps（落在任一“端区间”里）：
        # [gmin, s*_l] => 该区间内的 gap 满足 right <= s*_l  (因为 right 是该 gap 的右端点)
        # [s*_r, gmax] => 该区间内的 gap 满足 left  >= s*_r  (因为 left 是该 gap 的左端点)
        need = (
                (right <= s1_l) | (left >= s1_r) |
                (right <= s2_l) | (left >= s2_r)
        )

        # 只对需要的 gap 且 gap > interval 才补点
        mask = need & (gaps > interval)
        if not np.any(mask):
            merged = U
            return merged

        left_m = left[mask]
        gaps_m = gaps[mask]

        # 每个 gap 最少补点数 k = (d-1)//interval
        k = (gaps_m - 1) // interval
        total = int(k.sum())

        # 生成所有补点：left + interval * [1..k]（向量化拼接变长序列）
        idx = np.repeat(np.arange(k.size), k)  # 每个 gap 的编号重复 k 次
        group_start = np.repeat(np.cumsum(k) - k, k)
        step = (np.arange(total) - group_start) + 1  # 每组从 1 到 k

        inserted = left_m[idx] + interval * step

        # 合并最终结果
        result_unique = np.union1d(U.astype(int), inserted)

        # 是否保留重复点：若 unique=False，则把原始 concat 的重复也带回去（再 union 插入点）
        result = result_unique
        return result

    @staticmethod
    def add(eed1 : "EED", eed2 : "EED", interval = DefaultInterval, max_function = None):
        if eed1.n != eed2.n:
            raise ValueError(f"EED Add Error: The addition of two variables of different dimensions, {eed1.n}, {eed2.n}")
        if eed1.discrete_mask != eed2.discrete_mask:
            raise ValueError("Discrete dimension masks must match for addition")
        merged_breakpoints = []
        for i, (s1, s2) in enumerate(zip(eed1.S, eed2.S)):
            if eed1.discrete_mask[i]:
                merged_breakpoints.append(np.asarray(_sorted_unique(list(s1) + list(s2)), dtype=object))
            else:
                merged_breakpoints.append(EED.merge_breakpoints(s1, s2, interval))
        eed1, eed2 = eed1.align_to(merged_breakpoints), eed2.align_to(merged_breakpoints)
        if max_function:
            max2 = np.frompyfunc(max_function, 2, 1)
            return EED(merged_breakpoints, eed1.P + eed2.P, max2(eed1.alpha, eed2.alpha), max2(eed1.beta, eed2.beta), eed1.discrete_mask)
        return EED(merged_breakpoints, eed1.P + eed2.P, np.maximum(eed1.alpha, eed2.alpha), np.maximum(eed1.beta, eed2.beta), eed1.discrete_mask)

    def __add__(self, other):
        return EED.add(self, other, EED.DefaultInterval)

    @staticmethod
    def build_constraint(
            eed1 : "EED",
            eed2 : "EED",
            constraint_function):
        """
        :param eed1: left element
        :param eed2: right element
        :param constraint_function: constraint function
        :param return_list: whether to return the constraint list
        :param and_function: AND-function
        :param true: True element
        :param false: False element
        :return:
        """
        if eed1.n != eed2.n:
            raise ValueError(f"EED Less Error: The addition of two variables of different dimensions, {eed1.n}, {eed2.n}")
        if eed1.discrete_mask != eed2.discrete_mask:
            raise ValueError("Discrete dimension masks must match for comparison")
        if eed1.n == eed2.n == 0:
            return []
        merged_breakpoints = []
        for i, (s1, s2) in enumerate(zip(eed1.S, eed2.S)):
            if eed1.discrete_mask[i]:
                merged_breakpoints.append(np.asarray(_sorted_unique(list(s1) + list(s2)), dtype=object))
            else:
                merged_breakpoints.append(EED.merge_breakpoints(s1, s2, -1))
        eed1, eed2 = eed1.align_to(merged_breakpoints, exp_approx="max"), eed2.align_to(merged_breakpoints, exp_approx="min")
        lt = np.frompyfunc(constraint_function, 2, 1)
        constraint_list = (lt(eed1.P, eed2.P).ravel().tolist() +
                           lt(eed1.alpha, eed2.alpha).ravel().tolist() + lt(eed1.beta, eed2.beta).ravel().tolist())
        return constraint_list

    @staticmethod
    def leq(eed1 : "EED", eed2 : "EED"):
        return EED.build_constraint(eed1, eed2, lambda a, b : a <= b)

    # for float
    def __le__(self, other):
        if not isinstance(other, EED) or self.n != other.n:
            return NotImplemented
        if self.n != other.n:
            return NotImplemented
        if self.discrete_mask != other.discrete_mask:
            return NotImplemented
        merged_breakpoints = []
        for i, (s1, s2) in enumerate(zip(self.S, other.S)):
            if self.discrete_mask[i]:
                merged_breakpoints.append(np.asarray(_sorted_unique(list(s1) + list(s2)), dtype=object))
            else:
                merged_breakpoints.append(EED.merge_breakpoints(s1, s2, -1))
        eed1, eed2 = self.align_to(merged_breakpoints, exp_approx="max"), other.align_to(merged_breakpoints, exp_approx="min")
        return np.all(eed1.P <= eed2.P)

    ###### apply if #####
    @staticmethod
    def _insert_breakpoint(si: np.ndarray, x: Any) -> np.ndarray:
        """插入断点 x（支持 Fraction/int），若已存在则不变"""
        if np.any(si == x):
            return si
        out = np.asarray(_sorted_unique(list(si.tolist()) + [x]), dtype=object)
        if len(out) >= 2 and not np.all(out[1:] > out[:-1]):
            raise ValueError("插入断点后不严格递增")
        return out

    @staticmethod
    def _index_of(si: np.ndarray, x: Any) -> int:
        """x 必须在 si 中，返回其下标"""
        k = int(np.searchsorted(si, x, side="left"))
        if k >= len(si) or si[k] != x:
            raise ValueError(f"断点 {x} 未出现在对齐后的 S 里（这不应该发生）")
        return k

    def restrict_ge(
            self,
            axis: int,
            a: int,
            *,
            exp_approx: str = "max",
    ) -> "EED":
        """
        限制到 x_axis >= a，其它区域赋 fill，并把多余的 0 区域裁掉（同步修改 S）。
        [l,r) 语义：x=a 被保留。
        """
        axis = int(axis)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")
        if self.discrete_mask[axis]:
            if not _is_int_value(a):
                raise ValueError("Discrete axis requires integer bound")
            a = int(a)
            P_new = np.array(self.P, copy=True)
            si = self.S[axis]
            for idx, v in enumerate(si):
                if int(v) < a:
                    slicer = [slice(None)] * P_new.ndim
                    slicer[axis] = idx
                    P_new[tuple(slicer)] = 0
            return EED(self.S, P_new, self.alpha, self.beta, self.discrete_mask)
        a = a

        # 1) 插入断点并对齐
        S_align = [si.copy() for si in self.S]
        S_align[axis] = self._insert_breakpoint(S_align[axis], a)
        g = self.align_to(S_align, exp_approx=exp_approx)

        si = g.S[axis]
        k = self._index_of(si, a)  # si[k] == a

        # 2) 裁剪：保留从断点 a 开始的部分（含 a），即 S_axis := si[k:]
        #    对应 cell（含 ring）长度应为 len(S_axis)+1 = (len(si)-k)+1，
        #    而 g.P 该轴长度是 len(si)+1，所以取 slice(k, None) 正好匹配
        S_new = [arr.copy() for arr in g.S]
        S_new[axis] = si[k:]

        indexer = [slice(None)] * g.P.ndim
        indexer[axis] = slice(k, None)
        P_new = np.array(g.P[tuple(indexer)], copy=True)

        # 3) 新函数的“外侧”是 x<a，对应新轴的左 ring（index 0），设为 fill
        indexer2 = [slice(None)] * P_new.ndim
        indexer2[axis] = 0
        P_new[tuple(indexer2)] = 0

        # 4) 衰减率：左侧无效，设 alpha[axis]=0（可用 zero 适配 SMT）
        alpha2 = list(g.alpha)
        beta2 = list(g.beta)
        alpha2[axis] = 0

        return EED(S_new, P_new, alpha2, beta2, self.discrete_mask)

    def restrict_lt(
            self,
            axis: int,
            b: int,
            *,
            exp_approx: str = "max",
    ) -> "EED":
        """
        限制到 x_axis < b，其它区域赋 fill，并把多余的 0 区域裁掉（同步修改 S）。
        [l,r) 语义：x=b 不被保留。
        """
        axis = int(axis)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")
        if self.discrete_mask[axis]:
            if not _is_int_value(b):
                raise ValueError("Discrete axis requires integer bound")
            b = int(b)
            P_new = np.array(self.P, copy=True)
            si = self.S[axis]
            for idx, v in enumerate(si):
                if int(v) >= b:
                    slicer = [slice(None)] * P_new.ndim
                    slicer[axis] = idx
                    P_new[tuple(slicer)] = 0
            return EED(self.S, P_new, self.alpha, self.beta, self.discrete_mask)
        b = b

        # 1) 插入断点并对齐
        S_align = [si.copy() for si in self.S]
        S_align[axis] = self._insert_breakpoint(S_align[axis], b)
        g = self.align_to(S_align, exp_approx=exp_approx)

        si = g.S[axis]
        k = self._index_of(si, b)  # si[k] == b

        # 2) 裁剪：保留到断点 b 为止（含 b），即 S_axis := si[:k+1]
        #    新轴 cell（含 ring）长度 = len(S_axis)+1 = (k+1)+1 = k+2
        #    对应 g.P 该轴取 slice(None, k+2)（保留 0..k+1，其中 k+1 是从 b 开始的那格，作为新右 ring）
        S_new = [arr.copy() for arr in g.S]
        S_new[axis] = si[: k + 1]

        indexer = [slice(None)] * g.P.ndim
        indexer[axis] = slice(None, k + 2)
        P_new = np.array(g.P[tuple(indexer)], copy=True)

        # 3) 新函数的“外侧”是 x>=b，对应新轴的右 ring（最后一格），设为 fill
        indexer2 = [slice(None)] * P_new.ndim
        indexer2[axis] = -1
        P_new[tuple(indexer2)] = 0

        # 4) 衰减率：右侧无效，设 beta[axis]=0
        alpha2 = list(g.alpha)
        beta2 = list(g.beta)
        beta2[axis] = 0

        return EED(S_new, P_new, alpha2, beta2, self.discrete_mask)

    def restrict_interval(
            self,
            axis: int,
            intervals,
            *,
            exp_approx: str = "max",
            max_function=None,
    ) -> "EED":
        """
        Restrict to a union of intervals on a given axis.
        intervals: list of (lo, hi), using None for -inf / +inf.
                   Semantics: [lo, hi), with None as unbounded.
        """
        axis = int(axis)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")

        def normalize(ints):
            cleaned = []
            for it in ints:
                if it is None or len(it) != 2:
                    raise ValueError("interval must be (lo, hi)")
                lo, hi = it
                if self.discrete_mask[axis]:
                    if lo is not None and not _is_int_value(lo):
                        raise ValueError("Discrete axis requires integer bounds")
                    if hi is not None and not _is_int_value(hi):
                        raise ValueError("Discrete axis requires integer bounds")
                    if lo is not None:
                        lo = int(lo)
                    if hi is not None:
                        hi = int(hi)
                if lo is not None and hi is not None and lo >= hi:
                    raise ValueError("interval must satisfy lo < hi")
                cleaned.append((lo, hi))
            if not cleaned:
                return []

            def lo_key(x):
                return -float("inf") if x[0] is None else x[0]

            cleaned.sort(key=lo_key)
            merged = []
            for lo, hi in cleaned:
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

        intervals = normalize(intervals)
        if not intervals:
            # empty: everything becomes fill
            return self.restrict_ge(axis, 0).restrict_lt(axis, 0)

        res = None
        for lo, hi in intervals:
            cur = self
            if lo is not None:
                cur = cur.restrict_ge(axis, lo, exp_approx=exp_approx)
            if hi is not None:
                cur = cur.restrict_lt(axis, hi, exp_approx=exp_approx)
            if res is None:
                res = cur
            else:
                res = EED.add(res, cur, interval=EED.DefaultInterval, max_function=max_function)

        return res

    def times_constant(self, constant: float) -> "EED":
        return EED(self.S, self.P * constant, self.alpha, self.beta, self.discrete_mask)

    def _tail_mass(
        self,
        coeff: Any,
        rate: Any,
        approximate_step: Fraction = None,
        ln=None,
    ) -> Any:
        if approximate_step is None:
            if _is_plain_scalar(rate):
                if rate == 0:
                    return 0
                if rate == 1:
                    raise ValueError("Continuous tail with decay rate 1 has infinite mass")
                rate_for_ln = float(rate)
            else:
                if ln is None:
                    raise ValueError("Exact continuous tail mass with symbolic rate requires ln")
                rate_for_ln = rate

            ln_fn = ln if ln is not None else math.log
            log_rate = ln_fn(rate_for_ln)
            return coeff / (0 - log_rate)

        if approximate_step <= 0:
            raise ValueError("approximate_step must be positive")
        if _is_plain_scalar(rate) and rate == 1:
            raise ValueError("Continuous tail with decay rate 1 has infinite mass")
        return coeff * approximate_step / (1 - rate ** approximate_step)

    def _axis_slice_mass(
        self,
        axis: int,
        fixed_index: Sequence[Any],
        approximate_step: Fraction = None,
        ln=None,
    ) -> Any:
        mass = 0
        idx = list(fixed_index)

        if self.discrete_mask[axis]:
            for point_idx in range(len(self.S[axis])):
                idx[axis] = point_idx
                mass = mass + self.P[tuple(idx)]
            return mass

        si = self.S[axis]
        m = len(si)

        idx[axis] = 0
        mass = mass + self._tail_mass(self.P[tuple(idx)], self.alpha[axis], approximate_step, ln)

        for block_idx in range(1, m):
            idx[axis] = block_idx
            width = si[block_idx] - si[block_idx - 1]
            mass = mass + self.P[tuple(idx)] * width

        idx[axis] = m
        mass = mass + self._tail_mass(self.P[tuple(idx)], self.beta[axis], approximate_step, ln)
        return mass

    def assign_eed(
        self,
        axis: int,
        eed: "EED",
        approximate_step: Fraction = None,
        ln=None,
    ) -> "EED":
        axis = int(axis)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")
        if not isinstance(eed, EED):
            raise TypeError("eed must be an EED")
        if eed.n != 1:
            raise ValueError("assign_eed expects a one-dimensional EED")
        if self.discrete_mask[axis] != eed.discrete_mask[0]:
            raise ValueError("Assigned EED kind must match the target axis kind")
        if approximate_step is not None and self.discrete_mask[axis]:
            approximate_step = None
        if eed.P.shape[1:] != self.P.shape[self.n:]:
            raise ValueError("Assigned EED extra dimensions must match the target EED")

        spatial_shape_old = list(self.P.shape[: self.n])
        spatial_shape_new = list(spatial_shape_old)
        spatial_shape_new[axis] = eed.P.shape[0]
        extra_shape = self.P.shape[self.n :]

        P_new = np.empty(tuple(spatial_shape_new) + extra_shape, dtype=object)
        other_axes = [i for i in range(self.n) if i != axis]
        iter_shape = tuple(spatial_shape_old[i] for i in other_axes) + extra_shape
        iter_indices = np.ndindex(iter_shape) if iter_shape else [()]

        for combined_idx in iter_indices:
            split = len(other_axes)
            other_idx = combined_idx[:split]
            extra_idx = combined_idx[split:]

            src_index = [slice(None)] * self.n + list(extra_idx)
            dst_index = [slice(None)] * self.n + list(extra_idx)
            for other_axis, value in zip(other_axes, other_idx):
                src_index[other_axis] = value
                dst_index[other_axis] = value

            mass = self._axis_slice_mass(axis, src_index, approximate_step, ln)
            P_new[tuple(dst_index)] = mass * eed.P

        S_new = [arr.copy() for arr in self.S]
        S_new[axis] = eed.S[0].copy()
        alpha_new = list(self.alpha)
        beta_new = list(self.beta)
        alpha_new[axis] = eed.alpha[0]
        beta_new[axis] = eed.beta[0]
        return EED(S_new, P_new, alpha_new, beta_new, self.discrete_mask)

    def add_constant(self, axis, c):
        """
        add a constant in an axis.
        """
        S_new = [arr.copy() for arr in self.S]
        si = S_new[axis]
        if self.discrete_mask[axis]:
            if not _is_int_value(c):
                raise ValueError("Discrete axis requires integer shift")
            c = int(c)
        S_new[axis] = [x + c for x in si]
        return EED(S_new, self.P, self.alpha, self.beta, self.discrete_mask)

if __name__ == '__main__':

    a = EED([[Fraction(-1), Fraction(1), Fraction(4)], [Fraction(2), Fraction(3)]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.3, 0.2, 0.1]], [0.25, 0.5], [0.5, 0.25])
    print(a.S)
    print(a.P)
    b = a.align_to([[Fraction(-1), Fraction(1), Fraction(4)], [Fraction(-1), Fraction(2), Fraction(3)]], exp_approx="min")
    print(b.S)
    print(b.P)
    print(b.alpha)
    print(b.beta)
    c = b.restrict_lt(0, 1).restrict_lt(1, 3).restrict_ge(0, -3)
    print(c.S)
    print(c.P)
    print(c.alpha)
    print(c.beta)
    c = b.restrict_interval(0, [(-3, 1), (3, None)]).restrict_interval(1, [(None, 3)])
    print(c.S)
    print(c.P)
    print(c.alpha)
    print(c.beta)
