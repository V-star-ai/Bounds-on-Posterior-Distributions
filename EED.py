import numpy as np
from typing import Sequence, List, Any
from numpy.typing import ArrayLike, NDArray

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


class EED:
    """
    EED(S, P, alpha, beta)

    S: 每维断点（严格递增整数），区间统一 [l,r)
    P: ndarray，空间维形状为 (len(S0)+1, len(S1)+1, ..., len(Sn-1)+1)
       也就是：每维 [d_i + 2]，左右各一圈边界值
       索引语义（对第 i 维，m=len(S_i)）：
         0        : 左边界圈（x < S_i[0]）
         1..m-1   : 区间 [S_i[j-1], S_i[j])
         m        : 右边界圈（x >= S_i[-1]）
    alpha: 左侧衰减率向量（0<= <1）
    beta : 右侧衰减率向量（0<= <1）
    """

    DefaultInterval : int = -1

    def __init__(self, S: Sequence[Sequence[int]], P: Any,
                 alpha: Sequence[Any], beta: Sequence[Any]):
        self.S: List[np.ndarray] = [np.asarray(si, dtype=int) for si in S]
        self.P: np.ndarray = np.asarray(P)
        self.alpha = list(alpha)
        self.beta = list(beta)
        self.n = len(self.S)

        if len(self.alpha) != self.n or len(self.beta) != self.n:
            raise ValueError(f"alpha/beta 长度必须等于维度 n={self.n}")

        for i, si in enumerate(self.S):
            if si.ndim != 1 or len(si) < 1:
                raise ValueError(f"S[{i}] 必须至少有 1 个断点（允许退化：只有边界，没有常数块）")

            if len(si) >= 2 and not np.all(si[1:] > si[:-1]):
                raise ValueError(f"S[{i}] 断点必须严格递增")

        expected_spatial = tuple(len(si) + 1 for si in self.S)
        if self.P.shape[:self.n] != expected_spatial:
            raise ValueError(
                f"P 的前 {self.n} 维形状应为 {expected_spatial}，但得到 {self.P.shape[:self.n]}"
            )

    def align_to(self, S_new, *, one=1, zero=0,
                 exp_approx: str = "max",
                 ring_min: str = "boundary",
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
        ring_min = ring_min.lower()
        if exp_approx not in ("max", "min"):
            raise ValueError("exp_approx 只能是 'max' 或 'min'")
        if ring_min not in ("boundary", "infimum"):
            raise ValueError("ring_min 只能是 'boundary' 或 'infimum'")

        S_new = [np.asarray(si, dtype=int) for si in S_new]
        if len(S_new) != self.n:
            raise ValueError(f"S_new 维度数应为 {self.n}，但得到 {len(S_new)}")

        idx_list = []
        factor_list = []

        for axis, (so, sn) in enumerate(zip(self.S, S_new)):
            if len(sn) < 1:
                raise ValueError(f"S_new[{axis}] 至少要 1 个断点")
            if len(sn) >= 2 and not np.all(sn[1:] > sn[:-1]):
                raise ValueError(f"S_new[{axis}] 必须严格递增")

            if check_subset and (not np.isin(so, sn).all()):
                missing = so[~np.isin(so, sn)]
                raise ValueError(f"第 {axis} 维 S_old 不完全包含于 S_new，缺少断点: {missing}")

            b = int(so[0])
            e = int(so[-1])
            m = len(so)  # 旧断点数；旧该轴 P 长度 m+1；右圈索引 m

            # 新轴 cell 个数 = len(sn)+1
            starts = sn[:-1]
            ends = sn[1:]

            # --- interior cells -> old index（带 ring）
            left_mid = ends <= b
            right_mid = starts >= e
            inside_mid = ~(left_mid | right_mid)

            k_mid = np.empty(len(starts), dtype=int)
            k_mid[left_mid] = 0
            k_mid[right_mid] = m
            if np.any(inside_mid):
                j = np.searchsorted(so, starts[inside_mid], side="right") - 1  # 0..m-2
                k_mid[inside_mid] = 1 + j  # 1..m-1

            k = np.concatenate(([0], k_mid, [m]))
            idx_list.append(k)

            # --- 指数近似（左用 alpha，右用 beta；匹配你之前的例子）
            if exp_approx == "max":
                # 左指数 max 在 x->r^-：指数 b - end
                left_exp_mid = np.maximum(b - ends, 0)
                # 右指数 max 在 x=l：指数 start - e
                right_exp_mid = np.maximum(starts - e, 0)
            else:
                # 左指数 min 在 x=l：指数 b - start
                left_exp_mid = np.maximum(b - starts, 0)
                # 右指数 min 在 x->r^-：指数 end - e
                right_exp_mid = np.maximum(ends - e, 0)

            f_mid = _pow_int_array(self.alpha[axis], left_exp_mid, one=one) * \
                    _pow_int_array(self.beta[axis], right_exp_mid, one=one)

            # --- 无穷 ring 的处理
            # ring 的意义：边界面/边界点的“基值”（指数从这里开始延拓）
            # 因此 ring_min="boundary" 时，即使用边界点的指数值（紧下界）
            if exp_approx == "min" and ring_min == "infimum":
                f0 = zero
                flast = zero
            else:
                # 左 ring：边界在 sn[0]，旧左指数指数 = b - sn[0]
                f0 = _pow_int_array(self.alpha[axis], np.array([max(b - int(sn[0]), 0)]), one=one)[0]
                # 右 ring：边界在 sn[-1]，旧右指数指数 = sn[-1] - e
                flast = _pow_int_array(self.beta[axis], np.array([max(int(sn[-1]) - e, 0)]), one=one)[0]

            f = np.concatenate(([f0], f_mid, [flast]))
            factor_list.append(f)

        # --- 拉伸旧 P 到新网格
        P_new = self.P
        for axis in range(self.n):
            P_new = np.take(P_new, idx_list[axis], axis=axis)

        # --- 乘上各维衰减因子
        extra = P_new.ndim - self.n
        for axis, f in enumerate(factor_list):
            shape = [1] * self.n
            shape[axis] = len(f)
            f_rs = f.reshape(shape)
            if extra > 0:
                f_rs = f_rs[(...,) + (None,) * extra]
            P_new = P_new * f_rs

        return EED(S_new, P_new, self.alpha, self.beta)

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
        S1 = np.asarray(S1, dtype=np.int64)
        S2 = np.asarray(S2, dtype=np.int64)

        if S1.ndim != 1 or S2.ndim != 1:
            raise ValueError("S1 和 S2 必须是一维数组")
        if S1.size == 0 or S2.size == 0:
            raise ValueError("S1 和 S2 不能为空（若需要支持空数组我也可以改）")

        # interval = -1 => 只合并排序
        if interval == -1:
            merged = np.union1d(S1, S2)
            return merged

        if interval <= 0:
            raise ValueError("interval 必须为 -1 或者正整数")

        # 合并骨架：用去重的骨架计算 gaps（重复点不影响 gap）
        U = np.union1d(S1, S2)  # sorted unique
        if U.size <= 1:
            merged = U
            return merged

        gmin, gmax = U[0], U[-1]
        s1_l, s1_r = S1[0], S1[-1]
        s2_l, s2_r = S2[0], S2[-1]

        left = U[:-1]
        right = U[1:]
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
        result_unique = np.union1d(U, inserted)

        # 是否保留重复点：若 unique=False，则把原始 concat 的重复也带回去（再 union 插入点）
        result = result_unique
        return result

    @staticmethod
    def add(eed1 : "EED", eed2 : "EED", interval = DefaultInterval, max_function = None):
        if eed1.n != eed2.n:
            raise ValueError(f"EED Add Error: The addition of two variables of different dimensions, {eed1.n}, {eed2.n}")
        merged_breakpoints = [EED.merge_breakpoints(s1, s2, interval) for s1, s2 in zip(eed1.S, eed2.S)]
        eed1, eed2 = eed1.align_to(merged_breakpoints), eed2.align_to(merged_breakpoints)
        if max_function:
            max2 = np.frompyfunc(max_function, 2, 1)
            return EED(merged_breakpoints, eed1.P + eed2.P, max2(eed1.alpha, eed2.alpha), max2(eed1.beta, eed2.beta))
        return EED(merged_breakpoints, eed1.P + eed2.P, np.maximum(eed1.alpha, eed2.alpha), np.maximum(eed1.beta, eed2.beta))

    def __add__(self, other):
        return EED.add(self, other, EED.DefaultInterval)

    @staticmethod
    def build_constraint(
            eed1 : "EED",
            eed2 : "EED",
            constraint_function,
            return_list : bool = False,
            and_function = None,
            true = True,
            false = False):
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
        if eed1.n == eed2.n == 0:
            return false
        merged_breakpoints = [EED.merge_breakpoints(s1, s2, -1) for s1, s2 in zip(eed1.S, eed2.S)]
        eed1, eed2 = eed1.align_to(merged_breakpoints, exp_approx="max"), eed2.align_to(merged_breakpoints, exp_approx="min")
        lt = np.frompyfunc(constraint_function, 2, 1)
        constraint_list = (lt(eed1.P, eed2.P).ravel().tolist() +
                           lt(eed1.alpha, eed2.alpha).ravel().tolist() + lt(eed1.beta, eed2.beta).ravel().tolist())
        if return_list:
            return constraint_list
        res = true
        if and_function:
            for element in constraint_list:
                res = and_function(res, element)
        else:
            for element in constraint_list:
                res = res & element
        return res

    @staticmethod
    def leq(eed1 : "EED", eed2 : "EED", **args):
        return EED.build_constraint(eed1, eed2, lambda a, b : a <= b, **args)

    # for float
    def __le__(self, other):
        if not isinstance(other, EED) or self.n != other.n:
            return NotImplemented
        if self.n != other.n:
            return NotImplemented
        merged_breakpoints = [EED.merge_breakpoints(s1, s2, -1) for s1, s2 in zip(self.S, other.S)]
        eed1, eed2 = self.align_to(merged_breakpoints, exp_approx="max"), other.align_to(merged_breakpoints, exp_approx="min")
        return np.all(eed1.P <= eed2.P)

    ###### apply if #####
    @staticmethod
    def _insert_breakpoint(si: np.ndarray, x: int) -> np.ndarray:
        """把整数断点 x 插入严格递增断点数组 si（若已存在则不变）"""
        x = int(x)
        if np.any(si == x):
            return si
        out = np.sort(np.concatenate([si, np.array([x], dtype=int)]))
        if not np.all(out[1:] > out[:-1]):
            raise ValueError("插入断点后不严格递增")
        return out

    @staticmethod
    def _index_of(si: np.ndarray, x: int) -> int:
        """x 必须在 si 中，返回其下标"""
        x = int(x)
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
            ring_min: str = "boundary",
            one: Any = 1,
            zero: Any = 0,
            fill: Any = 0,
    ) -> "EED":
        """
        限制到 x_axis >= a，其它区域赋 fill，并把多余的 0 区域裁掉（同步修改 S）。
        [l,r) 语义：x=a 被保留。
        """
        axis = int(axis)
        a = int(a)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")

        # 1) 插入断点并对齐
        S_align = [si.copy() for si in self.S]
        S_align[axis] = self._insert_breakpoint(S_align[axis], a)
        g = self.align_to(S_align, exp_approx=exp_approx, ring_min=ring_min, one=one, zero=zero)

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
        P_new[tuple(indexer2)] = fill

        # 4) 衰减率：左侧无效，设 alpha[axis]=0（可用 zero 适配 SMT）
        alpha2 = list(g.alpha)
        beta2 = list(g.beta)
        alpha2[axis] = zero

        return EED(S_new, P_new, alpha2, beta2)

    def restrict_lt(
            self,
            axis: int,
            b: int,
            *,
            exp_approx: str = "max",
            ring_min: str = "boundary",
            one: Any = 1,
            zero: Any = 0,
            fill: Any = 0,
    ) -> "EED":
        """
        限制到 x_axis < b，其它区域赋 fill，并把多余的 0 区域裁掉（同步修改 S）。
        [l,r) 语义：x=b 不被保留。
        """
        axis = int(axis)
        b = int(b)
        if not (0 <= axis < self.n):
            raise IndexError(f"axis 越界：{axis}")

        # 1) 插入断点并对齐
        S_align = [si.copy() for si in self.S]
        S_align[axis] = self._insert_breakpoint(S_align[axis], b)
        g = self.align_to(S_align, exp_approx=exp_approx, ring_min=ring_min, one=one, zero=zero)

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
        P_new[tuple(indexer2)] = fill

        # 4) 衰减率：右侧无效，设 beta[axis]=0
        alpha2 = list(g.alpha)
        beta2 = list(g.beta)
        beta2[axis] = zero

        return EED(S_new, P_new, alpha2, beta2)

    def restrict_interval(
            self,
            axis: int,
            intervals,
            *,
            exp_approx: str = "max",
            ring_min: str = "boundary",
            one: Any = 1,
            zero: Any = 0,
            fill: Any = 0,
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
            return self.restrict_ge(axis, 0, fill=fill).restrict_lt(axis, 0, fill=fill)

        res = None
        for lo, hi in intervals:
            cur = self
            if lo is not None:
                cur = cur.restrict_ge(axis, lo, exp_approx=exp_approx, ring_min=ring_min, one=one, zero=zero, fill=fill)
            if hi is not None:
                cur = cur.restrict_lt(axis, hi, exp_approx=exp_approx, ring_min=ring_min, one=one, zero=zero, fill=fill)
            if res is None:
                res = cur
            else:
                res = EED.add(res, cur, interval=EED.DefaultInterval, max_function=max_function)

        return res

    def times_constant(self, constant: float) -> "EED":
        return EED(self.S, self.P * constant, self.alpha, self.beta)

    def add_constant(self, axis, c):
        """
        add a constant in an axis.
        """
        S_new = [arr.copy() for arr in self.S]
        si = S_new[axis]
        S_new[axis] = [x + c for x in si]
        return EED(S_new, self.P, self.alpha, self.beta)

if __name__ == '__main__':

    a = EED([[-1, 1, 4], [2, 3]], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.3, 0.2, 0.1]], [0.25, 0.5], [0.5, 0.25])
    print(a.S)
    print(a.P)
    b = a
    print(b.S)
    print(b.P)
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
