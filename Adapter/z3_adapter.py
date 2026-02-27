import numpy as np
import z3

from EED import EED
from Adapter.adapter import Adapter


class Z3Adapter(Adapter):
    def build_leq(self, eed_constant: EED, S, *, name_prefix: str = "x", solver=None):
        if len(S) != eed_constant.n:
            raise ValueError("The dimensions of S and eed_constant do not match.")

        solver = solver or z3.Solver()

        spatial_shape = tuple(len(si) + 1 for si in S)
        extra_shape = eed_constant.P.shape[eed_constant.n :]
        full_shape = spatial_shape + extra_shape

        P_z3 = np.empty(full_shape, dtype=object)
        for idx in np.ndindex(full_shape):
            name = "{}_p_{}".format(name_prefix, "_".join(map(str, idx)))
            P_z3[idx] = z3.Real(name)

        alpha = [z3.Real(f"{name_prefix}_alpha_{i}") for i in range(eed_constant.n)]
        beta = [z3.Real(f"{name_prefix}_beta_{i}") for i in range(eed_constant.n)]

        eed_z3 = EED(S, P_z3, alpha, beta)

        for a in alpha:
            solver.add(a >= 0, a < 1)
        for b in beta:
            solver.add(b >= 0, b < 1)

        c = EED.leq(
            eed_constant,
            eed_z3,
            return_list=False,
            and_function=z3.And,
            true=z3.BoolVal(True),
            false=z3.BoolVal(False),
        )
        solver.add(c)

        return eed_z3, solver

    def solve(self, eed_z3: EED, solver) -> EED:
        if solver.check() != z3.sat:
            raise RuntimeError(f"Solver did not return SAT, but {solver.check()}")

        model = solver.model()

        def eval_or_zero(v):
            if not z3.is_expr(v):
                try:
                    return float(v)
                except Exception:
                    return 0.0
            if model.eval(v, model_completion=True) is None:
                return 0.0
            val = model.eval(v, model_completion=True)
            if z3.is_algebraic_value(val):
                return float(val.approx(20))
            if z3.is_rational_value(val):
                return float(val.numerator_as_long()) / float(val.denominator_as_long())
            try:
                return float(val.as_decimal(20).replace("?", ""))
            except Exception:
                return 0.0

        P_val = np.vectorize(eval_or_zero, otypes=[float])(eed_z3.P)
        alpha_val = [eval_or_zero(a) for a in eed_z3.alpha]
        beta_val = [eval_or_zero(b) for b in eed_z3.beta]

        return EED(eed_z3.S, P_val, alpha_val, beta_val)

    def max(self, a, b):
        return z3.If(a >= b, a, b)

    def restrict_leq(self, eed1, eed2, solver):
        c = EED.leq(
            eed1,
            eed2,
            return_list=False,
            and_function=z3.And,
            true=z3.BoolVal(True),
            false=z3.BoolVal(False),
        )
        solver.add(c)
        return solver
