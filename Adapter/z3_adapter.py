import z3

from Adapter.adapter import Adapter


class Z3Adapter(Adapter):
    def build_var(self, name):
        return z3.Real(name)

    def solve(self, vars, constraints):
        solver = z3.Solver()
        solver.add(*constraints)
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
        return {key : eval_or_zero(val) for key, val in vars.items()}

    def var_max(self, a, b):
        return z3.If(a >= b, a, b)
