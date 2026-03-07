import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from Adapter.adapter import Adapter
from Adapter.expr import (
    Expr,
    Var,
    Const,
    FractionConst,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Constraint,
    CompareOp,
)


@dataclass
class _ConstraintSpec:
    left: Expr
    right: Expr
    op: CompareOp


class IpoptAdapter(Adapter):
    """
    Ipopt adapter using cyipopt. It solves feasibility problems by minimizing 0
    subject to constraints derived from Expr.
    """

    def __init__(
        self,
        *,
        max_iter: int = 500,
        tol: float = 1e-6,
        constraint_eps: float = 1e-8,
        smooth_max_eps: float = 0.0,
        fd_eps: float = 1e-6,
        print_level: int = 0,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.constraint_eps = float(constraint_eps)
        self.smooth_max_eps = float(smooth_max_eps)
        self.fd_eps = float(fd_eps)
        self.print_level = int(print_level)

    def build_var(self, name):
        # Store name only; ipopt uses indexed vectors internally.
        return name

    def _smooth_max(self, a: float, b: float) -> float:
        # Use exact max by default to avoid over-approximation that can
        # introduce artificial infeasibility. If smooth_max_eps > 0,
        # fall back to a differentiable approximation.
        if self.smooth_max_eps > 0.0:
            return 0.5 * (a + b + math.sqrt((a - b) ** 2 + self.smooth_max_eps))
        return a if a >= b else b

    def _eval_expr(self, expr: Expr, x: np.ndarray, var_index: Dict[str, int]) -> float:
        if isinstance(expr, Var):
            return float(x[var_index[expr.name]])
        if isinstance(expr, Const):
            return float(expr.value)
        if isinstance(expr, FractionConst):
            return float(expr.value)
        if isinstance(expr, Add):
            return self._eval_expr(expr.left, x, var_index) + self._eval_expr(expr.right, x, var_index)
        if isinstance(expr, Sub):
            return self._eval_expr(expr.left, x, var_index) - self._eval_expr(expr.right, x, var_index)
        if isinstance(expr, Mul):
            return self._eval_expr(expr.left, x, var_index) * self._eval_expr(expr.right, x, var_index)
        if isinstance(expr, Div):
            return self._eval_expr(expr.left, x, var_index) / self._eval_expr(expr.right, x, var_index)
        if isinstance(expr, Pow):
            return self._eval_expr(expr.left, x, var_index) ** self._eval_expr(expr.right, x, var_index)
        if isinstance(expr, Max):
            return self._smooth_max(
                self._eval_expr(expr.left, x, var_index),
                self._eval_expr(expr.right, x, var_index),
            )
        raise TypeError(expr)

    def solve(self, vars, constraints):
        try:
            import cyipopt  # type: ignore
        except Exception as exc:
            raise ImportError("cyipopt is required for IpoptAdapter") from exc

        var_names = list(vars.keys())
        var_index = {name: i for i, name in enumerate(var_names)}
        n = len(var_names)

        constraint_specs: List[_ConstraintSpec] = []
        for c in constraints:
            if isinstance(c, (bool, np.bool_)):
                if not bool(c):
                    raise RuntimeError("Constraints are infeasible (constant False)")
                continue
            if isinstance(c, Constraint):
                if c.op == CompareOp.NE:
                    raise ValueError("IpoptAdapter does not support '!=' constraints")
                constraint_specs.append(_ConstraintSpec(c.left, c.right, c.op))
                continue
            raise TypeError(c)

        m = len(constraint_specs)

        # Variable bounds (use wide bounds; tighten for alpha/beta for stability)
        lb = np.full(n, -1e20, dtype=float)
        ub = np.full(n, 1e20, dtype=float)
        for i, name in enumerate(var_names):
            if "_alpha_" in name or "_beta_" in name:
                lb[i] = 0.0
                ub[i] = 1.0 - self.constraint_eps

        # Constraint bounds
        cl = np.full(m, -1e20, dtype=float)
        cu = np.full(m, 1e20, dtype=float)
        for i, spec in enumerate(constraint_specs):
            if spec.op == CompareOp.LE:
                cu[i] = 0.0
            elif spec.op == CompareOp.LT:
                cu[i] = -self.constraint_eps
            elif spec.op == CompareOp.GE:
                cl[i] = 0.0
            elif spec.op == CompareOp.GT:
                cl[i] = self.constraint_eps
            elif spec.op == CompareOp.EQ:
                cl[i] = 0.0
                cu[i] = 0.0
            else:
                raise TypeError(spec.op)

        def _constraints(x):
            if m == 0:
                return np.zeros(0, dtype=float)
            vals = np.zeros(m, dtype=float)
            for i, spec in enumerate(constraint_specs):
                left_val = self._eval_expr(spec.left, x, var_index)
                right_val = self._eval_expr(spec.right, x, var_index)
                vals[i] = left_val - right_val
            return vals

        def _jacobian(x):
            if m == 0:
                return np.zeros(0, dtype=float)
            base = _constraints(x)
            jac = np.zeros((m, n), dtype=float)
            step = self.fd_eps
            for j in range(n):
                x2 = np.array(x, copy=True)
                x2[j] += step
                jac[:, j] = (_constraints(x2) - base) / step
            return jac.reshape(-1)

        class _Problem:
            def objective(self, x):
                return 0.0

            def gradient(self, x):
                return np.zeros(n, dtype=float)

            def constraints(self, x):
                return _constraints(x)

            def jacobian(self, x):
                return _jacobian(x)

            def jacobianstructure(self):
                if m == 0 or n == 0:
                    return (np.array([], dtype=int), np.array([], dtype=int))
                rows, cols = np.nonzero(np.ones((m, n), dtype=int))
                return rows, cols

            def hessian(self, x, lagrange, obj_factor):
                return np.zeros(0, dtype=float)

            def hessianstructure(self):
                return (np.array([], dtype=int), np.array([], dtype=int))

        x0 = np.zeros(n, dtype=float)
        for i, name in enumerate(var_names):
            if "_alpha_" in name or "_beta_" in name:
                x0[i] = 0.5
            else:
                x0[i] = 0.1

        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=_Problem(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
        nlp.add_option("max_iter", self.max_iter)
        nlp.add_option("tol", self.tol)
        nlp.add_option("print_level", self.print_level)
        nlp.add_option("hessian_approximation", "limited-memory")

        x, info = nlp.solve(x0)

        status = info.get("status") if isinstance(info, dict) else None
        if status not in (0, 1):
            raise RuntimeError(f"Ipopt failed to solve (status={status}, info={info})")

        return {name: float(x[idx]) for name, idx in var_index.items()}

    def solve_expr(self, eed_expr, envs):
        # Override to bypass walk_constraint and use raw Expr constraints.
        solved_vars = self.solve(envs.vars, envs.constraints_list)

        def expr_to_float(expr):
            return self.eval_expr(expr, solved_vars)

        P_val = np.vectorize(expr_to_float, otypes=[float])(eed_expr.P)
        alpha_val = [expr_to_float(a) for a in eed_expr.alpha]
        beta_val = [expr_to_float(b) for b in eed_expr.beta]
        return type(eed_expr)(eed_expr.S, P_val, alpha_val, beta_val)

    def var_max(self, a, b):
        # Not used in IpoptAdapter path (constraints evaluated from Expr directly).
        return self._smooth_max(a, b)
