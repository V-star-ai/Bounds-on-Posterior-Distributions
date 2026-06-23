import math
import time
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
    ensure_expr,
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
        compile_expr: bool = True,
        profile: bool = True,
    ):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.constraint_eps = float(constraint_eps)
        self.smooth_max_eps = float(smooth_max_eps)
        self.fd_eps = float(fd_eps)
        self.print_level = int(print_level)
        self.compile_expr = bool(compile_expr)
        self.profile = bool(profile)
        self.last_stats = {}
        self._active_stats = None

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
        return self._eval_expr_cached(expr, x, var_index, {})

    def _eval_expr_cached(
        self,
        expr: Expr,
        x: np.ndarray,
        var_index: Dict[str, int],
        cache: Dict[int, float],
    ) -> float:
        stats = self._active_stats
        if stats is not None:
            stats["recursive_expr_calls"] += 1
        key = id(expr)
        if key in cache:
            if stats is not None:
                stats["recursive_cache_hits"] += 1
            return cache[key]

        if isinstance(expr, Var):
            value = float(x[var_index[expr.name]])
        elif isinstance(expr, Const):
            value = float(expr.value)
        elif isinstance(expr, FractionConst):
            value = float(expr.value)
        elif isinstance(expr, Add):
            value = self._eval_expr_cached(
                expr.left, x, var_index, cache
            ) + self._eval_expr_cached(expr.right, x, var_index, cache)
        elif isinstance(expr, Sub):
            value = self._eval_expr_cached(
                expr.left, x, var_index, cache
            ) - self._eval_expr_cached(expr.right, x, var_index, cache)
        elif isinstance(expr, Mul):
            value = self._eval_expr_cached(
                expr.left, x, var_index, cache
            ) * self._eval_expr_cached(expr.right, x, var_index, cache)
        elif isinstance(expr, Div):
            value = self._eval_expr_cached(
                expr.left, x, var_index, cache
            ) / self._eval_expr_cached(expr.right, x, var_index, cache)
        elif isinstance(expr, Pow):
            value = self._eval_expr_cached(
                expr.left, x, var_index, cache
            ) ** self._eval_expr_cached(expr.right, x, var_index, cache)
        elif isinstance(expr, Max):
            value = self._smooth_max(
                self._eval_expr_cached(expr.left, x, var_index, cache),
                self._eval_expr_cached(expr.right, x, var_index, cache),
            )
        else:
            raise TypeError(expr)

        cache[key] = value
        return value

    def _compile_expr_source(self, result_expr, var_index: Dict[str, int], name: str):
        memo = {}
        lines = ["def _compiled(x, _smooth_max):"]
        counter = 0

        def emit(expr: Expr) -> str:
            nonlocal counter
            key = id(expr)
            if key in memo:
                return memo[key]

            if isinstance(expr, Var):
                return f"x[{var_index[expr.name]}]"
            if isinstance(expr, Const):
                return repr(float(expr.value))
            if isinstance(expr, FractionConst):
                return repr(float(expr.value))

            left = emit(expr.left)
            right = emit(expr.right)
            var_name = f"v{counter}"
            counter += 1

            if isinstance(expr, Add):
                source = f"{left}+{right}"
            elif isinstance(expr, Sub):
                source = f"{left}-{right}"
            elif isinstance(expr, Mul):
                source = f"{left}*{right}"
            elif isinstance(expr, Div):
                source = f"{left}/{right}"
            elif isinstance(expr, Pow):
                source = f"{left}**{right}"
            elif isinstance(expr, Max):
                source = f"_smooth_max({left},{right})"
            else:
                raise TypeError(expr)

            lines.append(f"    {var_name} = {source}")
            memo[key] = var_name
            return var_name

        result = emit(result_expr)
        lines.append(f"    return float({result})")
        source = "\n".join(lines)
        if len(source) > 5_000_000:
            raise ValueError(f"{name} compiled expression is too large")
        namespace = {}
        exec(
            compile(source, f"<ipopt:{name}>", "exec"),
            {"__builtins__": {}, "float": float},
            namespace,
        )
        fn = namespace["_compiled"]
        return lambda x: fn(x, self._smooth_max)

    def _compile_expr_callable(self, expr: Expr, var_index: Dict[str, int], name: str):
        return self._compile_expr_source(expr, var_index, name)

    def _compile_constraint_callable(
        self,
        spec: _ConstraintSpec,
        var_index: Dict[str, int],
        name: str,
    ):
        return self._compile_expr_source(Sub(spec.left, spec.right), var_index, name)

    def _new_stats(self):
        return {
            "objective_calls": 0,
            "gradient_calls": 0,
            "constraints_calls": 0,
            "jacobian_calls": 0,
            "compiled_expr_calls": 0,
            "recursive_expr_calls": 0,
            "recursive_cache_hits": 0,
            "objective_time": 0.0,
            "gradient_time": 0.0,
            "constraints_time": 0.0,
            "jacobian_time": 0.0,
            "compile_time": 0.0,
            "compiled_constraints": 0,
            "compiled_mode": False,
        }

    def _print_stats(self, stats):
        if not self.profile:
            return
        print(
            "[IpoptAdapter stats] callbacks: "
            f"objective={stats['objective_calls']}, "
            f"gradient={stats['gradient_calls']}, "
            f"constraints={stats['constraints_calls']}, "
            f"jacobian={stats['jacobian_calls']}",
            flush=True,
        )
        print(
            "[IpoptAdapter stats] eval: "
            f"compiled_mode={stats['compiled_mode']}, "
            f"compiled_constraints={stats['compiled_constraints']}, "
            f"compiled_expr_calls={stats['compiled_expr_calls']}, "
            f"recursive_expr_calls={stats['recursive_expr_calls']}, "
            f"recursive_cache_hits={stats['recursive_cache_hits']}",
            flush=True,
        )
        print(
            "[IpoptAdapter stats] time_sec: "
            f"objective={stats['objective_time']:.4f}, "
            f"gradient={stats['gradient_time']:.4f}, "
            f"constraints={stats['constraints_time']:.4f}, "
            f"jacobian={stats['jacobian_time']:.4f}, "
            f"compile={stats['compile_time']:.4f}",
            flush=True,
        )

    def solve(self, vars, constraints, objective=None):
        try:
            import cyipopt  # type: ignore
        except Exception as exc:
            raise ImportError("cyipopt is required for IpoptAdapter") from exc
        stats = self._new_stats()
        self._active_stats = stats

        var_names = list(vars.keys())
        var_index = {name: i for i, name in enumerate(var_names)}
        n = len(var_names)
        has_objective = objective is not None
        objective_expr = ensure_expr(0 if objective is None else objective)

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
        print(
            f"[IpoptAdapter] variables={n}, constraints={m}, "
            f"raw_constraints={len(constraints)}, "
            f"objective={'provided' if has_objective else 'zero'}",
            flush=True,
        )

        compiled_objective = None
        compiled_constraints = None
        if self.compile_expr:
            compile_start = time.perf_counter()
            try:
                objective_failed = False
                failed_constraints = 0
                try:
                    compiled_objective = self._compile_expr_callable(
                        objective_expr,
                        var_index,
                        "objective",
                    )
                except Exception as exc:
                    objective_failed = True
                    compiled_objective = None
                    print(
                        f"[IpoptAdapter] objective compilation disabled: {exc}",
                        flush=True,
                    )
                compiled_constraints = []
                for i, spec in enumerate(constraint_specs):
                    try:
                        compiled_constraints.append(
                            self._compile_constraint_callable(
                                spec,
                                var_index,
                                f"constraint_{i}",
                            )
                        )
                    except Exception:
                        compiled_constraints.append(None)
                        failed_constraints += 1
                stats["compiled_constraints"] = sum(
                    1 for fn in compiled_constraints if fn is not None
                )
                stats["compiled_mode"] = (
                    compiled_objective is not None
                    or stats["compiled_constraints"] > 0
                )
                if objective_failed or failed_constraints:
                    print(
                        "[IpoptAdapter] expression compilation fallback: "
                        f"objective_failed={objective_failed}, "
                        f"constraints_failed={failed_constraints}",
                        flush=True,
                    )
            finally:
                stats["compile_time"] += time.perf_counter() - compile_start

        # Variable bounds (use wide bounds; tighten for alpha/beta for stability)
        lb = np.full(n, -1e20, dtype=float)
        ub = np.full(n, 1e20, dtype=float)
        decay_upper = 1.0 - max(self.constraint_eps, 10.0 * self.fd_eps)
        for i, name in enumerate(var_names):
            if "_alpha_" in name or "_beta_" in name:
                lb[i] = 0.0
                ub[i] = decay_upper

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
            stats["constraints_calls"] += 1
            start = time.perf_counter()
            if m == 0:
                result = np.zeros(0, dtype=float)
            elif compiled_constraints is not None:
                vals = np.zeros(m, dtype=float)
                cache: Dict[int, float] = {}
                for i, spec in enumerate(constraint_specs):
                    fn = compiled_constraints[i]
                    if fn is None:
                        left_val = self._eval_expr_cached(spec.left, x, var_index, cache)
                        right_val = self._eval_expr_cached(spec.right, x, var_index, cache)
                        vals[i] = left_val - right_val
                    else:
                        vals[i] = fn(x)
                        stats["compiled_expr_calls"] += 1
                result = vals
            else:
                vals = np.zeros(m, dtype=float)
                cache: Dict[int, float] = {}
                for i, spec in enumerate(constraint_specs):
                    left_val = self._eval_expr_cached(spec.left, x, var_index, cache)
                    right_val = self._eval_expr_cached(spec.right, x, var_index, cache)
                    vals[i] = left_val - right_val
                result = vals
            stats["constraints_time"] += time.perf_counter() - start
            return result

        def _objective(x):
            stats["objective_calls"] += 1
            start = time.perf_counter()
            if compiled_objective is not None:
                value = compiled_objective(x)
                stats["compiled_expr_calls"] += 1
            else:
                value = self._eval_expr_cached(objective_expr, x, var_index, {})
            stats["objective_time"] += time.perf_counter() - start
            return value

        def _finite_difference(fn, base, x, j):
            step = self.fd_eps
            if x[j] + step <= ub[j]:
                x2 = np.array(x, copy=True)
                x2[j] += step
                return (fn(x2) - base) / step
            if x[j] - step >= lb[j]:
                x2 = np.array(x, copy=True)
                x2[j] -= step
                return (base - fn(x2)) / step

            upper_step = ub[j] - x[j]
            lower_step = x[j] - lb[j]
            if upper_step > 0:
                x2 = np.array(x, copy=True)
                x2[j] = ub[j]
                return (fn(x2) - base) / upper_step
            if lower_step > 0:
                x2 = np.array(x, copy=True)
                x2[j] = lb[j]
                return (base - fn(x2)) / lower_step
            if np.isscalar(base):
                return 0.0
            return np.zeros_like(base, dtype=float)

        def _objective_gradient(x):
            stats["gradient_calls"] += 1
            start = time.perf_counter()
            grad = np.zeros(n, dtype=float)
            if n == 0:
                return grad
            base = _objective(x)
            for j in range(n):
                grad[j] = _finite_difference(_objective, base, x, j)
            stats["gradient_time"] += time.perf_counter() - start
            return grad

        def _jacobian(x):
            stats["jacobian_calls"] += 1
            start = time.perf_counter()
            if m == 0:
                return np.zeros(0, dtype=float)
            base = _constraints(x)
            jac = np.zeros((m, n), dtype=float)
            for j in range(n):
                jac[:, j] = _finite_difference(_constraints, base, x, j)
            result = jac.reshape(-1)
            stats["jacobian_time"] += time.perf_counter() - start
            return result

        class _Problem:
            def objective(self, x):
                return _objective(x)

            def gradient(self, x):
                return _objective_gradient(x)

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

        try:
            x, info = nlp.solve(x0)
        except Exception:
            self.last_stats = stats.copy()
            self._print_stats(stats)
            self._active_stats = None
            raise

        status = info.get("status") if isinstance(info, dict) else None
        if status not in (0, 1):
            self.last_stats = stats.copy()
            self._print_stats(stats)
            self._active_stats = None
            raise RuntimeError(f"Ipopt failed to solve (status={status}, info={info})")

        result = {name: float(x[idx]) for name, idx in var_index.items()}
        if has_objective:
            print(f"[IpoptAdapter] objective_value={_objective(x)}", flush=True)
        self.last_stats = stats.copy()
        self._print_stats(stats)
        self._active_stats = None
        return result

    def solve_bgd_expr(self, bgd_expr, envs):
        # Override to bypass walk_constraint and use raw Expr constraints.
        solved_vars = self.solve(
            envs.vars,
            envs.constraints_list,
            objective=bgd_expr.mass(),
        )
        return self._eval_bgd_expr_with_vars(bgd_expr, solved_vars)

    def var_max(self, a, b):
        # Not used in IpoptAdapter path (constraints evaluated from Expr directly).
        return self._smooth_max(a, b)
