import json
import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Adapter.adapter import Adapter
from Adapter.expr import (
    Add,
    CompareOp,
    Const,
    Constraint,
    Div,
    Expr,
    FractionConst,
    Max,
    Mul,
    Pow,
    Sub,
    Var,
    ensure_expr,
)


class RustAdapter(Adapter):
    """
    Adapter that sends the generated nonlinear problem to diabolo's Rust solver.

    The boundary is intentionally generic: Python keeps BGD construction and
    exports variable bounds, <= constraints, and the objective as JSON.
    """

    def __init__(
        self,
        *,
        diabolo_dir="diabolo",
        bin_name="solve_json",
        release=False,
        max_iter=500,
        tol=1e-6,
        constraint_eps=1e-8,
        constraint_margin=0.0,
        verbose=False,
        preprocess=False,
        save_problem=None,
    ):
        self.diabolo_dir = Path(diabolo_dir)
        self.bin_name = bin_name
        self.release = bool(release)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.constraint_eps = float(constraint_eps)
        self.constraint_margin = float(constraint_margin)
        self.verbose = bool(verbose)
        self.preprocess = bool(preprocess)
        self.save_problem = save_problem
        self._aux_counter = 0
        self._expr_nodes = []

    def build_var(self, name):
        return name

    def var_max(self, a, b):
        return max(a, b)

    def _is_decay_var(self, name: str) -> bool:
        return "_alpha_" in name or "_beta_" in name

    def _variable_spec(self, name: str, *, aux=False):
        if self._is_decay_var(name):
            decay_upper = 1.0 - max(self.constraint_eps, 10.0 * self.tol)
            return {
                "name": name,
                "lb": 0.0,
                "ub": decay_upper,
                "start": 0.5,
                "kind": "decay",
            }
        return {
            "name": name,
            "lb": -1e20 if aux else 0.0,
            "ub": 1e20,
            "start": 0.1,
            "kind": "aux" if aux else "block",
        }

    def _fresh_aux_name(self):
        name = f"__rust_aux_max_{self._aux_counter}"
        self._aux_counter += 1
        return name

    def _new_expr_node(self, node):
        index = len(self._expr_nodes)
        self._expr_nodes.append(node)
        return index

    def _export_expr(
        self,
        expr: Expr,
        variables: Dict[str, dict],
        extra_constraints: List[dict],
        expr_cache: Dict[int, Tuple[Expr, int]],
    ):
        expr = ensure_expr(expr)
        cache_key = id(expr)
        cached = expr_cache.get(cache_key)
        if cached is not None and cached[0] is expr:
            return cached[1]

        if isinstance(expr, Var):
            if expr.name not in variables:
                variables[expr.name] = self._variable_spec(expr.name)
            result = self._new_expr_node({"op": "var", "name": expr.name})
            expr_cache[cache_key] = (expr, result)
            return result
        if isinstance(expr, Const):
            result = self._new_expr_node({"op": "const", "value": float(expr.value)})
            expr_cache[cache_key] = (expr, result)
            return result
        if isinstance(expr, FractionConst):
            result = self._new_expr_node({"op": "const", "value": float(expr.value)})
            expr_cache[cache_key] = (expr, result)
            return result

        if isinstance(expr, Add):
            op = "add"
        elif isinstance(expr, Sub):
            op = "sub"
        elif isinstance(expr, Mul):
            op = "mul"
        elif isinstance(expr, Div):
            op = "div"
        elif isinstance(expr, Pow):
            base = self._export_expr(expr.left, variables, extra_constraints, expr_cache)
            exponent = expr.right
            if isinstance(exponent, FractionConst):
                exp_value = exponent.value
            elif isinstance(exponent, Const):
                exp_value = Fraction(exponent.value).limit_denominator()
            else:
                raise ValueError("RustAdapter only supports constant exponents")
            if exp_value.denominator == 1:
                result = self._new_expr_node(
                    {"op": "pow_int", "base": base, "exp": int(exp_value)}
                )
            else:
                result = self._new_expr_node(
                    {
                        "op": "pow_frac",
                        "base": base,
                        "numer": int(exp_value.numerator),
                        "denom": int(exp_value.denominator),
                    }
                )
            expr_cache[cache_key] = (expr, result)
            return result
        elif isinstance(expr, Max):
            left = self._export_expr(expr.left, variables, extra_constraints, expr_cache)
            right = self._export_expr(expr.right, variables, extra_constraints, expr_cache)
            aux_name = self._fresh_aux_name()
            variables[aux_name] = self._variable_spec(aux_name, aux=True)
            aux = self._new_expr_node({"op": "var", "name": aux_name})
            extra_constraints.append({"lhs": left, "rhs": aux})
            extra_constraints.append({"lhs": right, "rhs": aux})
            expr_cache[cache_key] = (expr, aux)
            return aux
        else:
            raise TypeError(expr)

        result = self._new_expr_node(
            {
                "op": op,
                "left": self._export_expr(
                    expr.left, variables, extra_constraints, expr_cache
                ),
                "right": self._export_expr(
                    expr.right, variables, extra_constraints, expr_cache
                ),
            }
        )
        expr_cache[cache_key] = (expr, result)
        return result

    def _append_le(self, constraints_json, left, right, variables, extra_constraints, expr_cache):
        constraints_json.append(
            {
                "lhs": self._export_expr(left, variables, extra_constraints, expr_cache),
                "rhs": self._export_expr(right, variables, extra_constraints, expr_cache),
            }
        )

    def _export_constraint(
        self,
        constraint,
        constraints_json,
        variables,
        extra_constraints,
        expr_cache,
    ):
        if isinstance(constraint, (bool, np.bool_)):
            if not bool(constraint):
                raise RuntimeError("Constraints are infeasible (constant False)")
            return
        if not isinstance(constraint, Constraint):
            raise TypeError(constraint)
        if constraint.op == CompareOp.LE:
            self._append_le(
                constraints_json,
                constraint.left,
                constraint.right,
                variables,
                extra_constraints,
                expr_cache,
            )
        elif constraint.op == CompareOp.LT:
            self._append_le(
                constraints_json,
                constraint.left,
                constraint.right - self.constraint_eps,
                variables,
                extra_constraints,
                expr_cache,
            )
        elif constraint.op == CompareOp.GE:
            self._append_le(
                constraints_json,
                constraint.right,
                constraint.left,
                variables,
                extra_constraints,
                expr_cache,
            )
        elif constraint.op == CompareOp.GT:
            self._append_le(
                constraints_json,
                constraint.right,
                constraint.left - self.constraint_eps,
                variables,
                extra_constraints,
                expr_cache,
            )
        elif constraint.op == CompareOp.EQ:
            self._append_le(
                constraints_json,
                constraint.left,
                constraint.right,
                variables,
                extra_constraints,
                expr_cache,
            )
            self._append_le(
                constraints_json,
                constraint.right,
                constraint.left,
                variables,
                extra_constraints,
                expr_cache,
            )
        elif constraint.op == CompareOp.NE:
            raise ValueError("RustAdapter does not support '!=' constraints")
        else:
            raise TypeError(constraint.op)

    def _build_problem_json(self, vars, constraints, objective):
        self._aux_counter = 0
        self._expr_nodes = []
        variables = {name: self._variable_spec(name) for name in vars.keys()}
        constraints_json = []
        extra_constraints = []
        expr_cache = {}
        for constraint in constraints:
            self._export_constraint(
                constraint,
                constraints_json,
                variables,
                extra_constraints,
                expr_cache,
            )
        objective_json = self._export_expr(
            ensure_expr(0 if objective is None else objective),
            variables,
            extra_constraints,
            expr_cache,
        )

        return {
            "variables": list(variables.values()),
            "exprs": self._expr_nodes,
            "constraints": constraints_json + extra_constraints,
            "objective": objective_json,
            "options": {
                "tol": self.tol,
                "max_iter": self.max_iter,
                "constraint_margin": self.constraint_margin,
                "verbose": self.verbose,
                "preprocess": self.preprocess,
            },
        }

    def solve(self, vars, constraints, objective=None):
        problem = self._build_problem_json(vars, constraints, objective)
        print(
            "[RustAdapter] "
            f"variables={len(problem['variables'])}, "
            f"exprs={len(problem['exprs'])}, "
            f"constraints={len(problem['constraints'])}, "
            f"objective={'provided' if objective is not None else 'zero'}",
            flush=True,
        )
        payload = json.dumps(problem)
        if self.save_problem:
            Path(self.save_problem).write_text(payload, encoding="utf-8")

        args = [
            "cargo",
            "run",
            "--quiet",
            "--no-default-features",
            "--bin",
            self.bin_name,
        ]
        if self.release:
            args = [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--no-default-features",
                "--bin",
                self.bin_name,
            ]
        proc = subprocess.run(
            args,
            input=payload,
            text=True,
            capture_output=True,
            cwd=self.diabolo_dir,
            check=False,
            env=self._subprocess_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Rust solver failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError(f"Rust solver produced no JSON output; stderr:\n{proc.stderr}")
        response = json.loads(stdout_lines[-1])
        print(
            "[RustAdapter] "
            f"status={response.get('status')}, "
            f"raw_objective={response.get('objective')}, "
            f"max_violation={response.get('max_violation')}, "
            f"worst_constraint={response.get('worst_constraint')}, "
            f"worst_lhs={response.get('worst_lhs')}, "
            f"worst_rhs={response.get('worst_rhs')}",
            flush=True,
        )
        if response.get("status") != "solved":
            raise RuntimeError(
                "Rust solver did not satisfy constraints: "
                f"status={response.get('status')}, "
                f"objective={response.get('objective')}, "
                f"max_violation={response.get('max_violation')}, "
                f"variables={len(problem['variables'])}, "
                f"exprs={len(problem['exprs'])}, "
                f"constraints={len(problem['constraints'])}"
            )
        values = response["values"]
        if any(not math.isfinite(float(values[name])) for name in vars.keys()):
            raise RuntimeError("Rust solver returned a non-finite value")
        result = {}
        for name in vars.keys():
            value = float(values[name])
            spec = self._variable_spec(name)
            if spec["lb"] == 0.0 and -self.tol <= value < 0.0:
                value = 0.0
            if self._is_decay_var(name):
                upper = 1.0 - max(self.constraint_eps, 10.0 * self.tol)
                if upper < value < 1.0:
                    value = upper
            result[name] = value
        if objective is not None:
            print(
                "[RustAdapter] "
                f"post_objective={self.eval_expr(objective, result)}",
                flush=True,
            )
        return result

    def _subprocess_env(self):
        env = os.environ.copy()
        conda_lib = Path(sys.prefix) / "lib"
        if conda_lib.exists():
            for key in ("LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
                existing = env.get(key)
                env[key] = (
                    f"{conda_lib}{os.pathsep}{existing}"
                    if existing
                    else str(conda_lib)
                )
        return env

    def solve_bgd_expr(self, bgd_expr, envs):
        solved_vars = self.solve(
            envs.vars,
            envs.constraints_list,
            objective=bgd_expr.mass(),
        )
        return self._eval_bgd_expr_with_vars(bgd_expr, solved_vars)
