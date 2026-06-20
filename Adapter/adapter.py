from dataclasses import dataclass
from fractions import Fraction

from distributions import BGD
from abc import ABC, abstractmethod
from Adapter.expr import Expr, Var, Const, CompareOp, Constraint, FractionConst, ensure_expr
from Adapter.expr import Add, Sub, Mul, Div, Max, Pow
import numpy as np

@dataclass
class AdapterEnvs:
    vars: dict
    constraints_list: list

class Adapter(ABC):
    @abstractmethod
    def build_var(self, name):
        raise NotImplementedError

    def get_var_expr(self, name, envs : AdapterEnvs) -> Expr:
        if name in envs.vars:
            raise KeyError(f"Variable {name} already exists")
        envs.vars[name] = self.build_var(name)
        return Var(name)

    def ensure_var(self, x) -> any:
        return x

    def var_add(self, a, b):
        return self.ensure_var(a) + self.ensure_var(b)

    def var_sub(self, a, b):
        return self.ensure_var(a) - self.ensure_var(b)

    def var_mul(self, a, b):
        return self.ensure_var(a) * self.ensure_var(b)

    def var_div(self, a, b):
        return self.ensure_var(a) / self.ensure_var(b)

    def var_pow(self, a, b):
        return self.ensure_var(a) ** self.ensure_var(b)

    def safe_pow(self, a, b):
        if isinstance(b, (int, float, Fraction, np.number)):
            if b == 0:
                return self.ensure_var(1)
            if b == 1:
                return self.ensure_var(a)
        return self.var_pow(a, b)

    @abstractmethod
    def var_max(self, a, b):
        raise NotImplementedError

    def var_lt(self, a, b):
        return self.ensure_var(a) < self.ensure_var(b)

    def var_le(self, a, b):
        return self.ensure_var(a) <= self.ensure_var(b)

    def var_eq(self, a, b):
        return self.ensure_var(a) == self.ensure_var(b)

    def var_ne(self, a, b):
        return self.ensure_var(a) != self.ensure_var(b)

    def var_gt(self, a, b):
        return self.ensure_var(a) > self.ensure_var(b)

    def var_ge(self, a, b):
        return self.ensure_var(a) >= self.ensure_var(b)

    def walk_expr(self, expr : Expr, vars : dict):
        # if not isinstance(expr, Expr):
        #     return self.ensure_var(expr)
        if isinstance(expr, Var):
            return vars[expr.name]
        elif isinstance(expr, Const):
            return self.ensure_var(expr.value)
        elif isinstance(expr, FractionConst):
            return self.ensure_var(expr.value)
        elif isinstance(expr, Add):
            return self.var_add(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))
        elif isinstance(expr, Sub):
            return self.var_sub(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))
        elif isinstance(expr, Mul):
            return self.var_mul(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))
        elif isinstance(expr, Div):
            return self.var_div(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))
        elif isinstance(expr, Max):
            return self.var_max(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))
        elif isinstance(expr, Pow):
            return self.safe_pow(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))

        raise TypeError(expr)

    def walk_constraint(self, constraint: Constraint, vars : dict):
        match constraint.op:
            case CompareOp.LT: return self.var_lt(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))
            case CompareOp.LE: return self.var_le(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))
            case CompareOp.EQ: return self.var_eq(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))
            case CompareOp.NE: return self.var_ne(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))
            case CompareOp.GT: return self.var_gt(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))
            case CompareOp.GE: return self.var_ge(self.walk_expr(constraint.left, vars), self.walk_expr(constraint.right, vars))

        raise TypeError(constraint)

    def eval_expr(self, expr : Expr, vars : dict) -> float:
        # if not isinstance(expr, Expr):
        #     return float(expr)
        if isinstance(expr, Var):
            return vars[expr.name]
        elif isinstance(expr, Const):
            return expr.value
        elif isinstance(expr, FractionConst):
            return float(expr.value)
        elif isinstance(expr, Add):
            return self.eval_expr(expr.left, vars) + self.eval_expr(expr.right, vars)
        elif isinstance(expr, Sub):
            return self.eval_expr(expr.left, vars) - self.eval_expr(expr.right, vars)
        elif isinstance(expr, Mul):
            return self.eval_expr(expr.left, vars) * self.eval_expr(expr.right, vars)
        elif isinstance(expr, Div):
            return self.eval_expr(expr.left, vars) / self.eval_expr(expr.right, vars)
        elif isinstance(expr, Max):
            return max(self.eval_expr(expr.left, vars), self.eval_expr(expr.right, vars))
        elif isinstance(expr, Pow):
            return self.eval_expr(expr.left, vars) ** self.eval_expr(expr.right, vars)

        raise TypeError(expr)

    @abstractmethod
    def solve(self, vars, constraints, objective=None):
        raise NotImplementedError

    def _fresh_bgd_template(self, template: BGD, name_prefix: str, envs: AdapterEnvs) -> BGD:
        E_expr = np.empty(template.E.shape, dtype=object)
        for edge_index in np.ndindex(template.E.shape):
            mud = template.E[edge_index]
            P_expr = np.empty(mud.shape, dtype=object)
            for cell_index in np.ndindex(mud.shape):
                edge_name = "_".join(map(str, edge_index))
                cell_name = "_".join(map(str, cell_index))
                name = f"{name_prefix}_E_{edge_name}_p_{cell_name}"
                P_expr[cell_index] = self.get_var_expr(name, envs)
            E_expr[edge_index] = type(mud)(mud.S, P_expr)

        alpha = [
            self.get_var_expr(f"{name_prefix}_alpha_{i}", envs)
            for i in range(template.ndim)
        ]
        beta = [
            self.get_var_expr(f"{name_prefix}_beta_{i}", envs)
            for i in range(template.ndim)
        ]
        return BGD(E_expr, alpha, beta)

    def _print_bgd_template_shape_summary(self, template: BGD, name_prefix: str) -> None:
        def format_breakpoints(points):
            return "(" + ", ".join(str(point) for point in points) + ")"

        total_cells = 0
        print(f"[BGD template {name_prefix}] E tensor shape={template.E.shape}", flush=True)
        for edge_index in np.ndindex(template.E.shape):
            mud = template.E[edge_index]
            cell_count = int(np.prod(mud.shape, dtype=int)) if mud.shape else 1
            total_cells += cell_count
            shape_expr = " * ".join(str(size) for size in mud.shape) or "1"
            print(f"  E{edge_index} | {shape_expr} = {cell_count}", flush=True)
            if edge_index and edge_index[0] == 1:
                interval_counts = tuple(len(axis) - 1 for axis in mud.S)
                breakpoints = tuple(format_breakpoints(axis) for axis in mud.S)
                print(
                    f"    interval_counts={interval_counts}, S={breakpoints}",
                    flush=True,
                )

        decay_vars = 2 * template.ndim
        print(
            f"[BGD template {name_prefix}] "
            f"P_variables={total_cells}, decay_variables={decay_vars}, "
            f"template_variables={total_cells + decay_vars}",
            flush=True,
        )

    def _bgd_nonnegative_constraints(self, bgd_expr: BGD) -> list:
        constraints = []
        for dim in range(bgd_expr.ndim):
            constraints.append(0 <= bgd_expr.alpha[dim])
            constraints.append(bgd_expr.alpha[dim] < 1)
            constraints.append(0 <= bgd_expr.beta[dim])
            constraints.append(bgd_expr.beta[dim] < 1)

        for edge_index in np.ndindex(bgd_expr.E.shape):
            mud = bgd_expr.E[edge_index]
            for cell_index in np.ndindex(mud.shape):
                constraints.append(0 <= mud.P[cell_index])
        return constraints

    @staticmethod
    def _bgd_le_constraint(left, right, name: str):
        return ensure_expr(left) <= ensure_expr(right)

    def _bgd_le_constraints_same_frame(self, left: BGD, right: BGD) -> list:
        if left.ndim != right.ndim:
            raise ValueError("BGD dimensions do not match")
        if (
            left.center_lefts != right.center_lefts
            or left.center_rights != right.center_rights
            or left.left_lengths != right.left_lengths
            or left.right_lengths != right.right_lengths
        ):
            raise ValueError("BGD frames do not match")

        constraints = []
        for dim in range(left.ndim):
            constraints.append(self._bgd_le_constraint(left.alpha[dim], right.alpha[dim], f"alpha[{dim}]"))
            constraints.append(self._bgd_le_constraint(left.beta[dim], right.beta[dim], f"beta[{dim}]"))

        for edge_index in np.ndindex(left.E.shape):
            left_mud = left.E[edge_index]
            right_mud = right.E[edge_index]
            target_S = tuple(
                left_mud.S[dim]
                if left_mud.S[dim] == right_mud.S[dim]
                else self._merge_mud_breakpoints(left_mud.S[dim], right_mud.S[dim])
                for dim in range(left.ndim)
            )
            left_aligned = left_mud.align(target_S)
            right_aligned = right_mud.align(target_S)
            for cell_index in np.ndindex(left_aligned.shape):
                constraints.append(
                    self._bgd_le_constraint(
                        left_aligned.P[cell_index],
                        right_aligned.P[cell_index],
                        f"E{edge_index}.P{cell_index}",
                    )
                )
        return constraints

    @staticmethod
    def _merge_mud_breakpoints(left, right):
        from distributions.mud import merge_breakpoints

        return merge_breakpoints(left, right, preserve_dirac=True)

    def build_bgd_leq(
        self,
        bgd_constant: BGD,
        template: BGD = None,
        name_prefix="w",
        envs: AdapterEnvs = None,
    ):
        if not isinstance(bgd_constant, BGD):
            raise TypeError("bgd_constant must be a BGD")
        if template is None:
            template = bgd_constant
        if not isinstance(template, BGD):
            raise TypeError("template must be a BGD")
        if template.ndim != bgd_constant.ndim:
            raise ValueError("template and bgd_constant dimensions do not match")

        envs = envs or AdapterEnvs({}, [])
        template = self._close_template_frame_for_constant(bgd_constant, template)
        self._print_bgd_template_shape_summary(template, name_prefix)
        bgd_expr = self._fresh_bgd_template(template, name_prefix, envs)
        envs.constraints_list += self._bgd_nonnegative_constraints(bgd_expr)
        constant_aligned = bgd_constant.align_frame(
            bgd_expr.center_lefts,
            bgd_expr.center_rights,
            bgd_expr.left_lengths,
            bgd_expr.right_lengths,
        )
        envs.constraints_list += self._bgd_le_constraints_same_frame(
            constant_aligned,
            bgd_expr,
        )
        return bgd_expr, envs

    def _close_template_frame_for_constant(self, bgd_constant: BGD, template: BGD) -> BGD:
        result = template
        for _ in range(4):
            aligned = bgd_constant.align_frame(
                result.center_lefts,
                result.center_rights,
                result.left_lengths,
                result.right_lengths,
            )
            if (
                aligned.center_lefts == result.center_lefts
                and aligned.center_rights == result.center_rights
                and aligned.left_lengths == result.left_lengths
                and aligned.right_lengths == result.right_lengths
            ):
                return result
            result = result.align_frame(
                aligned.center_lefts,
                aligned.center_rights,
                aligned.left_lengths,
                aligned.right_lengths,
            )
        return result

    def restrict_leq_bgd(self, bgd1: BGD, bgd2: BGD, envs: AdapterEnvs) -> AdapterEnvs:
        envs.constraints_list += bgd1.le_constraints(
            bgd2,
            constraint_factory=self._bgd_le_constraint,
        )
        return envs

    def _eval_payload(self, value, solved_vars: dict):
        if isinstance(value, Expr):
            return self.eval_expr(value, solved_vars)
        return value

    def _eval_bgd_expr_with_vars(self, bgd_expr: BGD, solved_vars: dict) -> BGD:
        E_val = np.empty(bgd_expr.E.shape, dtype=object)
        for edge_index in np.ndindex(bgd_expr.E.shape):
            mud = bgd_expr.E[edge_index]
            P_val = np.empty(mud.shape, dtype=object)
            for cell_index in np.ndindex(mud.shape):
                P_val[cell_index] = self._eval_payload(mud.P[cell_index], solved_vars)
            E_val[edge_index] = type(mud)(mud.S, P_val)

        alpha_val = [
            self._eval_payload(alpha, solved_vars)
            for alpha in bgd_expr.alpha
        ]
        beta_val = [
            self._eval_payload(beta, solved_vars)
            for beta in bgd_expr.beta
        ]
        return BGD(E_val, alpha_val, beta_val)

    def solve_bgd_expr(self, bgd_expr: BGD, envs: AdapterEnvs) -> BGD:
        solved_vars = self.solve(
            envs.vars,
            [self.walk_constraint(r, envs.vars) for r in envs.constraints_list],
        )
        return self._eval_bgd_expr_with_vars(bgd_expr, solved_vars)
