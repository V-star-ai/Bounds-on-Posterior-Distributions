from dataclasses import dataclass
from typing import Tuple

from distributions import EED
from abc import ABC, abstractmethod
from Adapter.expr import Expr, Var, Const, CompareOp, Constraint, FractionConst
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

    def build_leq(self, eed_constant: EED, S, name_prefix = "w", envs : AdapterEnvs = None):
        if len(S) != eed_constant.n:
            raise ValueError("The dimensions of S and eed_constant do not match.")

        envs = envs or AdapterEnvs({}, [])

        spatial_shape = tuple((len(si) if is_dis else len(si) + 1) for (si, is_dis) in zip(S, eed_constant.discrete_mask))
        extra_shape = eed_constant.P.shape[eed_constant.n :]
        full_shape = spatial_shape + extra_shape

        P_expr = np.empty(full_shape, dtype=object)
        for idx in np.ndindex(full_shape):
            name = "{}_p_{}".format(name_prefix, "_".join(map(str, idx)))
            P_expr[idx] = self.get_var_expr(name, envs)

        alpha = [self.get_var_expr(f"{name_prefix}_alpha_{i}", envs) for i in range(eed_constant.n)]
        beta = [self.get_var_expr(f"{name_prefix}_beta_{i}", envs) for i in range(eed_constant.n)]

        eed_expr = EED(S, P_expr, alpha, beta, eed_constant.discrete_mask)

        for a in alpha:
            envs.constraints_list.append(0 <= a)
            envs.constraints_list.append(a < 1)
        for b in beta:
            envs.constraints_list.append(0 <= b)
            envs.constraints_list.append(b < 1)

        c = EED.leq(
            eed_constant,
            eed_expr,
        )
        
        envs.constraints_list += c

        return eed_expr, envs

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
            return self.var_pow(self.walk_expr(expr.left, vars), self.walk_expr(expr.right, vars))

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
    def solve(self, vars, constraints):
        raise NotImplementedError

    def solve_expr(self, eed_expr: EED, envs: AdapterEnvs) -> EED:
        solved_vars = self.solve(envs.vars, [self.walk_constraint(r, envs.vars) for r in envs.constraints_list])
        def expr_to_float(expr):
            return self.eval_expr(expr, solved_vars)
        P_val = np.vectorize(expr_to_float, otypes=[float])(eed_expr.P)
        alpha_val = [expr_to_float(a) for a in eed_expr.alpha]
        beta_val = [expr_to_float(b) for b in eed_expr.beta]
        return EED(eed_expr.S, P_val, alpha_val, beta_val)

    def restrict_leq(self, eed1, eed2, envs: AdapterEnvs) -> AdapterEnvs:
        c = EED.leq(
            eed1,
            eed2,
        )
        envs.constraints_list += c
        return envs

