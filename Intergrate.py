from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Optional, Set, Tuple

import attr

from probably.pgcl.ast import Program, RealType
from probably.pgcl.ast.expressions import (
    Binop,
    BinopExpr,
    Expr,
    NatLitExpr,
    RealLitExpr,
    UnopExpr,
    VarExpr,
)
from probably.pgcl.ast.instructions import (
    AsgnInstr,
    ChoiceInstr,
    IfInstr,
    Instr,
    LoopInstr,
    ObserveInstr,
    TickInstr,
    WhileInstr,
)


# -------------------------
# small math helpers
# -------------------------
def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return a // gcd(a, b) * b


def lcm_many(nums: Iterable[int]) -> int:
    res = 1
    for n in nums:
        res = lcm(res, n)
    return res


def is_numeric(e: Expr) -> bool:
    return isinstance(e, (NatLitExpr, RealLitExpr))


def as_fraction(e: Expr) -> Optional[Fraction]:
    """Evaluate a constant numeric expression to Fraction if possible."""
    if isinstance(e, NatLitExpr):
        return Fraction(e.value, 1)
    if isinstance(e, RealLitExpr):
        if e.is_infinite():
            return None
        return e.to_fraction()
    if isinstance(e, BinopExpr):
        a = as_fraction(e.lhs)
        b = as_fraction(e.rhs)
        if a is None or b is None:
            return None
        op = e.operator
        if op == Binop.PLUS:
            return a + b
        if op == Binop.MINUS:
            return a - b
        if op == Binop.TIMES:
            return a * b
        if op == Binop.DIVIDE:
            if b == 0:
                return None
            return a / b
        if op == Binop.POWER:
            # only support integer exponent
            if b.denominator != 1:
                return None
            exp = b.numerator
            if exp < 0:
                return None
            return a ** exp
        # modulo and boolean/comparisons are not numeric constants
        return None
    # UnopExpr in this library is used for "not"/Iverson, not numeric unary minus. :contentReference[oaicite:1]{index=1}
    return None


def lit_from_fraction(fr: Fraction) -> Expr:
    """Create NatLitExpr if integer else RealLitExpr as 'p/q'."""
    if fr.denominator == 1:
        n = fr.numerator
        return NatLitExpr(int(n)) if n >= 0 else RealLitExpr(str(fr))  # keep negatives as real fraction
    return RealLitExpr(str(fr))


# -------------------------
# substitution + simplification
# -------------------------
def subst_vars(expr: Expr, scales: Dict[str, int]) -> Expr:
    """
    Replace VarExpr(x) with (x / kx) if x has a scale.
    """
    if isinstance(expr, VarExpr):
        k = scales.get(expr.var, 1)
        if k == 1:
            return expr
        return BinopExpr(Binop.DIVIDE, expr, NatLitExpr(k))

    if isinstance(expr, BinopExpr):
        return BinopExpr(expr.operator, subst_vars(expr.lhs, scales), subst_vars(expr.rhs, scales))

    if isinstance(expr, UnopExpr):
        return UnopExpr(expr.operator, subst_vars(expr.expr, scales))

    # other Expr kinds (distributions, subst expr, etc.) are left structurally intact,
    # but their children (if any) will be handled by the generic instruction transform where needed.
    return expr


def simplify(expr: Expr) -> Expr:
    """
    Light constant folding / neutral element simplification.
    """
    if isinstance(expr, BinopExpr):
        lhs = simplify(expr.lhs)
        rhs = simplify(expr.rhs)
        op = expr.operator

        # constant fold
        if is_numeric(lhs) and is_numeric(rhs) and op in {Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.DIVIDE, Binop.POWER}:
            fr = as_fraction(BinopExpr(op, lhs, rhs))
            if fr is not None:
                return lit_from_fraction(fr)

        # neutral elements
        if op == Binop.TIMES:
            if isinstance(lhs, NatLitExpr) and lhs.value == 1:
                return rhs
            if isinstance(rhs, NatLitExpr) and rhs.value == 1:
                return lhs
            if isinstance(lhs, NatLitExpr) and lhs.value == 0:
                return lhs
            if isinstance(rhs, NatLitExpr) and rhs.value == 0:
                return rhs

        if op == Binop.PLUS:
            if isinstance(lhs, NatLitExpr) and lhs.value == 0:
                return rhs
            if isinstance(rhs, NatLitExpr) and rhs.value == 0:
                return lhs

        if op == Binop.MINUS:
            if isinstance(rhs, NatLitExpr) and rhs.value == 0:
                return lhs

        if op == Binop.DIVIDE:
            if isinstance(rhs, NatLitExpr) and rhs.value == 1:
                return lhs

        return BinopExpr(op, lhs, rhs)

    if isinstance(expr, UnopExpr):
        return UnopExpr(expr.operator, simplify(expr.expr))

    return expr


def mul_int(m: int, expr: Expr) -> Expr:
    """
    Build m * expr, distributing over +/-, and simplifying division by integers when possible.
    """
    assert m >= 1
    if m == 1:
        return expr

    expr = simplify(expr)

    if isinstance(expr, NatLitExpr):
        return NatLitExpr(m * expr.value)

    if isinstance(expr, RealLitExpr):
        fr = expr.to_fraction() * m
        return lit_from_fraction(fr)

    if isinstance(expr, BinopExpr):
        op = expr.operator

        if op in {Binop.PLUS, Binop.MINUS}:
            return simplify(BinopExpr(op, mul_int(m, expr.lhs), mul_int(m, expr.rhs)))

        # m * (a / d)
        if op == Binop.DIVIDE and isinstance(expr.rhs, NatLitExpr):
            d = expr.rhs.value
            g = gcd(m, d)
            new_num = mul_int(m // g, expr.lhs)
            new_den = d // g
            if new_den == 1:
                return simplify(new_num)
            return simplify(BinopExpr(Binop.DIVIDE, new_num, NatLitExpr(new_den)))

        # m * (c * a)  or  m * (a * c)
        if op == Binop.TIMES:
            if is_numeric(expr.lhs):
                fr = as_fraction(expr.lhs)
                if fr is not None:
                    return simplify(BinopExpr(Binop.TIMES, lit_from_fraction(fr * m), expr.rhs))
            if is_numeric(expr.rhs):
                fr = as_fraction(expr.rhs)
                if fr is not None:
                    return simplify(BinopExpr(Binop.TIMES, expr.lhs, lit_from_fraction(fr * m)))

    return simplify(BinopExpr(Binop.TIMES, NatLitExpr(m), expr))


# -------------------------
# scale collection
# -------------------------
def vars_in_expr(expr: Expr, real_vars: Set[str]) -> Set[str]:
    if isinstance(expr, VarExpr):
        return {expr.var} if expr.var in real_vars else set()
    if isinstance(expr, BinopExpr):
        return vars_in_expr(expr.lhs, real_vars) | vars_in_expr(expr.rhs, real_vars)
    if isinstance(expr, UnopExpr):
        return vars_in_expr(expr.expr, real_vars)
    return set()


def collect_denoms_in_expr(expr: Expr) -> Set[int]:
    denoms: Set[int] = set()

    if isinstance(expr, RealLitExpr) and not expr.is_infinite():
        denoms.add(expr.to_fraction().denominator)
        return denoms

    if isinstance(expr, NatLitExpr):
        return denoms

    if isinstance(expr, BinopExpr):
        denoms |= collect_denoms_in_expr(expr.lhs)
        denoms |= collect_denoms_in_expr(expr.rhs)
        return denoms

    if isinstance(expr, UnopExpr):
        denoms |= collect_denoms_in_expr(expr.expr)
        return denoms

    return denoms


def collect_scales(program: Program) -> Dict[str, int]:
    # Only scale REAL-typed variables (your examples都是 real). :contentReference[oaicite:2]{index=2}
    real_vars = {v for v, t in program.variables.items() if isinstance(t, RealType)}
    per_var_denoms: Dict[str, Set[int]] = {v: set() for v in real_vars}

    def note(vs: Set[str], denom: int) -> None:
        if denom <= 1:
            return
        for v in vs:
            if v in per_var_denoms:
                per_var_denoms[v].add(denom)

    def walk_expr_for_assoc(expr: Expr) -> Set[str]:
        """
        Traverse expr; when a numeric constant is combined/compared with vars,
        associate that constant's denominator to those vars.
        """
        if isinstance(expr, VarExpr):
            return {expr.var} if expr.var in real_vars else set()

        if isinstance(expr, (NatLitExpr, RealLitExpr)):
            return set()

        if isinstance(expr, UnopExpr):
            return walk_expr_for_assoc(expr.expr)

        if isinstance(expr, BinopExpr):
            vl = walk_expr_for_assoc(expr.lhs)
            vr = walk_expr_for_assoc(expr.rhs)

            # If one side has vars and the other side is a pure numeric constant expression, bind denom to vars.
            if expr.operator in {
                Binop.PLUS, Binop.MINUS, Binop.TIMES, Binop.DIVIDE,
                Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ, Binop.EQ,
            }:
                if vl and not vr:
                    fr = as_fraction(expr.rhs)
                    if fr is not None:
                        note(vl, fr.denominator)
                if vr and not vl:
                    fr = as_fraction(expr.lhs)
                    if fr is not None:
                        note(vr, fr.denominator)

            return vl | vr

        return set()

    def walk_instr(instr: Instr) -> None:
        if isinstance(instr, AsgnInstr):
            # assignment: all numeric denoms in rhs must be supported by lhs's scale
            if instr.lhs in real_vars:
                for d in collect_denoms_in_expr(instr.rhs):
                    note({instr.lhs}, d)
            return

        if isinstance(instr, WhileInstr):
            walk_expr_for_assoc(instr.cond)
            for s in instr.body:
                walk_instr(s)
            return

        if isinstance(instr, IfInstr):
            walk_expr_for_assoc(instr.cond)
            for s in instr.true:
                walk_instr(s)
            for s in instr.false:
                walk_instr(s)
            return

        if isinstance(instr, ObserveInstr):
            walk_expr_for_assoc(instr.cond)
            return

        if isinstance(instr, ChoiceInstr):
            # IMPORTANT: do not treat prob as "needs scaling" (it's a probability).
            for s in instr.lhs:
                walk_instr(s)
            for s in instr.rhs:
                walk_instr(s)
            return

        if isinstance(instr, LoopInstr):
            for s in instr.body:
                walk_instr(s)
            return

        # other instr kinds: ignore
        return

    for s in program.instructions:
        walk_instr(s)

    scales: Dict[str, int] = {}
    for v, ds in per_var_denoms.items():
        scales[v] = lcm_many(ds) if ds else 1
    return scales


# -------------------------
# main transformer
# -------------------------
def transform_condition(cond: Expr, scales: Dict[str, int], real_vars: Set[str]) -> Expr:
    """
    Transform boolean conditions:
    - for AND/OR/NOT: recurse
    - for comparisons: multiply both sides by L = lcm(scales of vars in that comparison)
    """
    if isinstance(cond, UnopExpr):
        return UnopExpr(cond.operator, transform_condition(cond.expr, scales, real_vars))

    if isinstance(cond, BinopExpr):
        op = cond.operator

        if op in {Binop.AND, Binop.OR}:
            return BinopExpr(op,
                             transform_condition(cond.lhs, scales, real_vars),
                             transform_condition(cond.rhs, scales, real_vars))

        if op in {Binop.LEQ, Binop.LT, Binop.GT, Binop.GEQ, Binop.EQ}:
            vs = vars_in_expr(cond.lhs, real_vars) | vars_in_expr(cond.rhs, real_vars)
            L = lcm_many(scales.get(v, 1) for v in vs) if vs else 1

            lhs = subst_vars(cond.lhs, scales)
            rhs = subst_vars(cond.rhs, scales)

            lhs2 = mul_int(L, lhs)
            rhs2 = mul_int(L, rhs)
            return BinopExpr(op, simplify(lhs2), simplify(rhs2))

    # fallback: leave as is
    return cond


def transform_rhs_for_lhs(lhs: str, rhs: Expr, scales: Dict[str, int]) -> Expr:
    k = scales.get(lhs, 1)
    if k == 1:
        return rhs
    # rhs' = k * rhs(x/k)
    rhs_sub = subst_vars(rhs, scales)
    return simplify(mul_int(k, rhs_sub))


def align_program_constants_to_integers(program: Program) -> Tuple[Program, Dict[str, int]]:
    scales = collect_scales(program)
    real_vars = {v for v, t in program.variables.items() if isinstance(t, RealType)}

    def tr_instr(instr: Instr) -> Instr:
        if isinstance(instr, AsgnInstr):
            if instr.lhs in real_vars:
                new_rhs = transform_rhs_for_lhs(instr.lhs, instr.rhs, scales)
                return attr.evolve(instr, rhs=new_rhs)
            return instr

        if isinstance(instr, WhileInstr):
            new_cond = transform_condition(instr.cond, scales, real_vars)
            new_body = [tr_instr(s) for s in instr.body]
            return attr.evolve(instr, cond=new_cond, body=new_body)

        if isinstance(instr, IfInstr):
            new_cond = transform_condition(instr.cond, scales, real_vars)
            new_true = [tr_instr(s) for s in instr.true]
            new_false = [tr_instr(s) for s in instr.false]
            return attr.evolve(instr, cond=new_cond, true=new_true, false=new_false)

        if isinstance(instr, ObserveInstr):
            new_cond = transform_condition(instr.cond, scales, real_vars)
            return attr.evolve(instr, cond=new_cond)

        if isinstance(instr, ChoiceInstr):
            # keep prob unchanged (概率不缩放)
            new_lhs = [tr_instr(s) for s in instr.lhs]
            new_rhs = [tr_instr(s) for s in instr.rhs]
            return attr.evolve(instr, lhs=new_lhs, rhs=new_rhs)

        if isinstance(instr, LoopInstr):
            new_body = [tr_instr(s) for s in instr.body]
            return attr.evolve(instr, body=new_body)

        if isinstance(instr, TickInstr):
            # optional: if tick depends on real vars you might want to transform too;
            # leaving it unchanged by default.
            return instr

        return instr

    new_instrs = [tr_instr(s) for s in program.instructions]
    new_prog = attr.evolve(program, instructions=new_instrs)
    return new_prog, scales
