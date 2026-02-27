from copy import deepcopy

from intergrate import align_constants_to_integers
from parser import parse_program
from prior import merge_prior
from EED import EED
from Adapter import Adapter

from probably.pgcl.ast.expressions import Binop, BinopExpr, UnopExpr, VarExpr, NatLitExpr, RealLitExpr
from intervals import const_int_value, interval_intersect, interval_union, interval_complement
from probably.pgcl.ast.instructions import (
    AsgnInstr,
    ChoiceInstr,
    IfInstr,
    LoopInstr,
    ObserveInstr,
    TickInstr,
    WhileInstr,
)

class ProgramStructure:
    def __init__(self, prog_str: str):
        prog = parse_program(prog_str)
        self.ori_prog = prog[1]
        self.var_order, exp = merge_prior(prog[0])
        self.var_map = {self.var_order[i] : i for i in range(len(self.var_order))}
        self.ori_eed, self.disc_prog, self.scales = align_constants_to_integers(self.var_order, exp, self.ori_prog)
        self.ctx_eed = deepcopy(self.ori_eed)

    def solve_eed(self, adapter : Adapter):
        """
        Traverse self.disc_prog (pgcl AST) and enforce assignments are x := x + c.
        For each valid assignment, apply self.ctx_eed.add_constant(var_index, c).
        """
        self.ctx_eed = deepcopy(self.ori_eed)

        def const_value(expr):
            if isinstance(expr, NatLitExpr):
                return int(expr.value)
            if isinstance(expr, RealLitExpr):
                fr = expr.to_fraction()
                if fr.denominator == 1:
                    return int(fr.numerator)
                return float(fr)
            raise ValueError("Assignment constant must be a numeric literal")

        def validate_if_condition(expr):
            if isinstance(expr, BinopExpr):
                op = expr.operator
                if op in (Binop.AND, Binop.OR):
                    v1, i1 = validate_if_condition(expr.lhs)
                    v2, i2 = validate_if_condition(expr.rhs)
                    if v1 != v2:
                        raise ValueError("If condition must use a single variable")
                    if op == Binop.AND:
                        return v1, interval_intersect(i1, i2)
                    return v1, interval_union(i1, i2)
                if op == Binop.LEQ:
                    if not isinstance(expr.lhs, (NatLitExpr, RealLitExpr)):
                        raise ValueError("If condition must be c <= x or x < c")
                    if not isinstance(expr.rhs, VarExpr):
                        raise ValueError("If condition must be c <= x or x < c")
                    c = const_int_value(expr.lhs)
                    return expr.rhs.var, [(c, None)]
                if op == Binop.LT:
                    if not isinstance(expr.lhs, VarExpr):
                        raise ValueError("If condition must be c <= x or x < c")
                    if not isinstance(expr.rhs, (NatLitExpr, RealLitExpr)):
                        raise ValueError("If condition must be c <= x or x < c")
                    c = const_int_value(expr.rhs)
                    return expr.lhs.var, [(None, c)]
                raise ValueError("If condition must be c <= x or x < c with logical combination")
            if isinstance(expr, UnopExpr):
                return validate_if_condition(expr.expr)
            raise ValueError("If condition must be c <= x or c < x with logical combination")

        def validate_assignment(instr):
            if isinstance(instr, AsgnInstr):
                lhs_name = instr.lhs
                rhs = instr.rhs

                if not isinstance(rhs, BinopExpr) or rhs.operator not in (Binop.PLUS, Binop.MINUS):
                    raise ValueError(f"Assignment must be of form {lhs_name} := {lhs_name} + c")

                if isinstance(rhs.lhs, VarExpr) and rhs.lhs.var == lhs_name and isinstance(rhs.rhs,
                                                                                           (NatLitExpr, RealLitExpr)):
                    c = const_value(rhs.rhs)
                    if rhs.operator == Binop.MINUS:
                        c = -c
                elif isinstance(rhs.rhs, VarExpr) and rhs.rhs.var == lhs_name and isinstance(rhs.lhs,
                                                                                             (NatLitExpr, RealLitExpr)):
                    if rhs.operator == Binop.MINUS:
                        raise ValueError(f"Assignment must be of form {lhs_name} := {lhs_name} + c")
                    c = const_value(rhs.lhs)
                else:
                    raise ValueError(f"Assignment must be of form {lhs_name} := {lhs_name} + c")

                if lhs_name not in self.var_map:
                    raise ValueError(f"Unknown variable in assignment: {lhs_name}")
                return lhs_name, c
            else:
                raise ValueError("Incorrect call function valid_assignment")

        def validate_choice_prob(expr):
            if isinstance(expr, NatLitExpr):
                val = int(expr.value)
                if not (0 <= val <= 1):
                    raise ValueError("Choice probability must satisfy 0 <= c <= 1")
                return val
            if isinstance(expr, RealLitExpr):
                if expr.is_infinite():
                    raise ValueError("Choice probability must be finite")
                fr = expr.to_fraction()
                if not (0 <= fr <= 1):
                    raise ValueError("Choice probability must satisfy 0 <= c <= 1")
                return fr
            raise ValueError("Choice probability must be a numeric literal")

        def walk_expr(expr):
            if isinstance(expr, BinopExpr):
                walk_expr(expr.lhs)
                walk_expr(expr.rhs)
            elif isinstance(expr, UnopExpr):
                walk_expr(expr.expr)
            elif isinstance(expr, (VarExpr, NatLitExpr, RealLitExpr)):
                pass

        def walk_instr(instr, ctx_eed, solver = None):
            if isinstance(instr, AsgnInstr):
                lhs_name, c = validate_assignment(instr)
                ctx_eed = ctx_eed.add_constant(self.var_map[lhs_name], c)
            elif isinstance(instr, WhileInstr):
                var_name, intervals = validate_if_condition(instr.cond)
                walk_expr(instr.cond)

                # test
                test_eed = ctx_eed.restrict_interval(self.var_map[var_name], intervals)
                for s in instr.body:
                    test_eed = walk_instr(s, test_eed, solver)

                # solve
                merged_S = [EED.merge_breakpoints(s1, s2, -1) for (s1, s2) in zip(ctx_eed.S, test_eed.S)]
                ctx_eed, solver = adapter.build_leq(ctx_eed, merged_S)
                true_eed = ctx_eed.restrict_interval(self.var_map[var_name], intervals)
                for s in instr.body:
                    true_eed = walk_instr(s, true_eed, solver)
                solver = adapter.restrict_leq(true_eed, ctx_eed, solver)
                ctx_eed = adapter.solve(true_eed, solver)
            elif isinstance(instr, IfInstr):
                var_name, intervals = validate_if_condition(instr.cond)
                neg_intervals = interval_complement(intervals)
                true_eed = ctx_eed.restrict_interval(self.var_map[var_name], intervals)
                false_eed = ctx_eed.restrict_interval(self.var_map[var_name], neg_intervals)
                walk_expr(instr.cond)
                for s in instr.true:
                    true_eed = walk_instr(s, true_eed, solver)
                for s in instr.false:
                    false_eed = walk_instr(s, false_eed, solver)
                ctx_eed = EED.add(true_eed, false_eed, max_function = adapter.max if solver else None)
            elif isinstance(instr, ObserveInstr):
                raise ValueError(f"Unsupported statement: observe")
                # walk_expr(instr.cond)
            elif isinstance(instr, ChoiceInstr):
                validate_choice_prob(instr.prob)
                left_eed = ctx_eed
                right_eed = ctx_eed
                for s in instr.lhs:
                    left_eed = walk_instr(s, left_eed, solver)
                for s in instr.rhs:
                    right_eed = walk_instr(s, right_eed, solver)
                val = validate_choice_prob(instr.prob)
                return EED.add(left_eed.times_constant(val), right_eed.times_constant(1. - val),
                               max_function = adapter.max if solver else None)
            elif isinstance(instr, LoopInstr):
                raise ValueError("Unsupported statement: loop { body }")
            elif isinstance(instr, TickInstr):
                raise ValueError(f"Unsupported statement: tick ( expr )")
            return ctx_eed

        result_eed = self.ctx_eed
        for s in self.disc_prog.instructions:
            result_eed = walk_instr(s, result_eed)
        return result_eed
