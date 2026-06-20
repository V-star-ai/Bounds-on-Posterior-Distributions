from copy import deepcopy
from fractions import Fraction
from itertools import product

from parsers import parse_src
from preprocessing.upper_prior_prep import (
    exponential_to_bgd,
    mapping_to_bgd,
    normal_to_bgd,
    num_to_bgd,
    prior_to_bgd,
    uniform_to_bgd,
)
from distributions import BGD
from Adapter import Adapter
from Adapter.expr import Expr

from probably.pgcl.ast.expressions import Binop, BinopExpr, UnopExpr, VarExpr, NatLitExpr, RealLitExpr
from intervals import (
    const_int_value,
    interval_complement,
    interval_intersect,
    interval_is_empty,
    interval_union,
)
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
    def __init__(
        self,
        prog_str: str,
        *,
        mode="MUD",
        center_subdivision=None,
        block_subdivision=None,
    ):
        self.prior, self.prog, self.distribution_map = parse_src(prog_str)
        self.mode = mode
        self.center_subdivision = center_subdivision
        self.block_subdivision = block_subdivision

        self.ori_bgd, self.var_order = prior_to_bgd(
            self.prior,
            mode,
            center_subdivision=center_subdivision,
            block_subdivision=block_subdivision,
        )
        self.var_map = {self.var_order[i] : i for i in range(len(self.var_order))}
        self.ctx_bgd = deepcopy(self.ori_bgd)

    def _distribution_spec_to_bgd(self, dist_spec) -> BGD:
        dist_name, params = dist_spec

        if dist_name == "Normal":
            mean, var = params
            return normal_to_bgd(
                mean,
                var,
                self.mode,
                center_subdivision=self.center_subdivision,
                block_subdivision=self.block_subdivision,
            )
        if dist_name == "Exponential":
            (lam,) = params
            return exponential_to_bgd(
                lam,
                self.mode,
                center_subdivision=self.center_subdivision,
                block_subdivision=self.block_subdivision,
            )
        if dist_name == "Uniform":
            a, b = params
            return uniform_to_bgd(a, b, self.mode)
        if dist_name == "Mapping":
            return mapping_to_bgd(params, self.mode)
        if dist_name == "Num":
            return num_to_bgd(params, self.mode)

        raise ValueError(f"Unsupported distribution assignment: {dist_name!r}")

    def _placeholder_distribution_bgd(self, expr):
        if not isinstance(expr, VarExpr):
            return None
        if expr.var not in self.distribution_map:
            return None
        return self._distribution_spec_to_bgd(self.distribution_map[expr.var])

    def _zero_restricted_bgd(self, bgd: BGD, dim: int) -> BGD:
        point = bgd.center_lefts[dim]
        return bgd.restrict(dim, ">=", point).restrict(dim, "<", point)

    def _restrict_intervals_bgd(self, bgd: BGD, dim: int, intervals, *, max_fn=None) -> BGD:
        pieces = []
        for lo, lo_closed, hi, hi_closed in interval_union(intervals, []):
            interval = (lo, lo_closed, hi, hi_closed)
            if interval_is_empty(interval):
                continue

            piece = bgd
            if lo is not None:
                piece = piece.restrict(dim, ">=" if lo_closed else ">", lo)
            if hi is not None:
                piece = piece.restrict(dim, "<=" if hi_closed else "<", hi)
            pieces.append(piece)

        if not pieces:
            return self._zero_restricted_bgd(bgd, dim)

        result = pieces[0]
        for piece in pieces[1:]:
            result = result.add(piece, max_fn=max_fn)
        return result

    @staticmethod
    def _common_frame_template(left: BGD, right: BGD, *, max_fn=None) -> BGD:
        return left.add(right.scale(0), max_fn=max_fn)

    @staticmethod
    def _solver_max(left, right, name: str):
        return Expr.max(left, right)

    @classmethod
    def _max_fn(cls, solver):
        return cls._solver_max if solver else None

    @staticmethod
    def _mud_is_static_zero(mud) -> bool:
        return all(value == 0 for value in mud.P.flat)

    @staticmethod
    def _bgd_left_side_is_empty_or_zero(bgd: BGD, dim: int) -> bool:
        if bgd.left_lengths[dim] == 0:
            return True

        for index in product(range(3), repeat=bgd.ndim):
            direction = bgd.index_to_direction(index)
            if direction[dim] < 0 and not ProgramStructure._mud_is_static_zero(bgd.E[index]):
                return False
        return True

    @staticmethod
    def _expand_empty_left_edges(bgd: BGD, length=Fraction(1)) -> BGD:
        left_lengths = list(bgd.left_lengths)
        changed = False
        for dim in range(bgd.ndim):
            if ProgramStructure._bgd_left_side_is_empty_or_zero(bgd, dim):
                if left_lengths[dim] != length:
                    left_lengths[dim] = length
                    changed = True
        if not changed:
            return bgd
        return bgd.align_frame(
            bgd.center_lefts,
            bgd.center_rights,
            left_lengths,
            bgd.right_lengths,
        )

    def solve_bgd(self, adapter : Adapter = None, method="Park"): # method = "Park" | "Diabolo"
        """
        Traverse the pGCL AST and compute a BGD upper bound.
        """
        self.ctx_bgd = deepcopy(self.ori_bgd)

        def const_value(expr):
            if isinstance(expr, NatLitExpr):
                return int(expr.value)
            if isinstance(expr, RealLitExpr):
                return expr.to_fraction()
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

                def atom(var_name, atom_op, constant):
                    if atom_op == Binop.LT:
                        return var_name, [(None, False, constant, False)]
                    if atom_op == Binop.LEQ:
                        return var_name, [(None, False, constant, True)]
                    if atom_op == Binop.GT:
                        return var_name, [(constant, False, None, False)]
                    if atom_op == Binop.GEQ:
                        return var_name, [(constant, True, None, False)]
                    if atom_op == Binop.EQ:
                        return var_name, [(constant, True, constant, True)]
                    raise ValueError(
                        "If condition must use <, <=, >, >=, or = with one variable and one numeric literal"
                    )

                if op in (Binop.LT, Binop.LEQ, Binop.GT, Binop.GEQ, Binop.EQ):
                    if isinstance(expr.lhs, VarExpr) and isinstance(expr.rhs, (NatLitExpr, RealLitExpr)):
                        return atom(expr.lhs.var, op, const_int_value(expr.rhs))
                    if isinstance(expr.lhs, (NatLitExpr, RealLitExpr)) and isinstance(expr.rhs, VarExpr):
                        reverse = {
                            Binop.LT: Binop.GT,
                            Binop.LEQ: Binop.GEQ,
                            Binop.GT: Binop.LT,
                            Binop.GEQ: Binop.LEQ,
                            Binop.EQ: Binop.EQ,
                        }[op]
                        return atom(expr.rhs.var, reverse, const_int_value(expr.lhs))
                    raise ValueError(
                        "If condition must compare one variable with one numeric literal"
                    )

                raise ValueError(
                    "If condition must use <, <=, >, >=, or = with logical combination"
                )
            if isinstance(expr, UnopExpr):
                return validate_if_condition(expr.expr)
            raise ValueError("If condition must compare a variable with a numeric literal")

        def validate_assignment(instr):
            if isinstance(instr, AsgnInstr):
                lhs_name = instr.lhs
                rhs = instr.rhs

                dist_bgd = self._placeholder_distribution_bgd(rhs)

                if (not isinstance(rhs, BinopExpr) or rhs.operator not in (Binop.PLUS, Binop.MINUS)) and \
                    dist_bgd is None:
                    raise ValueError(f"Assignment must be of form {lhs_name} := {lhs_name} + c or {lhs_name} := Distributions(...)")

                if dist_bgd is not None:
                    c = dist_bgd
                elif isinstance(rhs.lhs, VarExpr) and rhs.lhs.var == lhs_name and isinstance(rhs.rhs,
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
                    raise ValueError(f"Assignment must be of form {lhs_name} := {lhs_name} + c or {lhs_name} := Distributions(...)")

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

        def template_shift_periods(instructions):
            left_periods = [Fraction(0) for _ in range(len(self.var_order))]
            right_periods = [Fraction(0) for _ in range(len(self.var_order))]

            def record(instr):
                if isinstance(instr, AsgnInstr):
                    lhs_name, value = validate_assignment(instr)
                    if isinstance(value, BGD):
                        return
                    dim = self.var_map[lhs_name]
                    value = Fraction(value)
                    if value < 0:
                        left_periods[dim] = max(left_periods[dim], -value)
                    elif value > 0:
                        right_periods[dim] = max(right_periods[dim], value)
                elif isinstance(instr, IfInstr):
                    for child in instr.true:
                        record(child)
                    for child in instr.false:
                        record(child)
                elif isinstance(instr, ChoiceInstr):
                    for child in instr.lhs:
                        record(child)
                    for child in instr.rhs:
                        record(child)

            for instruction in instructions:
                record(instruction)
            return left_periods, right_periods

        def widen_template_periods(template: BGD, instructions) -> BGD:
            required_left, required_right = template_shift_periods(instructions)
            left_lengths = list(template.left_lengths)
            right_lengths = list(template.right_lengths)
            changed = False
            for dim in range(template.ndim):
                if left_lengths[dim] == 0 and required_left[dim] > 0:
                    left_lengths[dim] = required_left[dim]
                    changed = True
                if right_lengths[dim] == 0 and required_right[dim] > 0:
                    right_lengths[dim] = required_right[dim]
                    changed = True
            if not changed:
                return template
            return template.align_frame(
                template.center_lefts,
                template.center_rights,
                left_lengths,
                right_lengths,
            )

        while_counter = 0

        def walk_instr(instr, ctx_bgd, solver = None):
            nonlocal while_counter
            if isinstance(instr, AsgnInstr):
                lhs_name, c = validate_assignment(instr)
                if isinstance(c, BGD):
                    ctx_bgd = ctx_bgd.replace_dim(self.var_map[lhs_name], c)
                else:
                    ctx_bgd = ctx_bgd.add_constant(self.var_map[lhs_name], c)
            elif isinstance(instr, WhileInstr):
                if adapter is None:
                    raise ValueError("while requires an adapter")
                ctx_bgd = self._expand_empty_left_edges(ctx_bgd)
                if isinstance(instr.cond, RealLitExpr):
                    restrict = lambda bgd: bgd.scale(instr.cond.value)
                    restrict_neg = lambda bgd: bgd.scale(1. - instr.cond.value)
                else:
                    var_name, intervals = validate_if_condition(instr.cond)
                    walk_expr(instr.cond)
                    restrict = lambda bgd: self._restrict_intervals_bgd(
                        bgd,
                        self.var_map[var_name],
                        intervals,
                        max_fn=self._max_fn(solver),
                    )
                    neg_intervals = interval_complement(intervals)
                    restrict_neg = lambda bgd: self._restrict_intervals_bgd(
                        bgd,
                        self.var_map[var_name],
                        neg_intervals,
                        max_fn=self._max_fn(solver),
                    )

                need_solve = solver is None
                self_while_counter = while_counter
                while_counter += 1

                # test
                test_bgd = restrict(ctx_bgd)
                for s in instr.body:
                    test_bgd = walk_instr(s, test_bgd, solver)

                template = self._common_frame_template(
                    ctx_bgd,
                    test_bgd,
                    max_fn=self._max_fn(solver),
                )
                template = widen_template_periods(template, instr.body)

                # solve
                if method == "Park":
                    ori_bgd = deepcopy(ctx_bgd)
                    ctx_bgd, solver = adapter.build_bgd_leq(
                        ctx_bgd,
                        template=template,
                        name_prefix=f"w{self_while_counter}",
                    )
                    true_bgd = restrict(ctx_bgd)
                    for s in instr.body:
                        true_bgd = walk_instr(s, true_bgd, solver)
                    solver = adapter.restrict_leq_bgd(
                        true_bgd.add(ori_bgd, max_fn=self._solver_max),
                        ctx_bgd,
                        solver,
                    )
                    ctx_bgd = restrict_neg(ctx_bgd)
                elif method == "Diabolo":
                    ctx_bgd, solver = adapter.build_bgd_leq(
                        ctx_bgd,
                        template=template,
                        name_prefix=f"w{self_while_counter}",
                    )
                    c = adapter.get_var_expr(f"c_w{self_while_counter}", solver)
                    solver.constraints_list.append(0 < c)
                    solver.constraints_list.append(c < 1)
                    true_bgd = restrict(ctx_bgd)
                    for s in instr.body:
                        true_bgd = walk_instr(s, true_bgd, solver)
                    solver = adapter.restrict_leq_bgd(true_bgd, ctx_bgd.scale(c), solver)
                    ctx_bgd = restrict_neg(ctx_bgd).scale(1 / (1 - c))
                else:
                    raise ValueError(f"Unknown method: {method}")

                if need_solve:
                    ctx_bgd = adapter.solve_bgd_expr(ctx_bgd, solver)
            elif isinstance(instr, IfInstr):
                var_name, intervals = validate_if_condition(instr.cond)
                neg_intervals = interval_complement(intervals)
                true_bgd = self._restrict_intervals_bgd(
                    ctx_bgd,
                    self.var_map[var_name],
                    intervals,
                    max_fn=self._max_fn(solver),
                )
                false_bgd = self._restrict_intervals_bgd(
                    ctx_bgd,
                    self.var_map[var_name],
                    neg_intervals,
                    max_fn=self._max_fn(solver),
                )
                walk_expr(instr.cond)
                for s in instr.true:
                    true_bgd = walk_instr(s, true_bgd, solver)
                for s in instr.false:
                    false_bgd = walk_instr(s, false_bgd, solver)
                ctx_bgd = true_bgd.add(false_bgd, max_fn=self._max_fn(solver))
            elif isinstance(instr, ObserveInstr):
                var_name, intervals = validate_if_condition(instr.cond)
                ctx_bgd = self._restrict_intervals_bgd(
                    ctx_bgd,
                    self.var_map[var_name],
                    intervals,
                    max_fn=self._max_fn(solver),
                )
                walk_expr(instr.cond)
            elif isinstance(instr, ChoiceInstr):
                val = validate_choice_prob(instr.prob)
                left_bgd = ctx_bgd
                right_bgd = ctx_bgd
                for s in instr.lhs:
                    left_bgd = walk_instr(s, left_bgd, solver)
                for s in instr.rhs:
                    right_bgd = walk_instr(s, right_bgd, solver)
                return left_bgd.scale(val).add(
                    right_bgd.scale(1 - val),
                    max_fn=self._max_fn(solver),
                )
            elif isinstance(instr, LoopInstr):
                raise ValueError("Unsupported statement: loop { body }")
            elif isinstance(instr, TickInstr):
                raise ValueError(f"Unsupported statement: tick ( expr )")
            return ctx_bgd

        result_bgd = self.ctx_bgd
        for s in self.prog.instructions:
            result_bgd = walk_instr(s, result_bgd)
        self.ctx_bgd = result_bgd
        return result_bgd
