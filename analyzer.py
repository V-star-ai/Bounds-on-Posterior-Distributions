from copy import deepcopy
from fractions import Fraction
from itertools import product

import numpy as np

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
from distributions.mud import merge_breakpoints
from Adapter import Adapter
from Adapter.expr import Expr
from semantics.program import (
    ReplaceDistributionAssignment,
    ShiftAssignment,
    UniformConvolutionAssignment,
    build_polynomial_program_semantics,
    choice_probability,
    classify_assignment,
    condition_intervals,
    restrict_intervals,
)

from probably.pgcl.ast.expressions import (
    BinopExpr,
    NatLitExpr,
    RealLitExpr,
    UnopExpr,
    VarExpr,
)
from intervals import interval_complement
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
        template_dirac_iterations=2,
        loop_unroll_iterations=2,
        polynomial_loop_degree="infer",
        polynomial_loop_degree_increment=0,
        uniform_convolution_max_interval=Fraction(1, 2),
        loop_template_visualization=None,
    ):
        self.prior, self.prog, self.distribution_map = parse_src(prog_str)
        self.mode = mode
        self.center_subdivision = center_subdivision
        self.block_subdivision = block_subdivision
        self.template_dirac_iterations = template_dirac_iterations
        self.loop_unroll_iterations = loop_unroll_iterations
        self.polynomial_loop_degree = polynomial_loop_degree
        self.polynomial_loop_degree_increment = (
            polynomial_loop_degree_increment
        )
        self.uniform_convolution_max_interval = (
            None
            if uniform_convolution_max_interval is None
            else Fraction(uniform_convolution_max_interval)
        )
        self.loop_template_visualization = loop_template_visualization or {}

        self.ori_bgd, self.var_order = prior_to_bgd(
            self.prior,
            mode,
            center_subdivision=center_subdivision,
            block_subdivision=block_subdivision,
        )
        self.var_map = {self.var_order[i] : i for i in range(len(self.var_order))}
        self.ctx_bgd = deepcopy(self.ori_bgd)

    @staticmethod
    def _shape_only_bgd(template: BGD) -> BGD:
        E_shape = np.empty(template.E.shape, dtype=object)
        for edge_index in np.ndindex(template.E.shape):
            mud = template.E[edge_index]
            P = np.empty(mud.shape, dtype=object)
            P.fill(Fraction(1))
            E_shape[edge_index] = type(mud)(mud.S, P)
        return BGD(E_shape, [Fraction(1, 2)] * template.ndim, [Fraction(1, 2)] * template.ndim)

    @staticmethod
    def _template_visualization_specs(bgd: BGD, num: int):
        if bgd.ndim == 1:
            return [("var", {"num": num})]

        specs = []
        for dim in range(bgd.ndim):
            if dim < 2:
                specs.append(("var", {"num": num}))
            else:
                specs.append(("const", bgd.center_lefts[dim]))
        return specs

    def _maybe_visualize_loop_template(self, template: BGD, loop_index: int) -> None:
        config = self.loop_template_visualization
        if not config or not config.get("enabled", False):
            return

        from visualize_bgd import plot_bgd

        shape_bgd = self._shape_only_bgd(template)
        num = int(config.get("num", 160))
        fallback_html = config.get(
            "fallback_html",
            f"bgd_loop_template_w{loop_index}.html",
        )
        output_html = config.get("output_html")
        if isinstance(output_html, str):
            output_html = output_html.format(loop=loop_index)
        if isinstance(fallback_html, str):
            fallback_html = fallback_html.format(loop=loop_index)

        print(
            f"[BGD template visualization] loop=w{loop_index}, "
            f"show={bool(config.get('show', True))}, "
            f"output={output_html}, fallback={fallback_html}",
            flush=True,
        )
        plot_bgd(
            shape_bgd,
            self._template_visualization_specs(shape_bgd, num),
            mode=config.get("mode", "heatmap"),
            value=config.get("value", "cell_mass"),
            tail_blocks=int(config.get("tail_blocks", 2)),
            show_internal_grid=bool(config.get("show_internal_grid", True)),
            fallback_html=fallback_html,
            output_html=output_html,
            show=bool(config.get("show", True)),
        )

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

    def _restrict_intervals_bgd(self, bgd: BGD, dim: int, intervals, *, max_fn=None) -> BGD:
        return restrict_intervals(
            bgd,
            dim,
            intervals,
            max_fn=max_fn,
        )

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
    def _bgd_right_side_is_empty_or_zero(bgd: BGD, dim: int) -> bool:
        if bgd.right_lengths[dim] == 0:
            return True

        for index in product(range(3), repeat=bgd.ndim):
            direction = bgd.index_to_direction(index)
            if direction[dim] > 0 and not ProgramStructure._mud_is_static_zero(bgd.E[index]):
                return False
        return True

    @staticmethod
    def _expand_empty_edges(bgd: BGD, length=Fraction(1)) -> BGD:
        left_lengths = list(bgd.left_lengths)
        right_lengths = list(bgd.right_lengths)
        changed = False
        for dim in range(bgd.ndim):
            if ProgramStructure._bgd_left_side_is_empty_or_zero(bgd, dim):
                if left_lengths[dim] != length:
                    left_lengths[dim] = length
                    changed = True
            if ProgramStructure._bgd_right_side_is_empty_or_zero(bgd, dim):
                if right_lengths[dim] != length:
                    right_lengths[dim] = length
                    changed = True
        if not changed:
            return bgd
        return bgd.align_frame(
            bgd.center_lefts,
            bgd.center_rights,
            left_lengths,
            right_lengths,
        )

    @staticmethod
    def _mud_dirac_points_with_mass(mud, dim: int) -> set[Fraction]:
        points = set()
        for interval_index, (left, right) in enumerate(zip(mud.S[dim], mud.S[dim][1:])):
            if left != right:
                continue
            slicer = [slice(None)] * mud.ndim
            slicer[dim] = interval_index
            values = mud.P[tuple(slicer)]
            if any(value != 0 for value in np.asarray(values, dtype=object).flat):
                points.add(left)
        return points

    @staticmethod
    def _block_global_lefts(bgd: BGD, index) -> tuple[Fraction, ...]:
        center_index = (1,) * bgd.ndim
        if index == center_index:
            return bgd.center_lefts
        direction = bgd.index_to_direction(index)
        lefts = []
        for dim, value in enumerate(direction):
            if value < 0:
                lefts.append(bgd.center_lefts[dim] - bgd.left_lengths[dim])
            elif value > 0:
                lefts.append(bgd.center_rights[dim])
            else:
                lefts.append(bgd.center_lefts[dim])
        return tuple(lefts)

    @classmethod
    def _global_dirac_points(cls, bgd: BGD) -> list[set[Fraction]]:
        points = [set() for _ in range(bgd.ndim)]
        center_index = (1,) * bgd.ndim
        for index in product(range(3), repeat=bgd.ndim):
            mud = bgd.E[index]
            offsets = cls._block_global_lefts(bgd, index)
            for dim in range(bgd.ndim):
                for point in cls._mud_dirac_points_with_mass(mud, dim):
                    if index == center_index:
                        points[dim].add(point)
                    else:
                        points[dim].add(point + offsets[dim])
        return points

    @staticmethod
    def _template_local_dirac_for_dim(bgd: BGD, index, dim: int, point: Fraction):
        center_index = (1,) * bgd.ndim
        if index == center_index:
            if bgd.center_lefts[dim] <= point <= bgd.center_rights[dim]:
                return point
            return None

        direction = bgd.index_to_direction(index)
        if direction[dim] < 0:
            period = bgd.left_lengths[dim]
            if period <= 0 or point >= bgd.center_lefts[dim]:
                return None
            distance = bgd.center_lefts[dim] - point
            phase = distance % period
            if phase == 0:
                return Fraction(0)
            return period - phase
        if direction[dim] > 0:
            period = bgd.right_lengths[dim]
            if period <= 0 or point <= bgd.center_rights[dim]:
                return None
            return (point - bgd.center_rights[dim]) % period

        if bgd.center_lefts[dim] <= point <= bgd.center_rights[dim]:
            return point - bgd.center_lefts[dim]
        return None

    @classmethod
    def _add_probe_diracs_to_template(cls, template: BGD, probe: BGD) -> BGD:
        global_points = cls._global_dirac_points(probe)
        result_E = template._copy_E()

        for index in product(range(3), repeat=template.ndim):
            mud = result_E[index]
            target_S = list(mud.S)
            changed = False
            for dim in range(template.ndim):
                additions = []
                for point in global_points[dim]:
                    local_point = cls._template_local_dirac_for_dim(
                        template,
                        index,
                        dim,
                        point,
                    )
                    if local_point is not None:
                        additions.append((local_point, local_point))
                if additions:
                    merged = merge_breakpoints(
                        target_S[dim],
                        *additions,
                        preserve_dirac=True,
                    )
                    if merged != target_S[dim]:
                        target_S[dim] = merged
                        changed = True
            if changed:
                result_E[index] = mud.align(tuple(target_S))

        return BGD(result_E, template.alpha, template.beta)

    @staticmethod
    def _global_breakpoints_for_dim(
        bgd: BGD, dim: int, low: Fraction, high: Fraction
    ) -> set[Fraction]:
        points = set()
        if low > high:
            return points

        center_index = (1,) * bgd.ndim
        for index in product(range(3), repeat=bgd.ndim):
            mud = bgd.E[index]
            direction = bgd.index_to_direction(index)
            axis_direction = direction[dim]
            local_points = mud.S[dim]

            if index == center_index:
                offsets = [Fraction(0)]
            elif axis_direction < 0:
                length = bgd.left_lengths[dim]
                if length <= 0:
                    continue
                offsets = []
                block = -1
                while True:
                    offset = bgd.center_lefts[dim] + block * length
                    if offset + length < low:
                        break
                    offsets.append(offset)
                    block -= 1
            elif axis_direction > 0:
                length = bgd.right_lengths[dim]
                if length <= 0:
                    continue
                offsets = []
                block = 1
                while True:
                    offset = bgd.center_rights[dim] + (block - 1) * length
                    if offset > high:
                        break
                    offsets.append(offset)
                    block += 1
            else:
                offsets = [bgd.center_lefts[dim]]

            for offset in offsets:
                for point in local_points:
                    global_point = offset + point
                    if low <= global_point <= high:
                        points.add(global_point)

        return points

    @staticmethod
    def _add_center_axis_breakpoints(
        bgd: BGD, dim: int, points: set[Fraction]
    ) -> BGD:
        points = {
            point
            for point in points
            if bgd.center_lefts[dim] <= point <= bgd.center_rights[dim]
        }
        if not points:
            return bgd

        result_E = bgd._copy_E()
        center_index = (1,) * bgd.ndim
        center = result_E[center_index]
        target_S = list(center.S)
        target_S[dim] = merge_breakpoints(
            target_S[dim],
            tuple(sorted(points)),
            preserve_dirac=True,
        )
        result_E[center_index] = center.align(tuple(target_S))
        return BGD(result_E, bgd.alpha, bgd.beta)

    @classmethod
    def _apply_stable_probe_boundaries(
        cls, template: BGD, probes: list[BGD]
    ) -> BGD:
        if len(probes) < 2:
            return template

        result = template
        for dim in range(template.ndim):
            right_candidates = [
                (index, probe.center_rights[dim])
                for index, probe in enumerate(probes[:-1])
                if all(
                    later.center_rights[dim] <= probe.center_rights[dim]
                    for later in probes[index + 1 :]
                )
            ]
            if right_candidates:
                right_index, right = min(
                    right_candidates,
                    key=lambda item: item[1],
                )
                right_probe = probes[right_index]
            else:
                right = None
                right_probe = None
            if right is not None:
                old_right = result.center_rights[dim]
                threshold = max(right, old_right)
                result = result.restrict(dim, "<=", threshold)
                if threshold > old_right:
                    points = cls._global_breakpoints_for_dim(
                        right_probe, dim, old_right, threshold
                    )
                    points = {
                        point for point in points if old_right < point <= threshold
                    }
                    result = cls._add_center_axis_breakpoints(result, dim, points)

            left_candidates = [
                (index, probe.center_lefts[dim])
                for index, probe in enumerate(probes[:-1])
                if all(
                    later.center_lefts[dim] >= probe.center_lefts[dim]
                    for later in probes[index + 1 :]
                )
            ]
            if left_candidates:
                left_index, left = max(
                    left_candidates,
                    key=lambda item: item[1],
                )
                left_probe = probes[left_index]
            else:
                left = None
                left_probe = None
            if left is not None:
                old_left = result.center_lefts[dim]
                threshold = min(left, old_left)
                result = result.restrict(dim, ">=", threshold)
                if threshold < old_left:
                    points = cls._global_breakpoints_for_dim(
                        left_probe, dim, threshold, old_left
                    )
                    points = {
                        point for point in points if threshold <= point < old_left
                    }
                    result = cls._add_center_axis_breakpoints(result, dim, points)

        return result

    def build_polynomial_semantics(self):
        """Construct exact PolynomialBGD semantics without solving."""

        return build_polynomial_program_semantics(
            self.prior,
            self.prog,
            self.distribution_map,
            loop_template_degree=self.polynomial_loop_degree,
            loop_template_degree_increment=(
                self.polynomial_loop_degree_increment
            ),
            loop_unroll_iterations=max(1, self.loop_unroll_iterations),
        )

    def solve_bgd(self, adapter : Adapter = None, method="Park"): # method = "Park" | "Diabolo"
        """
        Traverse the pGCL AST and compute a BGD upper bound.
        """
        self.ctx_bgd = deepcopy(self.ori_bgd)

        def validate_if_condition(expr):
            return condition_intervals(expr)

        def validate_assignment(instr):
            action = classify_assignment(instr, self.distribution_map)
            if action.variable not in self.var_map:
                raise ValueError(
                    f"Unknown variable in assignment: {action.variable}"
                )
            if isinstance(action, ShiftAssignment):
                value = action.offset
            elif isinstance(action, UniformConvolutionAssignment):
                value = ("add_uniform", (action.low, action.high))
            elif isinstance(action, ReplaceDistributionAssignment):
                value = self._distribution_spec_to_bgd(action.distribution)
            else:
                raise TypeError("unsupported assignment action")
            return action.variable, value

        def validate_choice_prob(expr):
            return choice_probability(expr)

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
                    if isinstance(value, tuple) and value[0] == "add_uniform":
                        low, high = value[1]
                        width = Fraction(high) - Fraction(low)
                        if width <= 0:
                            raise ValueError("Uniform addition requires low < high")
                        if Fraction(low) < 0:
                            left_periods[dim] = max(left_periods[dim], -Fraction(low))
                        if Fraction(high) > 0:
                            right_periods[dim] = max(right_periods[dim], Fraction(high))
                        return
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
                elif isinstance(c, tuple) and c[0] == "add_uniform":
                    dim = self.var_map[lhs_name]
                    low, high = c[1]
                    ctx_bgd = ctx_bgd.convolve_uniform(
                        dim,
                        low,
                        high,
                        max_fn=self._max_fn(solver),
                        max_interval=self.uniform_convolution_max_interval,
                    )
                else:
                    ctx_bgd = ctx_bgd.add_constant(self.var_map[lhs_name], c)
            elif isinstance(instr, WhileInstr):
                if adapter is None:
                    raise ValueError("while requires an adapter")
                ctx_bgd = self._expand_empty_edges(ctx_bgd)
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

                def run_loop_body_once(bgd):
                    result = restrict(bgd)
                    for body_instr in instr.body:
                        result = walk_instr(body_instr, result, solver)
                    return result

                # Template shape heuristic: include the entry distribution, the
                # guarded entry distribution, and a bounded number of concrete
                # loop unrolls.
                guarded_bgd = restrict(ctx_bgd)
                unrolled_bgds = []
                probe_bgd = ctx_bgd
                for _ in range(max(0, self.loop_unroll_iterations)):
                    probe_bgd = run_loop_body_once(probe_bgd)
                    unrolled_bgds.append(probe_bgd)

                template = self._common_frame_template(
                    ctx_bgd,
                    guarded_bgd,
                    max_fn=self._max_fn(solver),
                )
                for unrolled_bgd in unrolled_bgds:
                    template = self._common_frame_template(
                        template,
                        unrolled_bgd,
                        max_fn=self._max_fn(solver),
                    )
                template = widen_template_periods(template, instr.body)
                probe_bgds = list(unrolled_bgds)
                if self.template_dirac_iterations > 0:
                    if not probe_bgds:
                        probe_bgd = guarded_bgd
                    for _ in range(self.template_dirac_iterations):
                        probe_bgd = run_loop_body_once(probe_bgd)
                        probe_bgds.append(probe_bgd)
                    template = self._apply_stable_probe_boundaries(
                        template,
                        probe_bgds,
                    )
                    template = self._add_probe_diracs_to_template(template, probe_bgd)
                template = template.standardize(
                    skip_static_zero=False
                ).align_center_subdivisions()
                self._maybe_visualize_loop_template(template, self_while_counter)

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
                if isinstance(instr.cond, (NatLitExpr, RealLitExpr)):
                    val = validate_choice_prob(instr.cond)
                    left_bgd = ctx_bgd
                    right_bgd = ctx_bgd
                    for s in instr.true:
                        left_bgd = walk_instr(s, left_bgd, solver)
                    for s in instr.false:
                        right_bgd = walk_instr(s, right_bgd, solver)
                    return left_bgd.scale(val).add(
                        right_bgd.scale(1 - val),
                        max_fn=self._max_fn(solver),
                    )
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
