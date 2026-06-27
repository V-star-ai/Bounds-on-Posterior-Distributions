from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go

from distributions import BGD
from distributions.mud import AffineCell


SpecType = Union[
    float,
    int,
    Fraction,
    Tuple[str, Any],
    Dict[str, Any],
]


@dataclass
class _Spec:
    kind: str
    value: Any


def _parse_spec(spec: SpecType) -> _Spec:
    if isinstance(spec, (int, float, Fraction)):
        return _Spec("const", spec)
    if isinstance(spec, dict):
        kind = spec.get("type")
        if kind not in ("const", "enum", "var"):
            raise ValueError(f"Invalid spec type: {kind}")
        return _Spec(kind, spec)
    if isinstance(spec, tuple) and len(spec) == 2:
        kind, val = spec
        if kind not in ("const", "enum", "var"):
            raise ValueError(f"Invalid spec type: {kind}")
        return _Spec(kind, val)
    raise ValueError(f"Invalid spec: {spec}")


def _to_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def _to_float(value, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric before visualization; got {value!r}"
        ) from exc


def _safe_show(fig, fallback_html: str | Path | None = None):
    try:
        fig.show()
    except (PermissionError, OSError):
        if fallback_html is not None:
            fig.write_html(str(fallback_html))
        return fig
    return fig


def _auto_num(span: float, *, min_n: int, max_n: int, scale: float) -> int:
    return int(max(min_n, min(max_n, span * scale)))


def _axis_structural_points(
    bgd: BGD,
    axis: int,
    *,
    tail_blocks: int,
    xmin: Fraction,
    xmax: Fraction,
):
    points = set()
    for edge_index in np.ndindex(bgd.E.shape):
        mud = bgd.E[edge_index]
        direction = bgd.index_to_direction(edge_index)
        local_points = mud.S[axis]

        if direction[axis] < 0:
            length = bgd.left_lengths[axis]
            if length <= 0:
                continue
            translations = [
                bgd.center_lefts[axis] - block_number * length
                for block_number in range(1, tail_blocks + 1)
            ]
        elif direction[axis] > 0:
            length = bgd.right_lengths[axis]
            if length <= 0:
                continue
            translations = [
                bgd.center_rights[axis] + (block_number - 1) * length
                for block_number in range(1, tail_blocks + 1)
            ]
        else:
            if edge_index == (1,) * bgd.ndim:
                translations = [Fraction(0)]
            else:
                translations = [bgd.center_lefts[axis]]

        for translation in translations:
            for point in local_points:
                global_point = translation + point
                if xmin <= global_point <= xmax:
                    points.add(global_point)

    return sorted(points)


def _axis_points_for_bgd(
    bgd: BGD,
    axis: int,
    spec_value: Any,
    *,
    tail_blocks: int,
    default_num: int,
):
    if spec_value is None:
        left_ext = bgd.left_lengths[axis] * tail_blocks
        right_ext = bgd.right_lengths[axis] * tail_blocks
        xmin = bgd.center_lefts[axis] - left_ext
        xmax = bgd.center_rights[axis] + right_ext
        if xmin == xmax:
            xmin -= 1
            xmax += 1
        span = _to_float(xmax - xmin, f"axis {axis} span")
        num = _auto_num(span, min_n=default_num, max_n=max(default_num, 600), scale=40.0)
        base_points = np.linspace(float(xmin), float(xmax), num)
        structural_points = _axis_structural_points(
            bgd,
            axis,
            tail_blocks=tail_blocks,
            xmin=xmin,
            xmax=xmax,
        )
        return np.array(
            sorted(set(base_points.tolist()) | {float(point) for point in structural_points}),
            dtype=float,
        )

    value = spec_value if isinstance(spec_value, dict) else {}
    xmin = value.get(
        "min",
        bgd.center_lefts[axis] - bgd.left_lengths[axis] * tail_blocks,
    )
    xmax = value.get(
        "max",
        bgd.center_rights[axis] + bgd.right_lengths[axis] * tail_blocks,
    )
    num = int(value.get("num", default_num))
    if num <= 0:
        raise ValueError("var axis num must be positive")
    xmin = _to_fraction(xmin)
    xmax = _to_fraction(xmax)
    base_points = np.linspace(float(xmin), float(xmax), num)
    structural_points = _axis_structural_points(
        bgd,
        axis,
        tail_blocks=tail_blocks,
        xmin=xmin,
        xmax=xmax,
    )
    return np.array(
        sorted(set(base_points.tolist()) | {float(point) for point in structural_points}),
        dtype=float,
    )


def _enum_points_for_axis(bgd: BGD, axis: int, spec_value: Any):
    value = spec_value if isinstance(spec_value, dict) else {}
    source = value.get("source", "center")
    if source != "center":
        raise ValueError("BGD enum currently supports only source='center'")

    points = []
    S = bgd.C.S[axis]
    for left, right in zip(S, S[1:]):
        if left == right:
            points.append((f"x{axis} = {left}", float(left)))
        else:
            mid = (left + right) / 2
            points.append((f"x{axis} in [{left},{right}]", float(mid)))
    return points


def _block_coord_for_axis(bgd: BGD, axis: int, x: Fraction) -> Optional[int]:
    center_left = bgd.center_lefts[axis]
    center_right = bgd.center_rights[axis]
    if center_left <= x <= center_right:
        return 0

    if x < center_left:
        length = bgd.left_lengths[axis]
        if length <= 0:
            return None
        return _floor_fraction((x - center_left) / length)

    length = bgd.right_lengths[axis]
    if length <= 0:
        return None
    return _floor_fraction((x - center_right) / length) + 1


def _block_coord_candidates_for_axis(bgd: BGD, axis: int, x: Fraction) -> list[int]:
    candidates = set()
    primary = _block_coord_for_axis(bgd, axis, x)
    if primary is not None:
        candidates.add(primary)

    center_left = bgd.center_lefts[axis]
    center_right = bgd.center_rights[axis]
    if x == center_left and bgd.left_lengths[axis] > 0:
        candidates.add(-1)
    if x == center_right and bgd.right_lengths[axis] > 0:
        candidates.add(1)

    if x < center_left and bgd.left_lengths[axis] > 0:
        distance = center_left - x
        if distance % bgd.left_lengths[axis] == 0:
            q = distance // bgd.left_lengths[axis]
            candidates.add(-q)
            candidates.add(-(q + 1))

    if x > center_right and bgd.right_lengths[axis] > 0:
        distance = x - center_right
        if distance % bgd.right_lengths[axis] == 0:
            q = distance // bgd.right_lengths[axis]
            candidates.add(q)
            candidates.add(q + 1)

    return sorted(candidates)


def _floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def _cell_index_for_point(S: Sequence[Fraction], x: Fraction) -> Optional[int]:
    if len(S) < 2 or x < S[0] or x > S[-1]:
        return None

    for index, (left, right) in enumerate(zip(S, S[1:])):
        if left == right and x == left:
            return index

    for index, (left, right) in enumerate(zip(S, S[1:])):
        if left == right:
            continue
        if left <= x < right:
            return index
        if index == len(S) - 2 and x == right:
            return index
    return None


def _payload_at(payload, interval: tuple[Fraction, Fraction], point: Fraction):
    if isinstance(payload, AffineCell):
        left, right = interval
        if left == right:
            return payload.left
        ratio = (point - left) / (right - left)
        return payload.left + (payload.right - payload.left) * ratio
    return payload


def _eval_mud_at_with_intervals(mud, local_x: Sequence[Fraction], *, value: str):
    if mud.is_empty:
        return 0.0, None

    index = []
    for axis, x in enumerate(local_x):
        cell_index = _cell_index_for_point(mud.S[axis], x)
        if cell_index is None:
            return 0.0, None
        index.append(cell_index)
    index_tuple = tuple(index)

    payload = mud.P[index_tuple]
    intervals = tuple(
        (mud.S[axis][cell_index], mud.S[axis][cell_index + 1])
        for axis, cell_index in enumerate(index_tuple)
    )
    if isinstance(payload, AffineCell):
        affine_dim = getattr(mud, "affine_dim", 0)
        payload = _payload_at(
            payload, intervals[affine_dim], local_x[affine_dim]
        )

    if value == "cell_mass":
        return _to_float(payload, "cell payload"), intervals
    if value != "density":
        raise ValueError("value must be 'density' or 'cell_mass'")

    volume = Fraction(1)
    for left, right in intervals:
        length = right - left
        if length > 0:
            volume *= length
    return _to_float(payload / volume, "cell density"), intervals


def _eval_mud_at(mud, local_x: Sequence[Fraction], *, value: str):
    result, _intervals = _eval_mud_at_with_intervals(mud, local_x, value=value)
    return result


def _eval_bgd_at(bgd: BGD, x: Sequence[float], *, value: str) -> float:
    if len(x) != bgd.ndim:
        raise ValueError("Point dimension does not match BGD")

    point = tuple(_to_fraction(v) for v in x)
    coord_candidates = []
    primary_coord = []
    for axis, axis_value in enumerate(point):
        primary = _block_coord_for_axis(bgd, axis, axis_value)
        primary_coord.append(primary)
        candidates = _block_coord_candidates_for_axis(bgd, axis, axis_value)
        if not candidates:
            return 0.0
        coord_candidates.append(candidates)

    total = 0.0
    for block_coord in product(*coord_candidates):
        block = bgd.block_at(tuple(block_coord))
        if all(coord == 0 for coord in block_coord):
            local_x = point
        else:
            local_x = tuple(
                point[axis] - block.translation[axis] for axis in range(bgd.ndim)
            )
        base, intervals = _eval_mud_at_with_intervals(
            block.distribution,
            local_x,
            value=value,
        )
        if intervals is None:
            continue
        if any(
            coord != primary_coord[axis]
            and intervals[axis][0] != intervals[axis][1]
            for axis, coord in enumerate(block_coord)
        ):
            continue
        total += base * _to_float(block.decay_factor, "BGD decay factor")
    return total


def _fixed_point_from_specs(parsed: Sequence[_Spec], enum_axes, combo, ndim: int):
    x = [None] * ndim
    for axis, spec in enumerate(parsed):
        if spec.kind == "const":
            x[axis] = float(spec.value)
    for axis, (_label, value) in zip(enum_axes, combo):
        x[axis] = value
    return x


def _axis_boundary_points(bgd: BGD, axis: int, *, tail_blocks: int) -> list[Fraction]:
    values = [bgd.center_lefts[axis], bgd.center_rights[axis]]
    for block in range(1, tail_blocks + 1):
        if bgd.left_lengths[axis] > 0:
            values.append(bgd.center_lefts[axis] - block * bgd.left_lengths[axis])
        if bgd.right_lengths[axis] > 0:
            values.append(bgd.center_rights[axis] + block * bgd.right_lengths[axis])
    return sorted(values)


def _axis_boundaries(bgd: BGD, axis: int, *, tail_blocks: int) -> list[float]:
    return [
        float(value)
        for value in _axis_boundary_points(bgd, axis, tail_blocks=tail_blocks)
    ]




def _axis_translations_for_edge(
    bgd: BGD,
    edge_index,
    axis: int,
    *,
    tail_blocks: int,
) -> list[Fraction]:
    direction = bgd.index_to_direction(edge_index)
    if direction[axis] < 0:
        length = bgd.left_lengths[axis]
        if length <= 0:
            return []
        return [
            bgd.center_lefts[axis] - block_number * length
            for block_number in range(1, tail_blocks + 1)
        ]
    if direction[axis] > 0:
        length = bgd.right_lengths[axis]
        if length <= 0:
            return []
        return [
            bgd.center_rights[axis] + (block_number - 1) * length
            for block_number in range(1, tail_blocks + 1)
        ]
    if edge_index == (1,) * bgd.ndim:
        return [Fraction(0)]
    return [bgd.center_lefts[axis]]


def _dirac_side(points: Sequence[Fraction], index: int) -> str:
    point = points[index]
    if index > 0 and points[index - 1] < point:
        return "left"
    if index + 2 < len(points) and point < points[index + 2]:
        return "right"
    if index == 0:
        return "right"
    return "left"


def _axis_structure_lines(
    bgd: BGD,
    axis: int,
    *,
    tail_blocks: int,
) -> tuple[set[Fraction], set[Fraction], list[tuple[Fraction, str]]]:
    xmin, xmax = _axis_plot_range(bgd, axis, tail_blocks=tail_blocks)
    big_lines = set(_axis_boundary_points(bgd, axis, tail_blocks=tail_blocks))
    small_lines = set()
    dirac_lines = []

    for edge_index in np.ndindex(bgd.E.shape):
        mud = bgd.E[edge_index]
        local_points = mud.S[axis]
        translations = _axis_translations_for_edge(
            bgd,
            edge_index,
            axis,
            tail_blocks=tail_blocks,
        )
        for translation in translations:
            for point in local_points[1:-1]:
                global_point = translation + point
                if xmin <= global_point <= xmax:
                    small_lines.add(global_point)
            for index, (left, right) in enumerate(zip(local_points, local_points[1:])):
                if left != right:
                    continue
                global_point = translation + left
                if xmin <= global_point <= xmax:
                    dirac_lines.append((global_point, _dirac_side(local_points, index)))

    return big_lines, small_lines, dirac_lines


def _axis_line_offset(
    bgd: BGD,
    axis: int,
    *,
    tail_blocks: int,
) -> Fraction:
    xmin, xmax = _axis_plot_range(bgd, axis, tail_blocks=tail_blocks)
    span = xmax - xmin
    if span <= 0:
        return Fraction(1, 1000)
    return span / 1000


def _offset_dirac_line(
    value: Fraction,
    side: str,
    *,
    offset: Fraction,
    occupied: set[Fraction],
) -> Fraction:
    if value not in occupied:
        return value
    if side == "left":
        return value - offset
    return value + offset


def _add_vline(fig, value, *, color: str, width: float, dash: str):
    fig.add_vline(
        x=float(value),
        line_width=width,
        line_dash=dash,
        line_color=color,
    )


def _add_hline(fig, value, *, color: str, width: float, dash: str):
    fig.add_hline(
        y=float(value),
        line_width=width,
        line_dash=dash,
        line_color=color,
    )


def _axis_plot_range(bgd: BGD, axis: int, *, tail_blocks: int) -> tuple[Fraction, Fraction]:
    return (
        bgd.center_lefts[axis] - bgd.left_lengths[axis] * tail_blocks,
        bgd.center_rights[axis] + bgd.right_lengths[axis] * tail_blocks,
    )


def _add_1d_boundaries(
    fig,
    bgd: BGD,
    axis: int,
    *,
    tail_blocks: int,
    show_internal_grid: bool,
):
    if show_internal_grid:
        big_lines, small_lines, dirac_lines = _axis_structure_lines(
            bgd,
            axis,
            tail_blocks=tail_blocks,
        )
        occupied = big_lines | small_lines
        offset = _axis_line_offset(bgd, axis, tail_blocks=tail_blocks)
        for value in sorted(small_lines):
            _add_vline(fig, value, color="rgba(0,150,70,0.70)", width=0.8, dash="solid")
        for value in sorted(big_lines):
            _add_vline(fig, value, color="rgba(210,30,30,0.85)", width=1.6, dash="solid")
        for value, side in dirac_lines:
            value = _offset_dirac_line(value, side, offset=offset, occupied=occupied)
            _add_vline(fig, value, color="rgba(40,90,230,0.90)", width=1.2, dash="solid")

    for value in _axis_boundaries(bgd, axis, tail_blocks=tail_blocks):
        if not show_internal_grid:
            _add_vline(fig, value, color="rgba(60,60,60,0.45)", width=1, dash="dot")


def _add_2d_heatmap_boundaries(
    fig,
    bgd: BGD,
    ax0: int,
    ax1: int,
    *,
    tail_blocks: int,
    show_internal_grid: bool,
):
    if show_internal_grid:
        x_big, x_small, x_dirac = _axis_structure_lines(
            bgd,
            ax0,
            tail_blocks=tail_blocks,
        )
        y_big, y_small, y_dirac = _axis_structure_lines(
            bgd,
            ax1,
            tail_blocks=tail_blocks,
        )
        x_occupied = x_big | x_small
        y_occupied = y_big | y_small
        x_offset = _axis_line_offset(bgd, ax0, tail_blocks=tail_blocks)
        y_offset = _axis_line_offset(bgd, ax1, tail_blocks=tail_blocks)

        for value in sorted(x_small):
            _add_vline(fig, value, color="rgba(0,150,70,0.70)", width=0.8, dash="solid")
        for value in sorted(y_small):
            _add_hline(fig, value, color="rgba(0,150,70,0.70)", width=0.8, dash="solid")
        for value in sorted(x_big):
            _add_vline(fig, value, color="rgba(210,30,30,0.85)", width=1.6, dash="solid")
        for value in sorted(y_big):
            _add_hline(fig, value, color="rgba(210,30,30,0.85)", width=1.6, dash="solid")
        for value, side in x_dirac:
            value = _offset_dirac_line(value, side, offset=x_offset, occupied=x_occupied)
            _add_vline(fig, value, color="rgba(40,90,230,0.90)", width=1.2, dash="solid")
        for value, side in y_dirac:
            value = _offset_dirac_line(value, side, offset=y_offset, occupied=y_occupied)
            _add_hline(fig, value, color="rgba(40,90,230,0.90)", width=1.2, dash="solid")

    for value in _axis_boundaries(bgd, ax0, tail_blocks=tail_blocks):
        if not show_internal_grid:
            _add_vline(fig, value, color="rgba(60,60,60,0.45)", width=1, dash="dot")
    for value in _axis_boundaries(bgd, ax1, tail_blocks=tail_blocks):
        if not show_internal_grid:
            _add_hline(fig, value, color="rgba(60,60,60,0.45)", width=1, dash="dot")


def plot_bgd(
    bgd: BGD,
    specs: Sequence[SpecType],
    *,
    mode: str = "surface",
    value: str = "density",
    tail_blocks: int = 2,
    show_blocks: bool = True,
    show_internal_grid: bool = False,
    output_html: str | Path | None = None,
    fallback_html: str | Path | None = None,
    show: bool = True,
):
    """Visualize a numeric BGD by point sampling.

    specs uses the same shape as visualize.plot_eed: each dimension is a
    constant, ("var", options), or ("enum", options). At most two dimensions may
    be variable. Dirac cells are shown as point mass; continuous cells are shown
    as average density when value="density". output_html writes unconditionally;
    fallback_html writes only if browser display fails. By default the sampled
    range includes the center block and two left/right geometric decay blocks.
    """
    if len(specs) != bgd.ndim:
        raise ValueError("specs length must equal BGD dimension")
    if tail_blocks < 0:
        raise ValueError("tail_blocks must be non-negative")
    if mode not in ("line", "heatmap", "surface"):
        raise ValueError("mode must be 'line', 'heatmap', or 'surface'")
    if value not in ("density", "cell_mass"):
        raise ValueError("value must be 'density' or 'cell_mass'")

    parsed = [_parse_spec(spec) for spec in specs]
    var_axes = [axis for axis, spec in enumerate(parsed) if spec.kind == "var"]
    if len(var_axes) > 2:
        raise ValueError("At most two variable dimensions are allowed")
    if mode == "line" and len(var_axes) > 1:
        raise ValueError("line mode requires at most one variable dimension")
    if len(var_axes) == 1 and mode in ("heatmap", "surface"):
        mode = "line"

    enum_axes = [axis for axis, spec in enumerate(parsed) if spec.kind == "enum"]
    enum_lists = [
        _enum_points_for_axis(bgd, axis, parsed[axis].value) for axis in enum_axes
    ]
    enum_combos = list(product(*enum_lists)) if enum_lists else [()]

    if len(var_axes) == 0:
        results = []
        for combo in enum_combos:
            x = _fixed_point_from_specs(parsed, enum_axes, combo, bgd.ndim)
            if any(value is None for value in x):
                raise ValueError("all non-enum dimensions must be const without var axes")
            results.append((_eval_bgd_at(bgd, x, value=value), combo))
        return results

    if len(var_axes) == 1:
        axis = var_axes[0]
        xs = _axis_points_for_bgd(
            bgd,
            axis,
            parsed[axis].value,
            tail_blocks=tail_blocks,
            default_num=240,
        )
        fig = go.Figure()
        for combo in enum_combos:
            ys = []
            for xv in xs:
                x = _fixed_point_from_specs(parsed, enum_axes, combo, bgd.ndim)
                x[axis] = float(xv)
                if any(value is None for value in x):
                    raise ValueError("non-variable dimensions must be const or enum")
                ys.append(_eval_bgd_at(bgd, x, value=value))
            nonzero_count = sum(1 for yv in ys if yv != 0)
            max_value = max(ys) if ys else 0.0
            print(
                f"[BGD visualization] samples={len(ys)}, "
                f"nonzero={nonzero_count}, max={max_value}",
                flush=True,
            )
            label = ", ".join(label for (label, _value) in combo) if combo else None
            trace_mode = "lines+markers" if len(xs) <= 2 else "lines"
            fig.add_trace(go.Scatter(x=xs, y=ys, mode=trace_mode, name=label))
        if show_blocks:
            _add_1d_boundaries(
                fig,
                bgd,
                axis,
                tail_blocks=tail_blocks,
                show_internal_grid=show_internal_grid,
            )
        fig.update_layout(
            title="BGD",
            xaxis_title=f"axis {axis}",
            yaxis_title=value,
        )
        if output_html is not None:
            fig.write_html(str(output_html))
        return _safe_show(fig, fallback_html) if show else fig

    ax0, ax1 = var_axes
    xs = _axis_points_for_bgd(
        bgd,
        ax0,
        parsed[ax0].value,
        tail_blocks=tail_blocks,
        default_num=130,
    )
    ys = _axis_points_for_bgd(
        bgd,
        ax1,
        parsed[ax1].value,
        tail_blocks=tail_blocks,
        default_num=130,
    )
    X, Y = np.meshgrid(xs, ys)
    degenerate_grid = len(xs) < 2 or len(ys) < 2
    titles = [
        ", ".join(label for (label, _value) in combo) if combo else "BGD"
        for combo in enum_combos
    ]
    fig = go.Figure()
    for combo_index, combo in enumerate(enum_combos):
        Z = np.zeros_like(X, dtype=float)
        for row in range(len(ys)):
            for col in range(len(xs)):
                x = _fixed_point_from_specs(parsed, enum_axes, combo, bgd.ndim)
                x[ax0] = float(X[row, col])
                x[ax1] = float(Y[row, col])
                if any(value is None for value in x):
                    raise ValueError("non-variable dimensions must be const or enum")
                Z[row, col] = _eval_bgd_at(bgd, x, value=value)
        print(
            f"[BGD visualization] grid={len(xs)}x{len(ys)}, samples={Z.size}, "
            f"nonzero={int(np.count_nonzero(Z))}, max={float(np.max(Z)) if Z.size else 0.0}",
            flush=True,
        )

        trace_visible = combo_index == 0
        if mode == "surface" and degenerate_grid:
            fig.add_trace(
                go.Scatter3d(
                    x=X.reshape(-1),
                    y=Y.reshape(-1),
                    z=Z.reshape(-1),
                    mode="lines+markers",
                    marker=dict(
                        size=4,
                        color=Z.reshape(-1),
                        colorscale="Viridis",
                        colorbar=dict(title=value),
                    ),
                    line=dict(width=5, color="rgba(30,90,180,0.75)"),
                    visible=trace_visible,
                    name=titles[combo_index],
                )
            )
        elif mode == "surface":
            fig.add_trace(
                go.Surface(
                    x=xs,
                    y=ys,
                    z=Z,
                    colorbar=dict(title=value),
                    visible=trace_visible,
                    name=titles[combo_index],
                )
            )
        elif degenerate_grid:
            fig.add_trace(
                go.Scatter(
                    x=X.reshape(-1),
                    y=Y.reshape(-1),
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=Z.reshape(-1),
                        colorscale="Viridis",
                        colorbar=dict(title=value),
                    ),
                    visible=trace_visible,
                    name=titles[combo_index],
                )
            )
        else:
            fig.add_trace(
                go.Heatmap(
                    x=xs,
                    y=ys,
                    z=Z,
                    colorbar=dict(title=value),
                    visible=trace_visible,
                    name=titles[combo_index],
                )
            )

    if mode == "surface":
        fig.update_layout(
            title=titles[0],
            scene=dict(xaxis_title=f"axis {ax0}", yaxis_title=f"axis {ax1}"),
        )
    else:
        if show_blocks:
            _add_2d_heatmap_boundaries(
                fig,
                bgd,
                ax0,
                ax1,
                tail_blocks=tail_blocks,
                show_internal_grid=show_internal_grid,
            )
        fig.update_layout(
            title=titles[0],
            xaxis_title=f"axis {ax0}",
            yaxis_title=f"axis {ax1}",
        )

    if len(enum_combos) > 1:
        buttons = []
        for index, title in enumerate(titles):
            visible = [False] * len(titles)
            visible[index] = True
            buttons.append(
                dict(
                    label=title,
                    method="update",
                    args=[{"visible": visible}, {"title": title}],
                )
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=1.02,
                    xanchor="left",
                    y=1.0,
                    yanchor="top",
                )
            ]
        )

    if output_html is not None:
        fig.write_html(str(output_html))
    return _safe_show(fig, fallback_html) if show else fig
