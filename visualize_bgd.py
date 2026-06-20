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
        return np.linspace(float(xmin), float(xmax), num)

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
    return np.linspace(float(xmin), float(xmax), num)


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


def _eval_mud_at(mud, local_x: Sequence[Fraction], *, value: str):
    if mud.is_empty:
        return 0.0

    index = []
    for axis, x in enumerate(local_x):
        cell_index = _cell_index_for_point(mud.S[axis], x)
        if cell_index is None:
            return 0.0
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
        return _to_float(payload, "cell payload")
    if value != "density":
        raise ValueError("value must be 'density' or 'cell_mass'")

    volume = Fraction(1)
    for left, right in intervals:
        length = right - left
        if length > 0:
            volume *= length
    return _to_float(payload / volume, "cell density")


def _eval_bgd_at(bgd: BGD, x: Sequence[float], *, value: str) -> float:
    if len(x) != bgd.ndim:
        raise ValueError("Point dimension does not match BGD")

    point = tuple(_to_fraction(v) for v in x)
    block_coord = []
    for axis, axis_value in enumerate(point):
        coord = _block_coord_for_axis(bgd, axis, axis_value)
        if coord is None:
            return 0.0
        block_coord.append(coord)

    block = bgd.block_at(tuple(block_coord))
    local_x = tuple(point[axis] - block.translation[axis] for axis in range(bgd.ndim))
    base = _eval_mud_at(block.distribution, local_x, value=value)
    return base * _to_float(block.decay_factor, "BGD decay factor")


def _fixed_point_from_specs(parsed: Sequence[_Spec], enum_axes, combo, ndim: int):
    x = [None] * ndim
    for axis, spec in enumerate(parsed):
        if spec.kind == "const":
            x[axis] = float(spec.value)
    for axis, (_label, value) in zip(enum_axes, combo):
        x[axis] = value
    return x


def _axis_boundaries(bgd: BGD, axis: int, *, tail_blocks: int) -> list[float]:
    values = [bgd.center_lefts[axis], bgd.center_rights[axis]]
    for block in range(1, tail_blocks + 1):
        if bgd.left_lengths[axis] > 0:
            values.append(bgd.center_lefts[axis] - block * bgd.left_lengths[axis])
        if bgd.right_lengths[axis] > 0:
            values.append(bgd.center_rights[axis] + block * bgd.right_lengths[axis])
    return sorted(float(value) for value in values)


def _add_1d_boundaries(fig, bgd: BGD, axis: int, *, tail_blocks: int):
    for value in _axis_boundaries(bgd, axis, tail_blocks=tail_blocks):
        fig.add_vline(
            x=value,
            line_width=1,
            line_dash="dot",
            line_color="rgba(60,60,60,0.45)",
        )


def _add_2d_heatmap_boundaries(fig, bgd: BGD, ax0: int, ax1: int, *, tail_blocks: int):
    for value in _axis_boundaries(bgd, ax0, tail_blocks=tail_blocks):
        fig.add_vline(
            x=value,
            line_width=1,
            line_dash="dot",
            line_color="rgba(60,60,60,0.45)",
        )
    for value in _axis_boundaries(bgd, ax1, tail_blocks=tail_blocks):
        fig.add_hline(
            y=value,
            line_width=1,
            line_dash="dot",
            line_color="rgba(60,60,60,0.45)",
        )


def plot_bgd(
    bgd: BGD,
    specs: Sequence[SpecType],
    *,
    mode: str = "surface",
    value: str = "density",
    tail_blocks: int = 2,
    show_blocks: bool = True,
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
            label = ", ".join(label for (label, _value) in combo) if combo else None
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label))
        if show_blocks:
            _add_1d_boundaries(fig, bgd, axis, tail_blocks=tail_blocks)
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

        trace_visible = combo_index == 0
        if mode == "surface":
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
            _add_2d_heatmap_boundaries(fig, bgd, ax0, ax1, tail_blocks=tail_blocks)
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
