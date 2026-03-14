from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from distributions import EED


SpecType = Union[
    float,
    int,
    Tuple[str, Any],
    Dict[str, Any],
]


@dataclass
class _Spec:
    kind: str  # "const" | "enum" | "var"
    value: Any


def _parse_spec(spec: SpecType) -> _Spec:
    if isinstance(spec, (int, float)):
        return _Spec("const", float(spec))
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


def _enum_points_for_axis(si: Sequence[int], is_discrete: bool = False) -> List[Tuple[str, float]]:
    if len(si) < 1:
        raise ValueError("S must have at least one breakpoint")
    pts: List[Tuple[str, float]] = []
    if is_discrete:
        s0 = int(si[0])
        sl = int(si[-1])
        pts.append((f"x < {si[0]}", float(s0 - 1)))
        for v in si:
            pts.append((f"x = {v}", float(v)))
        pts.append((f"x > {si[-1]}", float(sl + 1)))
        return pts
    s0 = float(si[0])
    sl = float(si[-1])
    pts.append((f"x < {si[0]}", s0 - 1.0))
    for i in range(1, len(si)):
        a = float(si[i - 1])
        b = float(si[i])
        mid = (a + b) / 2.0
        pts.append((f"[{si[i-1]},{si[i]})", mid))
    pts.append((f"x >= {si[-1]}", sl + 1.0))
    return pts


def _eval_eed_at(eed: EED, x: Sequence[float]) -> float:
    if eed.P.ndim != eed.n:
        raise ValueError("plot_eed only supports scalar EED (P has no extra dims)")
    if len(x) != eed.n:
        raise ValueError("Point dimension does not match EED")

    idx = []
    factor = 1.0
    for axis, (xi, si, a, b, is_discrete) in enumerate(zip(x, eed.S, eed.alpha, eed.beta, eed.discrete_mask)):
        if is_discrete:
            xi_int = int(round(xi))
            if abs(xi - xi_int) > 1e-9:
                return 0.0
            left = int(si[0])
            right = int(si[-1])
            if xi_int < left:
                idx.append(0)
                factor *= float(a) ** (left - xi_int)
            elif xi_int > right:
                idx.append(len(si) - 1)
                factor *= float(b) ** (xi_int - right)
            else:
                hits = np.where(si == xi_int)[0]
                if len(hits) == 0:
                    return 0.0
                idx.append(int(hits[0]))
            continue
        s0 = float(si[0])
        sl = float(si[-1])
        m = len(si)
        if xi < s0:
            idx.append(0)
            factor *= float(a) ** (s0 - xi)
        elif xi >= sl:
            idx.append(m)
            factor *= float(b) ** (xi - sl)
        else:
            j = int(np.searchsorted(si, xi, side="right") - 1)  # 0..m-2
            idx.append(1 + j)
    base = float(eed.P[tuple(idx)])
    return base * factor


def _auto_num(span: float, *, min_n: int, max_n: int, scale: float) -> int:
    n = int(max(min_n, min(max_n, span * scale)))
    return n


def _var_points_for_axis(eed: EED, axis: int, spec_value: Any):
    si = eed.S[axis]
    if eed.discrete_mask[axis]:
        if spec_value is None:
            left = int(si[0]) - 3
            right = int(si[-1]) + 3
            return np.arange(left, right + 1, dtype=float)
        v = spec_value if isinstance(spec_value, dict) else {}
        xmin = int(v.get("min", int(si[0]) - 3))
        xmax = int(v.get("max", int(si[-1]) + 3))
        step = int(v.get("step", 1))
        if step <= 0:
            raise ValueError("Discrete var axis step must be positive")
        return np.arange(xmin, xmax + 1, step, dtype=float)

    if spec_value is None:
        span = float(si[-1] - si[0])
        ext = max(1.0, span * 0.5)
        num = _auto_num(span + 2 * ext, min_n=80, max_n=600, scale=40.0)
        v = {
            "min": float(si[0]) - ext,
            "max": float(si[-1]) + ext,
            "num": num,
        }
    else:
        v = spec_value if isinstance(spec_value, dict) else {}
    vmin = v.get("min", float(si[0]) - 3.0)
    vmax = v.get("max", float(si[-1]) + 3.0)
    num = int(v.get("num", 200))
    return np.linspace(vmin, vmax, num)


def _expand_surface_discrete_axis(points: np.ndarray, values: np.ndarray, axis: int, eps: float = 0.25):
    if eps <= 0 or eps >= 0.5:
        raise ValueError("Discrete surface epsilon must satisfy 0 < eps < 0.5")

    expanded_points = []
    if axis == 0:
        expanded_values = np.zeros((values.shape[0], values.shape[1] * 3), dtype=float)
        for j, p in enumerate(points):
            expanded_points.extend([p - eps, p, p + eps])
            expanded_values[:, 3 * j + 1] = values[:, j]
        return np.asarray(expanded_points, dtype=float), expanded_values

    expanded_values = np.zeros((values.shape[0] * 3, values.shape[1]), dtype=float)
    for i, p in enumerate(points):
        expanded_points.extend([p - eps, p, p + eps])
        expanded_values[3 * i + 1, :] = values[i, :]
    return np.asarray(expanded_points, dtype=float), expanded_values


def plot_eed(
    eed: EED,
    specs: Sequence[SpecType],
    *,
    mode: str = "surface",  # "heatmap" | "surface"
):
    if len(specs) != eed.n:
        raise ValueError("specs length must equal EED dimension")

    parsed = [_parse_spec(s) for s in specs]
    var_axes = [i for i, s in enumerate(parsed) if s.kind == "var"]
    if len(var_axes) > 2:
        raise ValueError("At most two variable dimensions are allowed")
    # Build fixed values and enum combinations
    enum_axes = [i for i, s in enumerate(parsed) if s.kind == "enum"]
    enum_lists = []
    for i in enum_axes:
        enum_lists.append(_enum_points_for_axis(eed.S[i], eed.discrete_mask[i]))

    enum_combos = list(product(*enum_lists)) if enum_lists else [()]

    def get_const_for_axis(i: int) -> float:
        spec = parsed[i]
        if spec.kind == "const":
            return float(spec.value)
        if spec.kind == "enum":
            raise ValueError("enum axis requires combo value")
        if spec.kind == "var":
            raise ValueError("var axis is not a constant")
        raise ValueError("invalid spec kind")

    # Variable axis grids
    if mode not in ("heatmap", "surface"):
        raise ValueError("mode must be 'heatmap' or 'surface'")

    if len(var_axes) == 0:
        # Single value, possibly multiple enum combos
        results = []
        for combo in enum_combos:
            x = [None] * eed.n
            for i, spec in enumerate(parsed):
                if spec.kind == "const":
                    x[i] = float(spec.value)
            for (axis, (label, val)) in zip(enum_axes, combo):
                x[axis] = val
            results.append((_eval_eed_at(eed, x), combo))
        return results

    if len(var_axes) == 1:
        axis = var_axes[0]
        xs = _var_points_for_axis(eed, axis, parsed[axis].value)

        fig = go.Figure()
        for combo in enum_combos:
            ys = []
            for xv in xs:
                x = [None] * eed.n
                for i, spec in enumerate(parsed):
                    if spec.kind == "const":
                        x[i] = float(spec.value)
                for (axis_e, (label, val)) in zip(enum_axes, combo):
                    x[axis_e] = val
                x[axis] = float(xv)
                ys.append(_eval_eed_at(eed, x))
            label = ", ".join(lbl for (lbl, _) in combo) if combo else None
            if eed.discrete_mask[axis]:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=label))
            else:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label))
        fig.update_layout(xaxis_title=f"axis {axis}", yaxis_title="EED value")
        fig.show()
        return fig

    # len(var_axes) == 2
    ax0, ax1 = var_axes
    xs = _var_points_for_axis(eed, ax0, parsed[ax0].value)
    ys = _var_points_for_axis(eed, ax1, parsed[ax1].value)
    n0 = len(xs)
    n1 = len(ys)
    X, Y = np.meshgrid(xs, ys)

    rows = len(enum_combos)
    titles = [
        ", ".join(lbl for (lbl, _) in combo) if combo else "EED"
        for combo in enum_combos
    ]

    # If multiple enum combos, use a dropdown to show one at a time.
    use_dropdown = rows > 1
    fig = go.Figure()

    for r, combo in enumerate(enum_combos, start=1):
        Z = np.zeros_like(X, dtype=float)
        for i in range(n1):
            for j in range(n0):
                x = [None] * eed.n
                for k, spec in enumerate(parsed):
                    if spec.kind == "const":
                        x[k] = float(spec.value)
                for (axis_e, (label, val)) in zip(enum_axes, combo):
                    x[axis_e] = val
                x[ax0] = float(X[i, j])
                x[ax1] = float(Y[i, j])
                Z[i, j] = _eval_eed_at(eed, x)
        if mode == "surface":
            plot_xs = xs
            plot_ys = ys
            plot_Z = Z
            if eed.discrete_mask[ax0]:
                plot_xs, plot_Z = _expand_surface_discrete_axis(plot_xs, plot_Z, axis=0)
            if eed.discrete_mask[ax1]:
                plot_ys, plot_Z = _expand_surface_discrete_axis(plot_ys, plot_Z, axis=1)
            fig.add_trace(
                go.Surface(
                    z=plot_Z,
                    x=plot_xs,
                    y=plot_ys,
                    showscale=True,
                    colorbar=dict(title="EED"),
                    visible=(r == 1),
                    name=titles[r - 1],
                )
            )
        else:
            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    x=xs,
                    y=ys,
                    colorbar=dict(title="EED"),
                    visible=(r == 1),
                    name=titles[r - 1],
                )
            )

    if mode == "surface":
        fig.update_layout(scene=dict(xaxis_title=f"axis {ax0}", yaxis_title=f"axis {ax1}"))
    else:
        fig.update_xaxes(title_text=f"axis {ax0}")
        fig.update_yaxes(title_text=f"axis {ax1}")

    if use_dropdown:
        buttons = []
        for i, title in enumerate(titles):
            vis = [False] * len(titles)
            vis[i] = True
            buttons.append(
                dict(
                    label=title,
                    method="update",
                    args=[{"visible": vis}, {"title": title}],
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
            ],
            title=titles[0],
        )
    else:
        fig.update_layout(title=titles[0])
    fig.show()
    return fig
