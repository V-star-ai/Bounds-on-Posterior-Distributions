from distributions import MUD, BGD
from collections.abc import Sequence
from fractions import Fraction
import numpy as np
from numbers import Real
import math


def _as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def empty_1d_bgd(mode) -> BGD:
    """Create a one-dimensional empty BGD."""

    if mode == 'MUD':
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        center = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[0], beta=[0])


def normal_to_bgd(mean, var, mode, center_subdivision=None, block_subdivision=None) -> BGD:
    """Convert Normal(mean, var) to a one-dimensional BGD upper approximation.

    center_subdivision:
        Global breakpoints of the center block.
        If None, use the default:
            1) scaled_std_trunc = truncate(1.5 * std, 1 decimal)
            2) T                = max(scaled_std_trunc, 1.0)
            3) mean_trunc       = truncate(mean, 1 decimal)
            4) interval         = [mean_trunc - T, mean_trunc + T]
            5) breakpoints every 0.1, inclusive endpoints

    block_subdivision:
        Local breakpoints of both tail blocks.
        If None, use (0, 0.1, 0.2, ..., 1).
    """
    if var <= 0:
        raise ValueError("var must be positive")

    if mode == 'MUD':
        mean_f = float(mean)
        var_f = float(var)
        std = math.sqrt(var_f)

        normal_const = 1.0 / math.sqrt(2.0 * math.pi * var_f)

        # ---------------- center block ----------------
        if center_subdivision is None:
            # 1) scaled_std_trunc = truncate(1.5 * std, 1 decimal)
            # 2) T                = max(scaled_std_trunc, 1.0)
            # 3) mean_trunc       = truncate(mean, 1 decimal)
            # 4) interval         = [mean_trunc - T, mean_trunc + T]
            # 5) breakpoints every 0.1, inclusive endpoints

            scaled_std_tick = math.trunc(1.5 * std * 10)
            T_tick = max(scaled_std_tick, 10)

            mean_tick = math.trunc(mean_f * 10)

            left_tick = mean_tick - T_tick
            right_tick = mean_tick + T_tick

            center_points = tuple(
                Fraction(i, 10) for i in range(left_tick, right_tick + 1)
            )
        else:
            if len(center_subdivision) < 2:
                raise ValueError("center_subdivision must contain at least two points")

            center_points = tuple(_as_fraction(x) for x in center_subdivision)

            for left, right in zip(center_points, center_points[1:]):
                if not left < right:
                    raise ValueError("center_subdivision must be strictly increasing")

            if not float(center_points[0]) <= mean_f <= float(center_points[-1]):
                raise ValueError("center_subdivision must satisfy left <= mean <= right")

        center_left = center_points[0]
        center_right = center_points[-1]

        center_masses = []

        for left, right in zip(center_points, center_points[1:]):
            left_f = float(left)
            right_f = float(right)

            # The maximum on [left, right] is:
            #   - at mean, if mean is inside the interval;
            #   - otherwise at the endpoint closer to mean.
            if left_f <= mean_f <= right_f:
                max_point = mean_f
            elif right_f < mean_f:
                max_point = right_f
            else:
                max_point = left_f

            upper_density = normal_const * math.exp(
                -((max_point - mean_f) ** 2) / (2.0 * var_f)
            )

            mass = upper_density * float(right - left)
            center_masses.append(mass)

        center = MUD([center_points], np.array(center_masses, dtype=object))

        # ---------------- tail block subdivision ----------------
        if block_subdivision is None:
            # local breakpoints: 0, 0.1, ..., 1
            block_points = tuple(Fraction(i, 10) for i in range(11))
        else:
            if not isinstance(block_subdivision, (tuple, list)):
                raise ValueError("block_subdivision must be a tuple or list")

            if len(block_subdivision) < 2:
                raise ValueError("block_subdivision must contain at least two points")

            block_points = tuple(_as_fraction(x) for x in block_subdivision)

            if block_points[0] != 0:
                raise ValueError("block_subdivision must start at 0")

            if not block_points[-1] > 0:
                raise ValueError("block_subdivision must end with a positive value")

            for left, right in zip(block_points, block_points[1:]):
                if not left < right:
                    raise ValueError("block_subdivision must be strictly increasing")

        block_length = block_points[-1]
        block_length_f = float(block_length)

        # ---------------- left exponential tail upper bound ----------------
        # For x <= center_left:
        #   f(x) <= P0_left * exp(-rate_left * (center_left - x))
        #
        # u = mean - center_left >= 0
        # d = (u + sqrt(u^2 + 4 * var)) / 2
        # rate = d / var
        # P0 = f(center_left) * exp(var / (2 * d^2))

        u_left = mean_f - float(center_left)

        if u_left < 0:
            raise ValueError("center left endpoint must be <= mean")

        d_left = (u_left + math.sqrt(u_left * u_left + 4.0 * var_f)) / 2.0
        rate_left = d_left / var_f

        f_left = normal_const * math.exp(
            -((float(center_left) - mean_f) ** 2) / (2.0 * var_f)
        )

        P0_left = f_left * math.exp(var_f / (2.0 * d_left * d_left))

        left_masses = []

        for local_left, local_right in zip(block_points, block_points[1:]):
            # Left first block is local [0, L].
            # Globally it is [center_left - L, center_left].
            #
            # local interval [a,b] maps to global:
            #   [center_left - L + a, center_left - L + b]
            #
            # Let y = center_left - x.
            # Then y ranges from L-b to L-a.
            # The envelope P0 * exp(-rate*y) is decreasing in y,
            # so the maximum is at y = L-b.
            y_min = float(block_length - local_right)

            upper_density = P0_left * math.exp(-rate_left * y_min)
            mass = upper_density * float(local_right - local_left)

            left_masses.append(mass)

        left = MUD([block_points], np.array(left_masses, dtype=object))

        # ---------------- right exponential tail upper bound ----------------
        # For x >= center_right:
        #   f(x) <= P0_right * exp(-rate_right * (x - center_right))

        u_right = float(center_right) - mean_f

        if u_right < 0:
            raise ValueError("center right endpoint must be >= mean")

        d_right = (u_right + math.sqrt(u_right * u_right + 4.0 * var_f)) / 2.0
        rate_right = d_right / var_f

        f_right = normal_const * math.exp(
            -((float(center_right) - mean_f) ** 2) / (2.0 * var_f)
        )

        P0_right = f_right * math.exp(var_f / (2.0 * d_right * d_right))

        right_masses = []

        for local_left, local_right in zip(block_points, block_points[1:]):
            # Right first block local [a,b] maps globally to:
            #   [center_right + a, center_right + b]
            #
            # Let y = x - center_right.
            # Then y ranges from a to b.
            # The envelope P0 * exp(-rate*y) is decreasing in y,
            # so the maximum is at y = a.
            y_min = float(local_left)

            upper_density = P0_right * math.exp(-rate_right * y_min)
            mass = upper_density * float(local_right - local_left)

            right_masses.append(mass)

        right = MUD([block_points], np.array(right_masses, dtype=object))

        alpha = math.exp(-rate_left * block_length_f)
        beta = math.exp(-rate_right * block_length_f)

        E = np.array([left, center, right], dtype=object)

    else:
        pass

    return BGD(E, alpha=[alpha], beta=[beta])


def exponential_to_bgd(lam, mode, center_subdivision=None, block_subdivision=None) -> BGD:
    """Convert Exp(lam) to a one-dimensional BGD upper approximation.

    center_subdivision:
        Global breakpoints of the center block.
        If None, the center block is empty.

    block_subdivision:
        Local breakpoints of the right tail block.
        If None, use (0, 0.1, 0.2, ..., 1).
    """
    if not isinstance(lam, Real) or isinstance(lam, bool):
        raise ValueError("lam must be a positive numeric value")

    if lam <= 0:
        raise ValueError("lam must be positive")

    lam_f = float(lam)

    if mode == 'MUD':
        # ---------------- center block ----------------
        if center_subdivision is None:
            center = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
            center_right = Fraction(0)
        else:
            if len(center_subdivision) < 2:
                raise ValueError("center_subdivision must contain at least two points")
    
            center_points = tuple(_as_fraction(x) for x in center_subdivision)
    
            for left, right in zip(center_points, center_points[1:]):
                if not left < right:
                    raise ValueError("center_subdivision must be strictly increasing")
    
            if not center_points[0] <= 0:
                raise ValueError("center_subdivision must start with a non-positive value")
    
            if not center_points[-1] > 0:
                raise ValueError("center_subdivision must end with a positive value")
    
            center_masses = []
    
            for left, right in zip(center_points, center_points[1:]):
                if right <= 0:
                    mass = 0
                else:
                    start = max(left, Fraction(0))
                    upper_density = lam_f * math.exp(-lam_f * float(start))
                    mass = upper_density * float(right - left)
    
                center_masses.append(mass)
    
            center = MUD([center_points], np.array(center_masses, dtype=object))
            center_right = center_points[-1]
    
        # ---------------- right tail block ----------------
        if block_subdivision is None:
            # local breakpoints: 0, 0.1, ..., 1
            block_points = tuple(Fraction(i, 10) for i in range(11))
        else:
            if len(block_subdivision) < 2:
                raise ValueError("block_subdivision must contain at least two points")
    
            block_points = tuple(_as_fraction(x) for x in block_subdivision)
            if block_points[0] != 0:
                raise ValueError("block_subdivision must start at 0")
            if not block_points[-1] > 0:
                raise ValueError("block_subdivision must end with a positive value")
    
            for left, right in zip(block_points, block_points[1:]):
                if not left < right:
                    raise ValueError("block_subdivision must be strictly increasing")
    
        block_length = block_points[-1]
    
        right_masses = []
    
        for local_left, local_right in zip(block_points, block_points[1:]):
            global_left = center_right + local_left
            global_right = center_right + local_right
    
            start = global_left
            upper_density = lam_f * math.exp(-lam_f * float(start))
            mass = upper_density * float(global_right - global_left)
    
            right_masses.append(mass)
    
        right = MUD([block_points], np.array(right_masses, dtype=object))
    
        # ---------------- left tail block ----------------
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
    
        alpha = 0
        beta = math.exp(-lam_f * float(block_length))
    
        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[alpha], beta=[beta])


def mapping_to_bgd(mapping, mode) -> BGD:
    """Convert a finite discrete distribution mapping to a one-dimensional BGD."""

    if not mapping:
        return empty_1d_bgd(mode)
        
    items = sorted(mapping.items(), key=lambda item: _as_fraction(item[0]))
    
    breakpoints = []
    masses = []
    
    first_point, first_mass = items[0]
    first_point = _as_fraction(first_point)
    
    # First Dirac interval [x0, x0].
    breakpoints.extend([first_point, first_point])
    masses.append(first_mass)
    
    for point, mass in items[1:]:
        point = _as_fraction(point)
    
        # Zero-mass gap from previous point to current point.
        breakpoints.append(point)
        masses.append(0)
    
        # Dirac interval [point, point].
        breakpoints.append(point)
        masses.append(mass)

    if mode == 'MUD':
        center = MUD([tuple(breakpoints)], np.array(masses, dtype=object))

        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[0], beta=[0])


def uniform_to_bgd(a, b, mode) -> BGD:
    """Convert ('Uniform', (a, b)) to a one-dimensional BGD."""
    a = _as_fraction(a)
    b = _as_fraction(b)

    if not a < b:
        raise ValueError("Uniform requires a < b")

    if mode == 'MUD':
        # Center block: Uniform[a,b] with total mass 1.
        center = MUD([(a, b)], [1])

        # Empty tail-start blocks.
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))

        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[0], beta=[0])


def num_to_bgd(num, mode) -> BGD:
    """Convert x = num to a one-dimensional BGD Dirac distribution."""
    point = _as_fraction(num)

    if mode == 'MUD':
        center = MUD([(point, point)], np.array([1], dtype=object))
    
        left = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
        right = MUD([(Fraction(0),)], np.empty((0,), dtype=object))
    
        E = np.array([left, center, right], dtype=object)
    else:
        pass

    return BGD(E, alpha=[0], beta=[0])


def merge_bgds(bgds: Sequence[BGD]) -> BGD:
    """
    Merge a list of independent BGD components into one joint BGD.
    The dimension order follows the list order.
    """
    if len(bgds) == 0:
        raise ValueError("bgds must be non-empty")

    result = bgds[0]
    if not isinstance(result, BGD):
        raise TypeError("all elements of bgds must be BGD")

    for bgd in bgds[1:]:
        if not isinstance(bgd, BGD):
            raise TypeError("all elements of bgds must be BGD")
        result = result.independent_product(bgd)

    return result


def prior_to_bgd(prior: dict, mode, center_subdivision=None, block_subdivision=None) -> tuple[BGD, tuple]:
    """Convert a prior dict to a multi-dimensional BGD.

    prior format example:
        {
            ('x',): ('Normal', (0, 1)),
            ('y',): ('Uniform', (0, 1)),
            ('t',): ('Exponential', (1,)),
            ('w',): ('Mapping', {0: 1/2, 1: 1/2}),
            ('z',): ('Num', 0)
        }

    Return:
        (joint_bgd, variable_order)
    """
    if not prior:
        return empty_1d_bgd(mode), ('x',)
        
    bgds = []
    variable_order = []

    for variables, dist_spec in prior.items():
        # variables should be a tuple of variable names.
        # Example: ('x',)
        variable_order.extend(variables)

        dist_name, params = dist_spec

        if dist_name == 'Normal':
            mean, var = params
            bgd = normal_to_bgd(
                mean,
                var,
                mode,
                center_subdivision=center_subdivision,
                block_subdivision=block_subdivision,
            )

        elif dist_name == 'Exponential':
            (lam,) = params
            bgd = exponential_to_bgd(
                lam,
                mode,
                center_subdivision=center_subdivision,
                block_subdivision=block_subdivision,
            )

        elif dist_name == 'Uniform':
            (a, b) = params
            bgd = uniform_to_bgd(a, b, mode)

        elif dist_name == 'Mapping':
            bgd = mapping_to_bgd(params, mode)

        elif dist_name == 'Num':
            bgd = num_to_bgd(params, mode)

        else:
            raise ValueError(f"unsupported prior distribution: {dist_name!r}")

        if len(variables) != bgd.ndim:
            raise ValueError(
                f"variables {variables} has length {len(variables)}, "
                f"but converted BGD has dimension {bgd.ndim}"
            )

        bgds.append(bgd)

    result = bgds[0]
    for bgd in bgds[1:]:
        result = result.independent_product(bgd)

    return result, tuple(variable_order)
    
