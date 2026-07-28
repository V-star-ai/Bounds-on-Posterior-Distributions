from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from numbers import Integral, Rational
from typing import Iterable, Mapping, Sequence


def _as_fraction(value) -> Fraction:
    if isinstance(value, bool):
        raise TypeError("boolean values are not polynomial coefficients")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    if isinstance(value, Rational):
        return Fraction(value)
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"expected an exact rational value, got {value!r}") from exc


@dataclass(frozen=True, order=True)
class ParameterVariable:
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter variable name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("parameter variable name must not have surrounding whitespace")


ParameterMonomial = tuple[tuple[ParameterVariable, int], ...]


def _normalize_parameter_monomial(monomial) -> ParameterMonomial:
    if isinstance(monomial, Mapping):
        items = monomial.items()
    else:
        items = monomial

    exponents: dict[ParameterVariable, int] = {}
    for variable, exponent in items:
        if isinstance(variable, str):
            variable = ParameterVariable(variable)
        if not isinstance(variable, ParameterVariable):
            raise TypeError("parameter monomial keys must be ParameterVariable objects")
        if not isinstance(exponent, Integral) or isinstance(exponent, bool):
            raise TypeError("parameter exponents must be integers")
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("parameter exponents must be nonnegative")
        if exponent:
            exponents[variable] = exponents.get(variable, 0) + exponent
    return tuple(sorted(exponents.items(), key=lambda item: item[0].name))


def _multiply_parameter_monomials(
    left: ParameterMonomial,
    right: ParameterMonomial,
) -> ParameterMonomial:
    return _normalize_parameter_monomial(left + right)


class ParameterPolynomial:
    """Sparse exact polynomial in symbolic parameter variables."""

    __slots__ = ("_terms",)

    def __init__(self, terms: Mapping | Iterable[tuple] | None = None):
        combined: dict[ParameterMonomial, Fraction] = {}
        items = () if terms is None else (terms.items() if isinstance(terms, Mapping) else terms)
        for monomial, coefficient in items:
            normalized = _normalize_parameter_monomial(monomial)
            coefficient = _as_fraction(coefficient)
            if coefficient:
                combined[normalized] = combined.get(normalized, Fraction(0)) + coefficient
        self._terms = tuple(
            sorted(
                (
                    (monomial, coefficient)
                    for monomial, coefficient in combined.items()
                    if coefficient
                ),
                key=lambda item: tuple(
                    (variable.name, exponent) for variable, exponent in item[0]
                ),
            )
        )

    @classmethod
    def zero(cls) -> "ParameterPolynomial":
        return cls()

    @classmethod
    def constant(cls, value) -> "ParameterPolynomial":
        coefficient = _as_fraction(value)
        return cls({(): coefficient}) if coefficient else cls.zero()

    @classmethod
    def variable(cls, variable: ParameterVariable | str) -> "ParameterPolynomial":
        if isinstance(variable, str):
            variable = ParameterVariable(variable)
        if not isinstance(variable, ParameterVariable):
            raise TypeError("variable must be a ParameterVariable")
        return cls({((variable, 1),): Fraction(1)})

    @classmethod
    def coerce(cls, value) -> "ParameterPolynomial":
        if isinstance(value, cls):
            return value
        if isinstance(value, ParameterVariable):
            return cls.variable(value)
        return cls.constant(value)

    @property
    def terms(self) -> dict[ParameterMonomial, Fraction]:
        return dict(self._terms)

    @property
    def variables(self) -> tuple[ParameterVariable, ...]:
        return tuple(
            sorted(
                {
                    variable
                    for monomial, _coefficient in self._terms
                    for variable, _exponent in monomial
                },
                key=lambda variable: variable.name,
            )
        )

    @property
    def is_zero(self) -> bool:
        return not self._terms

    @property
    def is_constant(self) -> bool:
        return all(not monomial for monomial, _coefficient in self._terms)

    @property
    def constant_value(self) -> Fraction:
        if not self.is_constant:
            raise ValueError("polynomial is not constant")
        if not self._terms:
            return Fraction(0)
        return self._terms[0][1]

    def degree(self, variable: ParameterVariable | str | None = None) -> int:
        if not self._terms:
            return -1
        if variable is None:
            return max(
                sum(exponent for _var, exponent in monomial)
                for monomial, _coefficient in self._terms
            )
        if isinstance(variable, str):
            variable = ParameterVariable(variable)
        return max(
            dict(monomial).get(variable, 0)
            for monomial, _coefficient in self._terms
        )

    def coefficient(self, monomial=()) -> Fraction:
        normalized = _normalize_parameter_monomial(monomial)
        return dict(self._terms).get(normalized, Fraction(0))

    def evaluate(self, values: Mapping[ParameterVariable | str, object]) -> Fraction:
        by_name = {
            key.name if isinstance(key, ParameterVariable) else str(key): _as_fraction(value)
            for key, value in values.items()
        }
        result = Fraction(0)
        for monomial, coefficient in self._terms:
            term = coefficient
            for variable, exponent in monomial:
                if variable.name not in by_name:
                    raise KeyError(f"missing value for parameter {variable.name!r}")
                term *= by_name[variable.name] ** exponent
            result += term
        return result

    def differentiate(self, variable: ParameterVariable | str) -> "ParameterPolynomial":
        if isinstance(variable, str):
            variable = ParameterVariable(variable)
        result = {}
        for monomial, coefficient in self._terms:
            exponent_map = dict(monomial)
            exponent = exponent_map.get(variable, 0)
            if exponent == 0:
                continue
            if exponent == 1:
                del exponent_map[variable]
            else:
                exponent_map[variable] = exponent - 1
            result[_normalize_parameter_monomial(exponent_map)] = coefficient * exponent
        return ParameterPolynomial(result)

    def __add__(self, other) -> "ParameterPolynomial":
        if isinstance(other, StatePolynomial):
            return NotImplemented
        other = self.coerce(other)
        return ParameterPolynomial(self._terms + other._terms)

    def __radd__(self, other) -> "ParameterPolynomial":
        return self.__add__(other)

    def __neg__(self) -> "ParameterPolynomial":
        return ParameterPolynomial(
            (monomial, -coefficient) for monomial, coefficient in self._terms
        )

    def __sub__(self, other) -> "ParameterPolynomial":
        if isinstance(other, StatePolynomial):
            return NotImplemented
        return self + (-self.coerce(other))

    def __rsub__(self, other) -> "ParameterPolynomial":
        return self.coerce(other) - self

    def __mul__(self, other) -> "ParameterPolynomial":
        if isinstance(other, StatePolynomial):
            return NotImplemented
        other = self.coerce(other)
        return ParameterPolynomial(
            (
                _multiply_parameter_monomials(left_monomial, right_monomial),
                left_coefficient * right_coefficient,
            )
            for left_monomial, left_coefficient in self._terms
            for right_monomial, right_coefficient in other._terms
        )

    def __rmul__(self, other) -> "ParameterPolynomial":
        return self.__mul__(other)

    def __truediv__(self, divisor) -> "ParameterPolynomial":
        if isinstance(divisor, ParameterPolynomial):
            if not divisor.is_constant:
                raise TypeError("division by a nonconstant parameter polynomial is not allowed")
            divisor = divisor.constant_value
        elif isinstance(divisor, ParameterVariable):
            raise TypeError("division by a parameter variable is not allowed")
        divisor = _as_fraction(divisor)
        if divisor == 0:
            raise ZeroDivisionError("polynomial division by zero")
        return ParameterPolynomial(
            (monomial, coefficient / divisor)
            for monomial, coefficient in self._terms
        )

    def __pow__(self, exponent: int) -> "ParameterPolynomial":
        if not isinstance(exponent, Integral) or isinstance(exponent, bool):
            raise TypeError("polynomial exponent must be an integer")
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = ParameterPolynomial.constant(1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power //= 2
        return result

    def __eq__(self, other) -> bool:
        try:
            other = self.coerce(other)
        except (TypeError, ValueError):
            return False
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash(self._terms)

    def __repr__(self) -> str:
        return f"ParameterPolynomial({self.terms!r})"


StateMonomial = tuple[int, ...]


def _normalize_state_exponents(exponents: Sequence[int], ndim: int) -> StateMonomial:
    exponents = tuple(exponents)
    if len(exponents) != ndim:
        raise ValueError(f"state monomial must contain {ndim} exponents")
    result = []
    for exponent in exponents:
        if not isinstance(exponent, Integral) or isinstance(exponent, bool):
            raise TypeError("state exponents must be integers")
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("state exponents must be nonnegative")
        result.append(exponent)
    return tuple(result)


class StatePolynomial:
    """Sparse polynomial in local state variables with parameter-polynomial coefficients."""

    __slots__ = ("_ndim", "_terms")

    def __init__(
        self,
        ndim: int,
        terms: Mapping[Sequence[int], object] | Iterable[tuple[Sequence[int], object]] | None = None,
    ):
        if not isinstance(ndim, Integral) or isinstance(ndim, bool):
            raise TypeError("ndim must be an integer")
        ndim = int(ndim)
        if ndim < 0:
            raise ValueError("ndim must be nonnegative")

        combined: dict[StateMonomial, ParameterPolynomial] = {}
        items = () if terms is None else (terms.items() if isinstance(terms, Mapping) else terms)
        for exponents, coefficient in items:
            exponents = _normalize_state_exponents(exponents, ndim)
            coefficient = ParameterPolynomial.coerce(coefficient)
            if not coefficient.is_zero:
                combined[exponents] = combined.get(
                    exponents, ParameterPolynomial.zero()
                ) + coefficient

        self._ndim = ndim
        self._terms = tuple(
            sorted(
                (
                    (exponents, coefficient)
                    for exponents, coefficient in combined.items()
                    if not coefficient.is_zero
                ),
                key=lambda item: item[0],
            )
        )

    @classmethod
    def zero(cls, ndim: int) -> "StatePolynomial":
        return cls(ndim)

    @classmethod
    def constant(cls, ndim: int, value) -> "StatePolynomial":
        coefficient = ParameterPolynomial.coerce(value)
        return cls(ndim, {(0,) * ndim: coefficient}) if not coefficient.is_zero else cls.zero(ndim)

    @classmethod
    def variable(
        cls,
        ndim: int,
        dim: int,
        coefficient=1,
    ) -> "StatePolynomial":
        if dim < 0 or dim >= ndim:
            raise ValueError("dim out of range")
        exponents = [0] * ndim
        exponents[dim] = 1
        return cls(ndim, {tuple(exponents): coefficient})

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def terms(self) -> dict[StateMonomial, ParameterPolynomial]:
        return dict(self._terms)

    @property
    def parameter_variables(self) -> tuple[ParameterVariable, ...]:
        return tuple(
            sorted(
                {
                    variable
                    for _exponents, coefficient in self._terms
                    for variable in coefficient.variables
                },
                key=lambda variable: variable.name,
            )
        )

    @property
    def is_zero(self) -> bool:
        return not self._terms

    def degree(self, dim: int | None = None) -> int:
        if dim is not None and (dim < 0 or dim >= self.ndim):
            raise ValueError("dim out of range")
        if not self._terms:
            return -1
        if dim is None:
            return max(sum(exponents) for exponents, _coefficient in self._terms)
        return max(exponents[dim] for exponents, _coefficient in self._terms)

    def depends_on(self, dim: int) -> bool:
        return self.degree(dim) > 0

    def coefficient(self, exponents: Sequence[int]) -> ParameterPolynomial:
        normalized = _normalize_state_exponents(exponents, self.ndim)
        return dict(self._terms).get(normalized, ParameterPolynomial.zero())

    def _coerce(self, value) -> "StatePolynomial":
        if isinstance(value, StatePolynomial):
            if value.ndim != self.ndim:
                raise ValueError("state polynomial dimensions do not match")
            return value
        return StatePolynomial.constant(self.ndim, value)

    def evaluate(
        self,
        state_values: Sequence[object],
        parameter_values: Mapping[ParameterVariable | str, object] | None = None,
    ) -> Fraction:
        parameter_values = {} if parameter_values is None else parameter_values
        return self.evaluate_state(state_values).evaluate(parameter_values)

    def evaluate_state(
        self,
        state_values: Sequence[object],
    ) -> ParameterPolynomial:
        """Substitute state coordinates while preserving parameter symbols."""
        if len(state_values) != self.ndim:
            raise ValueError(f"state_values must contain {self.ndim} values")
        state_values = tuple(_as_fraction(value) for value in state_values)
        result = ParameterPolynomial.zero()
        for exponents, coefficient in self._terms:
            term = coefficient
            for value, exponent in zip(state_values, exponents):
                term *= value**exponent
            result += term
        return result

    def affine_substitute(
        self,
        dim: int,
        offset,
        scale,
    ) -> "StatePolynomial":
        """Substitute u_dim = offset + scale * v_dim in the same variable slot."""
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        offset = ParameterPolynomial.coerce(offset)
        scale = ParameterPolynomial.coerce(scale)
        result: dict[StateMonomial, ParameterPolynomial] = {}

        for exponents, coefficient in self._terms:
            old_exponent = exponents[dim]
            for new_exponent in range(old_exponent + 1):
                new_exponents = list(exponents)
                new_exponents[dim] = new_exponent
                new_exponents = tuple(new_exponents)
                contribution = (
                    coefficient
                    * comb(old_exponent, new_exponent)
                    * (offset ** (old_exponent - new_exponent))
                    * (scale**new_exponent)
                )
                result[new_exponents] = result.get(
                    new_exponents, ParameterPolynomial.zero()
                ) + contribution
        return StatePolynomial(self.ndim, result)

    def antiderivative(self, dim: int) -> "StatePolynomial":
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        result = {}
        for exponents, coefficient in self._terms:
            new_exponents = list(exponents)
            new_exponents[dim] += 1
            result[tuple(new_exponents)] = coefficient / new_exponents[dim]
        return StatePolynomial(self.ndim, result)

    def integrate_unit(self, dim: int, *, remove: bool = False) -> "StatePolynomial":
        if dim < 0 or dim >= self.ndim:
            raise ValueError("dim out of range")
        result_ndim = self.ndim - 1 if remove else self.ndim
        result: dict[StateMonomial, ParameterPolynomial] = {}
        for exponents, coefficient in self._terms:
            integrated = coefficient / (exponents[dim] + 1)
            if remove:
                new_exponents = exponents[:dim] + exponents[dim + 1 :]
            else:
                new_exponents_list = list(exponents)
                new_exponents_list[dim] = 0
                new_exponents = tuple(new_exponents_list)
            result[new_exponents] = result.get(
                new_exponents, ParameterPolynomial.zero()
            ) + integrated
        return StatePolynomial(result_ndim, result)

    def permute_dims(self, order: Sequence[int]) -> "StatePolynomial":
        order = tuple(order)
        if len(order) != self.ndim or set(order) != set(range(self.ndim)):
            raise ValueError("order must be a permutation of state dimensions")
        return StatePolynomial(
            self.ndim,
            (
                (
                    tuple(exponents[dim] for dim in order),
                    coefficient,
                )
                for exponents, coefficient in self._terms
            ),
        )

    def independent_product(self, other: "StatePolynomial") -> "StatePolynomial":
        if not isinstance(other, StatePolynomial):
            raise TypeError("other must be a StatePolynomial")
        return StatePolynomial(
            self.ndim + other.ndim,
            (
                (
                    left_exponents + right_exponents,
                    left_coefficient * right_coefficient,
                )
                for left_exponents, left_coefficient in self._terms
                for right_exponents, right_coefficient in other._terms
            ),
        )

    def __add__(self, other) -> "StatePolynomial":
        other = self._coerce(other)
        return StatePolynomial(self.ndim, self._terms + other._terms)

    def __radd__(self, other) -> "StatePolynomial":
        return self.__add__(other)

    def __neg__(self) -> "StatePolynomial":
        return StatePolynomial(
            self.ndim,
            (
                (exponents, -coefficient)
                for exponents, coefficient in self._terms
            ),
        )

    def __sub__(self, other) -> "StatePolynomial":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "StatePolynomial":
        return self._coerce(other) - self

    def __mul__(self, other) -> "StatePolynomial":
        other = self._coerce(other)
        return StatePolynomial(
            self.ndim,
            (
                (
                    tuple(
                        left_power + right_power
                        for left_power, right_power in zip(
                            left_exponents, right_exponents
                        )
                    ),
                    left_coefficient * right_coefficient,
                )
                for left_exponents, left_coefficient in self._terms
                for right_exponents, right_coefficient in other._terms
            ),
        )

    def __rmul__(self, other) -> "StatePolynomial":
        return self.__mul__(other)

    def __truediv__(self, divisor) -> "StatePolynomial":
        return StatePolynomial(
            self.ndim,
            (
                (exponents, coefficient / divisor)
                for exponents, coefficient in self._terms
            ),
        )

    def __pow__(self, exponent: int) -> "StatePolynomial":
        if not isinstance(exponent, Integral) or isinstance(exponent, bool):
            raise TypeError("polynomial exponent must be an integer")
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = StatePolynomial.constant(self.ndim, 1)
        base = self
        power = exponent
        while power:
            if power & 1:
                result = result * base
            base = base * base
            power //= 2
        return result

    def __eq__(self, other) -> bool:
        try:
            other = self._coerce(other)
        except (TypeError, ValueError):
            return False
        return self._terms == other._terms

    def __hash__(self) -> int:
        return hash((self.ndim, self._terms))

    def __repr__(self) -> str:
        return f"StatePolynomial(ndim={self.ndim}, terms={self.terms!r})"
