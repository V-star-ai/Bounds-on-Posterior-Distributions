from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

from semantics.polynomial import (
    ParameterPolynomial,
    ParameterVariable,
    StatePolynomial,
)


class Relation(Enum):
    EQ = "=="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"


@dataclass(frozen=True)
class UnitBoxDomain:
    ndim: int
    active_dims: tuple[int, ...]

    def __init__(self, ndim: int, active_dims: Sequence[int] | None = None):
        if not isinstance(ndim, int) or isinstance(ndim, bool):
            raise TypeError("ndim must be an integer")
        if ndim < 0:
            raise ValueError("ndim must be nonnegative")
        dims = tuple(range(ndim)) if active_dims is None else tuple(active_dims)
        if len(set(dims)) != len(dims):
            raise ValueError("active_dims must not contain duplicates")
        if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in dims):
            raise TypeError("active_dims entries must be integers")
        if any(dim < 0 or dim >= ndim for dim in dims):
            raise ValueError("active_dims entry out of range")

        object.__setattr__(self, "ndim", ndim)
        object.__setattr__(self, "active_dims", tuple(sorted(dims)))


@dataclass(frozen=True)
class ParameterConstraint:
    polynomial: ParameterPolynomial
    relation: Relation

    def __init__(self, polynomial, relation: Relation):
        if not isinstance(relation, Relation):
            raise TypeError("relation must be a Relation")
        object.__setattr__(self, "polynomial", ParameterPolynomial.coerce(polynomial))
        object.__setattr__(self, "relation", relation)

    def evaluate(self, parameter_values) -> bool:
        return _compare(self.polynomial.evaluate(parameter_values), self.relation)


@dataclass(frozen=True)
class DomainPolynomialConstraint:
    polynomial: StatePolynomial
    relation: Relation
    domain: UnitBoxDomain

    def __init__(
        self,
        polynomial: StatePolynomial,
        relation: Relation,
        domain: UnitBoxDomain | None = None,
    ):
        if not isinstance(polynomial, StatePolynomial):
            raise TypeError("polynomial must be a StatePolynomial")
        if not isinstance(relation, Relation):
            raise TypeError("relation must be a Relation")
        domain = UnitBoxDomain(polynomial.ndim) if domain is None else domain
        if not isinstance(domain, UnitBoxDomain):
            raise TypeError("domain must be a UnitBoxDomain")
        if domain.ndim != polynomial.ndim:
            raise ValueError("domain and polynomial dimensions do not match")
        for dim in range(polynomial.ndim):
            if dim not in domain.active_dims and polynomial.depends_on(dim):
                raise ValueError(
                    f"polynomial depends on inactive domain dimension {dim}"
                )

        object.__setattr__(self, "polynomial", polynomial)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "domain", domain)

    def evaluate_at(self, state_values, parameter_values=None) -> bool:
        if len(state_values) != self.polynomial.ndim:
            raise ValueError(
                f"state_values must contain {self.polynomial.ndim} values"
            )
        for dim in self.domain.active_dims:
            value = state_values[dim]
            if value < 0 or value > 1:
                raise ValueError(
                    f"state value in active dimension {dim} is outside [0, 1]"
                )
        return _compare(
            self.polynomial.evaluate(state_values, parameter_values),
            self.relation,
        )


@dataclass(frozen=True)
class PolynomialIdentity:
    left: StatePolynomial
    right: StatePolynomial

    def __post_init__(self) -> None:
        if not isinstance(self.left, StatePolynomial) or not isinstance(
            self.right, StatePolynomial
        ):
            raise TypeError("identity sides must be StatePolynomial objects")
        if self.left.ndim != self.right.ndim:
            raise ValueError("identity polynomial dimensions do not match")

    @property
    def difference(self) -> StatePolynomial:
        return self.left - self.right

    def coefficient_constraints(self) -> tuple[ParameterConstraint, ...]:
        return tuple(
            ParameterConstraint(coefficient, Relation.EQ)
            for coefficient in self.difference.terms.values()
        )


SemanticConstraint = (
    ParameterConstraint | DomainPolynomialConstraint | PolynomialIdentity
)


@dataclass(frozen=True)
class ConstraintProblem:
    variables: tuple[ParameterVariable, ...]
    constraints: tuple[SemanticConstraint, ...]


class ConstraintContext:
    """Mutable builder for an immutable semantic constraint problem."""

    def __init__(self):
        self._variables: dict[str, ParameterVariable] = {}
        self._constraints: list[SemanticConstraint] = []
        self._fresh_counters: dict[str, int] = {}

    @property
    def variables(self) -> tuple[ParameterVariable, ...]:
        return tuple(self._variables.values())

    @property
    def constraints(self) -> tuple[SemanticConstraint, ...]:
        return tuple(self._constraints)

    def declare(self, name: str) -> ParameterVariable:
        variable = ParameterVariable(name)
        if variable.name in self._variables:
            raise KeyError(f"parameter variable {variable.name!r} already exists")
        self._variables[variable.name] = variable
        return variable

    def fresh(self, prefix: str = "aux") -> ParameterVariable:
        if not isinstance(prefix, str) or not prefix.strip():
            raise ValueError("fresh variable prefix must be a non-empty string")
        prefix = prefix.strip()
        counter = self._fresh_counters.get(prefix, 0)
        while True:
            name = f"{prefix}_{counter}"
            counter += 1
            if name not in self._variables:
                self._fresh_counters[prefix] = counter
                return self.declare(name)

    def add(self, constraint: SemanticConstraint) -> SemanticConstraint:
        if not isinstance(
            constraint,
            (ParameterConstraint, DomainPolynomialConstraint, PolynomialIdentity),
        ):
            raise TypeError("unsupported semantic constraint")
        self._constraints.append(constraint)
        return constraint

    def constrain_parameter(
        self,
        left,
        relation: Relation,
        right=0,
    ) -> ParameterConstraint:
        constraint = ParameterConstraint(
            ParameterPolynomial.coerce(left) - ParameterPolynomial.coerce(right),
            relation,
        )
        self.add(constraint)
        return constraint

    def constrain_domain(
        self,
        left: StatePolynomial,
        relation: Relation,
        right=0,
        *,
        domain: UnitBoxDomain | None = None,
    ) -> DomainPolynomialConstraint:
        constraint = DomainPolynomialConstraint(
            left - right,
            relation,
            domain,
        )
        self.add(constraint)
        return constraint

    def constrain_identity(
        self,
        left: StatePolynomial,
        right: StatePolynomial,
    ) -> PolynomialIdentity:
        identity = PolynomialIdentity(left, right)
        self.add(identity)
        return identity

    def exact_positive_quotient(
        self,
        numerator,
        denominator,
        *,
        prefix: str = "quotient",
    ) -> ParameterPolynomial:
        numerator = ParameterPolynomial.coerce(numerator)
        denominator = ParameterPolynomial.coerce(denominator)
        if denominator.is_zero:
            raise ZeroDivisionError("quotient denominator is zero")

        quotient_variable = self.fresh(prefix)
        quotient = ParameterPolynomial.variable(quotient_variable)
        self.constrain_parameter(denominator, Relation.GT, 0)
        self.constrain_parameter(
            denominator * quotient,
            Relation.EQ,
            numerator,
        )
        return quotient

    def build(self) -> ConstraintProblem:
        declared = set(self._variables.values())
        referenced = {
            variable
            for constraint in self._constraints
            for variable in _constraint_variables(constraint)
        }
        undeclared = sorted(
            referenced - declared,
            key=lambda variable: variable.name,
        )
        if undeclared:
            names = ", ".join(variable.name for variable in undeclared)
            raise ValueError(f"constraints reference undeclared parameters: {names}")
        return ConstraintProblem(self.variables, self.constraints)


def _compare(value, relation: Relation) -> bool:
    if relation is Relation.EQ:
        return value == 0
    if relation is Relation.GE:
        return value >= 0
    if relation is Relation.GT:
        return value > 0
    if relation is Relation.LE:
        return value <= 0
    if relation is Relation.LT:
        return value < 0
    raise TypeError(relation)


def _constraint_variables(
    constraint: SemanticConstraint,
) -> tuple[ParameterVariable, ...]:
    if isinstance(constraint, ParameterConstraint):
        return constraint.polynomial.variables
    if isinstance(constraint, DomainPolynomialConstraint):
        return constraint.polynomial.parameter_variables
    if isinstance(constraint, PolynomialIdentity):
        return tuple(
            sorted(
                set(constraint.left.parameter_variables)
                | set(constraint.right.parameter_variables),
                key=lambda variable: variable.name,
            )
        )
    raise TypeError(constraint)
