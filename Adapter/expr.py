from __future__ import annotations

from _ast import Lt
from dataclasses import dataclass
from typing import Dict, Any, Sequence
from enum import Enum

from fractions import Fraction


def to_fraction(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return Fraction(x).limit_denominator()
    raise TypeError(x)


class Expr:
    def eval(self, env: Dict[str, Any]):
        raise NotImplementedError()

    @staticmethod
    def max(a, b):
        return Max(ensure_expr(a), ensure_expr(b))

    def __add__(self, other):
        return Add(self, ensure_expr(other))

    def __radd__(self, other):
        return Add(ensure_expr(other), self)

    def __sub__(self, other):
        return Sub(self, ensure_expr(other))

    def __rsub__(self, other):
        return Sub(ensure_expr(other), self)

    def __mul__(self, other):
        return Mul(self, ensure_expr(other))

    def __rmul__(self, other):
        return Mul(ensure_expr(other), self)

    def __pow__(self, exponent):
        return Pow(self, ensure_fraction(exponent))

    def __truediv__(self, other):
        return Div(self, ensure_expr(other))

    def __rtruediv__(self, other):
        return Div(ensure_expr(other), self)

    def __lt__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.LT)

    def __le__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.LE)

    def __eq__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.EQ)

    def __ne__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.NE)

    def __gt__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.GT)

    def __ge__(self, other):
        return Constraint(self, ensure_expr(other), CompareOp.GE)


def ensure_expr(v):
    if isinstance(v, Expr):
        return v
    return Const(v)

def ensure_fraction(v) -> FractionConst:
    if isinstance(v, Expr):
        if isinstance(v, Const):
            return FractionConst(Fraction(v.value))
        elif isinstance(v, FractionConst):
            return v
        raise TypeError(v)
    return FractionConst(v)

@dataclass
class Const(Expr):
    value: float

    def __init__(self, val):
        if isinstance(val, Expr):
            raise TypeError(val)
        self.value = float(val)

    def eval(self, env):
        return self.value

@dataclass
class FractionConst(Expr):
    value: Fraction

    def __init__(self, val):
        if isinstance(val, Fraction):
            self.value = val
        else:
            self.value = to_fraction(val)

    def eval(self, env):
        return self.value


@dataclass
class Var(Expr):
    name: str

    def eval(self, env):
        return env[self.name]

@dataclass
class BinaryOp(Expr):
    left: Expr
    right: Expr

    def op(self, a, b):
        raise NotImplementedError()

    def eval(self, env):
        return self.op(
            self.left.eval(env),
            self.right.eval(env)
        )

class Add(BinaryOp):
    def op(self, a, b):
        return a + b


class Sub(BinaryOp):
    def op(self, a, b):
        return a - b


class Mul(BinaryOp):
    def op(self, a, b):
        return a * b


class Div(BinaryOp):
    def op(self, a, b):
        return a / b


class Pow(BinaryOp):
    def op(self, a, b):
        return a ** b


class Max(BinaryOp):
    def op(self, a, b):
        return max(a, b)


class CompareOp(Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="

@dataclass
class Constraint:
    left: Expr
    right: Expr
    op: CompareOp