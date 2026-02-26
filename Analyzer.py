from Intergrate import align_constants_to_integers
from parser import parse_program
from prior import merge_prior

from probably.pgcl.ast.expressions import BinopExpr, UnopExpr, VarExpr, NatLitExpr, RealLitExpr
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
        self.eed, self.disc_prog, self.scales = align_constants_to_integers(self.var_order, exp, self.ori_prog)

    class DiscProgVisitor:
        def enter_program(self, program):
            pass

        def exit_program(self, program):
            pass

        def enter_instr(self, instr):
            pass

        def exit_instr(self, instr):
            pass

        def enter_expr(self, expr):
            pass

        def exit_expr(self, expr):
            pass

    def traverse_disc_prog(self, visitor=None):
        """
        Traverse self.disc_prog (pgcl AST) with a simple visitor.
        Override DiscProgVisitor methods or pass your own visitor instance.
        """
        visitor = visitor or ProgramStructure.DiscProgVisitor()

        def walk_expr(expr):
            visitor.enter_expr(expr)
            if isinstance(expr, BinopExpr):
                walk_expr(expr.lhs)
                walk_expr(expr.rhs)
            elif isinstance(expr, UnopExpr):
                walk_expr(expr.expr)
            elif isinstance(expr, (VarExpr, NatLitExpr, RealLitExpr)):
                pass
            visitor.exit_expr(expr)

        def walk_instr(instr):
            visitor.enter_instr(instr)

            if isinstance(instr, AsgnInstr):
                walk_expr(instr.rhs)
            elif isinstance(instr, WhileInstr):
                walk_expr(instr.cond)
                for s in instr.body:
                    walk_instr(s)
            elif isinstance(instr, IfInstr):
                walk_expr(instr.cond)
                for s in instr.true:
                    walk_instr(s)
                for s in instr.false:
                    walk_instr(s)
            elif isinstance(instr, ObserveInstr):
                walk_expr(instr.cond)
            elif isinstance(instr, ChoiceInstr):
                for s in instr.lhs:
                    walk_instr(s)
                for s in instr.rhs:
                    walk_instr(s)
            elif isinstance(instr, LoopInstr):
                for s in instr.body:
                    walk_instr(s)
            elif isinstance(instr, TickInstr):
                # tick expression is optional; only walk if present
                if hasattr(instr, "expr"):
                    walk_expr(instr.expr)

            visitor.exit_instr(instr)

        visitor.enter_program(self.disc_prog)
        for s in self.disc_prog.instructions:
            walk_instr(s)
        visitor.exit_program(self.disc_prog)
