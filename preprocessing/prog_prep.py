from distributions import EED
from probably.pgcl.ast import (
    Program,
    AsgnInstr,
    ChoiceInstr,
    IfInstr,
    VarExpr,
    WhileInstr,
)


def reverse_replace_distributions(syntax_tree: Any, distribution_map: dict[str, EED]) -> None:
    """
    Reverse the placeholder substitution performed by `replace_distributions`.

    This traverses the parsed pGCL syntax tree and replaces RHS placeholder
    variables such as `distribution_0` with the corresponding distributions.
    """
    
    if isinstance(syntax_tree, list):
        for item in syntax_tree:
            reverse_replace_distributions(item, distribution_map)

    elif isinstance(syntax_tree, WhileInstr):
        reverse_replace_distributions(syntax_tree.body, distribution_map)

    elif isinstance(syntax_tree, IfInstr):
        reverse_replace_distributions(syntax_tree.true, distribution_map)
        reverse_replace_distributions(syntax_tree.false, distribution_map)

    elif isinstance(syntax_tree, ChoiceInstr):
        reverse_replace_distributions(syntax_tree.lhs, distribution_map)
        reverse_replace_distributions(syntax_tree.rhs, distribution_map)

    elif isinstance(syntax_tree, AsgnInstr):
        # Replace placeholder variable: VarExpr("distribution_i").
        if isinstance(syntax_tree.rhs, VarExpr) and re.match(r"^distribution_\d+$", syntax_tree.rhs.var):
          dist = distribution_map[syntax_tree.rhs.var]
          if not isinstance(dist, EED):
            dist = dist.to_eed()
          syntax_tree.rhs = dist
          
        elif isinstance(syntax_tree.rhs, BinopExpr):
            if isinstance(syntax_tree.rhs.lhs, VarExpr) and re.match(r"^distribution_\d+$", syntax_tree.rhs.lhs.var):
                syntax_tree.rhs.lhs = distribution_map[syntax_tree.rhs.lhs.var]
            if isinstance(syntax_tree.rhs.rhs, VarExpr) and re.match(r"^distribution_\d+$", syntax_tree.rhs.rhs.var):
                syntax_tree.rhs.rhs = distribution_map[syntax_tree.rhs.rhs.var]

reverse_replace_distributions(prog.instructions, distribution_map)


def reverse_replace_program(prog, distribution_map):
  """Reverse the placeholder substitution performed by `replace_distributions`."""
  
  reverse_replace_distributions(prog.instructions, distribution_map)
  return prog
