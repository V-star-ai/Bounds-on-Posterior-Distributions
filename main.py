from probably.pgcl.parser import parse_pgcl
from Intergrate import align_program_constants_to_integers

code = """
real x1;
real x2;
while(x1 >= 0.6 & x2 < 1.4 || x1 > 0.2) {
    {x1 := x1 + 2.1} [0.6] {x2 := 0.4};
}
"""

ast = parse_pgcl(code)
new_ast, scales = align_program_constants_to_integers(ast)

print(scales)
print(new_ast)
