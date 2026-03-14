from probably.pgcl.parser import parse_pgcl
from analyzer import ProgramStructure
from Adapter.ipopt_adapter import IpoptAdapter
from Adapter.z3_adapter import Z3Adapter

from visualize import plot_eed

simple_test1 = '''
    prior:
        x1 = Normal(0, 1)
        x3 = Uniform(0,1)
    program:
        while(1/3) {
            {x1 := x1 - 1} [0.5] {x3 := x3 - 0.1}
            x3 := Uniform(0,1)
        }
        '''

simple_test2 = '''
    prior:
        x1 = Normal(0, 1)
        x3 = {0:1.0}
    program:
        while(1/3) {
            x1 := x1 - 1
            x3 := x3 - 1
        }
        '''

prog = ProgramStructure(simple_test2)
print(prog.prog)
print("ori S: ", prog.ori_eed.S)
result = prog.solve_eed(IpoptAdapter(), method="Park")
print(result.S)
print(result.P)
print(result.alpha)
print(result.beta)

plot_eed(result, specs=[("var", None), ("var", None)], mode="surface")