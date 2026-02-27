from probably.pgcl.parser import parse_pgcl
from analyzer import ProgramStructure
from Adapter.z3_adapter import Z3Adapter

test_prog_str = '''
    prior:
        x1 = Normal(0, 1)
        x2 = UniformBox([[0, 1]], [1])
        x3,x4 = EventualExp([[0, 1/2], [0, 1]], [[0.2, 0.3, 0.3], [0.1, 0.5, 0.2], [0.2, 0.1, 0.1]], [0.3, 0.2], [0.4, 0.6])
    program:
        while(0 <= x1) {
            if(0.5 <= x2) {
                x1 := x1 - 0.1;
                x3 := x3 - 0.2;
            } else {
                x1 := x1 - 0.1;
                x4 := x4 - 0.3;
            }
        }
        '''

simple_test = '''
    prior:
        x1 = Normal(0, 1)
        x2 = UniformBox([[0, 1]], [1])
        x3 = UniformBox([[0, 1]], [1])
    program:
        while(0 <= x1) {
            if(0.5 <= x2) {
                x1 := x1 - 0.1;
            } else {
                x1 := x1 - 0.1;
                x3 := x3 + 0.1;
            }
        }
        '''

prog = ProgramStructure(simple_test)
print(prog.disc_prog)
print(prog.var_order)
print(prog.ori_eed.S)
result = prog.solve_eed(Z3Adapter())
print(result.S)
print(result.P)
print(result.alpha)
print(result.beta)