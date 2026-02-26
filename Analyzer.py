from Intergrate import align_constants_to_integers
from parser import parse_program
from prior import merge_prior

class ProgramStructure:
    def __init__(self, prog_str: str):
        prog = parse_program(prog_str)
        self.ori_prog = prog[1]
        self.var_order, exp = merge_prior(prog[0])
        self.eed, self.disc_prog, self.scales = align_constants_to_integers(self.var_order, exp, self.ori_prog)