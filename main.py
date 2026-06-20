import argparse
import json
from pathlib import Path

from analyzer import ProgramStructure
from Adapter.ipopt_adapter import IpoptAdapter
from Adapter.z3_adapter import Z3Adapter
from visualize_bgd import plot_bgd

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

simple_test3 = '''
    prior:
        x1 = Normal(0, 1)
        x3 = Uniform(0,1)
    program:
        while(0 <= x1) {
            {x1 := x1 - 1} [0.5] {x3 := x3 - 0.1}
        }
    '''

def load_config(path):
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_program(path):
    program_path = Path(path)
    with program_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def get_prior_approximation_config(config):
    approximation = config.get("prior_approximation", {})
    normal = config.get("normal", {})
    return {
        "center_subdivision": normal.get(
            "center_subdivision",
            approximation.get("center_subdivision"),
        ),
        "block_subdivision": normal.get(
            "block_subdivision",
            approximation.get("block_subdivision"),
        ),
    }


def get_visualization_config(config):
    visualization = config.get("visualization", {})
    return {
        "num": visualization.get("num", 160),
        "mode": visualization.get("mode", "surface"),
        "value": visualization.get("value", "density"),
        "fallback_html": visualization.get("fallback_html", "bgd_visualization.html"),
        "show": visualization.get("show", True),
    }


def get_template_config(config):
    template = config.get("template", {})
    return {
        "dirac_iterations": int(template.get("dirac_iterations", 2)),
    }


def get_solver_config(config):
    solver = config.get("solver", {})
    ipopt = solver.get("ipopt", {})
    return {
        "name": solver.get("name", "ipopt"),
        "ipopt": {
            "max_iter": int(ipopt.get("max_iter", 500)),
            "tol": float(ipopt.get("tol", 1e-6)),
            "constraint_eps": float(ipopt.get("constraint_eps", 1e-8)),
            "smooth_max_eps": float(ipopt.get("smooth_max_eps", 0.0)),
            "fd_eps": float(ipopt.get("fd_eps", 1e-6)),
            "print_level": int(ipopt.get("print_level", 0)),
        },
    }


def build_solver(config):
    name = config["name"].lower()
    if name == "ipopt":
        return IpoptAdapter(**config["ipopt"])
    if name == "z3":
        return Z3Adapter()
    raise ValueError(f"Unknown solver: {config['name']}")


def solve_with_config(prog, config):
    solver = build_solver(config)
    return prog.solve_bgd(solver, method="Park")


def build_visualization_specs(bgd, num):
    if bgd.ndim == 1:
        return [("var", {"num": num})]

    specs = []
    for dim in range(bgd.ndim):
        if dim < 2:
            specs.append(("var", {"num": num}))
        else:
            specs.append(("const", bgd.center_lefts[dim]))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="bgd_config.json",
        help="JSON file for BGD approximation and visualization parameters",
    )
    parser.add_argument(
        "--program",
        "-p",
        default="./benchmarks/PLDI22/add_uniform.txt",
        help="Program source file containing prior: and program: sections",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prior_config = get_prior_approximation_config(config)
    visualization_config = get_visualization_config(config)
    template_config = get_template_config(config)
    solver_config = get_solver_config(config)
    program_source = load_program(args.program)

    prog = ProgramStructure(
        program_source,
        center_subdivision=prior_config["center_subdivision"],
        block_subdivision=prior_config["block_subdivision"],
        template_dirac_iterations=template_config["dirac_iterations"],
    )
    print(prog.prog)
    print("ori center S: ", prog.ori_bgd.C.S)
    result = solve_with_config(prog, solver_config)
    print(result.C.S)
    print(result.C.P)
    print(result.alpha)
    print(result.beta)

    plot_bgd(
        result,
        build_visualization_specs(result, visualization_config["num"]),
        mode=visualization_config["mode"],
        value=visualization_config["value"],
        fallback_html=visualization_config["fallback_html"],
        show=visualization_config["show"],
    )
    print(
        "BGD visualization opened in browser, "
        f"or written to {visualization_config['fallback_html']} if browser display failed"
    )


if __name__ == "__main__":
    main()
