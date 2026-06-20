import argparse
import json
from pathlib import Path

from probably.pgcl.parser import parse_pgcl
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="bgd_config.json",
        help="JSON file for BGD approximation and visualization parameters",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prior_config = get_prior_approximation_config(config)
    visualization_config = get_visualization_config(config)

    prog = ProgramStructure(
        simple_test3,
        center_subdivision=prior_config["center_subdivision"],
        block_subdivision=prior_config["block_subdivision"],
    )
    print(prog.prog)
    print("ori center S: ", prog.ori_bgd.C.S)
    result = prog.solve_bgd(IpoptAdapter(), method="Park")
    print(result.C.S)
    print(result.C.P)
    print(result.alpha)
    print(result.beta)

    plot_bgd(
        result,
        [
            ("var", {"num": visualization_config["num"]}),
            ("var", {"num": visualization_config["num"]}),
        ],
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
