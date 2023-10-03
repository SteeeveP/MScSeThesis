import argparse
import json
import logging
import os
import sys
from pathlib import Path
from timeit import timeit

from typeguard import check_type

sys.path.append(str(Path.cwd().parent))
from ape_to_asp.init_asp_input import (init_core_config, init_run_config,
                                       to_asp_instance, write_to_asp_file)
from ape_to_asp.read_config import APEConfig, load_json_from_file
from asp_solver.incremental_solver import IncASPSolver
from workflow_output.APE_to_notebook import solution_to_notebook
from workflow_output.workflow_parsing import asp_to_workflow_dump


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('run_ape_housing.log')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def fix_use_case_paths(
    config: APEConfig,
    config_path: Path,
    ontology_path: Path,
) -> None:
    """Adjust use case paths to be relative to their iteration folder."""
    config['ontology_path'] = (ontology_path / 'ontology_v2_DIM_2.owl').as_posix()
    config['tool_annotations_path'] = (ontology_path / 'tool_annotations_v2_DIM_2.json').as_posix()
    config['constraints_path'] = (config_path.parent / 'constraints_run.json').as_posix()
    config['solutions_dir_path'] = (config_path.parent / 'solutions').as_posix()

    # assert all paths exist
    try:
        assert Path(config['ontology_path']).exists()
        assert Path(config['tool_annotations_path']).exists()
        assert Path(config['constraints_path']).exists()
        assert Path(config['solutions_dir_path']).exists()
    except AssertionError as exc:
        logger.exception(exc)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "iteration_number",
        help="Number of iteration to run",
        type=int,
    )
    parser.add_argument(
        "--output",
        help="Output notebooks",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--solutions",
        help="Number of solutions to generate",
        type=int,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--use_domain_asp",
        help="Use domain asp",
        action="store_true",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--runs",
        help="Number of runs",
        type=int,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--validate",
        help="Validate runs",
        action="store_true",
        default=False,
        required=False,
    )

    args = parser.parse_args()

    logger.debug(f"args: {args}")

    try:
        check_type('iteration_number', args.iteration_number, int)
        iteration_number: int = args.iteration_number
        check_type('output', args.output, bool)
        output: bool = args.output
        check_type('solutions', args.solutions, int)
        solutions: int = args.solutions
        check_type('use_domain_asp', args.use_domain_asp, bool)
        use_domain_asp: bool = args.use_domain_asp
        check_type('runs', args.runs, int)
        runs: int = args.runs
        check_type('validate', args.validate, bool)
        validate: bool = args.validate

        assert iteration_number > 0, f"Expected iteration_number > 0, got {iteration_number}"
        assert solutions > 0, f"Expected solutions > 0, got {solutions}"
        assert runs >= 0, f"Expected runs > 0, got {runs}"
    except TypeError as exc:
        logger.exception(exc)
        sys.exit(1)
    except AssertionError as exc:
        logger.exception(exc)
        sys.exit(1)

    # ape_use_cases/thesis_use_cases/house_prices/out/iteration_1/config_run.json
    config_path = Path.cwd().parent / "ape_use_cases" / "thesis_use_cases" / "house_prices" / "out" / f"iteration_{iteration_number}" / "config_run.json"
    assert config_path.exists(), f"Config file not found at {config_path}"
    # ape_use_cases/thesis_use_cases/ontology
    ontology_path = Path.cwd().parent / "ape_use_cases" / "thesis_use_cases" / "ontology"
    assert ontology_path.exists(), f"Ontology file not found at {ontology_path}"

    # notebook data mapping
    input_mapping = [{
        'label': 'housing_train',
        'source': (config_path.parent.parent.parent / 'train.csv').as_posix(),
        'type': 'csv',
        'DataClass': 'MixedDataFrame',
        'StatisticalRelevance': 'NoRelevance'
    }]

    # ! hacky: append "strict_tool_annotations": "true" to config_run.json
    with open(config_path.as_posix(), "r") as f:
        # edit opened file
        config = json.load(f)
        # "strict_tool_annotations": "true",
        config['strict_tool_annotations'] = "true"
    with open(config_path.as_posix(), "w") as f:
        # write back to file
        json.dump(config, f, indent=4)
    # read config
    config = load_json_from_file(config_path.as_posix())

    # override solution number
    config.solutions = solutions

    # fix paths to absolute paths
    fix_use_case_paths(config, config_path, ontology_path)

    # read ontology
    rel_tuples, tool_modes = init_core_config(config)

    # read constraints and create solver config
    constraints, config_flags, solver_config = init_run_config(config)

    logger.info('Instance read.')

    # create ASP instance
    instance_tuple = to_asp_instance(
        rel_tuples,
        config.tools_tax_root,
        config.data_dim_tax_roots,
        tool_modes,
        constraints,
        config_flags,
        config.inputs,
        config.outputs,
    )

    # create working dir
    working_dir = config_path.parent / "asp"
    working_dir.mkdir(exist_ok=True)

    # write ASP instance to file
    asp_ontology_path: Path = working_dir / "ontology.lp"
    asp_tool_path: Path = working_dir / "tools.lp"
    asp_cst_path: Path = working_dir / "cst.lp"
    asp_io_path: Path = working_dir / "wf_io.lp"
    write_to_asp_file(
        *instance_tuple,
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )

    # additional asp files
    domain_asp_path: Path = ontology_path / "domain_heuristics.lp"

    logger.info('Instance written to asp.')

    # create solver instance
    def get_solver():
        inc_solver_inst = IncASPSolver(
            (Path.cwd().parent / 'asp_encoding').as_posix(),
            heuristics=use_domain_asp,
            _logger=logger,
        )
        inc_solver_inst.load_instance_from_files(
            str(asp_ontology_path),
            str(asp_tool_path),
            str(asp_cst_path),
            str(asp_io_path),
            str(domain_asp_path) if use_domain_asp else '',
        )
        logger.debug('Solver instance loaded.')
        return inc_solver_inst

    if runs > 0:
        logger.info('Starting timing runs.')
        # time solver
        try:
            runtime = timeit(
                stmt="inc_solver_inst = get_solver(); inc_solver_inst.solve_instance(solver_config)",
                globals=locals(),
                number=runs,
            )
        except Exception as exc:
            logger.exception(exc)
            sys.exit(1)

        logger.info('Timing finished.')
        logger.info("Runtime: %.2fs, Mean: %.2fs", runtime, runtime/runs)

    if validate:
        # validate runs
        inc_solver_inst = get_solver()
        logger.info('Starting validation run solver.')
        inc_solver_inst.solve_instance(solver_config)
        logger.info('Solver finished.')

        try:
            assert len(inc_solver_inst.models) == config.solutions, f"Expected {config.solutions} solutions, got {len(inc_solver_inst.models)}"
        except AssertionError as exc:
            logger.exception(exc)
            sys.exit(1)

        if output:
            # workflow_set
            workflows = [
                asp_to_workflow_dump(inc_solver_inst.models.pop(), tool_modes)
                for _ in range(len(inc_solver_inst.models))
            ]

            # notebooks
            notebooks = [
                solution_to_notebook(workflow, input_mapping, ix)
                for ix, workflow
                in enumerate(workflows, start=1)
            ]

            for ix, nb in enumerate(notebooks, start=1):
                with open(os.path.join(config.solutions_dir_path, f"nb_{ix}.ipynb"), "w") as f:
                    json.dump(nb, f, indent=4)


if __name__ == "__main__":
    main()
