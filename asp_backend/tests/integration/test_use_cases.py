"""Integration tests for existing APE usecases"""

from pathlib import Path
import sys
import pytest

from ape_to_asp.init_asp_input import init_core_config, init_run_config, to_asp_instance, write_to_asp_file
from ape_to_asp.read_config import APEConfig, load_json_from_file
from asp_solver.incremental_solver import IncASPSolver
from workflow_output.workflow_parsing import asp_to_workflow_dump


def fix_use_case_paths(config: APEConfig):
    """hacky way to add use case path as parent folder for use cases

    Args:
        config (APEConfig): config to fix
    """
    config.ontology_path = 'ape_use_cases' + config['ontology_path'][1:]
    config.tool_annotations_path = 'ape_use_cases' + config['tool_annotations_path'][1:]
    config.constraints_path = 'ape_use_cases' + config['constraints_path'][1:]
    config.solutions_dir_path = 'ape_use_cases' + config['solutions_dir_path'][1:] + 'asp/'


@pytest.mark.integration_test
def test_simple_config(
    tmp_path: Path,
    inc_solver_inst: IncASPSolver
):
    config = load_json_from_file('ape_use_cases/simple_use_case/E0/config.json')
    fix_use_case_paths(config)
    rel_tuples, tool_modes = init_core_config(config)
    constraints, config_flags, solver_config = init_run_config(config)
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
    # tmp_path = Path('/Users/stevep/Documents/code/APE_thesis/ape_asp/tests/integration/temp')
    asp_ontology_path: Path = tmp_path / "ontology.lp"
    asp_tool_path: Path = tmp_path / "tools.lp"
    asp_cst_path: Path = tmp_path / "cst.lp"
    asp_io_path: Path = tmp_path / "wf_io.lp"
    write_to_asp_file(
        *instance_tuple,
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )

    inc_solver_inst.load_instance_from_files(
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )
    inc_solver_inst.solve_instance(solver_config)
    inc_solver_inst.save_models(solver_config)
    assert inc_solver_inst._step == 4
    assert len(inc_solver_inst.models) == config.solutions

@pytest.mark.parametrize(
    'config_path,param_prefixed,min_models,output_to_temp',
    [
        ('geo_gmt_config_e0', False, 10, False), # APE: 1.064 sec
        ('geo_gmt_config_e1', False, 1, False), # APE: 3.21 sec
        ('image_magick_config_e1', False, 1, False), # APE: 3.026 sec
        ('image_magick_config_e2', False, 1, False), # APE: 3.256 sec
        ('mass_spec_config_no1', True, 1, False), # APE: 110.78 sec
        # ('mass_spec_config_no2', True, 0, False), # APE: 247.38 sec
        # ('mass_spec_config_no3', True, 10, False), # APE: 
        # ('mass_spec_config_no4', True, 10, False), # APE:
        # ('mass_spec_config_no1_full', True, 1000, False), # APE: 
        # ('mass_spec_config_no2_full', True, 10, False), # APE: 
        # ('mass_spec_config_no3_full', True, 10, False), # APE: 
        # ('mass_spec_config_no4_full', True, 10, False), # APE: 
        ('ds_config_inplace_full', False, 1, True), # APE: 110.78 sec
    ]
)
@pytest.mark.integration_test
def test_ape_use_case(
    tmp_path: Path,
    config_path: str,
    param_prefixed: bool,
    min_models: int,
    inc_solver_inst: IncASPSolver,
    output_to_temp: bool,
    request: pytest.FixtureRequest,
    benchmark,
):
    config = load_json_from_file(request.getfixturevalue(config_path))
    #!temp
    config.solutions = min_models
    fix_use_case_paths(config)
    rel_tuples, tool_modes = init_core_config(config, param_prefixed)
    constraints, config_flags, solver_config = init_run_config(config)
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
    if output_to_temp:
        tmp_path = Path('/Users/stevep/Documents/code/APE_thesis/ape_asp/tests/integration/temp')
    asp_ontology_path: Path = tmp_path / "ontology.lp"
    asp_tool_path: Path = tmp_path / "tools.lp"
    asp_cst_path: Path = tmp_path / "cst.lp"
    asp_io_path: Path = tmp_path / "wf_io.lp"
    write_to_asp_file(
        *instance_tuple,
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )

    inc_solver_inst.load_instance_from_files(
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )
    benchmark.pedantic(
        target=inc_solver_inst.solve_instance,
        args=[solver_config],
        rounds=1,
        iterations=1,
    )
    inc_solver_inst.save_models(solver_config)
    assert len(inc_solver_inst.models) >= min_models, f'Insufficient models after {inc_solver_inst._step} iterations.'
    if inc_solver_inst.models:
        wf_dump = asp_to_workflow_dump(inc_solver_inst.models.pop(), tool_modes)
