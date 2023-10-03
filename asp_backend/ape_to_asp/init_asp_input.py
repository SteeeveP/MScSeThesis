"""Some functions to generate ASP / clingo input files from APE inputs."""


import os
from sys import stderr

from typing import Dict, List, Set, Tuple, cast

import clingo

from ape_to_asp.read_constraints import Constraint, read_constraints_json
from ape_to_asp.read_config import APEConfig, SolverConfig
from ape_to_asp.read_owl import load_owl
from ape_to_asp.read_tool_annotations import ToolMode, read_tool_annoation_json


def ontology_to_asp(
    rel_tuples: Set[Tuple[str, str]],
    tool_root: str,
    data_dim_roots: List[str],
) -> List[clingo.Symbol]:
    taxonomy_functions = []
    for tax_type, tax_par_type in rel_tuples:
        taxonomy_functions.append(clingo.Function(
            'taxonomy',
            [clingo.String(tax_type), clingo.String(tax_par_type)],
        ))
    roots = [
        clingo.Function(
            'taxonomy_tool_root',
            [clingo.String(tool_root)],
        ),
    ] + [
        clingo.Function(
            'taxonomy_type_root',
            [clingo.String(type_root)],
        )
        for type_root
        in data_dim_roots
    ] + [
        clingo.Function(
            'taxonomy_type_root',
            [clingo.String('APE_label')],
        ),
        clingo.Function(
            'taxonomy',
            [
                clingo.String('APE_label'),
                clingo.String('TypesTaxonomy'),
            ],
        )
    ]
    return taxonomy_functions + roots


def tool_modes_to_asp(
    tool_modes: List[ToolMode],
) -> List[clingo.Symbol]:
    taxonomy_functions = []
    mode_inputs = []
    mode_outputs = []
    for mode in tool_modes:
        single_mode = all(
            mode.mode_id != taxop
            for taxop in mode.tax_ops
        )
        taxonomy_functions.append(
            clingo.Function(
                'annotated',
                [clingo.String(mode.mode_id)]
            )
        )
        if single_mode:
            taxonomy_functions += [
                clingo.Function(
                    'taxonomy',
                    [clingo.String(mode.mode_id), clingo.String(tax_op)]
                )
                for tax_op
                in mode.tax_ops
            ]
        for ix, input in enumerate(mode.inputs):
            for dim, types in input.items():
                mode_inputs += [
                    clingo.Function(
                        'tool_input_',
                        [
                            clingo.String(mode.mode_id),
                            clingo.Number(ix+1),
                            clingo.String(dim),
                            clingo.String(dim_type),
                        ]
                    )
                    for dim_type
                    in types
                ]
        for ix, output in enumerate(mode.outputs):
            for dim, types in output.items():
                mode_outputs += [
                    clingo.Function(
                        'tool_output_',
                        [
                            clingo.String(mode.mode_id),
                            clingo.Number(ix+1),
                            clingo.String(dim),
                            clingo.String(dim_type),
                        ]
                    )
                    for dim_type
                    in types
                ]
    return taxonomy_functions + mode_inputs + mode_outputs


def constraints_to_asp(
        constraint_list: List[Constraint],
        tool_root: str,
        dim_roots: List[str]
) -> List[clingo.Symbol]:
    constraint_functions: List[clingo.Symbol] = []
    for constraint_ix, constraint in enumerate(constraint_list):
        constraint_functions.append(clingo.Function(
            'constraint',
            [clingo.Number(constraint_ix), clingo.String(constraint.constraintid)],
        ))
        for param_ix, parameter in enumerate(constraint.parameters, start=1):
            if len(parameter.keys()) == 1 and list(parameter.keys())[0] == tool_root:
                for tool in parameter[tool_root]:
                    constraint_functions.append(clingo.Function(
                        'constraint_tool_param',
                        [
                            clingo.Number(constraint_ix),
                            clingo.Number(param_ix),
                            clingo.String(tool),
                        ],
                    ))
            elif all(dim in (dim_roots + ['APE_label']) for dim in parameter.keys()):
                for dim, type_list in parameter.items():
                    for dim_type in type_list:
                        constraint_functions.append(clingo.Function(
                            'constraint_type_param',
                            [
                                clingo.Number(constraint_ix),
                                clingo.Number(param_ix),
                                clingo.String(dim),
                                clingo.String(dim_type),
                            ],
                        ))
    return constraint_functions


def flags_to_asp(flag_list: List[str]) -> List[clingo.Symbol]:
    return [clingo.Function(flag) for flag in flag_list]


def io_to_asp(
    inputs: List[Dict[str, List[str]]],
    outputs: List[Dict[str, List[str]]],
) -> List[clingo.Symbol]:
    io_functions: List[clingo.Symbol] = []
    for ix_in, wf_input in enumerate(inputs, start=1):
        for dim, type_list in wf_input.items():
            for dim_type in type_list:
                if dim == 'APE_label':
                    io_functions.append(clingo.Function(
                        'taxonomy',
                        [
                            clingo.String(dim_type),
                            clingo.String('APE_label'),
                        ],
                    ))
                io_functions.append(clingo.Function(
                    'out_',
                    [
                        clingo.Function('', [
                            clingo.Number(0),
                            clingo.Number(ix_in),
                        ]),
                        clingo.String(dim),
                        clingo.String(dim_type),
                    ],
                ))
    for ix_out, wf_output in enumerate(outputs, start=1):
        for dim, type_list in wf_output.items():
            for dim_type in type_list:
                io_functions.append(clingo.Function(
                    'in_',
                    [
                        clingo.Function('', [
                            clingo.Number(-1),
                            clingo.Number(ix_out),
                        ]),
                        clingo.String(dim),
                        clingo.String(dim_type),
                    ],
                ))
    return io_functions


def init_core_config(
    config: APEConfig,
    parameters_prefixed: bool=False,
) -> Tuple[Set[Tuple[str, str]], List[ToolMode]]:
    """Initializes core config objects and performs some typechecking.

    Args:
        config (APEConfig): Config dataclass parsed from json file.
        parameters_prefixed (bool, optional):
            Whether tool annotation parameters and tax ops have a prefix. Defaults to False.

    Raises:
        FileNotFoundError: A file was not found (ontology or tool annotations).
        ValueError: The stated tool taxonomy root was not found in the ontology.
        ValueError: The stated type taxonomy roots were not found in the ontology.

    Returns:
        Tuple[Set[Tuple[str, str]], List[ToolMode]]:
        - Taxonomy as edge tuples
        - Tool modes in tool annotation file
    """
    # check if paths exist
    for path_key in [
        'ontology_path',
        'tool_annotations_path',
    ]:
        if not os.path.exists(cast(str, config[path_key])):
            raise FileNotFoundError(f"File for {path_key} not found: {config[path_key]}")

    # load owl
    rel_tuples = load_owl(
        config.ontology_path,
        config.ontology_prefix,
    )
    ontology_set = {t for t, _ in rel_tuples} | {t for _, t in rel_tuples}
    if config.tools_tax_root not in ontology_set:
        raise ValueError(f"Tools taxonomy root '{config.tools_tax_root}' not found in ontolgy.")
    for type_root in config.data_dim_tax_roots:
        if type_root not in ontology_set:
            raise ValueError(f"Data taxonomy root '{type_root}' not found in ontolgy.")
    #? ignore strict_tool_inheritance option for now. no mention in paper.

    # load tool annotations
    tool_modes = read_tool_annoation_json(
        config.tool_annotations_path,
        prefix=config.ontology_prefix if parameters_prefixed else None,
    )

    return rel_tuples, tool_modes


def init_run_config(
        config: APEConfig,
) -> Tuple[List[Constraint], List[str], SolverConfig]:
    """Initiliazed run config objects, performs some type checking
    and wraps solver config attributes.

    Args:
        config (APEConfig): Config dataclass parsed from json file.

    Raises:
        FileNotFoundError: Constraint file was not found.
        ValueError: One of the input or outputs contained an unknown dimension.

    Returns:
        Tuple[List[Constraint], List[str], SolverConfig]:
        - Constraints from constraints file
        - Flags and modes from config file
        - Partial APEConfig for solver
    """
    # load constraints / flags
    if config.constraints_path:
        if not os.path.exists(config.constraints_path):
            raise FileNotFoundError(f"Constraint file not found: {config.constraints_path}")
        constraints = read_constraints_json(config.constraints_path)
    else:
        constraints = []
    constraint_flags = [
        'use_inputs_' + config.use_workflow_input.lower(),
        'use_gen_' + config.use_all_generated_data.lower(),
    ] + (
        ['not_connected_ident_op']
        if config.not_connected_ident_op
        else []
    )
    # check inputs / outputs
    for wf_io in config.inputs + config.outputs:
        for dim in wf_io.keys():
            if not dim in config.data_dim_tax_roots + ['APE_label']:
                raise ValueError(f'Dimension {dim} not found in config.')
    
    # smaller solver config
    solver_config = SolverConfig(
        config.solutions_dir_path,
        config.solution_length_min,
        config.solution_length_max,
        config.solutions,
        config.timeout,
    )
    return constraints, constraint_flags, solver_config


def to_asp_instance(
    rel_tuples: Set[Tuple[str, str]],
    tool_root: str,
    data_dim_roots: List[str],
    tool_modes: List[ToolMode],
    constraints: List[Constraint],
    constraint_flags: List[str],
    wf_inputs: List[Dict[str, List[str]]],
    wf_outputs: List[Dict[str, List[str]]],
):
    return (
        ontology_to_asp(rel_tuples, tool_root, data_dim_roots),
        tool_modes_to_asp(tool_modes),
        constraints_to_asp(constraints, tool_root, data_dim_roots),
        flags_to_asp(constraint_flags),
        io_to_asp(wf_inputs, wf_outputs),
    )


def write_to_asp_file(
    taxonomy_list: List[clingo.Symbol],
    tool_modes_list: List[clingo.Symbol],
    constraints_list: List[clingo.Symbol],
    flags_list: List[clingo.Symbol],
    io_list: List[clingo.Symbol],
    asp_ontology_path: str,
    asp_tool_annotation_path: str,
    asp_constraint_path: str,
    asp_wf_io_path: str,
):
    def write_asp_function(function: clingo.Symbol):
        try:
            return str(function.number)
        except RuntimeError:
            pass
        try:
            return f'"{function.string}"'
        except RuntimeError:
            pass
        str_args = [write_asp_function(arg) for arg in function.arguments]
        if len(str_args) == 0:
            return function.name
        return f'{function.name}({", ".join(str_args)})'

    for symbol_list, path in [
        (taxonomy_list, asp_ontology_path),
        (tool_modes_list, asp_tool_annotation_path),
        (constraints_list + flags_list, asp_constraint_path),
        (io_list, asp_wf_io_path),
    ]:
        with open(path, 'w', encoding='utf-8') as asp_f:
            for symbol in symbol_list:
                asp_f.write(write_asp_function(symbol) + '.\n')
