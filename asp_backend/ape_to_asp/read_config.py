"""Some utility functions to load APE config files."""

from dataclasses import dataclass
import json

from typing import Any, Dict, List, Literal, Optional

import typeguard


@dataclass(init=False, kw_only=True)
class APEConfig:
    """Dataclass containing all the information by APE"""
    ontology_path: str
    ontology_prefix: str
    tools_tax_root: str
    data_dim_tax_roots: List[str]
    tool_annotations_path: str
    strict_tool_annotations: bool #? ignored
    solutions_dir_path: str
    solution_length_min: int
    solution_length_max: int
    solutions: int
    constraints_path: Optional[str]
    inputs: List[Dict[str, List[str]]]
    outputs: List[Dict[str, List[str]]]
    number_of_execution_scripts: int #? ignored
    # number_of_generated_graphs: int #? ignored
    tool_seq_repeat: bool #? ignored
    timeout: int
    use_workflow_input: Literal['ALL', 'ONE', 'NONE']
    use_all_generated_data: Literal['ALL', 'ONE', 'NONE']
    not_connected_ident_op: bool

    @typeguard.typechecked
    def __init__(
        self,
        ontology_path: str,
        ontologyPrefixIRI: str,
        toolsTaxonomyRoot: str,
        dataDimensionsTaxonomyRoots: List[str],
        tool_annotations_path: str,
        strict_tool_annotations: Literal['true', 'false', 'TRUE', 'FALSE'],
        solutions_dir_path: str,
        solution_length: Dict[Literal['min'] | Literal['max'], int],
        solutions: str | int,
        constraints_path: Optional[str]=None,
        inputs: Optional[List[Dict[str, List[str]]]]=None,
        outputs: Optional[List[Dict[str, List[str]]]]=None,
        number_of_execution_scripts: str | int="0",
    #     number_of_generated_graphs: str="0",
        tool_seq_repeat: Literal['true', 'false', 'TRUE', 'FALSE']='true',
        timeout: str | int=300,
        use_workflow_input: Literal['ALL', 'ONE', 'NONE', 'all', 'one', 'none']='ONE',
        use_all_generated_data: Literal['ALL', 'ONE', 'NONE', 'all', 'one', 'none']='ALL',
        not_connected_ident_op: Literal['true', 'false', 'TRUE', 'FALSE']='true',
        **_,
    ) -> None:
        self.ontology_path = ontology_path
        self.ontology_prefix = ontologyPrefixIRI
        self.tools_tax_root = toolsTaxonomyRoot
        self.data_dim_tax_roots = dataDimensionsTaxonomyRoots
        self.tool_annotations_path = tool_annotations_path
        self.strict_tool_annotations = strict_tool_annotations.lower() == 'true'
        self.solutions_dir_path = solutions_dir_path
        self.solution_length_min = solution_length['min']
        self.solution_length_max = solution_length['max']
        self.solutions = int(solutions)
        self.constraints_path = constraints_path
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.number_of_execution_scripts = int(number_of_execution_scripts)
        # self.number_of_generated_graphs = int(number_of_generated_graphs)
        self.tool_seq_repeat = tool_seq_repeat.lower() == 'true'
        self.timeout = int(timeout)
        self.use_workflow_input = use_workflow_input.upper() # type: ignore
        self.use_all_generated_data = use_all_generated_data.upper() # type: ignore
        self.not_connected_ident_op = not_connected_ident_op.lower() == 'true'


    def __getitem__(self, key: str):
        return self.__dict__[key]


    def __setitem__(self, key: str, item: Any):
        self.__dict__[key] = item


@dataclass()
class SolverConfig:
    """Partial APEConfig with solver specific attributes."""
    solutions_dir_path: str
    solution_length_min: int
    solution_length_max: int
    solutions: int
    timeout: int


def load_json_from_file(path: str) -> APEConfig:
    """JSON loader for typechecked APEConfig.

    Args:
        path (str): Path to config file.

    Raises:
        TypeError: Path does not belong to JSON file.

    Returns:
        APEConfig: Typechecked ape config.
    """
    with open(path, 'r', encoding='utf-8') as json_f:
        try:
            return APEConfig(**json.load(json_f))
        except json.decoder.JSONDecodeError as exc:
            raise TypeError('Failed to parse. Not a JSON file?') from exc
