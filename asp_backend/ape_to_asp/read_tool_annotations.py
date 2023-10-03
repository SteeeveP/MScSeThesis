"""Some utility functions / dataclass to load APE tool annotation files."""

import json

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from typeguard import typechecked


@typechecked
@dataclass(init=False, kw_only=True)
class ToolMode:
    """Wrapper with typeguard for json objects"""
    label: str
    mode_id: str
    tax_ops: List[str]
    inputs: List[Dict[str, List[str]]]
    outputs: List[Dict[str, List[str]]]
    implementation: str | None

    def __init__(
        self,
        label: str,
        id: str,
        taxonomyOperations: List[str],
        inputs: Optional[List[Dict[str, List[str]]]]=None,
        outputs: Optional[List[Dict[str, List[str]]]]=None,
        implementation: Optional[Dict[Literal['code'], str]]=None,
        prefix: Optional[str]=None,
        **_,
    ) -> None:
        def drop_prefix_from_params(
            params: List[Dict[str, List[str]]]
        ) -> List[Dict[str, List[str]]]:
            new_params = []
            for param in params:
                new_param = {}
                for dim, type_list in param.items():
                    new_param[dim] = [dim_type.split(prefix)[1] for dim_type in type_list]
                new_params.append(new_param)
            return new_params

        self.label = label
        self.mode_id = id
        self.tax_ops = taxonomyOperations
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        if prefix is not None:
            self.tax_ops = [op.split(prefix)[1] for op in self.tax_ops]
            self.inputs = drop_prefix_from_params(self.inputs)
            self.outputs = drop_prefix_from_params(self.outputs)
        if implementation is not None and 'code' in implementation:
            self.implementation = implementation['code']
        else:
            self.implementation = None


def read_tool_annoation_json(path: str, prefix: Optional[str]=None) -> List[ToolMode]:
    """Typeguard wrapper json load from tool annoation file.
    Taxonomy types are not checked by this function.

    Args:
        path (str): Path to tool annotation file.
        prefix (Optional[str]): Prefix of parameter types and tax ops.
    
    Raises:
        TypeError: Path does not belong to JSON file.

    Returns:
        List[ToolMode]: List of typechecked tools.
    """
    @typechecked
    def load_json(path: str) -> Dict[
        Literal['functions'],
        List[Dict]
    ]:
        with open(path, 'r', encoding='utf-8') as json_f:
            try:
                return json.load(json_f)
            except json.decoder.JSONDecodeError as exc:
                raise TypeError('Failed to parse. Not a JSON file?') from exc

    tool_annoation_json = load_json(path)
    tool_list: List[ToolMode] = []
    for mode in tool_annoation_json['functions']:
        tool_list.append(ToolMode(**mode, prefix=prefix))
    return tool_list
