"""Some utility functions / dataclass to load APE constraints."""

import json

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from typeguard import typechecked

CONSTRAINTID = Literal[
        'ite_m',
        'itn_m',
        'depend_m',
        'next_m',
        'use_m',
        'nuse_m',
        'last_m',
        'use_t',
        'gen_t',
        'nuse_t',
        'ngen_t',
        'use_ite_t',
        'gen_ite_t',
        'use_itn_t',
        'gen_itn_t',
        'operation_input',
        'operation_output',
        'connected_op',
        'not_connected_op',
        'not_repeat_op',
    ]


@typechecked
@dataclass(init=False, kw_only=True)
class Constraint:
    """Typeguard wrapper dataclass for json input constraints."""
    constraintid: CONSTRAINTID
    parameters: List[Dict[str, List[str]]]

    def __init__(
        self,
        constraintid: CONSTRAINTID,
        parameters: Optional[List[Dict[str, List[str]]]]
    ) -> None:
        self.constraintid = constraintid
        self.parameters = parameters if parameters else []

    def __hash__(self) -> int:
        return hash(self.constraintid + str(self.parameters))


def read_constraints_json(path: str) -> List[Constraint]:
    """Typeguard wrapper json load from constraint file.
    Parameter count is not checked by this function.

    Args:
        path (str): Path to constraint file.
    
    Raises:
        TypeError: Path does not belong to JSON file.

    Returns:
        List[Constraint]: List of typechecked constraints.
    """
    @typechecked
    def load_json(path: str) -> Dict[
        Literal['constraints'],
        List[
            Dict[
                Literal['constraintid', 'parameters'],
                CONSTRAINTID | List[
                    Dict[str, List[str]]
                ]
            ]
        ]
    ]:
        with open(path, 'r', encoding='utf-8') as json_f:
            try:
                return json.load(json_f)
            except json.decoder.JSONDecodeError as exc:
                raise TypeError('Failed to parse. Not a JSON file?') from exc

    constraint_json = load_json(path)
    constraint_list = []
    for constraint in constraint_json['constraints']:
        constraint_list.append(Constraint(**constraint)) # type: ignore
    return constraint_list
