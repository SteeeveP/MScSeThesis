from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from clingo import Symbol, parse_term

from ape_to_asp.read_tool_annotations import ToolMode


@dataclass
class WorkflowInput:
    dims: Dict[str, str]
    src: Tuple[int, int]


@dataclass
class ToolStep:
    tool: str
    mode_id: str
    inputs: List[WorkflowInput]
    outputs: List[Dict[str, str]]


@dataclass
class WorkflowDump:
    inputs: List[Dict[str, str]]
    outputs: List[WorkflowInput]
    steps: List[ToolStep]


def asp_to_workflow_dump(model: FrozenSet[str], tool_modes: List[ToolMode]) -> WorkflowDump:
    tool_states: Dict[int, Tuple[str, str]] = {}
    bind_dict: Dict[Tuple[int, int], Tuple[int, int]] = {}
    type_state_dict: Dict[Tuple[int, str, int], Dict[str, str]] = defaultdict(dict)

    tool_label_lookup = {tool.mode_id: tool.tax_ops for tool in tool_modes}

    for symbol_ in model:
        symbol = parse_term(symbol_)
        if symbol.name in ['in', 'out']:
            try:
                # eps
                symbol.arguments[1].string
            except RuntimeError:
                continue
            # non empty param -> normal dim
            type_state_dict[(
                symbol.arguments[0].arguments[0].number,
                symbol.name,
                symbol.arguments[0].arguments[1].number,
            )][symbol.arguments[1].string] = symbol.arguments[2].string
        elif symbol.name == 'bind':
            bind_dict[(
                symbol.arguments[0].arguments[0].number,
                symbol.arguments[0].arguments[1].number,
            )] = (
                symbol.arguments[1].arguments[0].number,
                symbol.arguments[1].arguments[1].number,
            )
        elif symbol.name == 'use_tool':
            tool_states[symbol.arguments[0].number] = (
                tool_label_lookup[symbol.arguments[1].string][0],
                symbol.arguments[1].string,
            )

    max_out = max([k[2] for k in type_state_dict.keys() if k[1] == 'out'])
    max_inp = max([k[2] for k in type_state_dict.keys() if k[1] == 'in'])
    wf_length = max([k[0] for k in type_state_dict.keys()])

    type_nodes: Dict[int, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda : {
        'out': [{} for _ in range(max_out)],
        'inp': [{} for _ in range(max_inp)],
    })
    for k, v in type_state_dict.items():
        if v:
            # ! temp fix
            inp_out = 'inp' if k[1] == 'in' else 'out'
            type_nodes[k[0]][inp_out][k[2]-1] = v

    return WorkflowDump(
        inputs=[param for param in type_nodes[0]['out'] if param],
        outputs=[
            WorkflowInput(
                dims=param,
                src=bind_dict[(-1, i)],
            )
            for i, param
            in enumerate(type_nodes[-1]['inp'], start=1)
            if len(param)
        ],
        steps=[
            ToolStep(
                tool=tool_states[step][0],
                mode_id=tool_states[step][1],
                inputs=[
                    WorkflowInput(
                        dims=param,
                        src=bind_dict[(step, i)],
                    )
                    for i, param
                    in enumerate(type_nodes[step]['inp'], start=1)
                    if len(param)
                ],
                outputs=[
                    param
                    for param
                    in type_nodes[step]['out']
                    if len(param)
                ]
            )
            for step
            in range(1, wf_length+1)
        ]
    )

