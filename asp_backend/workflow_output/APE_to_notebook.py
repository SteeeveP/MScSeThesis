import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, cast
import joblib

from workflow_output.workflow_parsing import (ToolStep, WorkflowDump,
                                              WorkflowInput)

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import wrapper_functions


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
)


# Types and tools

TERM_TYPES = {
    'DataClass': {
        'AdaBoostClassifier',
        'AdaBoostRegressor',
        'Axes',
        'Bool',
        'BoolArray',
        'BoolColumn',
        'BoolDataFrame',
        'BoolNDArray',
        'BoolSeries',
        'ClassificationReport',
        'CountVectorizer',
        'DBScanClustor',
        'DateTime',
        'DateTimeArray',
        'DateTimeColumn',
        'DateTimeNDArray',
        'DateTimeSeries',
        'DecisionTreeClassifier',
        'DecisionTreeRegressor',
        'DescribeDataFrame',
        'DescribeSeries',
        'DummyClassifier',
        'DummyRegressor',
        'ElasticNetRegressor',
        'EmbeddingMatrix',
        'Figure',
        'Float',
        'FloatColumn',
        'FloatDataFrame',
        'FloatFlatArray',
        'FloatNDArray',
        'FloatSeries',
        'GridSearchCV',
        'HalvingGridSearchCV',
        'Int',
        'IntColumn',
        'IntDataFrame',
        'IntFlatArray',
        'IntNDArray',
        'IntSeries',
        'IterativeImputer',
        'KMeansClustor',
        'KNNImputer',
        'KNeighborsClassifier',
        'KNeighborsRegressor',
        'KernelRidgeRegressor',
        'LinearRegressor',
        'LinearSVClassifier',
        'LinearSVRregressor',
        'LogisticRegressionClassifier',
        'MixedColumn',
        'MixedDataFrame',
        'MixedSeries',
        'PCA',
        'PerceptronClassifier',
        'RandomForestClassifier',
        'RandomForestRegressor',
        'RidgeRegressor',
        'SimpleImputer',
        'Str',
        'StrColumn',
        'StrFlatArray',
        'StrNDArray',
        'StrSeries',
        'TfidfVectorizer',
        'TrunctuatedSVD',
        'VotingClassifier',
        'VotingRegressor',
        'Word2Vec',
    },
    'StatisticalRelevance': {
        'DependentVariable',
        'IndependentVariable',
        'NoRelevance',
        'Prediction',
    }
}

try:
    TOOL_TERM_TO_LABEL = joblib.load(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tool_term_to_label.pkl'
    ))
    logging.debug('Loaded tool mapping.')
except FileNotFoundError:
    logging.debug('Regen tool mapping.')
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..',
            'ape_use_cases',
            'thesis_use_cases',
            'ontology',
            'tool_annotations_v2_DIM_2.json', #! with modeling tools
        ),
        'r',
        encoding='utf-8',
    ) as file_:
        tool_annotations = json.load(file_)
    TOOL_TERM_TO_LABEL = {
        tool['label']: tool['taxonomyOperations'][0]
        for tool
        in tool_annotations['functions']
    }
    joblib.dump(TOOL_TERM_TO_LABEL, os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'tool_term_to_label.pkl'
    ))


# Notebook templates

NOTEBOOK_TEMPLATE = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "APE",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.10"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

CODE_TEMPLATE = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [],
}

MARKDOWN_TEMPLATE = {
   "attachments": {},
   "cell_type": "markdown",
   "metadata": {},
   "source": [],
}


# Parse APE graphs

def parse_line(
    line: str,
    prefix: str,
    alias_io: Dict[str, str],
    type_state_dict: Dict[Tuple[int, str, int], Dict[str, str|List[str]]],
    bind_dict: Dict[Tuple[int, int], Tuple[int, int]|None],
    tool_states: Dict[int, str],
):
    line_ = line[:-1].replace(prefix, '')
    if line == '\n':
        return True
    # labels
    elif line_.startswith('&'):
        pass
    # type and tool states
    elif match := re.fullmatch(
        r'#(?P<ty>.+)\((?P<io>Mem|Used)T(?P<time>[0-9]+)\.(?P<id>[0-9]+)\)',
        line_,
    ):
        item = cast(str, match.group('ty'))
        node_type = alias_io[cast(str, match.group('io'))]
        node_time = int(match.group('time'))
        node_paramid = int(match.group('id'))
        try:
            for dim, terms in TERM_TYPES.items():
                if item in terms:
                    try:
                        type_state_dict[(node_time, node_type, node_paramid)][dim]
                        raise ValueError(str((
                            line_,
                            type_state_dict[(node_time, node_type, node_paramid)],
                        )))
                    except KeyError:
                        type_state_dict[(node_time, node_type, node_paramid)][dim] = item
                        raise StopIteration
            # raise ValueError(line_)
        except StopIteration:
            pass
    elif match := re.fullmatch(
        r'#(?P<op>.+)\(Tool(?P<time>[1-9][0-9]*)\)',
        line_,
    ):
        item = cast(str, match.group('op'))
        node_time = int(match.group('time'))
        if item in TOOL_TERM_TO_LABEL.keys():
            try:
                tool_states[node_time]
                raise ValueError(line_)
            except KeyError:
                tool_states[node_time] = item
        else:
            pass
            # raise ValueError(line_)
    elif match := re.fullmatch(
        r'empty\((?P<io>Mem|Used)T(?P<time>[0-9]+)\.(?P<id>[0-9]+)\)',
        line_,
    ):
        node_type = alias_io[cast(str, match.group('io'))]
        node_time = int(match.group('time'))
        node_paramid = int(match.group('id'))
        try:
            assert (node_time, node_type, node_paramid) in type_state_dict.keys()
            raise ValueError(line_)
        except AssertionError:
            type_state_dict[(node_time, node_type, node_paramid)] = {}
    # bind input to source
    elif match := re.fullmatch(r'memRef\(MemT([0-9]+)\.([0-9]+),UsedT([0-9]+)\.([0-9]+)\)', line_):
        src_t, src_id, inp_t, inp_id = (int(item) for item in match.groups())
        try:
            bind_dict[(inp_t, inp_id)]
            raise ValueError(line_)
        except KeyError:
            bind_dict[(inp_t, inp_id)] = (src_t, src_id)  
    # bind empty input
    elif match := re.fullmatch(r'memRef\(nullMem,UsedT([0-9]+)\.([0-9]+)\)', line_):
        inp_t, inp_id = (int(item) for item in match.groups())
        try:
            bind_dict[(inp_t, inp_id)]
            raise ValueError(line_)
        except KeyError:
            bind_dict[(inp_t, inp_id)] = None
    elif match := re.fullmatch(r'emptyLabel\(.+\)', line_):
        pass
    elif match := re.fullmatch(r'r_rel\(.+\)', line_):
        pass
    elif match := re.fullmatch(r'is_rel\(.+\)', line_):
        pass
    elif match := re.fullmatch(r'APE_label\(.+\)', line_):
        pass
    elif match := re.fullmatch(
        r'(?P<label>.+)\((?P<io>Mem|Used)T(?P<time>[0-9]+)\.(?P<id>[0-9]+)\)',
        line_,
    ):
        item = cast(str, match.group('label'))
        node_type = alias_io[cast(str, match.group('io'))]
        node_time = int(match.group('time'))
        node_paramid = int(match.group('id'))
        try:
            cast(
                List[str],
                type_state_dict[(node_time, node_type, node_paramid)]['APE_label'],
            ).append(item)
        except KeyError:
            type_state_dict[(node_time, node_type, node_paramid)]['APE_label'] = [item]
    else:
        raise ValueError(line_)
    return False


def parse_ape_solutions(
    path: Path,
    prefix: str='http://www.co-ode.org/ontologies/ont.owl',
    alias_io: Dict[str, str]={'Mem': 'out', 'Used': 'inp'},
    input_step: int=0,
) -> List[Dict]:
    def shift_bind(bind: Tuple[int, int]) -> Tuple[int, int]:
        return (bind[0] + input_step if bind[0] else 0, bind[1])

    line: str
    sol_type_nodes: Dict[int, Dict[str, List[Dict[str, str|List[str]]]]]

    bind_dict: Dict[Tuple[int, int], Tuple[int, int]|None] = {}
    type_state_dict: Dict[Tuple[int, str, int], Dict[str, str|List[str]]] = defaultdict(dict)
    tool_states: Dict[int, str] = {}
    solutions: List[Dict] = []

    with open(path, 'r', encoding='utf-8') as file_:
        for line in file_:
            end_of_sol = parse_line(line, prefix, alias_io, type_state_dict, bind_dict, tool_states)
            if end_of_sol:
                sol_max_out = max([k[2] for k in type_state_dict.keys() if k[1] == 'out']) + 1
                sol_max_inp = max([k[2] for k in type_state_dict.keys() if k[1] == 'inp']) + 1
                sol_length = max([k[0] for k in type_state_dict.keys()])
                # order inputs and outputs
                sol_type_nodes = defaultdict(lambda : {
                    'out': [{} for _ in range(sol_max_out)],
                    'inp': [{} for _ in range(sol_max_inp)],
                })
                for k, v in type_state_dict.items():
                    if v:
                        try:
                            sol_type_nodes[k[0]][k[1]][k[2]] = {
                                dim: v[dim]
                                for dim
                                in TERM_TYPES
                            }
                        except KeyError as k_err:
                            raise KeyError(str((k, v))) from k_err
                        try:
                            sol_type_nodes[k[0]][k[1]][k[2]].update({'APE_label': v['APE_label']})
                        except KeyError:
                            pass
                # connect add sources from binds
                try:
                    sol_workflow = {
                        'input': [
                            param
                            for param
                            in sol_type_nodes[0]['out']
                            if param
                        ] if 0 in sol_type_nodes else [],
                        'steps': [
                            {
                                'tool': TOOL_TERM_TO_LABEL[tool_states[step]],
                                'input': [
                                    param | {'src': shift_bind(cast(
                                        Tuple[int, int],
                                        bind_dict[(step-1, i)],
                                    ))}
                                    for i, param
                                    in enumerate(sol_type_nodes[step-1]['inp'])
                                    if param
                                ],
                                'output': [
                                    param
                                    for param
                                    in sol_type_nodes[step]['out']
                                    if param
                                ],
                            }
                            for step
                            in range(1, sol_length+1)
                        ],
                        'output': [
                            param | {'src': shift_bind(cast(
                                Tuple[int, int],
                                bind_dict[(sol_length, i)],
                            ))}
                            for i, param
                            in enumerate(sol_type_nodes[sol_length]['inp'])
                            if param
                        ] if sol_length in sol_type_nodes else [],
                    }
                except KeyError as k_err:
                    print(json.dumps(tool_states, indent=2))
                    print(json.dumps(dict(sol_type_nodes), indent=2))
                    raise k_err
                # save solution
                solutions.append(sol_workflow)
                # reset dictionaries
                bind_dict = {}
                type_state_dict = defaultdict(dict)
                tool_states = {}
    return solutions


def lower_first_letter(x: str) -> str:
    return x[:1].lower() + x[1:]


def get_output_id(param: Dict[str, str|List[str]|List[int]], step: int, ix: int):
    return f"{lower_first_letter(cast(str, param['DataClass']))}_{step}_{ix}"


def get_input_id(param: WorkflowInput) -> str:
    # column references
    try:
        if 'Column' in cast(str, param.dims['DataClass']):
            return "'" + param.dims['APE_label'] + "'"
        # objects
        # inputs
        if cast(int, param.src[0]) == 0:
            if param.dims['DataClass'] in ['Str']:
                return "'" + param.dims['APE_label'] + "'"
            return param.dims['APE_label']
    except KeyError:
        # Column is not a APELabel but produced by a tool
        assert 'Column' in param.dims['DataClass']
        assert 'APE_label' not in param.dims.keys()
    # tool outputs
    return get_output_id(
        param.dims, # type: ignore
        cast(int, param.src[0]),
        cast(int, param.src[1]),
    )


#! hacky, fixme
def match_arg(param_type: str, arg_type: str):
    """Check if argument type matches parameter type."""
    if param_type in ['IntColumn', 'FloatColumn']:
        return 'NumberColumn' in arg_type or 'str' in arg_type
    if param_type == 'BoolColumn':
        return 'NumberColumn' in arg_type or 'BoolColumn' in arg_type or 'str' in arg_type
    if param_type == 'DateTimeColumn':
        return 'DateTimeColumn' in arg_type or 'str' in arg_type
    if param_type in ['IntSeries', 'FloatSeries']:
        if 'NumberSeries' in arg_type:
            return True
    if param_type == 'BoolSeries':
        return 'NumberSeries' in arg_type or 'BoolSeries' in arg_type
    if param_type == 'DateTimeSeries':
        return 'DateTimeSeries' in arg_type or 'str' in arg_type
    if 'Column' in param_type:
        return 'str' in arg_type or 'Column' in arg_type
    if 'DataFrame' in param_type:
        return 'DataFrame' in arg_type
    if 'Series' in param_type:
        return 'Series' in arg_type
    if 'Figure' == param_type:
        return "Figure" in arg_type
    if 'Axes' == param_type:
        return "Axes" in arg_type
    if param_type in arg_type:
        return True
    if param_type.replace('Classifier', 'SVC') in arg_type:
        return True
    if param_type.replace('Regressor', 'SVR') in arg_type:
        return True
    if param_type.replace('Classifier', '') in arg_type:
        return True
    if param_type.replace('Regressor', '') in arg_type:
        return True
    if param_type.replace('Clustor', '') in arg_type:
        return True
    if param_type == 'Int' and 'int' in arg_type:
        return True
    if param_type == 'Float' and 'float' in arg_type:
        return True
    if param_type == 'Bool' and 'bool' in arg_type:
        return True
    if param_type == 'Str' and 'str' in arg_type:
        return True
    return False


def get_kwrgs(
    tool: str,
    inputs: List[WorkflowInput],
) -> List[Tuple[str, str]]:
    """Match inputs to tool arguments."""
    matched: List[Tuple[str, str]] = []
    arg_list = list(wrapper_functions.__dict__[tool].__annotations__.items())[:-1]
    arg_ix = 0
    for input_ix, param in enumerate(inputs):
        try:
            if match_arg(cast(str, param.dims['DataClass']), str(arg_list[arg_ix][1])):
                matched.append((arg_list[arg_ix][0], get_input_id(param)))
                arg_ix += 1
            elif 'Optional' in str(arg_list[arg_ix][1]):
                arg_ix += 1
            else:
                raise TypeError(
                    tool + ': '
                    + f'input_ix: {input_ix} - {inputs[input_ix].dims["DataClass"]}, '
                    + f'arg_ix: {arg_ix} - {arg_list[arg_ix]}'
                )
        except IndexError as i_err:
            raise IndexError(
                tool + ': '
                + f'input_ix: {input_ix} - {inputs[input_ix].dims["DataClass"]}'
                + str(arg_list)
            ) from i_err
    return matched


def get_function_call(step: ToolStep, step_num: int) -> str:
    """Get function call string from step"""
    # input params
    if step.tool == 'init_sklearn_estimator':
        kwrgs_str = f'estimator="{step.outputs[0]["DataClass"]}"'
    elif step.tool == 'init_sklearn_searchcv':
        kwrgs_str = f'search_cv="{step.outputs[0]["DataClass"]}", ' \
                    + f'estimator="{step.inputs[0].dims["APE_label"][0]}"'
    elif step.tool == 'init_sklearn_voting_estimator':
        kwrgs_str = f'voting_estimator="{step.outputs[0]["DataClass"]}", ' \
                    + f'estimator_list="{step.inputs[0].dims["APE_label"][0]}"'
    elif step.inputs != []:
        kwrgs_str = ', '.join(
            f'{k}={v}'
            for k, v
            in get_kwrgs(step.tool, step.inputs)
        )
    else:
        raise TypeError(
            f"Step {step_num} with tool {step.tool} has no inputs and is not a constructor"
        )
    # output params
    if step.outputs == []:
        return f"{step.tool}({kwrgs_str})"
    outputs_str = ', '.join(
        get_output_id(param, step_num, ix) # type: ignore
        for ix, param
        in enumerate(step.outputs, start=1)
    )
    return f"{outputs_str} = {step.tool}({kwrgs_str})"


def doc_str_to_md(tool: str) -> str:
    doc = wrapper_functions.__dict__[tool].__doc__
    return f"#### Notes\n{doc}" if doc else ''


def param_list_to_md(params: List) -> str:
    try:
        out = []
        for ix, param in enumerate(params, start=1):
            out.append(f"- {ix}\n" + '\n'.join(
                f'\t- {dim}: `{val}`'
                for dim, val
                in param.items()
            ))
    except:
        out = []
        for ix, param in enumerate(params, start=1):
            out.append(f"- {ix}\n" + '\n'.join(
                f'\t- {dim}: `{val}`'
                for dim, val
                in param.dims.items()
            ))
            out[-1] += '\n' + f'\t- src: `{param.src}`'
    return '\n'.join(out)


def solution_to_notebook(
    solution: WorkflowDump,
    input_mapping: List[Dict],
    solution_num: int,
    input_step: int=0,
) -> Dict:
    head_cells: List[Tuple[str, str]] = [
        (
            'markdown',
            f'![workflow graph](Figures/SolutionNo_{solution_num}_length_{len(solution.steps)}.png "Workflow Graph")'
        )
    ]
    setup_cells: List[Tuple[str, str]] = [
        (
            'code',
            f"""\
from pathlib import Path
import sys

import pandas as pd

sys.path.append('{os.path.abspath(os.path.join(__file__, '..', '..'))}')
from wrapper_functions import *\
    """,
        ),
        (
            'markdown',
            """## Workflow Input Objects""",
        ),
    ]

    input_loading_cells: List[Tuple[str, str]] = sum([
        [
            (
                'markdown',
                f"""\
### Table {i}
- id: `{input["label"]}`
- source: `{input["source"]}`
- DataClass: `{input['DataClass']}`
- DataClass: `{input['StatisticalRelevance']}`\
    """,
            ),
            (
                'code',
                f"{input['label']} = load_table_csv('{input['source']}')",
            )
        ]
        for i, input
        in enumerate(input_mapping, start=1)
        if input["type"] == 'csv'
    ], []) # type: ignore

    step_cells: List[Tuple[str, str]] = []
    for num, step in enumerate(solution.steps, start=input_step+1):
        step_cells.append((
            'markdown',
            f"""\
### Step {num}: `{step.tool}`
{doc_str_to_md(step.tool)}
#### inputs:
{param_list_to_md(step.inputs)}
#### outputs:
{param_list_to_md(step.outputs)}\
""",
        ))
        step_cells.append((
            'code',
            get_function_call(step, num),
        ))

    output_cells = [
        ('markdown', f"""\
## Output
{param_list_to_md(cast(List[Dict], solution.outputs))}\
    """),
        ('code', ''.join(
            f"display({get_output_id(param.dims, param.src[0], param.src[1]+1)})" # type: ignore
            for param
            in solution.outputs
        )),
    ] if solution.outputs else []

    notebook = NOTEBOOK_TEMPLATE.copy()
    notebook['cells'] = []
    for cell_ in head_cells + setup_cells + input_loading_cells + step_cells + output_cells:
        source = [line + '\n' for line in cell_[1].split('\n')]
        if cell_[0] == 'code':
            cell = CODE_TEMPLATE.copy()
        else:
            cell = MARKDOWN_TEMPLATE.copy()
        cell['source'] = source[:-1]+[source[-1][:-1]]
        notebook['cells'].append(cell)

    return notebook
