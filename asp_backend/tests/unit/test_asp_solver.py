"""Unit tests for incremental_solver"""

from pathlib import Path
import re
from typing import cast

import clingo

from ape_to_asp.read_config import SolverConfig
from asp_solver.incremental_solver import IncASPSolver, InstanceLoadedError


def test_inc_solver_invalid_path():
    try:
        IncASPSolver('INVALID_PATH')
        raise AssertionError('Ran with invalid dir path.')
    except FileNotFoundError:
        pass

def test_load_from_symbol_lists(inc_solver_inst):
    inc_solver_inst.load_instance_from_symbols(
        [
            clingo.Function('taxonomy', [clingo.String("toolparent"), clingo.String("toolroot")]),
            clingo.Function('tool_tax_root', [clingo.String("toolroot")]),
        ],
        [
            clingo.Function('tool_input_', [
                clingo.Number(1),
                clingo.String("dim1"),
                clingo.String("type1"),
            ]),
        ],
        [
            clingo.Function('use_gen_one', []),
            clingo.Function('constraint', [clingo.Number(0), clingo.String('use_m')]),
            clingo.Function('constraint_tool_param', [
                clingo.Number(0),
                clingo.Number(1),
                clingo.String("toolparent"),
            ]),
        ],
        [
            clingo.Function('in_', [
                clingo.Function('', [clingo.Number(-1), clingo.Number(1)]),
                clingo.String("dim1"),
                clingo.String("type1"),
            ])
        ],
    )
    assert inc_solver_inst._instance_loaded

def test_load_from_symbol_lists_twice_error(inc_solver_inst):
    inc_solver_inst.load_instance_from_symbols([], [], [], [])
    try:
        inc_solver_inst.load_instance_from_symbols([], [], [], [])
        raise AssertionError('Instance was loaded twice')
    except InstanceLoadedError:
        pass

def test_load_from_files(inc_solver_inst, simple_instance_files):
    inc_solver_inst.load_instance_from_files(*simple_instance_files)
    assert inc_solver_inst._instance_loaded

def test_load_from_files_twice_error(inc_solver_inst, simple_instance_files):
    inc_solver_inst.load_instance_from_files(*simple_instance_files)
    try:
        inc_solver_inst.load_instance_from_files(*simple_instance_files)
        raise AssertionError('Instance was loaded twice')
    except InstanceLoadedError:
        pass

def test_solve_simple_instance(inc_solver_inst):
    inc_solver_inst._ctl = clingo.Control(['0'])
    inc_solver_inst._ctl.add(
"""
#program base.

goal.
#show.
#show query/1.
"""
    )
    inc_solver_inst._ctl.add(
        'step',
        ['t'],
"""
holds(t) :- t > 5.
#show holds_(t) : holds(t).
"""
    )
    inc_solver_inst._ctl.add(
        'check',
        ['t'],
"""
#external query(t).
:- goal, query(t), not holds(t).
"""
    )
    inc_solver_inst._instance_loaded = True
    inc_solver_inst.solve_instance(SolverConfig(
        '',
        0,
        10,
        3,
        1000,
    ))
    assert len(inc_solver_inst.models) == 3
    assert frozenset({
        'holds_(6)',
        'query(6)',
    }) in inc_solver_inst.models

def test_save_models(inc_solver_inst, tmpdir):
    inc_solver_inst._ctl = clingo.Control(['0'])
    inc_solver_inst._ctl.add(
"""
#program base.

{ a(X): X=1..4 }.
"""
    )
    inc_solver_inst._instance_loaded = True
    test_config = SolverConfig(
        str(tmpdir),
        0,
        10,
        3,
        100,
    )
    inc_solver_inst.solve_instance(test_config)
    inc_solver_inst.save_models(test_config)
    empty_file = False
    for model_ix in range(1, 4):
        file_content = cast(Path, tmpdir / f'model_{model_ix}.lp').read_text(encoding='utf-8')
        assert (
            (not empty_file and file_content == '')
            or re.fullmatch(r'(a\([1234]\).\n)+', file_content)
        )
        if file_content == '':
            empty_file = True
