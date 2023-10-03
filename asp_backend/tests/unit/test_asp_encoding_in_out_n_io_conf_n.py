"""Unit tests for in_out_n.lp and io.lp"""

import os
from typing import List

import clingo

from tests.conftest import ASP_ENCODING_PATH
from tests.unit.conftest import sim_iterative_grounding


def test_in_n_sat(clingo_control, simple_tools):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'in_out_n.lp'))
    clingo_control.add(simple_tools)
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
type("dim1", "type3"). type("dim1", "type6").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
""")
    clingo_control.add("""
    use_tool("tool_mode", 1).
    in((1, 1), "dim1", "type1"). in((1, 1), "dim2", "type7").
    in((1, 2), "dim1", "type3"). in((1, 2), "dim2", "type8").
    in((1, 3), null, eps).
    out((1, 1), "dim1", "type6"). out((1, 1), "dim2", "type7").
    out((1, 2), null, eps).
    out((1, 3), null, eps).
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

def test_in_n_unsat_eps(clingo_control, simple_tools):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'in_out_n.lp'))
    clingo_control.add(simple_tools)
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
type("dim1", "type3"). type("dim1", "type6").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
""")
    clingo_control.add("""
    use_tool("tool_mode", 1).
    in((1, 1), "dim1", "type1"). in((1, 1), "dim2", "type7").
    in((1, 2), "dim1", "type3"). in((1, 2), "dim2", "type8").
    in((1, 3), null, eps).
    out((1, 1), "dim1", "type6").  out((1, 1), null, eps).
    out((1, 2), null, eps).
    out((1, 3), null, eps).
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_in_n_unsat_dim_missing(clingo_control, simple_tools):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'in_out_n.lp'))
    clingo_control.add(simple_tools)
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
type("dim1", "type3"). type("dim1", "type6").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
""")
    clingo_control.add("""
    use_tool("tool_mode", 1).
    in((1, 1), "dim1", "type1"). in((1, 1), "dim2", "type7").
    in((1, 2), "dim1", "type3"). in((1, 2), "dim2", "type8").
    in((1, 3), null, eps).
    out((1, 1), "dim1", "type6").
    out((1, 2), null, eps).
    out((1, 3), null, eps).
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_io(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'io.lp'))
    clingo_control.add("""
term_tool(1).
term_tool(2).
""")
    clingo_control.add('step', ['t'], """
use_tool(t, t).
bind((t,t),(t-1,t)).
bind((t,t),(t-2,t)).
bind((-1,t),(t,t)).
bind((-1,t),(t-1,t)).
""")
    sim_iterative_grounding(clingo_control, steps=2)

    def look_up_io_symbols(model: clingo.Model, symbols: List[clingo.Symbol]):
        for symbol in symbols:
            assert symbol in model.symbols(shown=True), \
                f'{symbol} not in {[str(s) for s in model.symbols(shown=True)]}'

    res: clingo.SolveResult = clingo_control.solve(
        on_model=lambda m: look_up_io_symbols(
            m,
            [
                clingo.Function('use_tool', [clingo.Number(1), clingo.Number(1)]),
                clingo.Function('use_tool', [clingo.Number(2), clingo.Number(2)]),
                clingo.Function('bind', [
                    clingo.Function('', [clingo.Number(1), clingo.Number(1)]),
                    clingo.Function('', [clingo.Number(0), clingo.Number(1)])
                ]),
                clingo.Function('bind', [
                    clingo.Function('', [clingo.Number(2), clingo.Number(2)]),
                    clingo.Function('', [clingo.Number(1), clingo.Number(2)])
                ]),
                clingo.Function('bind', [
                    clingo.Function('', [clingo.Number(-1), clingo.Number(2)]),
                    clingo.Function('', [clingo.Number(2), clingo.Number(2)])
                ]),
            ]
        )
    )
    assert res.satisfiable

def test_conf_n_sat(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'conf_n.lp'))
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
term_type("dim1", "type1"). term_type("dim1", "type2").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
term_type("dim2", "type7"). term_type("dim2", "type8").

term_tool(1).
term_tool(2).

tool_input_ix_max(2).
tool_output_ix_max(2).
""")
    clingo_control.add('step', ['t'], """
use_tool(t, t). use_tool(-t, t).

in((t, 1), "dim1", "type1").
in((t, 1), "dim2", "type7").
in((t, 2), "dim1", "type2").
in((t, 2), "dim2", "type8").

out((t, 1), "dim1", "type1").
out((t, 1), "dim2", "type7").
out((t, 2), "dim1", "type2").
out((t, 2), "dim2", "type8").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

def test_conf_n_unsat_multi_tool(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'conf_n.lp'))
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
term_type("dim1", "type1"). term_type("dim1", "type2").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
term_type("dim2", "type7"). term_type("dim2", "type8").

term_tool(1).
term_tool(2).
term_tool(-1).
term_tool(-2).

tool_input_ix_max(2).
tool_output_ix_max(2).
""")
    clingo_control.add('step', ['t'], """
use_tool(t, t). use_tool(-t, t).
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_conf_n_unsat_multi_in(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'conf_n.lp'))
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
term_type("dim1", "type1"). term_type("dim1", "type2").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
term_type("dim2", "type7"). term_type("dim2", "type8").

term_tool(1).
term_tool(2).

tool_input_ix_max(2).
tool_output_ix_max(2).
""")
    clingo_control.add('step', ['t'], """
use_tool(t, t). use_tool(-t, t).

in((t, 1), "dim1", "type1").
in((t, 1), "dim2", "type7").
in((t, 2), "dim1", "type2").
in((t, 2), "dim2", "type8").

in((t, 1), "dim1", "type2").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_conf_n_unsat_multi_out(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'conf_n.lp'))
    clingo_control.add("""
type("dim1", "dim1").
type("dim1", "type1"). type("dim1", "type2").
term_type("dim1", "type1"). term_type("dim1", "type2").

type("dim2", "dim2").
type("dim2", "type7"). type("dim2", "type8").
term_type("dim2", "type7"). term_type("dim2", "type8").

term_tool(1).
term_tool(2).

tool_input_ix_max(2).
tool_output_ix_max(2).
""")
    clingo_control.add('step', ['t'], """
use_tool(t, t). use_tool(-t, t).

out((t, 1), "dim1", "type1").
out((t, 1), "dim2", "type7").
out((t, 2), "dim1", "type2").
out((t, 2), "dim2", "type8").

out((t, 1), "dim1", "type2").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable
