"""Unit tests for asp encodings worfklow.lp, goal.lp, aux.lp and tax_t_op_n.lp."""

import os

import clingo

from tests.conftest import ASP_ENCODING_PATH
from tests.unit.conftest import sim_iterative_grounding


# AUX

def test_aux_tax(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'aux.lp'))
    clingo_control.add("""
taxonomy_tool_root("Tools").
taxonomy_type_root("dim1").
taxonomy_type_root("dim2").

taxonomy("toolparent", "Tools").
taxonomy("tool", "toolparent").
taxonomy("mode1", "tool").
taxonomy("mode2", "tool").
annotated("mode1").
annotated("mode2").

taxonomy("dim1", "Types").
taxonomy("type1", "dim1").
taxonomy("type2", "dim1").
taxonomy("type3", "type2").
taxonomy("type4", "type2").
taxonomy("type5", "dim2").
taxonomy("type6", "dim2").
""")
    clingo_control.add("""
:- not tool("tool").
:- not tool("mode1").
:- term_tool("tool").
:- not term_tool("mode1").
:- not term_tool("mode2").
:- not tool_tax("mode1", "tool").

:- not type("dim1", "type3").
:- term_type("dim1", "type2").
:- not term_type("dim1", "type3").
:- not term_type("dim2", "type5").
:- not type_tax("dim1", "type3", "type2").
""")
    clingo_control.ground([('base', [])])
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

def test_aux_io_ix_max(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'aux.lp'))
    clingo_control.add("""
tool_input_("mode1", 1, "dim1", "type1").
tool_input_("mode1", 1, "dim2", "type2").
tool_input_("mode1", 2, "dim2", "type2").
tool_input_("mode1", 3, "dim1", "type3").
tool_input_("mode2", 1, "dim1", "type3").
in_((-1, 1), "dim1", "type3").
in_((-1, 2), "dim1", "type3").
in_((-1, 3), "dim1", "type3").
in_((-1, 4), "dim1", "type3").

:- not tool_input_ix_max(4).

tool_output_("mode1", 1, "dim1", "type1").
tool_output_("mode1", 1, "dim2", "type2").
tool_output_("mode1", 2, "dim2", "type2").
tool_output_("mode1", 3, "dim1", "type3").
tool_output_("mode2", 1, "dim1", "type3").
out_((0, 1), "dim1", "type3").
out_((0, 2), "dim1", "type3").
out_((0, 3), "dim1", "type3").
out_((0, 4), "dim1", "type3").

:- not tool_output_ix_max(4).
""")
    clingo_control.ground([('base', [])])
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

def test_aux_io_completion(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'aux.lp'))
    clingo_control.add("""
tool_input_("mode1", 1, "dim1", "type1").
tool_input_("mode1", 1, "dim2", "type2").
tool_input_("mode1", 2, "dim2", "type2").
tool_input_("mode1", 3, "dim1", "type3").
tool_input_("mode2", 1, "dim1", "type3").
in_((-1, 1), "dim1", "type3").
in_((-1, 2), "dim1", "type3").
in_((-1, 3), "dim1", "type3").
in_((-1, 4), "dim1", "type3").

tool_output_("mode1", 1, "dim1", "type1").
tool_output_("mode1", 1, "dim2", "type2").
tool_output_("mode1", 2, "dim2", "type2").
tool_output_("mode1", 3, "dim1", "type3").
tool_output_("mode2", 1, "dim1", "type3").
out_((0, 1), "dim1", "type3").
out_((0, 2), "dim1", "type3").
out_((0, 3), "dim1", "type3").
out_((0, 4), "dim1", "type3").

term_tool("mode1"). term_tool("mode2").
type("dim1", "type1").
type("dim1", "type3").
type("dim2", "type2").

:- not 4 = #count { Ix : tool_input("mode1", Ix, _, _) }.
:- not 4 = #count { Ix : tool_output("mode1", Ix, _, _) }.
:- not 4 = #count { Ix : in((-1, Ix), _, _) }.
:- not 4 = #count { Ix : out((0, Ix), _, _) }.
""")
    clingo_control.ground([('base', [])])
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable


# Workflow + goal

def test_workflow_goal(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'workflow.lp'))
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'goal.lp'))
    clingo_control.add("""
tool("mode1").
tool("mode2").
tool("mode3").
term_tool("mode1").
term_tool("mode2").
term_tool("mode3").

type("dim1", "dim1").
type("dim1", "type1").
type("dim1", "type2").
type("dim2", "dim2").
type("dim2", "type3").
type("dim3", "dim3").
type("dim3", "type4").

tool_input_ix_max(3).
tool_output_ix_max(2).

out((0, 1), "dim1", "type1").
out((0, 1), "dim2", "type3").
out((0, 1), "dim3", "type4").
out((0, 2), null, eps).

in((-1, 1), "dim1", "dim1").
in((-1, 1), "dim2", "type3").
in((-1, 1), "dim3", "type4").
in((-1, 2), "dim2", "dim2").
in((-1, 2), "dim3", "dim3").
in((-1, 3), null, eps).
""")
    clingo_control.add("""
type_tax(null, eps, null).
type(null, eps).
term_type(null, eps).

#program check(t).

#external query(t).

:- in((t, Ix), Dim, Type),
   in((t, Ix), null, eps),
   (Dim, Type) != (null, eps).
:- out((t, Ix), Dim, Type),
   out((t, Ix), null, eps),
   (Dim, Type) != (null, eps).
""")
    clingo_control.add("""
#program check(t).

#external query(t).

:- not 1 { use_tool(_, t) }, t>0.
:- not 1 { in((t, _), _, _) }, t>0.
:- not 1 { out((t, _), _, _) }.
:- in((Step, IxIn), Dim, _), Dim != null, not bind((Step, IxIn), _).

:- bind((In, _), (Out, _)), Out >= In, In != -1.
""")               
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

# bind

def test_bind_n_sat(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'bind_n.lp'))
    clingo_control.add("""
    bind((1, 1), (0, 1)).
    bind((2, 1), (0, 2)).
    bind((-1, 2), (2, 3)).

    in((1, 1), "d1", "t1"). in((1, 1), "d2", "t2").
    out((0, 1), "d1", "t1"). out((0, 1), "d2", "t2").
    in((2, 1), "d1", "t3").
    out((0, 2), "d1", "t3").
    in((-1, 2), "d1", "t1").
    out((2, 3), "d1", "t1").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable

def test_bind_n_unsat_eps_in(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'bind_n.lp'))
    clingo_control.add("""
    bind((1, 1), (0, 1)).

    in((1, 1), null, eps).
    out((0, 1), "d1", "t1").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_bind_n_unsat_eps_out(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'bind_n.lp'))
    clingo_control.add("""
    bind((1, 1), (0, 1)).

    in((1, 1), "d1", "t1").
    out((0, 1), null, eps).
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_bind_n_unsat_dim_mismatch(clingo_control):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'bind_n.lp'))
    clingo_control.add("""
    bind((1, 1), (0, 1)).

    in((1, 1), "d1", "t1"). in((1, 1), "d2", "t2").
    out((0, 1), "d1", "t1"). out((0, 1), "d2", "t3").
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert not res.satisfiable

def test_tax_t_op_n(clingo_control, simple_ontology):
    clingo_control.load(os.path.join(ASP_ENCODING_PATH, 'tax_t_op_n.lp'))
    clingo_control.add(simple_ontology)
    clingo_control.add("""
use_tool("tool", 1).
use_tool("tool_mode_2", 2).

in((1, 2), "dim1", "type3").
out((1, 2), "dim2", "type7").

out((0, 2), "dim1", "type3").
in((-1, 2), "dim1", "type3").
""")
    clingo_control.add('check', ['t'], """
:- query(2), not use_tool("tool_parent", 1).
:- query(2), not 1 { use_tool("tool_mode", 1) ; use_tool("tool_mode2", 1) }.
:- query(2), not use_tool("tool", 2).
:- query(2), not use_tool("tool_parent", 2).

:- query(2), not in((1, 2), "dim1", "dim1").
:- query(2), not 1 { in((1, 2), "dim1", "type4") ; in((1, 2), "dim1", "type5") }.
:- query(2), not out((1, 2), "dim2", "dim2").
""")
    clingo_control.add("""
:- not out((0, 2), "dim1", "dim1").
:- not 1 { out((0, 2), "dim1", "type4") ; out((0, 2), "dim1", "type5") }.
:- not in((-1, 2), "dim1", "dim1").
:- not 1 { in((-1, 2), "dim1", "type4") ; in((-1, 2), "dim1", "type5") }.
""")
    sim_iterative_grounding(clingo_control, steps=2)
    res: clingo.SolveResult = clingo_control.solve()
    assert res.satisfiable
