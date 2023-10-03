"""Unit tests for asp_encoding folder"""

import clingo

from tests.unit.conftest import sim_iterative_grounding


# TOOLS

def test_constraint_ite_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "ite_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool2", 5).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_itn_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "itn_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool2", 5).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_ite_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "ite_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 5).
use_tool("Tool2", 1).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_itn_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "itn_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 5).
use_tool("Tool2", 1).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_next_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "next_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 2).
use_tool("Tool2", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_prev_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "prev_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 2).
use_tool("Tool2", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_next_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "next_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 3).
use_tool("Tool2", 2).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_prev_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "prev_m").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 3).
use_tool("Tool2", 2).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool1", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool2", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_nuse_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "nuse_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool2", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_nuse_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "nuse_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool1", 3).
""")
    sim_iterative_grounding(simple_constraint_control)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_last_m_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "last_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool1", 1).
use_tool("Tool1", 4).
use_tool("Tool1", 5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_last_m_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "last_m").
constraint_tool_param(0, 1, "Tool1").

use_tool("Tool1", 1).
use_tool("Tool1", 2).
use_tool("Tool1", 4).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_connected_op_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool2", 5).
bind((5, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_connected_op_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool3", 4).
use_tool("Tool2", 5).
bind((4, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_connected_op_unsat_tool_unused(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool3", 4).
bind((4, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_not_connected_op_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "not_connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool3", 4).
use_tool("Tool2", 5).
bind((4, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_not_connected_op_sat_tool_unused(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "not_connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool3", 4).
bind((4, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_not_connected_op_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "not_connected_op").
constraint_tool_param(0, 1, "Tool1").
constraint_tool_param(0, 2, "Tool2").

use_tool("Tool1", 1).
use_tool("Tool2", 5).
bind((5, 1), (1, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable


# TYPES

def test_constraint_use_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

in((5, 2), "dim1", "type1").
in((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

in((5, 1), "dim1", "type1").
in((5, 2), "dim2", "type2").
in((5, 2), "dim1", "type3").
out((5, 2), "dim1", "type1").
in((4, 2), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_gen_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

out((5, 2), "dim1", "type1").
out((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_gen_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

out((5, 1), "dim1", "type1").
out((5, 2), "dim2", "type2").
out((5, 2), "dim1", "type3").
in((5, 2), "dim1", "type1").
out((4, 2), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_nuse_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "nuse_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

in((5, 1), "dim1", "type1").
in((5, 2), "dim2", "type2").
in((5, 2), "dim1", "type3").
out((5, 2), "dim1", "type1").
in((4, 2), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_nuse_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "nuse_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

in((5, 2), "dim1", "type1").
in((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_ngen_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "ngen_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

out((5, 1), "dim1", "type1").
out((5, 2), "dim2", "type2").
out((5, 2), "dim1", "type3").
in((5, 2), "dim1", "type1").
out((4, 2), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_ngen_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "ngen_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").

out((5, 2), "dim1", "type1").
out((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_ite_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
in((4, 1), "dim2", "type2").
in((5, 2), "dim1", "type3").
in((5, 2), "dim2", "type4").

""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_ite_t_sat_precondition_not_met(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "use_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_ite_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
in((4, 1), "dim2", "type2").
in((5, 2), "dim1", "type3").
in((5, 2), "dim2", "type5").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_gen_ite_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
out((4, 1), "dim2", "type2").
out((5, 2), "dim1", "type3").
out((5, 2), "dim2", "type4").

""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_gen_ite_t_sat_precondition_not_met(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "gen_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_gen_ite_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_ite_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
out((4, 1), "dim2", "type2").
out((5, 2), "dim1", "type3").
out((5, 2), "dim2", "type5").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_itn_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
in((4, 1), "dim2", "type2").
in((5, 2), "dim1", "type3").
in((5, 2), "dim2", "type5").

""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_itn_t_sat_precondition_not_met(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "use_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
in((5, 2), "dim1", "type3").
in((5, 2), "dim2", "type5").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_itn_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "use_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

in((4, 1), "dim1", "type1").
in((4, 1), "dim2", "type2").
in((5, 2), "dim1", "type3").
in((5, 2), "dim2", "type4").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_gen_itn_t_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
out((4, 1), "dim2", "type2").
out((5, 2), "dim1", "type3").
out((5, 2), "dim2", "type5").

""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_gen_itn_t_sat_precondition_not_met(
    simple_constraint_control,
):
    simple_constraint_control.add("""
constraint(0, "gen_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
out((5, 2), "dim1", "type3").
out((5, 2), "dim2", "type5").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_gen_itn_t_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "gen_itn_t").
constraint_type_param(0, 1, "dim1", "type1").
constraint_type_param(0, 1, "dim2", "type2").
constraint_type_param(0, 2, "dim1", "type3").
constraint_type_param(0, 2, "dim2", "type4").

out((4, 1), "dim1", "type1").
out((4, 1), "dim2", "type2").
out((5, 2), "dim1", "type3").
out((5, 2), "dim2", "type4").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable


# MIXED

def test_constraint_operation_input_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "operation_input").
constraint_tool_param(0, 1, "Tool1").
constraint_type_param(0, 2, "dim1", "type1").
constraint_type_param(0, 2, "dim2", "type2").

use_tool("Tool1", 5).
in((5, 2), "dim1", "type1").
in((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_operation_input_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "operation_input").
constraint_tool_param(0, 1, "Tool1").
constraint_type_param(0, 2, "dim1", "type1").
constraint_type_param(0, 2, "dim2", "type2").

use_tool("Tool1", 5).
out((5, 2), "dim1", "type1").
in((5, 1), "dim1", "type1").
in((4, 2), "dim1", "type1").
in((5, 2), "dim2", "type2").

use_tool("Tool2", 4).
in((4, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_operation_output_sat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "operation_output").
constraint_tool_param(0, 1, "Tool1").
constraint_type_param(0, 2, "dim1", "type1").
constraint_type_param(0, 2, "dim2", "type2").

use_tool("Tool1", 5).
out((5, 2), "dim1", "type1").
out((5, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_operation_output_unsat(simple_constraint_control):
    simple_constraint_control.add("""
constraint(0, "operation_output").
constraint_tool_param(0, 1, "Tool1").
constraint_type_param(0, 2, "dim1", "type1").
constraint_type_param(0, 2, "dim2", "type2").

use_tool("Tool1", 5).
in((5, 2), "dim1", "type1").
out((5, 1), "dim1", "type1").
out((4, 2), "dim1", "type1").
out((5, 2), "dim2", "type2").

use_tool("Tool2", 4).
out((4, 2), "dim2", "type2").
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable


# CONFIG FLAGS

def test_constraint_not_connected_ident_op_sat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
not_connected_ident_op.

use_tool("Tool1", 4).
use_tool("Tool2", 5).
term_tool("Tool1").
term_tool("Tool2").
bind((4, 1), (5, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_not_connected_ident_op_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
not_connected_ident_op.

use_tool("Tool1", 4).
use_tool("Tool1", 5).
term_tool("Tool1").
bind((4, 1), (5, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_inputs_all_sat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_all.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
bind((4, 1), (0, 2)).
bind((3, 2), (0, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_inputs_all_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_all.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
out_((0, 3), "dim2", "type2").
bind((4, 1), (0, 2)).
bind((3, 2), (0, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_inputs_none_sat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_none.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
bind((4, 1), (2, 2)).
bind((3, 2), (1, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_inputs_none_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_none.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
out_((0, 3), "dim2", "type2").
bind((4, 1), (0, 2)).
bind((3, 2), (0, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_inputs_one_sat_one(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_one.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
bind((4, 1), (0, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_inputs_one_unsat_none(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_one.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
out_((0, 3), "dim2", "type2").
bind((4, 1), (2, 2)).
bind((3, 2), (1, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_inputs_one_sat_multiple(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_inputs_one.

out_((0, 1), "dim1", "type1").
out_((0, 1), "dim3", "type3").
out_((0, 2), "dim2", "type2").
out_((0, 3), "dim2", "type2").
bind((4, 1), (0, 2)).
bind((3, 2), (0, 1)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_gen_all_sat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_all.

out((1, 1), "dim1", "type1").
out((1, 2), "dim1", "type1").
out((4, 1), "dim1", "type1").
bind((2, 1), (1, 2)).
bind((2, 2), (1, 1)).
bind((6, 1), (4, 1)).
use_tool("temp", 1..5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_gen_all_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_all.

out((1, 1), "dim1", "type1").
out((1, 2), "dim1", "type1").
out((4, 1), "dim1", "type1").
out((5, 1), "dim1", "type1").
bind((2, 1), (1, 2)).
bind((2, 2), (1, 1)).
bind((6, 1), (4, 1)).
use_tool("temp", 1..5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_gen_none_sat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_none.

bind((1, 1), (0, 2)).
use_tool("temp", 1..5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_gen_none_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_none.

bind((6, 1), (4, 1)).
use_tool("temp", 1..5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_gen_one_sat_one(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_one.

bind((6, 1), (5, 2)).
bind((6, 1), (4, 1)).
bind((3, 1), (3, 2)).
bind((3, 1), (2, 2)).
bind((2, 1), (1, 1)).
bind((1, 1), (0, 2)).
use_tool("temp", 1..5).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable

def test_constraint_use_gen_one_unsat(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_one.

out((5, 2), "td", "tt").
out((4, 1), "td", "tt").
out((3, 2), "td", "tt").
out((2, 2), "td", "tt").
out((1, 1), "td", "tt").

bind((6, 1), (5, 2)).
bind((5, 1), (4, 1)).
bind((4, 1), (3, 2)).
bind((3, 1), (1, 2)).
bind((2, 1), (1, 1)).
bind((1, 1), (0, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert not res.satisfiable

def test_constraint_use_gen_one_sat_multiple(
    simple_constraint_control,
):
    simple_constraint_control.add("""
use_gen_one.

bind((6, 1), (5, 2)).
bind((5, 1), (4, 1)).
bind((4, 1), (3, 2)).
bind((3, 1), (2, 2)).
bind((2, 1), (1, 1)).
bind((1, 1), (0, 2)).
""")
    sim_iterative_grounding(simple_constraint_control, 5)
    res: clingo.SolveResult = simple_constraint_control.solve()
    assert res.satisfiable
