constraint(0, "use_m").
constraint_tool_param(0, 1, "describe").
constraint(1, "use_m").
constraint_tool_param(1, 1, "Distribution").
constraint(2, "operation_input").
constraint_tool_param(2, 1, "Distribution").
constraint_type_param(2, 2, "StatisticalRelevance", "DependentVariable").
constraint(3, "use_m").
constraint_tool_param(3, 1, "skew").
constraint(4, "operation_input").
constraint_tool_param(4, 1, "skew").
constraint_type_param(4, 2, "StatisticalRelevance", "DependentVariable").
constraint(5, "use_m").
constraint_tool_param(5, 1, "kurt").
constraint(6, "operation_input").
constraint_tool_param(6, 1, "kurt").
constraint_type_param(6, 2, "StatisticalRelevance", "DependentVariable").
use_inputs_one.
use_gen_none.
not_connected_ident_op.
