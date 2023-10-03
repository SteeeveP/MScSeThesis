constraint(0, "use_m").
constraint_tool_param(0, 1, "describe").
constraint(1, "operation_input").
constraint_tool_param(1, 1, "histplot").
constraint_type_param(1, 2, "StatisticalRelevance", "DependentVariable").
constraint(2, "operation_input").
constraint_tool_param(2, 1, "skew").
constraint_type_param(2, 2, "StatisticalRelevance", "DependentVariable").
constraint(3, "operation_input").
constraint_tool_param(3, 1, "kurt").
constraint_type_param(3, 2, "StatisticalRelevance", "DependentVariable").
use_inputs_one.
use_gen_one.
not_connected_ident_op.
