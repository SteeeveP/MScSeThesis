constraint(0, "use_m").
constraint_tool_param(0, 1, "log").
constraint(1, "use_m").
constraint_tool_param(1, 1, "normality_plots").
constraint(2, "operation_input").
constraint_tool_param(2, 1, "normality_plots").
constraint_type_param(2, 2, "DataClass", "Column").
constraint_type_param(2, 2, "StatisticalRelevance", "DependentVariable").
constraint(3, "operation_input").
constraint_tool_param(3, 1, "normality_plots").
constraint_type_param(3, 2, "DataClass", "Column").
constraint_type_param(3, 2, "StatisticalRelevance", "IndependentVariable").
constraint(4, "connected_op").
constraint_tool_param(4, 1, "log").
constraint_tool_param(4, 2, "normality_plots").
use_inputs_one.
use_gen_none.
not_connected_ident_op.
