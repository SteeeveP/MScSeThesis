constraint(0, "use_m").
constraint_tool_param(0, 1, "scatterplot").
constraint(1, "operation_input").
constraint_tool_param(1, 1, "scatterplot").
constraint_type_param(1, 2, "StatisticalRelevance", "DependentVariable").
constraint(2, "operation_input").
constraint_tool_param(2, 1, "scatterplot").
constraint_type_param(2, 2, "StatisticalRelevance", "IndependentVariable").
constraint(3, "use_m").
constraint_tool_param(3, 1, "Distribution").
constraint(4, "operation_input").
constraint_tool_param(4, 1, "Distribution").
constraint_type_param(4, 2, "StatisticalRelevance", "DependentVariable").
constraint(5, "operation_input").
constraint_tool_param(5, 1, "Distribution").
constraint_type_param(5, 2, "DataClass", "StrColumn").
constraint_type_param(5, 2, "StatisticalRelevance", "IndependentVariable").
constraint(6, "use_m").
constraint_tool_param(6, 1, "set_figure_size").
constraint(7, "operation_input").
constraint_tool_param(7, 1, "set_figure_size").
constraint_type_param(7, 2, "APE_label", "9").
constraint(8, "operation_input").
constraint_tool_param(8, 1, "set_figure_size").
constraint_type_param(8, 2, "APE_label", "16").
constraint(9, "connected_op").
constraint_tool_param(9, 1, "set_figure_size").
constraint_tool_param(9, 2, "Distribution").
constraint(10, "use_m").
constraint_tool_param(10, 1, "rotate_x_labels").
constraint(11, "connected_op").
constraint_tool_param(11, 1, "Distribution").
constraint_tool_param(11, 2, "rotate_x_labels").
constraint(12, "use_m").
constraint_tool_param(12, 1, "k_most_corr_indep_var_corr_matrix").
constraint(13, "operation_input").
constraint_tool_param(13, 1, "k_most_corr_indep_var_corr_matrix").
constraint_type_param(13, 2, "StatisticalRelevance", "DependentVariable").
constraint(14, "operation_input").
constraint_tool_param(14, 1, "k_most_corr_indep_var_corr_matrix").
constraint_type_param(14, 2, "APE_label", "10").
constraint(15, "use_m").
constraint_tool_param(15, 1, "heatmap").
constraint(16, "connected_op").
constraint_tool_param(16, 1, "k_most_corr_indep_var_corr_matrix").
constraint_tool_param(16, 2, "heatmap").
constraint(17, "use_m").
constraint_tool_param(17, 1, "pairplot").
constraint(18, "operation_input").
constraint_tool_param(18, 1, "pairplot").
constraint_type_param(18, 2, "StatisticalRelevance", "DependentVariable").
constraint(19, "operation_input").
constraint_tool_param(19, 1, "pairplot").
constraint_type_param(19, 2, "APE_label", "10").
use_inputs_one.
use_gen_none.
not_connected_ident_op.
