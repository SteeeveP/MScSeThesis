#program check(t).

#external query(t).

% TODO: change query(t) constraints to new fulfilled(ID, t) :- ..., add fulfilled to goal.

% =============================================================================
%                               Aux Rules
% =============================================================================

% A constrained dim is fulfilled by IN/OUT_ID
% if its type matches AT LEAST ONE of the constraint dim types.
used_constraint_type_dim(CST_ID, CST_Ix, Dim, (t, IN_Ix)) :-
    in((t, IN_Ix), Dim, Type),
    constraint_type_param(CST_ID, CST_Ix, Dim, Type).
generated_constraint_type_dim(CST_ID, CST_Ix, Dim, (t, OUT_Ix)) :-
    out((t, OUT_Ix), Dim, Type),
    constraint_type_param(CST_ID, CST_Ix, Dim, Type).

% A constrained type is fulfilled by (Step, Ix) if ALL constrained dims are fulfilled.
fulfilled_use(CST_ID, CST_Ix, t) :-
    Ix=1..Max, tool_input_ix_max(Max),
    constraint_type_param(CST_ID, CST_Ix, _, _),
    used_constraint_type_dim(CST_ID, CST_Ix, Dim, (t, Ix)):
        constraint_type_param(CST_ID, CST_Ix, Dim, _).
fulfilled_gen(CST_ID, CST_Ix, t) :-
    Ix=1..Max, tool_output_ix_max(Max),
    constraint_type_param(CST_ID, CST_Ix, _, _),
    generated_constraint_type_dim(CST_ID, CST_Ix, Dim, (t, Ix)):
        constraint_type_param(CST_ID, CST_Ix, Dim, _).

% =============================================================================
%                               Direct Constraint Encoding
% =============================================================================

% -----------------------------------------------------------------------------
%                               Natural Language Templates
% -----------------------------------------------------------------------------

%? Every Tool/Type Var represents one OPTION:
%? The constraint parameters are always lists

% ite_m: two params, both tool params
% If we use module ${parameter_1}, then use ${parameter_2} subsequently.
:- constraint(ID, "ite_m"),
   constraint_tool_param(ID, 1, Tool1),
   use_tool(Tool1, Step),
   not 1 {
        use_tool(Tool2, LaterStep) : constraint_tool_param(ID, 2, Tool2),
                                     LaterStep=Step+1..t
   },
   query(t).

% itn_m: two params, both tool params
% If we use module ${parameter_1}, then do not use ${parameter_2} subsequently.
:- constraint(ID, "itn_m"),
   constraint_tool_param(ID, 1, Tool1),
   use_tool(Tool1, Step),
   1 {
        use_tool(Tool2, LaterStep) : constraint_tool_param(ID, 2, Tool2),
                                     LaterStep=Step+1..t
   },
   query(t).

% next_m: two params, both tool params
% If we use module ${parameter_1}, then use ${parameter_2} as a next module in the sequence.
:- constraint(ID, "next_m"),
   constraint_tool_param(ID, 1, Tool1),
   use_tool(Tool1, t-1),
   not 1 { use_tool(Tool2, t) : constraint_tool_param(ID, 2, Tool2) }.

% prev_m: two params, both tool params
% If we use module ${parameter_1},
% then we must have used ${parameter_2} as a previous module in the sequence.
:- constraint(ID, "prev_m"),
   constraint_tool_param(ID, 1, Tool1),
   use_tool(Tool1, t),
   not 1 { use_tool(Tool2, t-1) : constraint_tool_param(ID, 2, Tool2) }.

% use_m: one tool param
% Use module ${parameter_1} in the solution.
:- constraint(ID, "use_m"),
   not 1 { use_tool(Tool, _) : constraint_tool_param(ID, 1, Tool) },
   query(t).

% nuse_m: one tool param
% Do not use module ${parameter_1} in the solution.
:- constraint(ID, "nuse_m"),
   1 { use_tool(Tool, _) : constraint_tool_param(ID, 1, Tool) },
   query(t).

% last_m: one tool param
% Use ${parameter_1} as last module in the solution.
:- constraint(ID, "last_m"),
   not 1 { use_tool(Tool, t) : constraint_tool_param(ID, 1, Tool) },
   query(t).

% use_t: one type param
% Use type ${parameter_1} in the solution.
:- constraint(ID, "use_t"),
   not fulfilled_use(ID, 1, _),
   query(t).

% gen_t: one type param
% Generate type ${parameter_1} in the solution.
:- constraint(ID, "gen_t"),
   not fulfilled_gen(ID, 1, _),
   query(t).

% nuse_t: one type param
% Do not use type ${parameter_1} in the solution.
:- constraint(ID, "nuse_t"),
   fulfilled_use(ID, 1, _),
   query(t).

% ngen_t: one type param
% Do not generate type ${parameter_1} in the solution.
:- constraint(ID, "ngen_t"),
   fulfilled_gen(ID, 1, _),
   query(t).

% use_ite_t: two type params
% If we have used data type ${parameter_1},
% then use type ${parameter_2} subsequently.
:- constraint(ID, "use_ite_t"),
   fulfilled_use(ID, 1, Step),
   not 1 { fulfilled_use(ID, 2, LaterStep) : LaterStep=Step+1..t },
   query(t).

% gen_ite_t: two type params
% If we have generated data type ${parameter_1},
% then generate type ${parameter_2} subsequently.
:- constraint(ID, "gen_ite_t"),
   fulfilled_gen(ID, 1, Step),
   not 1 { fulfilled_gen(ID, 2, LaterStep) : LaterStep=Step+1..t },
   query(t).

% use_itn_t: two type params
% If we have used data type ${parameter_1},
% then do not use type ${parameter_2} subsequently.
:- constraint(ID, "use_itn_t"),
   fulfilled_use(ID, 1, Step),
   1 { fulfilled_use(ID, 2, LaterStep) : LaterStep=Step+1..t },
   query(t).

% gen_itn_t: two type params
% If we have generated data type ${parameter_1},
% then do not generate type ${parameter_2} subsequently.
:- constraint(ID, "gen_itn_t"),
   fulfilled_gen(ID, 1, Step),
   1 { fulfilled_gen(ID, 2, LaterStep) : LaterStep=Step+1..t },
   query(t).

% operation_input: one tool param, one type param
% Use the operation with an input of the given type.
:- constraint(ID, "operation_input"),
   constraint_tool_param(ID, 1, Tool),
   not 1 { use_tool(Tool, Step) : fulfilled_use(ID, 2, Step) },
   query(t).

% operation_output: one tool param, one type param
% Use the operation to generate an output of the given type.
:- constraint(ID, "operation_output"),
   constraint_tool_param(ID, 1, Tool),
   not 1 { use_tool(Tool, Step) : fulfilled_gen(ID, 2, Step) },
   query(t).

% connected_op: two tool params
% The 1st operation should generate an output used by the 2nd operation.
:- constraint(ID, "connected_op"),
   constraint_tool_param(ID, 1, Tool1),
   constraint_tool_param(ID, 2, Tool2),
   not 1 { bind((Step2, _), (Step1, _)): use_tool(Tool1, Step1), use_tool(Tool2, Step2) },
   query(t).

% not_connected_op: two tool params
% The 1st operation should never generate an output used by the 2nd operation.
:- constraint(ID, "not_connected_op"),
   constraint_tool_param(ID, 1, Tool1),
   constraint_tool_param(ID, 2, Tool2),
   1 { bind((Step2, _), (Step1, _)): use_tool(Tool1, Step1), use_tool(Tool2, Step2) },
   query(t).

%! not_repeat_op
%! No operation that belongs to the subtree should be repeated over.

% -----------------------------------------------------------------------------
%                               Flags / Config Constraints
% -----------------------------------------------------------------------------

%? not_connected_ident_op: no params
%? Identical tools should not be connected via output/input
:- not_connected_ident_op,
   term_tool(Tool),
   use_tool(Tool, Step1),
   use_tool(Tool, Step2),
   bind((Step2, _), (Step1, _)),
   query(t).

% use_inputs_all
:- use_inputs_all,
   out_((0, Ix), Dim, _),
   Dim != null,
   not bind(_, (0, Ix)),
   query(t).
% % use_inputs_none
% :- use_inputs_none,
%    bind(_, (0, _)),
%    query(t).
% use_inputs_one
:- use_inputs_one,
   not bind(_, (0, _)),
   query(t).

% use_gen_all
:- use_gen_all,
   out((T, Ix), Dim, _),
   Dim != null,
   not bind(_, (T, Ix)),
   query(t).
% use_gen_none
% :- use_gen_none,
%    bind(_, (T, _)),
%    T=1..t,
%    query(t).
% use_gen_one
:- use_gen_one,
   not bind(_, (T, _)),
   out((T, _), Dim, Type),
   Dim != null,
   T=1..t,
   query(t).
   