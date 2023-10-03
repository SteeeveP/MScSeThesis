#program step(t).

% [[W]]_n
1 = { use_tool(Tool, t) : term_tool(Tool) }. % required for broken mass spec ontology
1 { use_tool(Tool, t) : tool(Tool) }.
1 {
    in((t, Ix), Dim, Type) : type(Dim, Type)
} :- Ix=1..MaxIn,
     tool_input_ix_max(MaxIn).
1 {
    out((t, Ix), Dim, Type) : type(Dim, Type)
} :- Ix=1..MaxOut,
     tool_output_ix_max(MaxOut).
1 {
    bind((t, IxIn), (0..t-1, 1..MaxOut))
} :- IxIn=1..MaxIn,
     tool_input_ix_max(MaxIn),
     tool_output_ix_max(MaxOut),
     not in((t, IxIn), null, eps).
{
    bind((-1, IxIn), (t, 1..MaxOut))
} :- IxIn=1..MaxIn,
     tool_input_ix_max(MaxIn),
     tool_output_ix_max(MaxOut),
     not in((-1, IxIn), null, eps).

#program base.

%! output at t=0
{
    bind((-1, IxIn), (0, 1..MaxOut))
} :- IxIn=1..MaxIn,
     tool_input_ix_max(MaxIn),
     tool_output_ix_max(MaxOut),
     not in((-1, IxIn), null, eps).

%! already in aux.lp
% 1 {
%     out((0, Ix), Dim, Type) : type(Dim, Type)
% } :- Ix=1..MaxOut,
%      tool_input_ix_max(MaxOut).
