#program base.

% split taxonomies
tool_tax(Tool, ToolRoot) :- taxonomy_tool_root(ToolRoot),
                            taxonomy(Tool, ToolRoot).
tool_tax(SubTool, Tool) :- tool_tax(Tool, _),
                           taxonomy(SubTool, Tool).

type_tax(Dim, Type, Dim) :- taxonomy_type_root(Dim),
                            taxonomy(Type, Dim).
type_tax(Dim, SubType, Type) :- type_tax(Dim, Type, _),
                                taxonomy(SubType, Type).

% ontology
tool(Tool) :- tool_tax(_, Tool).
tool(Tool) :- tool_tax(Tool, _).
% annotated is required for broken mass spec ontology
term_tool(Tool) :- tool(Tool), not tool_tax(_, Tool), annotated(Tool).

type(Dim, Type) :- type_tax(Dim, _, Type).
type(Dim, Type) :- type_tax(Dim, Type, _).
term_type(Dim, Type) :- type(Dim, Type), not type_tax(Dim, _, Type).

% exclusive empty type (exlusivity in in_out_n.lp)
type_tax(null, eps, null). type(null, eps). term_type(null, eps).

% tool I/O max index (workflow out step = -1)
tool_input_ix_max(Max) :- Max = #max {
    Ix : tool_input_(_, Ix, _, _) ;
    Ix : in_((-1, Ix), _, _)
}.
tool_output_ix_max(Max) :- Max = #max {
    Ix : tool_output_(_, Ix, _, _) ;
    Ix : out_((0, Ix), _, _)
}.

% complete implicit tool I/O: explicit, implicit dim, implicit eps
tool_input(TermTool, Ix, Dim, Type) :- tool_input_(TermTool, Ix, Dim, Type).
tool_input(TermTool, Ix, Dim, Dim) :- tool_input_(TermTool, Ix, _, _),
                                      type(Dim, _), Dim != null,
                                      not tool_input_(TermTool, Ix, Dim, _).
tool_input(TermTool, Ix, null, eps) :- term_tool(TermTool),
                                       Ix=1..Max, tool_input_ix_max(Max),
                                       not tool_input_(TermTool, Ix, _, _).
tool_output(TermTool, Ix, Dim, Type) :- tool_output_(TermTool, Ix, Dim, Type).
tool_output(TermTool, Ix, Dim, Dim) :- Ix=1..Max, tool_output_ix_max(Max),
                                       tool_output_(TermTool, Ix, _, _),
                                       type(Dim, _), Dim != null,
                                       not tool_output_(TermTool, Ix, Dim, _).
tool_output(TermTool, Ix, null, eps) :- term_tool(TermTool),
                                        Ix=1..Max, tool_output_ix_max(Max),
                                        not tool_output_(TermTool, Ix, _, _).

% complete implicit workflow input
out((0, Ix), Dim, Type) :- out_((0, Ix), Dim, Type).
out((0, Ix), null, eps) :- Ix=1..Max, tool_output_ix_max(Max),
                           not out_((0, Ix), _, _).
out((0, Ix), Dim, Dim) :- out_((0, Ix), _, _),
                          type(Dim, _), Dim != null,
                          not out_((0, Ix), Dim, _).
% specified ix and dim at workflow out step=-1
in((-1, Ix), Dim, Type) :- in_((-1, Ix), Dim, Type).
% empty ix
in((-1, Ix), null, eps) :- Ix=1..Max, tool_input_ix_max(Max),
                           not in_((-1, Ix), _, _).
% specified ix, empty dim
in((-1, Ix), Dim, Dim) :- in_((-1, Ix), _, _),
                          type(Dim, _), Dim != null,
                          not in_((-1, Ix), Dim, _).
