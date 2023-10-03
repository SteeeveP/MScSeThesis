#program step(t).

% Preserve tool taxonomy [[TaxOp]]_n
1 {
    use_tool(SubTool, t) : tool_tax(SubTool, Tool)
} :- use_tool(Tool, t),
     tool_tax(_, Tool).
use_tool(ParentTool, t) :- use_tool(Tool, t),
                           tool_tax(Tool, ParentTool).

% Preserve type taxonomy [[TaxT]]_n
1 {
    in((t, Ix), Dim, SubType) : type_tax(Dim, SubType, Type)
} :- in((t, Ix), Dim, Type),
     type_tax(Dim, _, Type),
     Type != eps.
in((t, Ix), Dim, ParentType) :- in((t, Ix), Dim, Type),
                                type_tax(Dim, Type, ParentType),
                                Type != eps.
1 {
    out((t, Ix), Dim, SubType) : type_tax(Dim, SubType, Type)
} :- out((t, Ix), Dim, Type),
     type_tax(Dim, _, Type),
     Type != eps.
out((t, Ix), Dim, ParentType) :- out((t, Ix), Dim, Type),
                                 type_tax(Dim, Type, ParentType),
                                 Type != eps.

#program base.

1 {
    in((-1, Ix), Dim, SubType) : type_tax(Dim, SubType, Type)
} :- in((-1, Ix), Dim, Type),
     type_tax(Dim, _, Type),
     Type != eps.
in((-1, Ix), Dim, ParentType) :- in((-1, Ix), Dim, Type),
                                 type_tax(Dim, Type, ParentType),
                                 Type != eps.
1 {
    out((0, Ix), Dim, SubType) : type_tax(Dim, SubType, Type)
} :- out((0, Ix), Dim, Type),
     type_tax(Dim, _, Type),
     Type != eps.
out((0, Ix), Dim, ParentType) :- out((0, Ix), Dim, Type),
                                 type_tax(Dim, Type, ParentType),
                                 Type != eps.
