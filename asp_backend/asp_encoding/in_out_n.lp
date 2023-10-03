#program step(t).

% [[In]]_n
:- use_tool(Tool, t),
   tool_input_(Tool, Ix, _, _),
   type(Dim, _), Dim != null,
   not 1 { in((t, Ix), Dim, Type) : tool_input(Tool, Ix, Dim, Type) }.
:- use_tool(Tool, t),
   tool_input(Tool, Ix, null, eps),
   not in((t, Ix), null, eps).

% [[Out(X)]]_n
:- use_tool(Tool, t),
   tool_output_(Tool, Ix, _, _),
   type(Dim, _), Dim != null,
   not 1 { out((t, Ix), Dim, Type) : tool_output(Tool, Ix, Dim, Type) }.
:- use_tool(Tool, t),
   tool_output(Tool, Ix, null, eps),
   not out((t, Ix), null, eps).

% empty type exlusivity
:- in((t, Ix), Dim, Type),
   in((t, Ix), null, eps),
   (Dim, Type) != (null, eps).
:- out((t, Ix), Dim, Type),
   out((t, Ix), null, eps),
   (Dim, Type) != (null, eps).
