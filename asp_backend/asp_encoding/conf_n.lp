#program step(t).

% [[Conf(X1, X2)]]_n
:- 2 { use_tool(Tool, Step) : term_tool(Tool) }, Step=t.
:- 2 { in((Step, Ix), Dim, Type) : term_type(Dim, Type) },
   Step=t,
   Ix=1..MaxIn, tool_input_ix_max(MaxIn),
   type(Dim, _), Dim != null.
:- 2 { out((Step, Ix), Dim, Type) : term_type(Dim, Type) },
   Step=t,
   Ix=1..MaxOut, tool_output_ix_max(MaxOut),
   type(Dim, _), Dim != null.

#program base.
:- 2 { in((Step, Ix), Dim, Type) : term_type(Dim, Type) },
   Step=-1,
   Ix=1..MaxIn, tool_input_ix_max(MaxIn),
   type(Dim, _), Dim != null.
:- 2 { out((Step, Ix), Dim, Type) : term_type(Dim, Type) },
   Step=0,
   Ix=1..MaxOut, tool_output_ix_max(MaxOut),
   type(Dim, _), Dim != null.
