#show.
% #show bind/2.
% #show in/3.
% #show out/3.
#show in((T, IX), Dim, Type): in((T, IX), Dim, Type), term_type(Dim, Type).
#show out((T, IX), Dim, Type): out((T, IX), Dim, Type), term_type(Dim, Type).

#program step(t).

#show use_tool(t, Tool): use_tool(Tool, t), term_tool(Tool).
#show in((t, IX), Dim, Type): in((t, IX), Dim, Type), term_type(Dim, Type).
#show out((t, IX), Dim, Type): out((t, IX), Dim, Type), term_type(Dim, Type).

#program check(t).

#external query(t).

1 = { 
    last_bind((T, IxIn), (Step, IxOut), t) : bind((T, IxIn), (Step, IxOut))
} :- bind((T, IxIn), (Step, _)),
     not bind((T, IxIn), (StepLater, _)): StepLater=Step+1..t.

#show bind(In, Out): last_bind(In, Out, t), query(t).
