#program check(t).

% [[W_bind]]_n
% (1)
:- bind((t, In), Out),
   in((t, In), DimIn, TypeIn),
   not out(Out, DimIn, TypeIn).
:- bind((t, In), Out),
   out(Out, DimOut, TypeOut),
   not in((t, In), DimOut, TypeOut).
:- bind((-1, In), (t, Out)),
   in((-1, In), DimIn, TypeIn),
   not out((t, Out), DimIn, TypeIn).
:- bind((-1, In), (t, Out)),
   out((t, Out), DimOut, TypeOut),
   not in((-1, In), DimOut, TypeOut).

% (2) redundant, alread in choice rule: (0..Step-1, 1..MaxOut)
