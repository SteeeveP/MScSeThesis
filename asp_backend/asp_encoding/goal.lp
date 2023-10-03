#program check(t).

#external query(t).

% workflow out step=-1
:- query(t), in((-1, Ix), Dim, Type), Dim != null, not bind((-1, Ix), _).
