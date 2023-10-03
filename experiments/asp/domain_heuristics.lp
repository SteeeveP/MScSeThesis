#program step(t).

% usually no modeling
#heuristic in((t, _), "DataClass", "SklearnObject"). [40, false]
#heuristic in((t, _), "DataClass", "TextEmbeddingObject"). [40, false]
#heuristic in((t, _), "DataClass", "NumpyObject"). [40, false]
#heuristic use_tool(t, "Modeling"). [40, false]
#heuristic use_tool(t, "Embedding"). [40, false]
#heuristic use_tool(t, "Encoding"). [40, false]

% decide tools before data
#heuristic use_tool(t, _). [20, level]


% training: make binding to train test split data more likely than input data


% Do not use the same input twice
#heuristic in((t, X), "APE_label", COL): in((t, Y), "APE_label", COL), X != Y. [10, false]

% plotting: make binding numeric data to y more likely than nonnumeric data

% plotting: make binding numeric data to vars in positions 4,5 less likely
% most cases: (1: df, 2: x, 3: y, 4: hue, 5: style)
#heuristic in((t, 4), "DataClass", "NumberColumn"): use_tool(t, "Plotting"). [10, false]

% categorical plotting: dont use numeric data for x axis
#heuristic in((t, 3), "DataClass", "NumberColumn"): use_tool(t, "countplot"). [15, false]
#heuristic in((t, 3), "DataClass", "NumberColumn"): use_tool(t, "boxplot"). [15, false]
#heuristic in((t, 3), "DataClass", "NumberColumn"): use_tool(t, "barplot"). [15, false]

% plotting with figure and axes: use same step for figure and axes
:- in((t, IxF), "DataClass", "Figure"), in((t, IxA), "DataClass", "Axes"),
   bind((t, IxA), (T1, _)), bind((t, IxF), (T2, _)),
   T1 != T2.
