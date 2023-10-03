# ASP APE Solver

Modules:
- APE input to ASP instance: [`ape_to_asp/`](ape_to_asp)
    - Reads OWL, tool annotations, constraints, and config
    - Creates ASP instance from internal representation
- ASP problem encoding: [`asp_encoding/`](asp_encoding)
    - Encodings for auxiliary predicates (`aux.lp`) and goal conditions (`goal.lp`)
    - Encodings of APE's constraints (`constraints.lp`)
    - Encodings for output filtering  (`io.lp`)
    - All other encodings and predicates are named after their SAT counterpart from APE
- Incremental solver: [`asp_solver/`](asp_solver)
- Output parsing: [`workflow_output`](workflow_output)
    - Parses ASP output to workflow representation
    - Creates notebook from workflow representation (adapted from [`native_ape_output_parsing/APE_to_notebook.py`](native_ape_output_parsing/APE_to_notebook.py))
- Some tests: [`tests`](tests)
