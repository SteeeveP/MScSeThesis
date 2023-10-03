# Toolset & Ontology

## Toolset

Python library: [`wrapper_functions.py`](wrapper_functions.py)

## Ontology

### Used Version

- OWL: [`ontology_v2_DIM_2.owl`](ontology_v2_DIM_2.owl)
- Tool annotations: [`tool_annotations_v2_DIM_2.json`](tool_annotations_v2_DIM_2.json)

### Other Variants

[`variants/`](variants)

- OWL and tool annotations for different numbers of dimensions (1, 3, 5)
- V1 OWL

### Source

- Raw markdown ontology: [`src/markdown/`](src/markdown)
    - DataClass dimension: `data_class_...md`
    - DataSetIndex dimension: `data_set_index_...md`
    - DataState dimension: `data_state_...md`
    - StatisticalRelevance dimension: `statistical_relevance_...md`
    - Tool dimension with annotations: `tools_...md`
- Scripts to generate OWL and tool annotations
    - Markdown parser module: [`src/scripts/md_to_json.py`](src/scripts/md_to_json.py)
    - Generates ontologies with different numbers of dimensions: [`src/scripts/multi_ontology.ipynb`](src/scripts/multi_ontology.ipynb)
