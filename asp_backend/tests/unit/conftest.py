"""Fixtures for unit tests."""

import os
from pathlib import Path

import clingo
import pytest

from ape_to_asp.read_config import load_json_from_file
from tests.conftest import ASP_ENCODING_PATH, FIXTURES_PATH, USE_CASES_PATH


@pytest.fixture
def quangis_owl_fixture():
    return 'http://geographicknowledge.de/vocab/CoreConceptData.rdf#'

@pytest.fixture
def geo_gmt_owl_fixture():
    return 'http://www.co-ode.org/ontologies/ont.owl#'

@pytest.fixture
def geo_gmt_owl():
    """Valid owl xml file with intersections as parent classes."""
    return os.path.join(
        USE_CASES_PATH,
        'GeoGmt',
        'GMT_UseCase_taxonomy.owl',
    )

@pytest.fixture
def image_magick_owl():
    """Valid owl xml with a single type dimension."""
    return os.path.join(
        USE_CASES_PATH,
        'ImageMagick',
        'imagemagick_taxonomy.owl',
    )

@pytest.fixture
def quangis_owl():
    """Valid owl rdf file with multiple type dimensions."""
    return os.path.join(
        USE_CASES_PATH,
        'QuAnGIS',
        'GISTaxonomy.rdf',
    )

@pytest.fixture
def empty_owl_xml():
    """Valid xml, but no class elements."""
    return os.path.join(FIXTURES_PATH, 'empty_owl.owl')

@pytest.fixture
def valid_xml_not_owl():
    """XML file but not an OWL ontology."""
    return os.path.join(
        USE_CASES_PATH,
        'ImageMagick',
        'catalog-v001.xml',
    )

@pytest.fixture
def image_magick_owl_prefix():
    return 'http://www.co-ode.org/ontologies/ont.owl#'

@pytest.fixture
def image_magick_config_e1_dataclass(image_magick_config_e1):
    config = load_json_from_file(image_magick_config_e1)
    config.ontology_path = 'ape_use_cases' + config['ontology_path'][1:]
    config.tool_annotations_path = 'ape_use_cases' + config['tool_annotations_path'][1:]
    config.constraints_path = 'ape_use_cases' + config['constraints_path'][1:]
    config.solutions_dir_path = 'ape_use_cases' + config['solutions_dir_path'][1:]
    return config


# ASP / clingo fixtures

def sim_iterative_grounding(ctl: clingo.Control, steps: int=5):
    ctl.ground([('base', []), ('check', [clingo.Number(0)])])
    ctl.assign_external(
        clingo.Function('query', [clingo.Number(0)]),
        True,
    )
    for i in range(1, steps+1):
        ctl.solve()
        ctl.release_external(clingo.Function('query', [clingo.Number(i-1)]))
        ctl.cleanup()
        ctl.ground([('step', [clingo.Number(i)]), ('check', [clingo.Number(i)])])
        ctl.assign_external(
            clingo.Function('query', [clingo.Number(i)]),
            True,
        )

@pytest.fixture
def simple_ontology():
    return """
% ontology

% tools
tool_tax("tool", "tool_parent").
tool_tax("tool_parent", "tool_root").
tool_tax("tool_mode", "tool").
tool_tax("tool_mode2", "tool").

% types
type_tax("dim1", "type1", "dim1").
type_tax("dim1", "type2", "dim1").
type_tax("dim1", "type3", "dim1").
    type_tax("dim1", "type4", "type3").
    type_tax("dim1", "type5", "type3").
type_tax("dim1", "type6", "dim1").

type_tax("dim2", "type7", "dim2").
"""

@pytest.fixture
def simple_tools():
    return """
% tool I/O

% tool 1
tool_input("tool_mode", 1, "dim1", "type1"). tool_input("tool_mode", 1, "dim1", "type2").
tool_input("tool_mode", 1, "dim2", "type7").
tool_input("tool_mode", 2, "dim1", "type3").
tool_input("tool_mode", 2, "dim2", "type8").
tool_input("tool_mode", 3, null, eps).

tool_output("tool_mode", 1, "dim1", "type6").
tool_output("tool_mode", 1, "dim2", "type7").
tool_output("tool_mode", 2, null, eps).
tool_output("tool_mode", 3, null, eps).

% tool 2
tool_input("tool_mode2", 1, "dim1", "type3").
tool_input("tool_mode2", 2, "dim1", "type1").
tool_input("tool_mode2", 3, "dim1", "type2").
tool_input("tool_mode2", 1, "dim2", "type7").
tool_input("tool_mode2", 2, "dim2", "type8").
tool_input("tool_mode2", 3, "dim2", "type8").

tool_output("tool_mode2", 1, null, eps).
tool_output("tool_mode2", 2, null, eps).
tool_output("tool_mode2", 3, null, eps).

tool_input_(Tool, Ix, Dim, Type) :- tool_input(Tool, Ix, Dim, Type), Dim != null.
tool_output_(Tool, Ix, Dim, Type) :- tool_output(Tool, Ix, Dim, Type), Dim != null.
"""

@pytest.fixture
def simple_wf_input():
    return """
% input, output constraints
out_((0, 2), "dim1", "type1").
out_((0, 2), "dim2", "type7").
out_((0, 1), "dim1", "type4").
out_((0, 3), "dim1", "type5").
"""

@pytest.fixture
def simple_wf_output():
    return """
in_((bound+1, 1), "dim1", "type6").
"""

@pytest.fixture
def clingo_control():
    return clingo.Control()

@pytest.fixture
def simple_constraint_control():
    ctl = clingo.Control(['--warn=none'])
    ctl.add("""
tool_input_ix_max(2).
tool_output_ix_max(2).
""")
    ctl.load(os.path.join(ASP_ENCODING_PATH, 'constraints.lp'))
    return ctl

@pytest.fixture
def simple_instance_files(tmp_path):
    tax_file: Path = tmp_path / "tax.lp"
    tool_file: Path = tmp_path / "tools.lp"
    constraint_file: Path = tmp_path / "cst.lp"
    io_file: Path = tmp_path / "wf_io.lp"

    with open(str(tax_file), 'w', encoding='utf-8') as tax_f:
        tax_f.write(
"""
taxonomy_tool_root("tool_root").
taxonomy_type_root("dim1").
taxonomy_type_root("dim2").

taxonomy("tool_parent", "tool_root").
taxonomy("tool", "tool_parent").

taxonomy("dim1", "type1").
taxonomy("dim1", "type2").
    taxonomy("type2", "type3").
    taxonomy("type2", "type4").
taxonomy("dim2", "type5").
taxonomy("dim2", "type6").
"""
    )

    with open(str(tool_file), 'w', encoding='utf-8') as tax_f:
        tax_f.write(
"""
taxonomy("tool_mode1", "tool").
taxonomy("tool_mode2", "tool").

tool_input_("tool_mode1", 1, "dim1", "type1").
tool_input_("tool_mode1", 1, "dim2", "type5").
tool_input_("tool_mode1", 2, "dim1", "type2").
tool_output_("tool_mode1", 1, "dim2", "type5").

tool_input_("tool_mode2", 1, "dim2", "type5").
tool_output_("tool_mode2", 1, "dim1", "type1").
tool_output_("tool_mode2", 1, "dim2", "type5").
tool_output_("tool_mode2", 2, "dim1", "type2").
"""
    )
    with open(str(constraint_file), 'w', encoding='utf-8') as tax_f:
        tax_f.write(
"""
constraint(0, "nuse_m").
constraint_tool_param(0, 1, "tool_mode2").

use_gen_one.
"""
    )
    with open(str(io_file), 'w', encoding='utf-8') as tax_f:
        tax_f.write(
"""
out_((0, 1), "dim1", "type1").
in_((-1, 1), "dim2", "type5").
"""
    )
    return str(tax_file), str(tool_file), str(constraint_file), str(io_file)
