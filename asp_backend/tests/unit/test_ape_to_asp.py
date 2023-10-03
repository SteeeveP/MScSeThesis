"""Unit tests for ape_to_asp module."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import clingo
import pytest

from ape_to_asp.read_constraints import Constraint
from ape_to_asp.read_owl import owl_xml_to_cls_elements, load_owl
from ape_to_asp.read_config import APEConfig, load_json_from_file
from ape_to_asp.init_asp_input import init_core_config, init_run_config, write_to_asp_file
from ape_to_asp.read_tool_annotations import ToolMode, read_tool_annoation_json


# read_owl

def test_owl_xml_to_cls_elements_xml(image_magick_owl):
    assert len(owl_xml_to_cls_elements(image_magick_owl)) == 40

def test_owl_xml_to_cls_elements_rdf(quangis_owl):
    assert len(owl_xml_to_cls_elements(quangis_owl)) == 75

def test_load_owl_empty_file(empty_owl_xml):
    try:
        load_owl(empty_owl_xml, '')
        raise AssertionError('Function continued on empty owl.')
    except ValueError:
        pass

def test_owl_xml_to_cls_elements_non_xml(image_magick_config_e1):
    try:
        owl_xml_to_cls_elements(image_magick_config_e1)
        raise AssertionError('Function ran on non xml')
    except TypeError:
        pass

def test_owl_xml_to_cls_elements_non_owl(valid_xml_not_owl):
    try:
        owl_xml_to_cls_elements(valid_xml_not_owl)
        raise AssertionError('Function ran on non owl')
    except TypeError:
        pass

def test_load_owl_image_magick(image_magick_owl, image_magick_owl_prefix):
    assert ('Descriptive', 'Tool') in load_owl(image_magick_owl, image_magick_owl_prefix)
    assert len(load_owl(image_magick_owl, image_magick_owl_prefix)) == 38

def test_load_owl_multi_inheritance(quangis_owl, quangis_owl_fixture):
    assert len(load_owl(quangis_owl, quangis_owl_fixture)) == 81

def test_load_owl_intersection_par_cls(geo_gmt_owl, geo_gmt_owl_fixture):
    assert len(load_owl(geo_gmt_owl, geo_gmt_owl_fixture)) == 164


# read_tool_annottions

def test_ToolMode_valid_types():
    ToolMode(**{
        "label": "mild_blur",
        "id": "mild_blur",
        "taxonomyOperations": ["Blur", "Mild"],
        "inputs": [
            { "Type": ["Image"] }
        ],
        "outputs": [
            { "Type": ["Image"], "Format": ["PNG"] }
        ],
        "implementation": { "code": "@output[0]='@output[0].png'\nconvert $@input[0] -blur 2x1 $@output[0]\n" }
    })

def test_ToolMode_valid_missing_inputs():
    ToolMode(**{
        "label": "mild_blur",
        "id": "mild_blur",
        "taxonomyOperations": ["Blur", "Mild"],
        "outputs": [
            { "Type": ["Image"], "Format": ["PNG"] }
        ],
        "implementation": { "code": "@output[0]='@output[0].png'\nconvert $@input[0] -blur 2x1 $@output[0]\n" }
    })

def test_ToolMode_valid_missing_implementation():
    ToolMode(**{
        "label": "mild_blur",
        "id": "mild_blur",
        "taxonomyOperations": ["Blur", "Mild"],
        "inputs": [
            { "Type": ["Image"] }
        ],
        "outputs": [
            { "Type": ["Image"], "Format": ["PNG"] }
        ],
    })

def test_ToolMode_invalid_type_implementation():
    try:
        ToolMode(**{
            "label": "mild_blur",
            "id": "mild_blur",
            "taxonomyOperations": ["Blur", "Mild"],
            "inputs": [
                { "Type": ["Image"] }
            ],
            "outputs": [
                { "Type": ["Image"], "Format": ["PNG"] }
            ],
            "implementation": "@output[0]='@output[0].png'\nconvert $@input[0] -blur 2x1 $@output[0]\n" 
        })
        raise AssertionError("Ran with wrong implementation type.")
    except TypeError:
        pass


def test_read_tool_annotation_json():
    read_tool_annoation_json(os.path.join('ape_use_cases','ImageMagick','tool_annotations.json'))


# read_config

def test_load_json_from_file_non_json(geo_gmt_owl):
    try:
        load_json_from_file(geo_gmt_owl)
        raise AssertionError('Function ran on a non json')
    except TypeError:
        pass

def test_load_json_from_file_valid_config(image_magick_config_e1):
    assert isinstance(load_json_from_file(image_magick_config_e1), APEConfig)

def test_init_core_config_valid_config(image_magick_config_e1_dataclass):
    init_core_config(image_magick_config_e1_dataclass)


@pytest.mark.parametrize(
    "core_config_required_attribute",
    [
        'ontology_path',
        'ontologyPrefixIRI',
        'toolsTaxonomyRoot',
        'dataDimensionsTaxonomyRoots',
        'tool_annotations_path',
        'strict_tool_annotations',
    ]
)

def test_APEConfig_missing_attribute(
    image_magick_config_e1,
    core_config_required_attribute,
):
    with open(image_magick_config_e1, 'r', encoding='utf-8') as json_f:
        config_json: Dict[str, Any] = json.load(json_f)
    try:
        APEConfig(**{
            k: v
            for k,v
            in config_json.items()
            if k != core_config_required_attribute
        })
        raise AssertionError(f'Ran with missing attribute {core_config_required_attribute}.')
    except TypeError:
        pass


@pytest.mark.parametrize(
    "core_config_path_attribute",
    [
        'ontology_path',
        'tool_annotations_path',
    ]
)

def test_init_core_config_invalid_paths(
    image_magick_config_e1_dataclass,
    core_config_path_attribute,
):
    image_magick_config_e1_dataclass[core_config_path_attribute] = 'INVAL_PATH'
    try:
        init_core_config(image_magick_config_e1_dataclass)
        raise AssertionError('Ran with invalid path.')
    except FileNotFoundError:
        pass


@pytest.mark.parametrize(
    "core_config_ontology_attribute, core_config_ontology_invalid_value",
    [
        ('tools_tax_root', 'ToolTax'),
        ('data_dim_tax_roots', ['TypesRoot']),
    ],
)

def test_init_core_config_invalid_ontology_attributes(
    image_magick_config_e1_dataclass,
    core_config_ontology_attribute,
    core_config_ontology_invalid_value,
):
    image_magick_config_e1_dataclass[core_config_ontology_attribute] = core_config_ontology_invalid_value
    try:
        init_core_config(image_magick_config_e1_dataclass)
        raise AssertionError(
            f"Ran with invalid attribute: {core_config_ontology_attribute}: "
            + f"{core_config_ontology_invalid_value}"
        )
    except ValueError:
        pass

def test_init_core_config(image_magick_config_e1_dataclass):
    _, tool_modes = init_core_config(
        image_magick_config_e1_dataclass
    )
    assert len(tool_modes) == 35

def test_init_run_config(image_magick_config_e1_dataclass):
    constraints, cst_flags, _ = init_run_config(
        image_magick_config_e1_dataclass
    )
    assert set(cst_flags) == {'use_inputs_all', 'use_gen_one', 'not_connected_ident_op'}
    assert [hash(cst) for cst in constraints] == [
        hash(Constraint('use_m', [{'Tool': ['Borders']}]))
    ]

def test_write_to_asp_file(tmp_path):
    asp_ontology_path: Path = tmp_path / "ontology.lp"
    asp_tool_path: Path = tmp_path / "tools.lp"
    asp_cst_path: Path = tmp_path / "cst.lp"
    asp_io_path: Path = tmp_path / "wf_io.lp"
    write_to_asp_file(
        [clingo.Function('taxonomy', [clingo.String('type1'), clingo.String('par_type1')])],
        [],
        [],
        [],
        [],
        str(asp_ontology_path),
        str(asp_tool_path),
        str(asp_cst_path),
        str(asp_io_path),
    )
    assert asp_ontology_path.read_text() == 'taxonomy("type1", "par_type1").\n'
