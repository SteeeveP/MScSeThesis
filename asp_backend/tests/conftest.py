"""Fixtures for unit and integration tests."""

import os
import sys

import pytest

from asp_solver.incremental_solver import IncASPSolver


sys.path.append(os.path.join(os.getcwd()))

USE_CASES_PATH = 'ape_use_cases'
FIXTURES_PATH = os.path.join(
    'tests',
    'fixtures',
)
ASP_ENCODING_PATH = 'asp_encoding'


@pytest.fixture
def inc_solver_inst():
    return IncASPSolver(ASP_ENCODING_PATH)

@pytest.fixture
def geo_gmt_config_e0():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'GeoGMT',
        'E0',
        'config.json',
    )

@pytest.fixture
def geo_gmt_config_e1():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'GeoGMT',
        'E1',
        'config.json',
    )

@pytest.fixture
def image_magick_config_e1():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'ImageMagick',
        'Example1',
        'config.json',
    )

@pytest.fixture
def image_magick_config_e2():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'ImageMagick',
        'Example2',
        'config.json',
    )

@pytest.fixture
def mass_spec_config_no1():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No1',
        'config_original.json',
    )

@pytest.fixture
def mass_spec_config_no2():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No2',
        'config_extended.json',
    )

@pytest.fixture
def mass_spec_config_no3():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No3',
        'config_extended.json',
    )

@pytest.fixture
def mass_spec_config_no4():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No4',
        'config_extended.json',
    )

@pytest.fixture
def mass_spec_config_no1_full():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No1',
        'config_full_bio.tools.json',
    )

@pytest.fixture
def mass_spec_config_no2_full():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No2',
        'config_full_bio.tools.json',
    )

@pytest.fixture
def mass_spec_config_no3_full():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No3',
        'config_full_bio.tools.json',
    )

@pytest.fixture
def mass_spec_config_no4_full():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'MassSpectometry',
        'No4',
        'config_full_bio.tools.json',
    )

@pytest.fixture
def ds_config_inplace_full():
    """Constraint file for ImageMagick usecase example 1."""
    return os.path.join(
        USE_CASES_PATH,
        'ds',
        'config.json',
    )
