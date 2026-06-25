#
# FRETBursts - A single-molecule FRET burst analysis toolkit.
#
"""
Pytest configuration: provide the datasets the suite needs *offline*.

Historically the tests downloaded several large HDF5 files from figshare/zenodo
into ``notebooks/data/``. Here we instead generate small synthetic Photon-HDF5
files (see :mod:`synthdata`) with matching names and structure, into a temporary
directory, and point each test module's ``DATASETS_DIR`` at it.

This keeps the existing test bodies completely unchanged while removing all
network dependencies. To run against real data instead, set the environment
variable ``FRETBURSTS_TEST_DATA`` to a directory containing the real files.
"""

import os
import sys
import importlib

import pytest

import synthdata


# filename -> generator producing that file. Distinct seeds keep the four
# ns-ALEX files (used as a "grouped" multi-file dataset) independent.
_GENERATORS = {
    '0023uLRpitc_NTP_20dT_0.5GndCl.hdf5':
        lambda p: synthdata.make_usalex(p, seed=1),
    'dsdna_d7_d17_50_50_1.hdf5': lambda p: synthdata.make_nsalex(p, seed=7),
    'HP3_TE150_SPC630.hdf5': lambda p: synthdata.make_nsalex(p, seed=150),
    'HP3_TE200_SPC630.hdf5': lambda p: synthdata.make_nsalex(p, seed=200),
    'HP3_TE250_SPC630.hdf5': lambda p: synthdata.make_nsalex(p, seed=250),
    'HP3_TE300_SPC630.hdf5': lambda p: synthdata.make_nsalex(p, seed=300),
    '12d_New_30p_320mW_steer_3.hdf5':
        lambda p: synthdata.make_multispot(p, seed=12),
}

# Test modules that hard-code a ``DATASETS_DIR`` module global.
_DATA_MODULES = ('test_burstlib', 'test_burstlib_ext', 'test_burst_plot')


@pytest.fixture(scope='session', autouse=True)
def _datasets(tmp_path_factory):
    """Make the required datasets available and redirect ``DATASETS_DIR``."""
    real = os.environ.get('FRETBURSTS_TEST_DATA')
    if real:
        data_dir = real
    else:
        out = tmp_path_factory.mktemp('synthdata')
        for name, generate in _GENERATORS.items():
            generate(str(out / name))
        data_dir = str(out)

    datasets_dir = os.path.join(data_dir, '')  # ensure a trailing separator
    for modname in _DATA_MODULES:
        module = sys.modules.get(modname)
        if module is None:
            module = importlib.import_module(modname)
        module.DATASETS_DIR = datasets_dir
    yield
