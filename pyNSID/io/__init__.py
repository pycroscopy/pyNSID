"""
Tools to read, write data in h5NSID files
This module is based on sidpy
Submodules
----------

.. autosummary::
    :toctree: _autosummary

    hdf_utils
    hdf_io
    nsi_data
    nsi_reader
"""
from . import hdf_utils, hdf_io, nsi_data
from .nsi_data import NSIDataset
from .nsi_reader import NSIDReader
from .hdf_io import *

__all__ = ['hdf_utils', 'hdf_io', 'NSIDataset', 'NSIDReader']
