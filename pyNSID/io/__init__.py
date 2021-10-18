"""
Tools to read, write data in h5NSID files

Submodules
----------

.. autosummary::
    :toctree: _autosummary

    hdf_utils
    hdf_io
    nsi_reader
"""
from . import hdf_utils, hdf_io
from .hdf_io import *
from .hdf_utils import make_nexus_compatible

__all__ = ['hdf_utils', 'hdf_io', 'make_nexus_compatible']