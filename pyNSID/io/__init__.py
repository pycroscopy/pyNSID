"""
Tools to read, write data in h5NSID files
This module is based on sidpy
Submodules
----------

.. autosummary::
    :toctree: _autosummary

    base
    model
    nsi_data
    other
    simple
    nsi_data
    translator
    write_utils

"""
from . import nsi_data

from .model import write_main_dataset, write_simple_attrs
from .simple import get_attr

from .nsi_data import NSIDataset
from .write_utils import read_nsid
from .write_utils import write_nsid
from .write_utils import create_empty_dataset
__all__ = ['create_empty_dataset', 'read_nsid', 'write_nsid', 'NSIDataset', 'write_main_dataset', 'write_simple_attrs']
