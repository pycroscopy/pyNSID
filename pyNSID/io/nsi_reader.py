# -*- coding: utf-8 -*-
"""
Reader capable of reading one or all NSID datasets present in a given HDF5 file

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somas
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np
from dask import array as da
from sidpy import Reader
from sidpy.base.num_utils import contains_integers
from sidpy.hdf.dtype_utils import validate_dtype
from sidpy.base.string_utils import validate_single_string_arg, validate_string_args
from sidpy.hdf.hdf_utils import write_simple_attrs, is_editable_h5
from sidpy.sid import Dimension

if sys.version_info.major == 3:
    unicode = str


class NSIDReader(Reader):

    def __init__(self, h5_path, dset_path=None):
        """
        dset_path - str or list of str.
            Path to a specific Main dataset that needs to be read in.
            If no path is specified, read all available NSID Main datasets
        """
        warn('This Reader will eventually be moved to the ScopeReaders package'
             '. Be prepared to change your import statements',
             FutureWarning)
        # TODO: Perhaps init may want to call can_read to see if this is a
        # legitimate file
        # Subsequently, Find all main datasets
        # Do the rest in read()
        # DO NOT close HDF5 file. Dask array will fail if you do so.
        # TODO: sidpy.Dataset may need the ability to close a HDF5 file
        # Perhaps this would be done by reading all contents into memory..
        pass

    def read(self):
        """
        Go through each of the identified
        """
        pass

    def can_read(self):
        pass
