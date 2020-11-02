# -*- coding: utf-8 -*-
"""
Reader capable of reading one or all NSID datasets present in a given HDF5 file

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import sidpy

from pyNSID.io.hdf_utils import get_all_main, read_h5py_dataset, check_if_main

if sys.version_info.major == 3:
    unicode = str


class NSIDReader(sidpy.Reader):

    def __init__(self, h5_object):
        """
        dset - str
            specific Main dataset that needs to be read in.
            # If no path is specified, read all available NSID Main datasets
        """
        warn('This Reader will eventually be moved to the ScopeReaders package'
             '. Be prepared to change your import statements',
             FutureWarning)

        if not isinstance(h5_object.file, h5py.File):
            raise TypeError('we can only read h5py datasets')

        super(NSIDReader, self).__init__(file_path=h5_object.file.name)

        self.dset = None
        self.main_datasets = []
        if isinstance(h5_object, h5py.Dataset):
            self.dset = h5_object
            self.h5_group = self.dset.parent

        elif isinstance(h5_object, h5py.Group):
            self.h5_group = h5_object
        else:
            raise TypeError('we can only read h5py datasets')

        # Find all main datasets is done in read as the file may change between readings
        # DO NOT close HDF5 file. Dask array will fail if you do so.
        # TODO: sidpy.Dataset may need the ability to close a HDF5 file
        # Perhaps this would be done by reading all contents into memory..

    def can_read(self):
        list_of_main = get_all_main(self.h5_group, verbose=False)
        return len(list_of_main) > 0

    def read(self, dataset=None):
        if not isinstance(self.h5_group, h5py.Group):
            raise TypeError('This function needs to be initialised with a hdf5 group or dataset first')

        if dataset is None:
            return self.read_all(recursive=True)
        else:
            if isinstance(dataset, h5py.Dataset):
                return read_h5py_dataset(dataset)

    def read_all(self, recursive=True, parent=None):

        if parent is None:
            h5_group = self.h5_group
        else:
            if isinstance(parent, h5py.Group):
                h5_group = parent
            else:
                raise TypeError('parent should be a h5py object')

        if recursive:
            list_of_main = get_all_main(h5_group, verbose=False)
        else:
            list_of_main = []
            for key in h5_group:
                if isinstance(h5_group[key], h5py.Dataset):
                    if check_if_main(h5_group[key]):
                        list_of_main.append(h5_group[key])
        # Go through each of the identified
        list_of_datasets = []
        for dset in list_of_main:
            list_of_datasets.append(read_h5py_dataset(dset))
        return list_of_datasets
