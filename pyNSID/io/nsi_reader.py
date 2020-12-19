# -*- coding: utf-8 -*-
"""
Reader capable of reading one or all NSID datasets present in a given HDF5 file

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somnath, Maxim Ziadtinov
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

    def __init__(self, file_path):
        """
        Creates an instance of NSIDReader which can read one or more HDF5
        datasets formatted according to NSID into sidpy.Dataset objects

        Parameters
        ----------
        file_path : str, h5py.File, or h5py.Group
            Path to a HDF5 file or a handle to an open HDF5 file or group
            object
        """

        warn('This Reader will eventually be moved to the ScopeReaders package'
             '. Be prepared to change your import statements',
             FutureWarning)

        super(NSIDReader, self).__init__(file_path)

        # Let h5py raise an OS error if a non-HDF5 file was provided
        self._h5_file = h5py.File(file_path, mode='w')

        # DO NOT close HDF5 file. Dask array will fail if you do so.

    def can_read(self):
        main_dsets = get_all_main(self._h5_file, verbose=False)
        return len(main_dsets) > 0

    def read(self, h5_object=None):
        if h5_object is None:
            return self.read_all(recursive=True)
        if not isinstance(h5_object, (h5py.Group, h5py.Dataset)):
            raise TypeError('Provided h5_object was not a h5py.Dataset or '
                            'h5py.Group object but was of type: {}'
                            ''.format(type(h5_object)))
        self.__validate_obj_in_same_file(h5_object)
        if isinstance(h5_object, h5py.Dataset):
            return read_h5py_dataset(h5_object)
        else:
            return self.read_all(parent=h5_object)

    def __validate_obj_in_same_file(self, h5_object):
        if h5_object.file != self._h5_file:
            raise OSError('The file containing the provided h5_object: {} is '
                          'not the same as provided HDF5 file when '
                          'instantiating this object: {}'
                          ''.format(h5_object.file.filename,
                                    self._h5_file.filename))

    def read_all(self, recursive=True, parent=None):

        if parent is None:
            h5_group = self._h5_file
        else:
            if not isinstance(parent, h5py.Group):
                raise TypeError('parent should be a h5py object')
            self.__validate_obj_in_same_file(parent)
            h5_group = parent

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
