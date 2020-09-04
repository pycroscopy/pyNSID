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
import numpy as np
from sidpy import Reader
from sidpy.sid import Dimension, Dataset

from .hdf_utils import get_all_main

if sys.version_info.major == 3:
    unicode = str


class NSIDReader(Reader):

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

    @staticmethod
    def read_h5py_dataset(dset):

        if not isinstance(dset, h5py.Dataset):
            raise TypeError('can only read single Dataset, use read_all_in_group or read_all function instead')
        # create vanilla dask array
        dataset = Dataset.from_array(np.array(dset))

        if 'title' in dset.attrs:
            dataset.title = dset.attrs['title']
        else:
            dataset.title = dset.name

        if 'units' in dset.attrs:
            dataset.units = dset.attrs['units']
        else:
            dataset.units = 'generic'

        if 'quantity' in dset.attrs:
            dataset.quantity = dset.attrs['quantity']
        else:
            dataset.quantity = 'generic'

        if 'data_type' in dset.attrs:
            dataset.data_type = dset.attrs['data_type']
        else:
            dataset.data_type = 'generic'

        if 'modality' in dset.attrs:
            dataset.modality = dset.attrs['modality']
        else:
            dataset.modality = 'generic'

        if 'source' in dset.attrs:
            dataset.source = dset.attrs['source']
        else:
            dataset.source = 'generic'

        dataset.axes = {}

        for dim in range(np.array(dset).ndim):
            try:
                label = dset.dims[dim].keys()[-1]
                name = dset.dims[dim][label].name
                dim_dict = {'quantity': 'generic', 'units': 'generic', 'dimension_type': 'generic'}
                dim_dict.update(dict(dset.parent[name].attrs))

                dataset.set_dimension(dim, Dimension(dset.dims[dim].label,
                                                     np.array(dset.parent[name][()]),
                                                     dim_dict['quantity'], dim_dict['units'],
                                                     dim_dict['dimension_type']))
            except ValueError:
                print('dimension {} not NSID type using generic'.format(dim))

        dataset.attrs = dict(dset.attrs)

        dataset.original_metadata = {}
        if 'original_metadata' in dset.parent:
            dataset.original_metadata = dict(dset.parent['original_metadata'].attrs)

        # hdf5 information
        dataset.h5_file = dset.file
        dataset.h5_filename = dset.file.filename
        dataset.h5_dataset = dset.name

        return dataset

    def can_read(self):
        pass

    def read(self):
        if not isinstance(self.h5_group, h5py.Group):
            raise TypeError('This function needs to be initialised with a hdf5 group or dataset first')
        list_of_main = get_all_main(self.h5_group, verbose=False)

        """
        Go through each of the identified
        """
        list_of_datasets = []
        for dset in list_of_main:
            list_of_datasets.append(self.read_h5py_dataset(dset))

        return list_of_datasets

    def read_all_in_group(self, recursive=True):
        pass
