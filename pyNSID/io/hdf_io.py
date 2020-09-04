# -*- coding: utf-8 -*-
"""
Utilities that assist in writing NSID related data to HDF5 files

Created on Thu August 20 2020

@author: Suhas Somnath, Gerd Duscher
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
from warnings import warn

import h5py
import numpy as np

__all__ = ['create_empty_dataset', 'write_nsid_dataset', 'write_results']

from dask import array as da

from sidpy import Dataset, Dimension
from sidpy.base.num_utils import contains_integers
# from sidpy.base.string_utils import validate_string_args
from sidpy.hdf.hdf_utils import is_editable_h5, write_simple_attrs
from sidpy.hdf.prov_utils import create_indexed_group
from sidpy.base.dict_utils import flatten_dict
# from sidpy.hdf.dtype_utils import validate_dtype

from .hdf_utils import link_as_main  # , validate_main_dimensions
from ..__version__ import *
if sys.version_info.major == 3:
    unicode = str


def create_empty_dataset(shape, h5_group, name='nDIM_Data'):
    """
    returns a NSID dataset filled with zeros according to required shape list.

    :param shape: list of integer
    :param h5_group: hdf5 group
    :param name: -optional- name of NSID dataset

    :return:
    NSID dataset
    """
    if not contains_integers(shape):
        raise ValueError('dimensions of shape need to be all integers')
    if not isinstance(h5_group, h5py.Group):
        raise TypeError('h5_group should be a h5py.Group object')

    return write_nsid_dataset(Dataset.from_array(np.zeros(shape)), h5_group, name)


def write_nsid_dataset(dataset, h5_group, main_data_name='', verbose=False, **kwargs):
    """
        Writes the provided sid dataset as a 'Main' dataset with all appropriate linking.

        Parameters
        ----------
        dataset: main_data : sidpy Dataset
        h5_group : class:`h5py.Group`
            Parent group under which the datasets will be created
        main_data_name : String / Unicode
            Name to give to the main dataset. This cannot contain the '-' character.
        verbose: boolean
        kwargs: additional h5py parameters

        Return
        ------
        h5py dataset
    """
    if not isinstance(dataset, Dataset):
        raise ValueError('data to write should be sidpy Dataset')

    if not isinstance(h5_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group object')
    if not is_editable_h5(h5_group):
        raise ValueError('The provided file is not editable')
    if verbose:
        print('h5 group and file OK')

    if main_data_name == '':
        if dataset.title.strip() == '':
            main_data_name = 'nDim_Data'
        else:
            main_data_name = dataset.title.split('/')[-1]

    main_data_name = main_data_name.strip()
    if '-' in main_data_name:
        warn('main_data_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(main_data_name, main_data_name.replace('-', '_')))
    main_data_name = main_data_name.replace('-', '_')

    #####################
    # Write Main Dataset
    ####################
    if h5_group.file.driver == 'mpio':
        if kwargs.pop('compression', None) is not None:
            warn('This HDF5 file has been opened wth the "mpio" communicator. '
                 'mpi4py does not allow creation of compressed datasets. Compression kwarg has been removed')

    print(h5_group, main_data_name)
    if main_data_name in h5_group:
        raise ValueError('h5 dataset of that name already exists, choose different name or delete first')

    _ = kwargs.pop('dtype', None)
    # step 1 - create the empty dataset:
    h5_main = h5_group.create_dataset(main_data_name, shape=dataset.shape, dtype=dataset.dtype, **kwargs)
    if verbose:
        print('Created empty dataset: {} for writing Dask dataset: {}'.format(h5_main, dataset))
        print('Dask array will be written to HDF5 dataset: "{}" in file: "{}"'.format(h5_main.name,
                                                                                      h5_main.file.filename))
    # Step 2 - now ask Dask to dump data to disk
    da.to_hdf5(h5_main.file.filename, {h5_main.name: dataset})

    if verbose:
        print('Created dataset for Main')

    #################
    # Add Dimensions
    #################
    dimensional_dict = {}

    for i, this_dim in dataset.axes.items():
        if not isinstance(this_dim, Dimension):
            raise ValueError('Dimensions {} is not a sidpy Dimension')

        this_dim_dset = h5_group.create_dataset(this_dim.name, data=this_dim.values)
        attrs_to_write = {'name': this_dim.name, 'units': this_dim.units, 'quantity': this_dim.quantity,
                          'dimension_type': this_dim.dimension_type, 'nsid_version': version}
        write_simple_attrs(this_dim_dset, attrs_to_write)
        dimensional_dict[i] = this_dim_dset

    attrs_to_write = {'quantity': dataset.quantity, 'units': dataset.units, 'nsid_version': version,
                      'main_data_name': dataset.title, 'data_type': dataset.data_type,
                      'modality': dataset.modality, 'source': dataset.source}
    write_simple_attrs(h5_main, attrs_to_write)
    # dset = write_main_dataset(h5_group, np.array(dataset), main_data_name,
    #                          dataset.quantity, dataset.units, dataset.data_type, dataset.modality,
    #                          dataset.source, dataset.axes, verbose=False)

    for key, item in dataset.attrs.items():
        if key not in attrs_to_write:
            # TODO: Check item to be simple
            h5_main.attrs[key] = item

    original_group = h5_group.create_group('original_metadata')
    for key, item in dataset.original_metadata.items():
        original_group.attrs[key] = item

    if hasattr(dataset, 'aberrations'):
        aberrations_group = h5_group.create_group('aberrations')
        for key, item in dataset.aberrations.items():
            aberrations_group.attrs[key] = item

    if hasattr(dataset, 'annotations'):
        annotations_group = h5_group.create_group('annotations')
        for key, item in dataset.annotations.items():
            annotations_group.attrs[key] = item

    # ToDo: check if we need  write_book_keeping_attrs(h5_main)
    # This will attach the dimensions
    nsid_data_main = link_as_main(h5_main, dimensional_dict)

    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    dataset.h5_dataset = nsid_data_main

    return nsid_data_main  # NSIDataset(h5_main)


def write_results(h5_group, dataset=None, attributes=None, process_name=None):

    found_valid_dataset = False
    if dataset is not None:
        if isinstance(dataset, Dataset):
            found_valid_dataset = True
    found_valid_attributes = False

    if attributes is not None:
        if isinstance(attributes, dict):
            if len(attributes) > 0:
                found_valid_attributes = True
    if not (found_valid_dataset or found_valid_attributes):
        raise ValueError('results should contain at least a sidpy Dataset or a dictionary in results')
    log_name = 'Log_'
    if process_name is not None:
        log_name = log_name+process_name
    log_group = create_indexed_group(h5_group, log_name)

    if found_valid_dataset:
        write_nsid_dataset(dataset, log_group)
    if found_valid_attributes:
        write_simple_attrs(log_group, flatten_dict(attributes))

    return log_group
