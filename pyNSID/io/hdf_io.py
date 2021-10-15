# -*- coding: utf-8 -*-
"""
Utilities that assist in writing NSID related data to HDF5 files

Created on Thu August 20 2020

@author: Suhas Somnath, Gerd Duscher
"""

from __future__ import (division, print_function, unicode_literals,
                        absolute_import)
import sys
from warnings import warn

import h5py
import numpy as np

__all__ = ['create_empty_dataset', 'write_nsid_dataset', 'write_results']

from dask import array as da

from sidpy import Dataset, Dimension
from sidpy.base.num_utils import contains_integers
from sidpy.hdf.hdf_utils import is_editable_h5, write_simple_attrs, \
    write_book_keeping_attrs, write_dict_to_h5_group
from sidpy.hdf.prov_utils import create_indexed_group
from sidpy.base.dict_utils import flatten_dict

from .hdf_utils import link_as_main, write_pynsid_book_keeping_attrs

if sys.version_info.major == 3:
    unicode = str


def create_empty_dataset(shape, h5_group, name='nDIM_Data'):
    """
    returns a h5py.Dataset filled with zeros according to required shape list.

    Parameters
    ----------
    shape: list
        List of integers denoting the shape of the main dataset
    h5_group: h5py.Group
        HDF5 group into which the datasets will be written into
    name: str, optional. Default: "nDIM_Data"
        Name of the main HDF5 dataset

    Returns
    -------
    h5py.Dataset
        HDF5 dataset of desired shape written according to NSID format
    """
    if not contains_integers(shape):
        raise ValueError('dimensions of shape need to be all integers')
    if not isinstance(h5_group, h5py.Group):
        raise TypeError('h5_group should be a h5py.Group object')

    return write_nsid_dataset(Dataset.from_array(np.zeros(shape)),
                              h5_group, name)


def write_nsid_dataset(dataset, h5_group, main_data_name='', verbose=False,
                       **kwargs):
    """
    Writes the provided sid dataset as a 'Main' dataset with all appropriate
    linking.

    Parameters
    ----------
    dataset : sidpy.Dataset
        Dataset to be written to HDF5 in NSID format
    h5_group : class:`h5py.Group`
        Parent group under which the datasets will be created
    main_data_name : String / Unicode
        Name to give to the main dataset. This cannot contain the '-' character
        Use this to provide better context about the dataset in the HDF5 file
    verbose : bool, Optional. Default = False
        Whether or not to write logs to standard out
    kwargs: dict
        additional keyword arguments passed on to h5py when writing data

    Return
    ------
    h5py dataset
    """
    if not isinstance(dataset, Dataset):
        raise TypeError('data to write should be sidpy Dataset')
    if not isinstance(h5_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group '
                        'object')
    if not isinstance(main_data_name, str):
        raise TypeError('main_data_name should be a string, but it instead  it'
                        ' is {}'.format(type(main_data_name)))

    if not is_editable_h5(h5_group):
        raise ValueError('The provided file is not editable')
    if verbose:
        print('h5 group and file OK')

    if not isinstance(main_data_name, str):
        raise TypeError('main_data_name must be a string')

    if main_data_name == '':
        if dataset.title.strip() == '':
            main_data_name = 'nDim_Data'
        else:
            main_data_name = dataset.title.split('/')[-1]

    main_data_name = main_data_name.strip()
    if '-' in main_data_name:
        warn('main_data_name should not contain the "-" character. Reformatted'
             ' name from:{} to '
             '{}'.format(main_data_name, main_data_name.replace('-', '_')))
    main_data_name = main_data_name.replace('-', '_')

    h5_group = h5_group.create_group(main_data_name)

    write_book_keeping_attrs(h5_group)
    write_pynsid_book_keeping_attrs(h5_group)

    #####################
    # Write Main Dataset
    ####################
    if h5_group.file.driver == 'mpio':
        if kwargs.pop('compression', None) is not None:
            warn('This HDF5 file has been opened wth the "mpio" communicator. '
                 'mpi4py does not allow creation of compressed datasets. '
                 'Compression kwarg has been removed')

    if main_data_name in h5_group:
        raise ValueError('h5 dataset of that name already exists, choose '
                         'different name or delete first')

    _ = kwargs.pop('dtype', None)

    # step 1 - create the empty dataset:
    h5_main = h5_group.create_dataset(main_data_name,
                                      shape=dataset.shape,
                                      dtype=dataset.dtype,
                                      **kwargs)
    if verbose:
        print('Created empty dataset: {} for writing Dask dataset: {}'
              ''.format(h5_main, dataset))
        print('Dask array will be written to HDF5 dataset: "{}" in file: "{}"'
              ''.format(h5_main.name, h5_main.file.filename))
    # Step 2 - now ask Dask to dump data to disk
    da.to_hdf5(h5_main.file.filename, {h5_main.name: dataset})

    if verbose:
        print('Created dataset for Main')

    #################
    # Add Dimensions
    #################
    dimensional_dict = {}

    for i, this_dim in dataset._axes.items():
        if not isinstance(this_dim, Dimension):
            raise ValueError('Dimensions {} is not a sidpy Dimension')

        this_dim_dset = h5_group.create_dataset(this_dim.name,
                                                data=this_dim.values)
        attrs_to_write = {'name': this_dim.name,
                          'units': this_dim.units,
                          'quantity': this_dim.quantity,
                          'dimension_type': this_dim.dimension_type.name}

        write_simple_attrs(this_dim_dset, attrs_to_write)
        dimensional_dict[i] = this_dim_dset

    attrs_to_write = {'quantity': dataset.quantity,
                      'units': dataset.units,
                      'main_data_name': dataset.title,
                      'data_type': dataset.data_type.name,
                      'modality': dataset.modality,
                      'source': dataset.source}

    write_simple_attrs(h5_main, attrs_to_write)
    write_pynsid_book_keeping_attrs(h5_main)

    for attr_name in dir(dataset):
        attr_val = getattr(dataset, attr_name)
        if isinstance(attr_val, dict) and attr_name[0] != '_':
            if verbose:
                print('Writing attributes from property: {} of the '
                      'sidpy.Dataset'.format(attr_name))
            write_dict_to_h5_group(h5_group, attr_val, attr_name)

    # This will attach the dimensions
    nsid_data_main = link_as_main(h5_main, dimensional_dict)

    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    dataset.h5_dataset = nsid_data_main

    return nsid_data_main


def write_results(h5_group, dataset=None, attributes=None, process_name=None):
    """
    Writes results of a processing step back to HDF5 in NSID format

    Parameters
    ----------
    h5_group : h5py.Group
        HDF5 Group into which results will be written
    dataset : sidpy.Dataset, optional. Default = None
        Dataset ??
    attributes : dict, optional. Default = None
        Metadata regarding processing step
    process_name : str, optional. Default = "Log_"
        Name of the prefix for group containing process results

    Returns
    -------
    log_group : h5py.Group
        HDF5 group containing results
    """

    found_valid_dataset = False

    if dataset is not None:

        if isinstance(dataset, Dataset):
            dataset = [dataset]

        if isinstance(dataset, list):
            if not all([isinstance(itm, Dataset) for itm in dataset]):
                raise TypeError('List contains non-Sidpy dataset entries! '
                                'Should only contain sidpy datasets')

            found_valid_dataset = True

    found_valid_attributes = False

    if attributes is not None:
        if isinstance(attributes, dict):
            if len(attributes) > 0:
                found_valid_attributes = True
        else:
            raise TypeError("Provided attributes is type {} but should be type"
                            " dict".format(type(attributes)))

    if not (found_valid_dataset or found_valid_attributes):
        raise ValueError('results should contain at least a sidpy Dataset or '
                         'a dictionary in results')
    log_name = 'Log_'
    if process_name is not None:
        log_name = log_name+process_name

    log_group = create_indexed_group(h5_group, log_name)
    write_book_keeping_attrs(log_group)
    write_pynsid_book_keeping_attrs(log_group)

    if found_valid_dataset:
        for dset in dataset:
            write_nsid_dataset(dset, log_group)

        if found_valid_attributes:
            write_simple_attrs(log_group, attributes)

    return log_group
