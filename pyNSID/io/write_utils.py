# -*- coding: utf-8 -*-
"""
Utilities that assist in writing NSID related data to HDF5 files

Created on Thu August 20 2020

@author: Suhas Somnath, Gerd Duscher
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import h5py
import numpy as np

__all__ = ['validate_dimensions', 'validate_main_dimensions']

if sys.version_info.major == 3:
    unicode = str

sys.path.append('../../../sidpy/')
import sidpy as sid
from sidpy.base.num_utils import contains_integers


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

    return write_nsid(sid.Dataset.from_array(np.zeros(shape)), h5_group, name)


def read_nsid(dset, chunks=None, name=None, lock=False):
    # create vanilla dask array
    dataset = sid.Dataset.from_array(np.array(dset), chunks, name, lock)

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

    # TODO: modality and source not yet properties
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
        # print(dim, dset.dims[dim].label)
        # print(dset.dims[dim][0][0])
        dim_dict = dict(dset.parent[dset.dims[dim].label].attrs)
        # print(dset.dims[dim].label, np.array(dset.dims[dim][0]))
        # print(dset.parent[dset.dims[0].label][()])
        # print(dim_dict['quantity'], dim_dict['units'], dim_dict['dimension_type'])
        dataset.set_dimension(dim,
                              sid.sid.Dimension(dset.dims[dim].label, np.array(dset.parent[dset.dims[dim].label][()]),
                                                dim_dict['quantity'], dim_dict['units'],
                                                dim_dict['dimension_type']))
    dataset.attrs = dict(dset.attrs)

    dataset.original_metadata = {}
    if 'original_metadata' in dset.parent:
        dataset.original_metadata = dict(dset.parent['original_metadata'].attrs)

    return dataset


def write_nsid(dataset, h5_group, main_data_name=''):
    if main_data_name == '':
        if dataset.title.strip() == '':
            main_data_name = 'nDim_Data'
        else:
            main_data_name = dataset.title
    from .model import write_main_dataset

    dset = write_main_dataset(h5_group, np.array(dataset), main_data_name,
                              dataset.quantity, dataset.units, dataset.data_type, dataset.modality,
                              dataset.source, dataset.axes, verbose=False)

    for key, item in dataset.attrs.items():
        # TODO: Check item to be simple
        dset.attrs[key] = item

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

    return dset

