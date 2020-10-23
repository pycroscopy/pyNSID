# -*- coding: utf-8 -*-
"""
Lower-level and simpler NSID-specific HDF5 utilities that facilitate
higher-level data operations

Created on Tue Aug  3 21:14:25 2020

@author: Gerd Duscher, and Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
import h5py
import numpy as np
import dask.array as da

from sidpy.hdf.hdf_utils import get_attr, copy_dataset
from sidpy.hdf import hdf_utils as hut
from sidpy import Dimension, Dataset

if sys.version_info.major == 3:
    unicode = str


def get_all_main(parent, verbose=False):
    """
    Simple function to recursively print the contents of an hdf5 group
    Parameters
    ----------
    parent : :class:`h5py.Group`
        HDF5 Group to search within
    verbose : bool, optional. Default = False
        If true, extra print statements (usually for debugging) are enabled
    Returns
    -------
    main_list : list of h5py.Dataset
        The datasets found in the file that meet the 'Main Data' criteria.
    """
    if not isinstance(parent, (h5py.Group, h5py.File)):
        raise TypeError('parent should be a h5py.File or h5py.Group object')

    main_list = list()

    def __check(name, obj):
        if verbose:
            print(name, obj)
        if isinstance(obj, h5py.Dataset):
            if verbose:
                print(name, 'is an HDF5 Dataset.')
            ismain = check_if_main(obj)
            if ismain:
                if verbose:
                    print(name, 'is a `Main` dataset.')
                main_list.append(obj)

    if verbose:
        print('Checking the group {} for `Main` datasets.'.format(parent.name))
    parent.visititems(__check)

    return main_list


def read_h5py_dataset(dset):
    if not isinstance(dset, h5py.Dataset):
        raise TypeError('can only read single Dataset, use read_all_in_group or read_all function instead')

    if not check_if_main(dset):
        raise TypeError('can only read NSID datasets, not general one, try to import with from_array')

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

            dataset.set_dimension(dim, Dimension(np.array(dset.parent[name][()]),
                                                 dset.dims[dim].label,
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
    try:
        dataset.h5_dataset = dset.name
    except ValueError:
        pass
    return dataset


def find_dataset(h5_group, dset_name):
    """
    Uses visit() to find all datasets with the desired name
    Parameters
    ----------
    h5_group : :class:`h5py.Group`
        Group to search within for the Dataset
    dset_name : str
        Name of the dataset to search for
    Returns
    -------
    datasets : list
        List of [Name, object] pairs corresponding to datasets that match `ds_name`.
    """
    datasets = []

    for obj in hut.find_dataset(h5_group, dset_name):
        # TODO: Can we cast the HDF5 object as a sidpy.Dataset object?
        # We would need the main portion of the NSIDReader as a function
        datasets.append(obj)

    return datasets


def validate_main_dset(h5_main, must_be_h5):
    """
    Checks to make sure that the provided object is a NSID main dataset
    Errors in parameters will result in Exceptions
    Parameters
    ----------
    h5_main : h5py.Dataset or numpy.ndarray or Dask.array.core.array
        object that represents the NSID main data
    must_be_h5 : bool
        Set to True if the expecting an h5py.Dataset object.
        Set to False if expecting a numpy.ndarray or Dask.array.core.array
    Returns
    -------
    """
    # Check that h5_main is a dataset
    if must_be_h5:
        if not isinstance(h5_main, h5py.Dataset):
            raise TypeError('{} is not an HDF5 Dataset object.'.format(h5_main))
    else:
        if not isinstance(h5_main, (np.ndarray, da.core.Array)):
            raise TypeError('raw_data should either be a np.ndarray or a '
                            'da.core.Array')

    # Check dimensionality
    if len(h5_main.shape) != len(h5_main.dims):
        raise ValueError('Main data does not have full set of dimensional '
                         'scales. Provided object has shape: {} but only {} '
                         'dimensional scales'
                         ''.format(h5_main.shape, len(h5_main.dims)))


def check_if_main(h5_main, verbose=False):
    """
    Checks the input dataset to see if it has all the necessary
    features to be considered a Main dataset.  This means it is
    dataset has dimensions of correct size and has the following attributes:
    * quantity
    * units
    * main_data_name
    * data_type
    * modality
    * source
    In addition, the shapes of the ancillary matrices should match with that of
    h5_main
    Parameters
    ----------
    h5_main : HDF5 Dataset
        Dataset of interest
    verbose : Boolean (Optional. Default = False)
        Whether or not to print statements
    Returns
    -------
    success : Boolean
        True if all tests pass
    """
    try:
        validate_main_dset(h5_main, True)
    except Exception as exep:
        if verbose:
            print(exep)
        return False

    # h5_name = h5_main.name.split('/')[-1]
    h5_group = h5_main.parent

    # success = True

    # Check for Datasets

    attrs_names = ['dimension_type', 'name', 'nsid_version', 'quantity', 'units', ]
    attr_success = []
    # Check for all required attributes in dataset
    main_attrs_names = ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source']
    main_attr_success = np.all([att in h5_main.attrs for att in main_attrs_names])
    if verbose:
        print('All Attributes in dataset: ', main_attr_success)
    if not main_attr_success:
        if verbose:
            print('{} does not have the mandatory attributes'.format(h5_main.name))
        return False

    for attr_name in main_attrs_names:
        val = get_attr(h5_main, attr_name)
        if not isinstance(val, (str, unicode)):
            if verbose:
                print('Attribute {} of {} found to be {}. Expected a string'.format(attr_name, h5_main.name, val))
            return False

    length_success = []
    dset_success = []
    # Check for Validity of Dimensional Scales
    for i, dimension in enumerate(h5_main.dims):
        # check for all required attributes
        h5_dim_dset = h5_group[dimension.label]
        attr_success.append(np.all([att in h5_dim_dset.attrs for att in attrs_names]))
        dset_success.append(np.all([attr_success, isinstance(h5_dim_dset, h5py.Dataset)]))
        # dimensional scale has to be 1D
        if len(h5_dim_dset.shape) == 1:
            # and of the same length as the shape of the dataset
            length_success.append(h5_main.shape[i] == h5_dim_dset.shape[0])
        else:
            length_success.append(False)
    # We have the list now and can get error messages according to which dataset is bad or not.
    if np.all([np.all(attr_success), np.all(length_success), np.all(dset_success)]):
        if verbose:
            print('Dimensions: All Attributes: ', np.all(attr_success))
            print('Dimensions: All Correct Length: ', np.all(length_success))
            print('Dimensions: All h5 Datasets: ', np.all(dset_success))
    else:
        print('length of dimension scale {length_success.index(False)} is wrong')
        print('attributes in dimension scale {attr_success.index(False)} are wrong')
        print('dimension scale {dset_success.index(False)} is not a dataset')
        return False

    return main_attr_success


def link_as_main(h5_main, dim_dict):
    """
    Attaches datasets as h5 Dimensional Scales to  `h5_main`
    used in hdf_io.py write_nsid_dataset
    Parameters
    ----------
    h5_main : h5py.Dataset
        N-dimensional Dataset which will have the references added as h5 Dimensional Scales
    dim_dict: dictionary with dimensional order as key and items are datasets to be used as h5 Dimensional Scales

    Returns
    -------
    pyNSID.NSIDataset
        NSIDataset version of h5_main now that it is a NSID Main dataset
    """
    if not isinstance(h5_main, h5py.Dataset):
        raise TypeError('h5_main should be a h5py.Dataset object')

    h5_parent_group = h5_main.parent
    main_shape = h5_main.shape
    ######################
    # Validate Dimensions
    ######################
    # An N dimensional dataset should have N items in the dimension dictionary
    if len(dim_dict) != len(main_shape):
        raise ValueError('Incorrect number of dimensions: {} provided to support main data, of shape: {}'
                         .format(len(dim_dict), main_shape))
    if set(range(len(main_shape))) != set(dim_dict.keys()):
        raise KeyError('')
    
    dim_names = []
    for index, dim_exp_size in enumerate(main_shape):
        this_dim = dim_dict[index]
        if isinstance(this_dim, h5py.Dataset):
            error_message = validate_dimensions(this_dim, main_shape[index])
            if len(error_message) > 0:
                raise TypeError('Dimension {} has the following error_message:\n'.format(index), error_message)
            else:
                # if this_dim.name not in dim_names:
                if this_dim.name not in dim_names:  # names must be unique
                    dim_names.append(this_dim.name)
                else:
                    raise TypeError('All dimension names must be unique, found {} twice'.format(this_dim.name))
                if this_dim.file != h5_parent_group.file:
                    copy_dataset(this_dim, h5_parent_group, verbose=False)
        else: 
            raise TypeError('Items in dictionary must all  be h5py.Datasets !')

    ################
    # Attach Scales
    ################
    for i, this_dim_dset in dim_dict.items():
        this_dim_dset.make_scale(this_dim_dset.attrs['name'])
        h5_main.dims[int(i)].label = this_dim_dset.attrs['name']
        h5_main.dims[int(i)].attach_scale(this_dim_dset)
        
    return h5_main


def validate_dimensions(this_dim, dim_shape):
    """
    Checks if the provided object is an h5 dataset.
    A valid dataset to be uses as dimension must be 1D not a compound data type but 'simple'.
    Such a dataset must have  ancillary attributes 'name', quantity', 'units', and 'dimension_type',
    which have to be of  types str, str, str, and bool respectively and not empty
    If it is not valid of dataset, Exceptions are raised.

    Parameters
    ----------
    this_dim : h5 dataset
        with non empty attributes 'name', quantity', 'units', and 'dimension_type'
    dim_shape : required length of dataset

    Returns
    -------
    error_message: string, empty if ok.
    """

    if not isinstance(this_dim, h5py.Dataset):
        error_message = 'this Dimension must be a h5 Dataset'
        return error_message

    error_message = ''
    # Is it 1D?
    if len(this_dim.shape) != 1:
        error_message += ' High dimensional datasets are not allowed as dimensions;\n'
    # Does this dataset have a "simple" dtype - no compound data type allowed!
    # is the shape matching with the main dataset?
    if len(this_dim) != dim_shape:
        error_message += ' Dimension has wrong length;\n'
    # Does it contain some ancillary attributes like 'name', quantity', 'units', and 'dimension_type'
    necessary_attributes = ['name', 'quantity', 'units', 'dimension_type']
    for key in necessary_attributes:
        if key not in this_dim.attrs:
            error_message += 'Missing {} attribute in dimension;\n '.format(key)
        elif not isinstance(this_dim.attrs[key], str):
            error_message += '{} attribute in dimension should be string;\n '.format(key)

    return error_message
