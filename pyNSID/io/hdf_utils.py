# -*- coding: utf-8 -*-
"""
Lower-level and simpler NSID-specific HDF5 utilities that facilitate
higher-level data operations

Created on Tue Aug  3 21:14:25 2020

@author: Gerd Duscher, and Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import sys
from warnings import warn
import h5py
import numpy as np
import datetime

from sidpy.hdf.hdf_utils import get_attr, copy_dataset, write_simple_attrs, \
    write_book_keeping_attrs, h5_group_to_dict
from sidpy.hdf import hdf_utils as hut
from sidpy import Dimension, Dataset

from pyNSID.__version__ import version as pynsid_version

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
                # TODO: Upconvert to sidpy.Dataset object with new function
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

    for key in dset.parent:
        if isinstance(dset.parent[key], h5py.Group):
            if key[0] != '_':
                setattr(dataset, key, h5_group_to_dict(dset.parent[key]))
                
    dataset.h5_dataset = dset
    dataset.h5_filename = dset.file.filename
    try:
        dataset.h5_dataset_name = dset.name
    except ValueError:
        dataset.h5_dataset_name = ''
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
        if check_if_main(obj):
            # TODO: Upconvert to sidpy.Dataset object with new function
            datasets.append(obj)
        else:
            datasets.append(obj)

    return datasets


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
    if not isinstance(h5_main, h5py.Dataset):
        if verbose:
            print('{} is not an HDF5 Dataset object.'.format(h5_main))
        return False

    number_of_dims = 0
    for dim in h5_main.dims:
        if np.array(dim.values()).size > 0:
            number_of_dims += 1

    if len(h5_main.shape) != number_of_dims:
        if verbose:
            print('Main data does not have full set of dimension scales. '
                  'Provided object has shape: {} but only {} dimensional '
                  'scales'.format(h5_main.shape, len(h5_main.dims)))
        return False

    # h5_name = h5_main.name.split('/')[-1]
    h5_group = h5_main.parent

    # success = True

    # Check for Datasets

    attrs_names = ['dimension_type', 'name', 'quantity', 'units']

    # Check for all required attributes in dataset
    main_attrs_names = ['quantity', 'units', 'main_data_name', 'pyNSID_version', 'data_type', 'modality', 'source']
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
    attr_success = []
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
        if False in length_success:
            print('length of dimension scale {} is wrong'.format(length_success.index(False)))
        if False in attr_success:
            print('attributes in dimension scale {} are wrong'.format(attr_success.index(False)))
        if False in dset_success:
            print('dimension scale {} is not a dataset'.format(dset_success.index(False)))
        return False

    return main_attr_success


def link_as_main(h5_main, dim_dict):
    """
    Attaches datasets as h5 Dimensional Scales to  `h5_main`

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
    if not isinstance(dim_dict, dict):
        raise TypeError("""dim_dict must be a dictionary"""
                        """ (keys: dimensional order, values: h5py.Datasets)""")

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
            error_message = validate_h5_dimension(this_dim, main_shape[index])
            if len(error_message) > 0:
                raise TypeError('Dimension {} has the following error_message:\n'.format(index), error_message)
            else:
                # if h5_dim.name not in dim_names:
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


def validate_h5_dimension(h5_dim, dim_length):
    """
    Validates a dimension already present in an HDF5 file.

    Parameters
    ----------
    h5_dim : h5py.Dataset
        HDF5 dataset which represents a scientific dimension.
        The dimension should have non empty attributes 'name', quantity',
        'units', and 'dimension_type'
    dim_length : int
        Expected length of dataset

    Returns
    -------
    error_message: string, empty if ok.

    Notes
    -----
    A valid dataset to be used as dimension must be 1D not a compound data type but 'simple'.
    Such a dataset must have  ancillary attributes 'name', quantity', 'units', and 'dimension_type',
    which have to be of  types str, str, str, and bool respectively and not empty
    If it is not valid of dataset, Exceptions are raised.
    """
    warn('validate_h5_dimension may be removed in a future version',
         FutureWarning)
    # TODO: Raise exceptions instead of returning strings that need to be parsed

    if not isinstance(h5_dim, h5py.Dataset):
        error_message = 'this Dimension must be a h5 Dataset'
        return error_message

    error_message = ''
    # Is it 1D?
    if len(h5_dim.shape) != 1:
        error_message += ' High dimensional datasets are not allowed as dimensions;\n'
    # Does this dataset have a "simple" dtype - no compound data type allowed!
    # is the shape matching with the main dataset?
    if len(h5_dim) != dim_length:
        error_message += ' Dimension has wrong length;\n'
    # TODO: Relax requirements for these attributes. Check against sidpy.Dataset
    # Does it contain some ancillary attributes like 'name', quantity', 'units', and 'dimension_type'
    necessary_attributes = ['name', 'quantity', 'units', 'dimension_type']
    for key in necessary_attributes:
        if key not in h5_dim.attrs:
            error_message += 'Missing {} attribute in dimension;\n '.format(key)
        elif not isinstance(h5_dim.attrs[key], str):
            error_message += '{} attribute in dimension should be string;\n '.format(key)

    return error_message


def validate_main_and_dims(main_shape, dim_dict, h5_parent_group):
    """
    Validates the shape of the main dataset against the dictionary of
    dimensions and the parent HDF5 group into which the data would be written.
    This method was written as a low-level validation check before
    sidpy.Dataset was conceived. It may still be relevant if one intends to
    reuse the Dimension HDF5 datasets already in the file

    Parameters
    ----------
    main_shape : list or tuple
        Shape of the Main dataset
    dim_dict : dict
        Dictionary of dimensions that could either be sidpy.Dimension or
        h5py.Dataset objects
    h5_parent_group : h5py.Group
        HDF5 group to write into

    Returns
    -------
    bool
        Whether or not the dimensions match the main data shape
    """
    warn('validate_main_and_dims may not exist in a future version',
         FutureWarning)

    # Each item could either be a Dimension object or a HDF5 dataset
    # Collect the file within which these ancillary HDF5 objects are present if they are provided
    which_h5_file = {}
    # Also collect the names of the dimensions. We want them to be unique
    dim_names = []

    dimensions_correct = []
    for index, dim_exp_size in enumerate(main_shape):
        this_dim = dim_dict[index]
        if isinstance(this_dim, h5py.Dataset):
            error_message = validate_h5_dimension(this_dim, main_shape[index])

            # All these checks should live in a helper function for cleanliness

            if len(error_message) > 0:
                print('Dimension {} has the following error_message:\n'.format(index), error_message)

            else:
                if this_dim.name not in dim_names:  # names must be unique
                    dim_names.append(this_dim.name)
                else:
                    raise TypeError('All dimension names must be unique, found'
                                    ' {} twice'.format(this_dim.name))

                # are all datasets in the same file?
                if this_dim.file != h5_parent_group.file:
                    copy_dataset(this_dim, h5_parent_group, verbose=True)

        elif isinstance(this_dim, Dimension):
            # is the shape matching with the main dataset?
            dimensions_correct.append(len(this_dim.values) == dim_exp_size)
            # Is there a HDF5 dataset with the same name already in the provided group
            # where this dataset will be created?
            if this_dim.name in h5_parent_group:
                # check if this object with the same name is a dataset and if it satisfies the above tests
                if isinstance(h5_parent_group[this_dim.name], h5py.Dataset):
                    print('needs more checking')
                    dimensions_correct[-1] = False
                else:
                    dimensions_correct[-1] = True
            # Otherwise, just append the dimension name for the uniqueness test
            elif this_dim.name not in dim_names:
                dim_names.append(this_dim.name)
            else:
                dimensions_correct[-1] = False
        else:
            raise TypeError('Values of dim_dict should either be h5py.Dataset '
                            'objects or Dimension. Object at index: {} was of '
                            'type: {}'.format(index, index))

        for dim in which_h5_file:
            if which_h5_file[dim] != h5_parent_group.file.filename:
                print('need to copy dimension', dim)
        for i, dim_name in enumerate(dim_names[:-1]):
            if dim_name in dim_names[i + 1:]:
                print(dim_name, ' is not unique')

    return all(dimensions_correct)


def write_pynsid_book_keeping_attrs(h5_object):
    """
    Writes book-keeping information to the HDF5 object

    Parameters
    ----------
    h5_object

    Returns
    -------

    """
    write_book_keeping_attrs(h5_object)
    write_simple_attrs(h5_object, {'pyNSID_version': pynsid_version})


def make_nexus_compatible(h5_dataset):
    """
    Makes a pyNSID file compatible with the NeXus file format
    by adding the approbriate attributes and by writing one group.

    Parameters
    ----------
    h5_dataset: h5py.Dataset
        h5py dataset with main data

    Returns
    -------

    """
    if not isinstance(h5_dataset, h5py.Dataset):
        raise ValueError('We need a h5py dataset for compatibility with NeXus file format')

    time_stamp = "T".join(str(datetime.datetime.now()).split())
    h5_file = h5_dataset.file
    h5_group = h5_dataset.parent.parent
    h5_file.attrs[u'default'] = h5_dataset.parent.parent.name

    # give the HDF5 root some more attributes

    h5_file.attrs[u'file_name'] = h5_file.filename
    h5_file.attrs[u'file_time'] = time_stamp
    h5_file.attrs[u'instrument'] = u'None'
    h5_file.attrs[u'creator'] = u'pyNSID'
    h5_file.attrs[u'NeXus_version'] = u'4.3.0'
    h5_file.attrs[u'HDF5_Version'] = h5py.version.hdf5_version
    h5_file.attrs[u'h5py_version'] = h5py.version.version

    h5_file.attrs[u'default'] = h5_group.name

    h5_group.attrs[u'NX_class'] = u'NXentry'
    h5_group.attrs[u'default'] = h5_dataset.name.split('/')[-1]

    if 'title' in h5_group:
        del h5_group['title']
    h5_group.create_dataset(u'title', data=h5_dataset.name.split('/')[-1])

    nxdata = h5_dataset.parent

    nxdata.attrs[u'NX_class'] = u'NXdata'
    nxdata.attrs[u'signal'] = h5_dataset.name.split('/')[-1]
    axes = []
    for dimension_label in h5_dataset.attrs['DIMENSION_LABELS']:
        axes.append(dimension_label.decode('utf8'))

    nxdata.attrs[u'axes'] = axes
    for i, axis in enumerate(axes):
        nxdata.attrs[axes[i] + '_indices'] = [i, ]
