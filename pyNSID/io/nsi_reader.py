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
# from sidpy.base.num_utils import contains_integers
# from sidpy.hdf.dtype_utils import validate_dtype
# from sidpy.base.string_utils import validate_single_string_arg, validate_string_args
from sidpy.hdf.hdf_utils import get_attr  # write_simple_attrs, is_editable_h5
from sidpy.sid import Dimension, Dataset

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
            try:
                dim_dict = dict(dset.parent[dset.dims[dim].label].attrs)

                dataset.set_dimension(dim, Dimension(dset.dims[dim].label,
                                                     np.array(dset.parent[dset.dims[dim].label][()]),
                                                     dim_dict['quantity'], dim_dict['units'],
                                                     dim_dict['dimension_type']))
            except ValueError:
                print('dimension {} not NSID type using generic'.format(dim))

        dataset.attrs = dict(dset.attrs)

        dataset.original_metadata = {}
        if 'original_metadata' in dset.parent:
            dataset.original_metadata = dict(dset.parent['original_metadata'].attrs)

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
