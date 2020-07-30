# -*- coding: utf-8 -*-
"""
Utilities for reading and writing USID datasets that are highly model-dependent (with or without N-dimensional form)

Created on Fri May 22 16:29:25 2020

@author: []]
ToDo update version

"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np
from dask import array as da
from ..write_utils import Dimension, validate_dimensions  # new
from ..dtype_utils import contains_integers, validate_dtype, validate_single_string_arg, validate_string_args, \
    validate_list_of_strings, lazy_load_array
from .base import get_attr, write_simple_attrs, is_editable_h5, write_book_keeping_attrs
from .simple import link_as_main, check_if_main, validate_dims_against_main, validate_anc_h5_dsets, copy_dataset
from pyNSID.io.write_utils import validate_dimensions
#from ..write_utils import INDICES_DTYPE, make_indices_matrix
from ... import Dimension

if sys.version_info.major == 3:
    unicode = str

def __check_anc_before_creation(aux_prefix, dim_type='pos'):
    aux_prefix = validate_single_string_arg(aux_prefix, 'aux_' + dim_type + '_prefix')
    if not aux_prefix.endswith('_'):
        aux_prefix += '_'
    if '-' in aux_prefix:
        warn('aux_' + dim_type + ' should not contain the "-" character. Reformatted name from:{} to '
                                    '{}'.format(aux_prefix, aux_prefix.replace('-', '_')))
    aux_prefix = aux_prefix.replace('-', '_')
    for dset_name in [aux_prefix + 'Indices', aux_prefix + 'Values']:
        if dset_name in h5_parent_group.keys():
            # TODO: What if the contained data was correct?
            raise KeyError('Dataset named: ' + dset_name + ' already exists in group: '
                                                            '{}. Consider passing these datasets using kwargs (if they are correct) instead of providing the pos_dims and spec_dims arguments'.format(h5_parent_group.name))
    return aux_prefix

""" New version much shorter and in validate_main_dimensions in file "simple.py"
def __ensure_anc_in_correct_file(h5_inds, h5_parent_group, prefix):
    h5_parent_group = h5_vals.parent
    if h5_inds.file != h5_parent_group.file:
        # Need to copy over the anc datasets to the new group
        if verbose:
            print('Need to copy over ancillary datasets: {} and {} to '
                    'destination group: {} which is in a different HDF5 '
                    'file'.format(h5_inds, h5_parent_group))
        ret_vals = [copy_dataset(x, h5_parent_group, verbose=verbose) for x in [h5_inds, h5_vals]]
    else:
        ret_vals = [h5_inds, h5_vals]
    return tuple(ret_vals)
"""



def write_main_dataset(h5_parent_group, main_data, main_data_name, 
                        quantity, units, data_type, modality, source, 
                        dim_dict, main_dset_attrs=None, verbose=False,
                        slow_to_fast=False, **kwargs):

    """

    #TODO: Suhas to think about this a lot more

    Writes the provided data as a 'Main' dataset with all appropriate linking.
    By default, the instructions for generating dimension should be provided as a dictionary containing pyNSID-Dimensions or 1-Dim datasets 
    The dimension-datasets can be shared with other main datasets; in this case, fresh datasets will not be generated.

    Parameters
    ----------
    h5_parent_group : :class:`h5py.Group`
        Parent group under which the datasets will be created
    main_data : numpy.ndarray, dask.array.core.Array, list or tuple
        2D matrix formatted as [position, spectral] or a list / tuple with the shape for an empty dataset.
        If creating an empty dataset - the dtype must be specified via a kwarg.
    main_data_name : String / Unicode
        Name to give to the main dataset. This cannot contain the '-' character.
    quantity : String / Unicode
        Name of the physical quantity stored in the dataset. Example - 'Current'
    units : String / Unicode
        Name of units for the quantity stored in the dataset. Example - 'A' for amperes
    data_type : `string : What kind of data this is. Example - image, image stack, video, hyperspectral image, etc.
    modality : `string : Experimental / simulation modality - scientific meaning of data. Example - photograph, TEM micrograph, SPM Force-Distance spectroscopy.
    source : `string : Source for dataset like the kind of instrument.
    dim_dict : Dictionary containing Dimension or h5PyDataset objects, that map each dimension to the specified dimension. E.g.
        {'0': position_X, '1': position_Y, 2: spectra} where position_X, position_Y, spectra can be either Dimensions or h5py datasets.

        Sequence of Dimension objects that provides all necessary instructions for constructing the indices and values
        datasets
        Object specifying the instructions necessary for building the Position indices and values datasets
    main_dset_attrs: dictionary, Optional, default = None
        flat dictionary of data to be added to the dataset, 
    verbose : bool, Optional, default=False
        If set to true - prints debugging logs
    kwargs will be passed onto the creation of the dataset. Please pass chunking, compression, dtype, and other
        arguments this way

    Returns
    -------
    h5_main : NSIDataset
        Reference to the main dataset

    """
    
    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group object')
    if not is_editable_h5(h5_parent_group):
        raise ValueError('The provided file is not editable')
    if verbose:
        print('h5 group and file OK')

    #####################
    # Validate Main Data
    #####################
    quantity, units, main_data_name, data_type, modality, source = validate_string_args([quantity, units, main_data_name, data_type, modality, source],
                                                           ['quantity', 'units', 'main_data_name','data_type', 'modality', 'source'])

    if verbose:
            print('quantity, units, main_data_name all OK')

    quantity = quantity.strip()
    units = units.strip()
    main_data_name = main_data_name.strip()
    if '-' in main_data_name:
        warn('main_data_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(main_data_name, main_data_name.replace('-', '_')))
    main_data_name = main_data_name.replace('-', '_')
    
    if  isinstance(main_data, (list, tuple)):
        if not contains_integers(main_data, min_val=1):
            raise ValueError('main_data if specified as a shape should be a list / tuple of integers >= 1')
        if len(main_data) < 1:
            raise ValueError('main_data if specified as a shape should contain at least 1 number for the singular dimension')
        if 'dtype' not in kwargs:
            raise ValueError('dtype must be included as a kwarg when creating an empty dataset')
        _ = validate_dtype(kwargs.get('dtype'))
        main_shape = main_data
        if verbose:
            print('Selected empty dataset creation. OK so far')
    elif isinstance(main_data, (np.ndarray, da.core.Array)):
        main_shape = main_data.shape
        if verbose:
            print('Provided numpy or Dask array for main_data OK so far')
    else:
        raise TypeError('main_data should either be a numpy array or a tuple / list with the shape of the data')
      
    ######################
    # Validate Dimensions
    ######################
    # An N dimensional dataset should have N items in the dimension dictionary
    if len(dim_dict) != len(main_shape):
        raise ValueError('Incorrect number of dimensions: {} provided to support main data, of shape: {}'.format(len(dim_dict), main_shape))
    if set(range(len(main_shape))) != set(dim_dict.keys()):
        raise KeyError('')
    
    if False in validate_main_dimensions(main_shape,dim_dict, h5_parent_group):
        print('Dimensions incorrect')
        return
    if verbose:
        print('Dimensions are correct!')

    #####################
    # Write Main Dataset
    ####################
    if h5_parent_group.file.driver == 'mpio':
        if kwargs.pop('compression', None) is not None:
            warn('This HDF5 file has been opened wth the "mpio" communicator. '
                 'mpi4py does not allow creation of compressed datasets. Compression kwarg has been removed')

    if main_data_name in h5_parent_group:
        print('Oops, dataset exits')
        #del h5_parent_group[main_data_name]
        return
    
    if isinstance(main_data, np.ndarray):
        # Case 1 - simple small dataset
        h5_main = h5_parent_group.create_dataset(main_data_name, data=main_data, **kwargs)
        if verbose:
            print('Created main dataset with provided data')
    elif isinstance(main_data, da.core.Array):
        # Case 2 - Dask dataset
        # step 0 - get rid of any automated dtype specification:
        _ = kwargs.pop('dtype', None)
        # step 1 - create the empty dataset:
        h5_main = h5_parent_group.create_dataset(main_data_name, shape=main_data.shape, dtype=main_data.dtype,
                                                 **kwargs)
        if verbose:
            print('Created empty dataset: {} for writing Dask dataset: {}'.format(h5_main, main_data))
            print('Dask array will be written to HDF5 dataset: "{}" in file: "{}"'.format(h5_main.name,
                                                                                          h5_main.file.filename))
        # Step 2 - now ask Dask to dump data to disk
        da.to_hdf5(h5_main.file.filename, {h5_main.name: main_data})
        # main_data.to_hdf5(h5_main.file.filename, h5_main.name)  # Does not work with python 2 for some reason
    else:
        # Case 3 - large empty dataset
        h5_main = h5_parent_group.create_dataset(main_data_name, main_data, **kwargs)
        if verbose:
            print('Created empty dataset for Main')

     #################
    # Add Dimensions
    #################
    dimensional_dict = {}
    for i, this_dim in dim_dict.items():
        if isinstance(this_dim, h5py.Dataset):
            this_dim_dset = this_dim
            if 'nsid_version' not in this_dim_dset.attrs:
                this_dim_dset.attrs['nsid_version'] = '0.0.1'
            #this_dim_dset[i] = this_dim
        elif isinstance(this_dim, Dimension):
            this_dim_dset = h5_parent_group.create_dataset(this_dim.name,data=this_dim.values)
            attrs_to_write={'name':  this_dim.name, 'units': this_dim.units, 'quantity':  this_dim.quantity, 'dimension_type': this_dim.dimension_type, 'nsid_version' : '0.0.1'}
            write_simple_attrs(this_dim_dset, attrs_to_write)

        else:
            print(i,' not a good dimension')
            pass
        dimensional_dict[i] = this_dim_dset
    
    
        
    attrs_to_write={'quantity': quantity, 'units': units, 'nsid_version' : '0.0.1'}
    attrs_to_write['main_data_name'] =  main_data_name
    attrs_to_write['data_type'] =  data_type
    attrs_to_write['modality'] =  modality
    attrs_to_write['source'] =  source
    
    write_simple_attrs(h5_main, attrs_to_write)

    
    if verbose:
        print('Wrote dimensions and attributes to main dataset')

    if isinstance(main_dset_attrs, dict):
        write_simple_attrs(h5_main, main_dset_attrs)
        if verbose:
            print('Wrote provided attributes to main dataset')

    #ToDo: check if we need  write_book_keeping_attrs(h5_main)
    NSID_data_main = link_as_main(h5_main, dimensional_dict)
    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    
    return NSID_data_main#NSIDataset(h5_main)


def validate_main_dimensions(main_shape, dim_dict, h5_parent_group ):
    # Each item could either be a Dimension object or a HDF5 dataset
    # Collect the file within which these ancillary HDF5 objectsa are present if they are provided
    which_h5_file = {}
    # Also collect the names of the dimensions. We want them to be unique
    dim_names = []

    dimensions_correct = []
    for index, dim_exp_size in enumerate(main_shape):
        this_dim = dim_dict[index]
        if isinstance(this_dim, h5py.Dataset):
            #print(f'{index} is a dataset')
            error_message = validate_dimensions(this_dim, main_shape[index])

            # All these checks should live in a helper function for cleaniness

            if len(error_message)>0:
                print('Dimension {} has the following error_message:\n'.format(index), error_message)

            else:
                if this_dim.name not in dim_names: ## names must be unique
                    dim_names.append(this_dim.name)
                else:
                    raise TypeError('All dimension names must be unique, found {} twice'.format(this_dim.name))

                # are all datasets in the same file?
                if this_dim.file != h5_parent_group.file:
                    this_dim = copy_dataset(this_dim, h5_parent_group, verbose=False)

        elif isinstance(this_dim, Dimension):
            #print('Dimension')
            #print(len(this_dim.values))
            # is the shape matching with the main dataset?
            dimensions_correct.append(len(this_dim.values) == dim_exp_size)
            # Is there a HDF5 dataset with the same name already in the provided group where this dataset will be created?
            if  this_dim.name in h5_parent_group:
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
            raise TypeError('Values of dim_dict should either be h5py.Dataset objects or Dimension. '
                            'Object at index: {} was of type: {}'.format(index, type(index)))

        for dim in which_h5_file:
            if which_h5_file[dim] != h5_parent_group.file.filename:
                print('need to copy dimension', dim)
        for i, dim_name in enumerate(dim_names[:-1]):
            if dim_name in  dim_names[i+1:]:
                print(dim_name, ' is not unique')

    return dimensions_correct