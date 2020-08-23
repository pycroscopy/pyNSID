# -*- coding: utf-8 -*-
"""
Utilities for reading and writingNUSID datasets that are highly model-dependent
Depends heavily on sidpy

Created on Fri May 22 16:29:25 2020

@author: Gerd Duscher, Suhas Somas
TODO: update version

"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np
from dask import array as da

sys.path.append('../../../sidpy/')
import sidpy as sid
from sidpy.base.num_utils import contains_integers
from sidpy.hdf.dtype_utils import validate_dtype
from sidpy.base.string_utils import validate_single_string_arg, validate_string_args
from sidpy.hdf.hdf_utils import write_simple_attrs, is_editable_h5  # , get_attr, write_book_keeping_attrs

from .simple import link_as_main  # , check_if_main, validate_dims_against_main, validate_anc_h5_dsets, copy_dataset
from .write_utils import validate_main_dimensions  # validate_dimensions ,

if sys.version_info.major == 3:
    unicode = str

def write_main_dataset(h5_parent_group, main_data, main_data_name,
                       quantity, units, data_type, modality, source,
                       dim_dict, main_dset_attrs=None, verbose=False, **kwargs):

    """

    #TODO: Suhas to think about this a lot more

    Writes the provided data as a 'Main' dataset with all appropriate linking.
    By default, the instructions for generating dimension should be provided as a dictionary containing
    pyNSID-Dimensions or 1-Dim datasets
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
    modality : `string : Experimental / simulation modality - scientific meaning of data.
                Example - photograph, TEM micrograph, SPM Force-Distance spectroscopy.
    source : `string : Source for dataset like the kind of instrument.
    dim_dict : Dictionary containing Dimension or h5PyDataset objects, that map each dimension to the specified
               dimension. E.g.
                {'0': position_X, '1': position_Y, 2: spectra} where position_X, position_Y,
                spectra can be either Dimensions or h5py datasets.

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
    quantity, units, main_data_name, data_type, modality, source \
        = validate_string_args([quantity, units, main_data_name, data_type, modality, source],
                               ['quantity', 'units', 'main_data_name', 'data_type', 'modality', 'source'])

    if verbose:
        print('quantity, units, main_data_name all OK')

    quantity = quantity.strip()
    units = units.strip()
    main_data_name = main_data_name.strip()
    if '-' in main_data_name:
        warn('main_data_name should not contain the "-" character. Reformatted name from:{} to '
             '{}'.format(main_data_name, main_data_name.replace('-', '_')))
    main_data_name = main_data_name.replace('-', '_')
    
    if isinstance(main_data, (list, tuple)):
        if not contains_integers(main_data, min_val=1):
            raise ValueError('main_data if specified as a shape should be a list / tuple of integers >= 1')
        if len(main_data) < 1:
            raise ValueError('main_data if specified as a shape should contain at least 1 number '
                             'for the singular dimension')
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
        raise ValueError('Incorrect number of dimensions: {} provided to support main data, of shape: {}'
                         .format(len(dim_dict), main_shape))
    if set(range(len(main_shape))) != set(dim_dict.keys()):
        raise KeyError('')
    
    if False in validate_main_dimensions(main_shape, dim_dict, h5_parent_group):
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
        print(h5_parent_group.name)
        print('Oops, dataset exits')
        del h5_parent_group[main_data_name]
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
    this_dim_dset = None
    dimensional_dict = {}
    for i, this_dim in dim_dict.items():
        if isinstance(this_dim, h5py.Dataset):
            this_dim_dset = this_dim
            if 'nsid_version' not in this_dim_dset.attrs:
                this_dim_dset.attrs['nsid_version'] = '0.0.1'
            # this_dim_dset[i] = this_dim
        elif isinstance(this_dim, sid.sid.Dimension):
            this_dim_dset = h5_parent_group.create_dataset(this_dim.name, data=this_dim.values)
            attrs_to_write = {'name': this_dim.name, 'units': this_dim.units, 'quantity': this_dim.quantity,
                              'dimension_type': this_dim.dimension_type, 'nsid_version': '0.0.1'}
            write_simple_attrs(this_dim_dset, attrs_to_write)

        else:
            print(i, ' not a good dimension')
            pass
        dimensional_dict[i] = this_dim_dset

    attrs_to_write = {'quantity': quantity, 'units': units, 'nsid_version': '0.0.1', 'main_data_name': main_data_name,
                      'data_type': data_type, 'modality': modality, 'source': source}

    write_simple_attrs(h5_main, attrs_to_write)
    
    if verbose:
        print('Wrote dimensions and attributes to main dataset')

    if isinstance(main_dset_attrs, dict):
        write_simple_attrs(h5_main, main_dset_attrs)
        if verbose:
            print('Wrote provided attributes to main dataset')

    # ToDo: check if we need  write_book_keeping_attrs(h5_main)
    nsid_data_main = link_as_main(h5_main, dimensional_dict)
    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    return nsid_data_main  # NSIDataset(h5_main)
