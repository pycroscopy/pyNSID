# -*- coding: utf-8 -*-
"""
Utilities for reading and writing USID datasets that are highly model-dependent (with or without N-dimensional form)

Created on Fri May 22 16:29:25 2020

@author: []]
"""
from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import sys
import h5py
import numpy as np
from dask import array as da

from ..dtype_utils import contains_integers, validate_dtype, validate_single_string_arg, validate_string_args, \
    validate_list_of_strings, lazy_load_array

from .base import get_attr, write_simple_attrs, is_editable_h5, write_book_keeping_attrs
from .simple import link_as_main, check_if_main, write_ind_val_dsets, validate_dims_against_main, validate_anc_h5_dsets, copy_dataset
from pyNSID.io.write_utils import validate_dimensions
from ..write_utils import INDICES_DTYPE, make_indices_matrix

if sys.version_info.major == 3:
    unicode = str

def write_main_dataset(h5_parent_group, main_data, main_data_name,
                       quantity, units, dims,
                       data_type, modality, source,
                       main_dset_attrs=None, verbose=False,
                      **kwargs):
    """

    #TODO: Suhas to think about this a lot more

    Writes the provided data as a 'Main' dataset with all appropriate linking.
    By default, the instructions for generating the ancillary datasets should be specified using the pos_dims and
    spec_dims arguments as dictionary objects. Alternatively, if both the indices and values datasets are already
    available for either/or the positions / spectroscopic, they can be specified using the keyword arguments. In this
    case, fresh datasets will not be generated.

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
    dims : Dictionary containing Dimension or h5PyDataset objects, that map each dimension to the specified dimension. E.g.
    {'0': position_X, '1': position_Y, 2: spectra} where position_X, position_Y, spectra can be either Dimensions or h5py datasets.

        Sequence of Dimension objects that provides all necessary instructions for constructing the indices and values
        datasets
        Object specifying the instructions necessary for building the Position indices and values datasets

    verbose : bool, Optional, default=False
        If set to true - prints debugging logs

    kwargs will be passed onto the creation of the dataset. Please pass chunking, compression, dtype, and other
        arguments this way

    Returns
    -------
    h5_main : USIDataset
        Reference to the main dataset

    """
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

    def __ensure_anc_in_correct_file(h5_inds, h5_vals, prefix):
        if h5_inds.file != h5_vals.file:
            raise ValueError('Provided ' + prefix + ' datasets are present in different HDF5 files!')

        if h5_inds.file != h5_parent_group.file:
            # Need to copy over the anc datasets to the new group
            if verbose:
                print('Need to copy over ancillary datasets: {} and {} to '
                      'destination group: {} which is in a different HDF5 '
                      'file'.format(h5_inds, h5_vals, h5_parent_group))
            ret_vals = [copy_dataset(x, h5_parent_group, verbose=verbose) for x in [h5_inds, h5_vals]]
        else:
            ret_vals = [h5_inds, h5_vals]
        return tuple(ret_vals)

    if not isinstance(h5_parent_group, (h5py.Group, h5py.File)):
        raise TypeError('h5_parent_group should be a h5py.File or h5py.Group object')
    if not is_editable_h5(h5_parent_group):
        raise ValueError('The provided file is not editable')
    if verbose:
        print('h5 group and file OK')

    quantity, units, main_data_name = validate_string_args([quantity, units, main_data_name],
                                                           ['quantity', 'units', 'main_data_name'])
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
        if len(main_data) != 2:
            raise ValueError('main_data if specified as a shape should contain 2 numbers')
        if 'dtype' not in kwargs:
            raise ValueError('dtype must be included as a kwarg when creating an empty dataset')
        _ = validate_dtype(kwargs.get('dtype'))
        main_shape = main_data
        if verbose:
            print('Selected empty dataset creation. OK so far')
    elif isinstance(main_data, (np.ndarray, da.core.Array)):
        if main_data.ndim != 2:
            raise ValueError('main_data should be a 2D array')
        main_shape = main_data.shape
        if verbose:
            print('Provided numpy or Dask array for main_data OK so far')
    else:
        raise TypeError('main_data should either be a numpy array or a tuple / list with the shape of the data')

    if h5_pos_inds is not None and h5_pos_vals is not None:
        # The provided datasets override fresh building instructions.
        validate_anc_h5_dsets(h5_pos_inds, h5_pos_vals, main_shape, is_spectroscopic=False)
        if verbose:
            print('The shapes of the provided h5 position indices and values are OK')
        h5_pos_inds, h5_pos_vals = __ensure_anc_in_correct_file(h5_pos_inds, h5_pos_vals, 'Position')
    else:
        aux_pos_prefix = __check_anc_before_creation(aux_pos_prefix, dim_type='pos')
        pos_dims = validate_dimensions(pos_dims, dim_type='Position')
        validate_dims_against_main(main_shape, pos_dims, is_spectroscopic=False)
        if verbose:
            print('Passed all pre-tests for creating position datasets')
        h5_pos_inds, h5_pos_vals = write_ind_val_dsets(h5_parent_group, pos_dims, is_spectral=False, verbose=verbose,
                                                       slow_to_fast=slow_to_fast, base_name=aux_pos_prefix)
        if verbose:
            print('Created position datasets!')

    if h5_spec_inds is not None and h5_spec_vals is not None:
        # The provided datasets override fresh building instructions.
        validate_anc_h5_dsets(h5_spec_inds, h5_spec_vals, main_shape, is_spectroscopic=True)
        if verbose:
            print('The shapes of the provided h5 position indices and values '
                  'are OK')
        h5_spec_inds, h5_spec_vals = __ensure_anc_in_correct_file(h5_spec_inds, h5_spec_vals,
                                         'Spectroscopic')
    else:
        aux_spec_prefix = __check_anc_before_creation(aux_spec_prefix, dim_type='spec')
        spec_dims = validate_dimensions(spec_dims, dim_type='Spectroscopic')
        validate_dims_against_main(main_shape, spec_dims, is_spectroscopic=True)
        if verbose:
            print('Passed all pre-tests for creating spectroscopic datasets')
        h5_spec_inds, h5_spec_vals = write_ind_val_dsets(h5_parent_group, spec_dims, is_spectral=True, verbose=verbose,
                                                         slow_to_fast=slow_to_fast, base_name=aux_spec_prefix)
        if verbose:
            print('Created Spectroscopic datasets')

    if h5_parent_group.file.driver == 'mpio':
        if kwargs.pop('compression', None) is not None:
            warn('This HDF5 file has been opened wth the "mpio" communicator. '
                 'mpi4py does not allow creation of compressed datasets. Compression kwarg has been removed')

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

    write_simple_attrs(h5_main, {'quantity': quantity, 'units': units})
    if verbose:
        print('Wrote quantity and units attributes to main dataset')

    if isinstance(main_dset_attrs, dict):
        write_simple_attrs(h5_main, main_dset_attrs)
        if verbose:
            print('Wrote provided attributes to main dataset')

    write_book_keeping_attrs(h5_main)

    # make it main
    link_as_main(h5_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)
    if verbose:
        print('Successfully linked datasets - dataset should be main now')

    from ..usi_data import USIDataset
    return USIDataset(h5_main)

