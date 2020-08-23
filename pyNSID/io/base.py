# -*- coding: utf-8 -*-
"""
Simple yet handy HDF5 utilities, independent of the  USID model

Created on Fri May 22, 2020

@author: Gerd Duscher, Suhas Somnath
"""
from __future__ import division, print_function, absolute_import, unicode_literals
import socket
import sys
from platform import platform
from warnings import warn

import dask as da
import h5py
import numpy as np
import collections


sys.path.append('../../../sidpy/')

from sidpy.base.string_utils import validate_single_string_arg, validate_list_of_strings, \
    clean_string_att, get_time_stamp

from sidpy.hdf.hdf_utils import get_auxiliary_datasets, link_h5_obj_as_alias, get_attr, write_book_keeping_attrs, \
    write_simple_attrs, validate_h5_objs_in_same_h5_file, lazy_load_array

if sys.version_info.major == 3:
    unicode = str

__all__ = ['check_and_link_ancillary']




def check_and_link_ancillary(h5_dset, anc_names, h5_main=None, anc_refs=None):
    """
    This function will add references to auxilliary datasets as attributes
    of an input dataset.
    If the entries in anc_refs are valid references, they will be added
    as attributes with the name taken from the corresponding entry in
    anc_names.
    If an entry in anc_refs is not a valid reference, the function will
    attempt to get the attribute with the same name from the h5_main
    dataset
    Parameters
    ----------
    h5_dset : HDF5 Dataset
        dataset to which the attributes will be written
    anc_names : list of str
        the attribute names to be used
    h5_main : HDF5 Dataset, optional
        dataset from which attributes will be copied if `anc_refs` is None
    anc_refs : list of HDF5 Object References, optional
        references that correspond to the strings in `anc_names`
    Returns
    -------
    None
    Notes
    -----
    Either `h5_main` or `anc_refs` MUST be provided and `anc_refs` has the
    higher priority if both are present.
    """
    if not isinstance(h5_dset, h5py.Dataset):
        raise TypeError('h5_dset should be a h5py.Dataset object')

    if isinstance(anc_names, (str, unicode)):
        anc_names = [anc_names]
    if isinstance(anc_refs, (h5py.Dataset, h5py.Group, h5py.File,
                             h5py.Reference)):
        anc_refs = [anc_refs]

    if not isinstance(anc_names, (list, tuple)):
        raise TypeError('anc_names should be a list / tuple')
    if h5_main is not None:
        if not isinstance(h5_main, h5py.Dataset):
            raise TypeError('h5_main should be a h5py.Dataset object')
        validate_h5_objs_in_same_h5_file(h5_dset, h5_main)
    if anc_refs is not None:
        if not isinstance(anc_refs, (list, tuple)):
            raise TypeError('anc_refs should be a list / tuple')

    if anc_refs is None and h5_main is None:
        raise ValueError('No objected provided to link as ancillary')

    def __check_and_link_single(h5_obj_ref, target_ref_name):
        if isinstance(h5_obj_ref, h5py.Reference):
            # TODO: Same HDF5 file?
            h5_dset.attrs[target_ref_name] = h5_obj_ref
        elif isinstance(h5_obj_ref, (h5py.Dataset, h5py.Group, h5py.File)):
            validate_h5_objs_in_same_h5_file(h5_obj_ref, h5_dset)
            h5_dset.attrs[target_ref_name] = h5_obj_ref.ref
        elif h5_main is not None:
            h5_anc = get_auxiliary_datasets(h5_main, aux_dset_name=[target_ref_name])
            if len(h5_anc) == 1:
                link_h5_obj_as_alias(h5_dset, h5_anc[0], target_ref_name)
        else:
            warnstring = '{} is not a valid h5py Reference and will be skipped.'.format(repr(h5_obj_ref))
            warn(warnstring)

    if bool(np.iterable(anc_refs) and not isinstance(anc_refs, h5py.Dataset)):
        """
        anc_refs can be iterated over
        """
        for ref_name, h5_ref in zip(anc_names, anc_refs):
            __check_and_link_single(h5_ref, ref_name)
    elif anc_refs is not None:
        """
        anc_refs is just a single value
        """
        __check_and_link_single(anc_refs, anc_names)
    elif isinstance(anc_names, str) or isinstance(anc_names, unicode):
        """
        Single name provided
        """
        __check_and_link_single(None, anc_names)
    else:
        """
        Iterable of names provided
        """
        for name in anc_names:
            __check_and_link_single(None, name)

    h5_dset.file.flush()
