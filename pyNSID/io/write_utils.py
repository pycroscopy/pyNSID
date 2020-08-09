# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""


from __future__ import division, print_function, unicode_literals, absolute_import
import sys
import h5py  # new

__all__ = ['validate_dimensions']

if sys.version_info.major == 3:
    unicode = str


def validate_dimensions(this_dim, dim_shape):
    """
    Checks if the provided object is an  h5 dataset. 
    A valid dataset to be uses as dimension must be 1D not a compound data type but 'simple'.
    Such a dataset must have  ancillary attributes 'name', quantity', 'units', and 'dimension_type',
    which have to be of  types str, str, str, and bool respectively and not empty
    If it is not valid of dataset, Exceptions are raised.

    Parameters
    ----------
    dimensions : h5 dataset
        with non empty attributes 'name', quantity', 'units', and 'dimension_type' 
    dim_shape : required length of dataset 

    Returns
    -------
    error_message: string, empty if ok. 
    """

    if not isinstance(this_dim, h5py.Dataset):
        error_message = 'this Dimension must be a h5 Dataset'
        return  error_message 
    
    error_message = ''
    # Is it 1D?
    if len(this_dim.shape)!=1:
        error_message += ' High dimensional datasets are not allowed as dimensions;\n'
    # Does this dataset have a "simple" dtype - no compound data type allowed!
    # is the shape matching with the main dataset?
    if len(this_dim) != dim_shape:
        error_message += ' Dimension has wrong length;\n'
    # Does it contain some ancillary attributes like 'name', quantity', 'units', and 'dimension_type' 
    necessary_attributes =  ['name', 'quantity', 'units', 'dimension_type']
    for key in necessary_attributes:
        if key not in this_dim.attrs:
            error_message += 'Missing {} attribute in dimension;\n '.format(key)
        # and are these of types str, str, str, and bool respectively and not empty?
        #elif key == 'dimension_type':
        #    if this_dim.attrs['dimension_type'] not in [True, False]: ## isinstance is here not working 
        #        error_message += f'{key} attribute in dimension should be boolean;\n ' 
        elif not isinstance(this_dim.attrs[key], str):
            error_message += '{} attribute in dimension should be string;\n '.format(key)
    
    return error_message
