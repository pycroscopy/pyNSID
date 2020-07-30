# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""


from __future__ import division, print_function, unicode_literals, absolute_import
import sys
from itertools import groupby
import numpy as np
import h5py #new
from .dtype_utils import validate_single_string_arg
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['clean_string_att', 'get_slope', 'Dimension', 'validate_dimensions',
           'to_ranges']

if sys.version_info.major == 3:
    unicode = str


class Dimension(object):
    """
    ..autoclass::Dimension
    """

    def __init__(self, name, values, quantity='generic', units='generic',  dimension_type='generic'):
        """
        Simple object that describes a dimension in a dataset by its name, units, and values
        Parameters
        ----------
        name : str or unicode
            Name of the dimension. For example 'X'
        quantity : str or unicode
            Quantity for this dimension. For example: 'Length'
        units : str or unicode
            Units for this dimension. For example: 'um'
        values : array-like or int
            Values over which this dimension was varied. A linearly increasing set of values will be generated if an
            integer is provided instead of an array.
        dimension_type : str or unicode for example: 'spectral' or 'spatial', 'time', 'frame', 'reciprocal'
            This will determine how the data are visualized. 'spatial' are image dimensions.
            'spectral' indicate spectroscopy data dimensions.
        """

        self.name = name
        self.values= values

        self.quantity = quantity
        self.units = units
        self.dimension_type =dimension_type


    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = validate_single_string_arg(value, 'name')

    @property
    def quantity(self):
        return self._quantity
    @quantity.setter
    def quantity(self, value):
        self._quantity = validate_single_string_arg(value, 'quantity')

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = validate_single_string_arg(value, 'units')


    @property
    def dimension_type(self):
        return self._dimension_type
    @dimension_type.setter
    def dimension_type(self, value):
        self._dimension_type  = validate_single_string_arg(value, 'dimension_type')

    @property
    def values(self):
        return self._values
    @values.setter
    def values(self, values):
        if isinstance(values, int):
            if values < 1:
                raise ValueError('values should at least be specified as a positive integer')
            values = np.arange(values)
        if not isinstance(values, (np.ndarray, list, tuple)):
            raise TypeError('values should be array-like')
        values = np.array(values)
        if values.ndim > 1:
            raise ValueError('Values for dimension: {} are not 1-dimensional'.format(self.name))

        self._values = values

    def __repr__(self):
        return '{} - {} ({}): {}'.format(self.name, self.quantity, self.units, self.values)

    def __eq__(self, other):
        if isinstance(other, Dimension):
            if self.name != other.name:
                return False
            if self.units != other.units:
                return False
            if self.quantity != other.quantity:
                return False
            if len(self.values) != len(other.values):
                return False
            if not np.allclose(self.values, other.values):
                return False
        return True


def validate_dimensions(this_dim,dim_shape):
    """
    Checks if the provided object is an  h5 dataset. 
    A valid dataset to be uses as dimension must be 1D not a comopound data type but 'simple'.
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


def clean_string_att(att_val):
    """
    Replaces any unicode objects within lists with their string counterparts to ensure compatibility with python 3.
    If the attribute is indeed a list of unicodes, the changes will be made in-place

    Parameters
    ----------
    att_val : object
        Attribute object

    Returns
    -------
    att_val : object
        Attribute object
    """
    try:
        if isinstance(att_val, Iterable):
            if type(att_val) in [unicode, str]:
                return att_val
            elif np.any([type(x) in [str, unicode, bytes, np.str_] for x in att_val]):
                return np.array(att_val, dtype='S')
        if type(att_val) == np.str_:
            return str(att_val)
        return att_val
    except TypeError:
        raise TypeError('Failed to clean: {}'.format(att_val))


def get_slope(values, tol=1E-3):
    """
    Attempts to get the slope of the provided values. This function will be handy
    for checking if a dimension has been varied linearly or not.
    If the values vary non-linearly, a ValueError will be raised

    Parameters
    ----------
    values : array-like
        List of numbers
    tol : float, optional. Default = 1E-3
        Tolerance in the variation of the slopes.
    Returns
    -------
    float
        Slope of the line
    """
    if not isinstance(tol, float):
        raise TypeError('tol should be a float << 1')
    step_size = np.unique(np.diff(values))
    if len(step_size) > 1:
        # often we end up here. In most cases,
        step_avg = step_size.max()
        step_size -= step_avg
        var = np.mean(np.abs(step_size))
        if var / step_avg < tol:
            step_size = [step_avg]
        else:
            # Non-linear dimension! - see notes above
            raise ValueError('Provided values cannot be expressed as a linear trend')
    return step_size[0]


def to_ranges(iterable):
    """
    Converts a sequence of iterables to range tuples

    From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python

    Credits: @juanchopanza and @luca

    Parameters
    ----------
    iterable : collections.Iterable object
        iterable object like a list

    Returns
    -------
    iterable : generator object
        Cast to list or similar to use
    """
    iterable = sorted(set(iterable))
    for key, group in groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        if sys.version_info.major == 3:
            yield range(group[0][1], group[-1][1]+1)
        else:
            yield xrange(group[0][1], group[-1][1]+1)