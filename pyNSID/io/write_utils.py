# -*- coding: utf-8 -*-
"""
Utilities that assist in writing USID related data to HDF5 files

Created on Thu Sep  7 21:14:25 2017

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import sys
from warnings import warn
from enum import Enum
from itertools import groupby
import numpy as np
from .dtype_utils import contains_integers, validate_list_of_strings, validate_single_string_arg
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['clean_string_att', 'get_aux_dset_slicing', 'make_indices_matrix', 'INDICES_DTYPE', 'VALUES_DTYPE', 'get_slope',
           'Dimension', 'build_ind_val_matrices', 'calc_chunks', 'create_spec_inds_from_vals', 'validate_dimensions', 'DimType',
           'to_ranges']

if sys.version_info.major == 3:
    unicode = str


class Dimension(object):
    """
    ..autoclass::Dimension
    """

    def __init__(self, name, quantity, units, values, is_position):
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
        is_position : bool
            Whether or not this is a position or spectroscopy dimensions
        """
        name = validate_single_string_arg(name, 'name')
        quantity = validate_single_string_arg(quantity, 'quantity')

        if not isinstance(units, (str, unicode)):
            raise TypeError('units should be a string')
        units = units.strip()

        if isinstance(values, int):
            if values < 1:
                raise ValueError('values should at least be specified as a positive integer')
            values = np.arange(values)
        if not isinstance(values, (np.ndarray, list, tuple)):
            raise TypeError('values should be array-like')

        if not isinstance(is_position, bool):
            raise TypeError('is_position should be a bool')

        self.name = name
        self.quantity = quantity
        self.units = units
        self.values = values
        self.is_position = is_position

    def __repr__(self):
        return '{} - {} ({}) mode:{} : {}'.format(self.name, self.quantity, self.units, self.values)

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


def validate_dimensions(dimensions, dim_type='Position'):
    """
    Checks if the provided object is an iterable with pyUSID.Dimension objects.
    If it is not full of Dimension objects, Exceptions are raised.

    Parameters
    ----------
    dimensions : iterable or pyUSID.Dimension
        Iterable containing pyUSID.Dimension objects
    dim_type : str, Optional. Default = "Position"
        Type of Dimensions in the iterable. Set to "Spectroscopic" if not Position dimensions.
        This string is only used for more descriptive Exceptions

    Returns
    -------
    list
        List containing pyUSID.Dimension objects
    """
    if isinstance(dimensions, Dimension):
        dimensions = [dimensions]
    if isinstance(dimensions, np.ndarray):
        if dimensions.ndim > 1:
            dimensions = dimensions.ravel()
            warn(dim_type + ' dimensions should be specified by a 1D array-like. Raveled this numpy array for now')
    if not isinstance(dimensions, (list, np.ndarray, tuple)):
        raise TypeError(dim_type + ' dimensions should be array-like of Dimension objects')
    if not np.all([isinstance(x, Dimension) for x in dimensions]):
        raise TypeError(dim_type + ' dimensions should be a sequence of Dimension objects')
    return dimensions


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