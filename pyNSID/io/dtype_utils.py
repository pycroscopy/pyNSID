# -*- coding: utf-8 -*-
"""
Utilities for transforming and validating data types

Given that many of the data transformations involve copying the data, they should
ideally happen in a lazy manner to avoid memory issues.

Created on Tue Nov  3 21:14:25 2015

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, absolute_import, unicode_literals, print_function
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
from itertools import groupby
if sys.version_info.major == 3:
    from collections.abc import Iterable
else:
    from collections import Iterable

__all__ = ['flatten_complex_to_real', 'get_compound_sub_dtypes', 'flatten_compound_to_real', 'check_dtype',
           'stack_real_to_complex', 'validate_dtype', 'integers_to_slices', 'get_exponent', 'is_complex_dtype',
           'stack_real_to_compound', 'stack_real_to_target_dtype', 'flatten_to_real', 'contains_integers',
           'validate_single_string_arg', 'validate_string_args', 'validate_list_of_strings',
           'lazy_load_array']

if sys.version_info.major == 3:
    unicode = str


def lazy_load_array(dataset):
    """
    Loads the provided object as a dask array (h5py.Dataset or numpy.ndarray

    Parameters
    ----------
    dataset : :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Array to laod as dask array

    Returns
    -------
    :class:`dask.array.core.Array`
        Dask array with appropriate chunks
    """
    if isinstance(dataset, da.core.Array):
        return dataset
    elif not isinstance(dataset, (h5py.Dataset, np.ndarray)):
        raise TypeError('Expected one of h5py.Dataset, dask.array.core.Array, or numpy.ndarray'
                        'objects. Provided object was of type: {}'.format(type(dataset)))
    # Cannot pass 'auto' for chunks for python 2!
    chunks = "auto" if sys.version_info.major == 3 else dataset.shape
    if isinstance(dataset, h5py.Dataset):
        chunks = chunks if dataset.chunks is None else dataset.chunks
    return da.from_array(dataset, chunks=chunks)


def contains_integers(iter_int, min_val=None):
    """
    Checks if the provided object is iterable (list, tuple etc.) and contains integers optionally greater than equal to
    the provided min_val

    Parameters
    ----------
    iter_int : :class:`collections.Iterable`
        Iterable (e.g. list, tuple, etc.) of integers
    min_val : int, optional, default = None
        The value above which each element of iterable must possess. By default, this is ignored.

    Returns
    -------
    bool
        Whether or not the provided object is an iterable of integers
    """
    if not isinstance(iter_int, Iterable):
        raise TypeError('iter_int should be an Iterable')
    if len(iter_int) == 0:
        return False

    if min_val is not None:
        if not isinstance(min_val, (int, float)):
            raise TypeError('min_val should be an integer. Provided object was of type: {}'.format(type(min_val)))
        if min_val % 1 != 0:
            raise ValueError('min_val should be an integer')

    try:
        if min_val is not None:
            return np.all([x % 1 == 0 and x >= min_val for x in iter_int])
        else:
            return np.all([x % 1 == 0 for x in iter_int])
    except TypeError:
        return False

def flatten_complex_to_real(dataset, lazy=False):
    """
    Stacks the real values followed by the imaginary values in the last dimension of the given N dimensional matrix.
    Thus a complex matrix of shape (2, 3, 5) will turn into a matrix of shape (2, 3, 10)

    Parameters
    ----------
    dataset : array-like or :class:`numpy.ndarray`, or :class:`h5py.Dataset`, or :class:`dask.array.core.Array`
        Dataset of complex data type
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    -------
    retval : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        real valued dataset
    """
    if not isinstance(dataset, (h5py.Dataset, np.ndarray, da.core.Array)):
        raise TypeError('dataset should either be a h5py.Dataset or numpy / dask array')
    if not is_complex_dtype(dataset.dtype):
        raise TypeError("Expected a complex valued dataset")

    if isinstance(dataset, da.core.Array):
        lazy = True

    xp = np
    if lazy:
        dataset = lazy_load_array(dataset)
        xp = da

    axis = xp.array(dataset).ndim - 1
    if axis == -1:
        return xp.hstack([xp.real(dataset), xp.imag(dataset)])
    else:  # along the last axis
        return xp.concatenate([xp.real(dataset), xp.imag(dataset)], axis=axis)


def stack_real_to_complex(ds_real, lazy=False):
    """
    Puts the real and imaginary sections of the provided matrix (in the last axis) together to make complex matrix

    Parameters
    ------------
    ds_real : :class:`numpy.ndarray`, :class:`dask.array.core.Array`, or :class:`h5py.Dataset`
        n dimensional real-valued numpy array or HDF5 dataset where data arranged as [instance, 2 x features],
        where the first half of the features are the real component and the
        second half contains the imaginary components
    lazy : bool, optional. Default = False
        If set to True, will use lazy Dask arrays instead of in-memory numpy arrays

    Returns
    ----------
    ds_compound : :class:`numpy.ndarray` or :class:`dask.array.core.Array`
        2D complex array arranged as [sample, features]
    """
    if not isinstance(ds_real, (np.ndarray, da.core.Array, h5py.Dataset)):
        if not isinstance(ds_real, (tuple, list)):
            raise TypeError("Expected at least an iterable like a list or tuple")
        ds_real = np.array(ds_real)
    if len(ds_real.dtype) > 0:
        raise TypeError("Array cannot have a compound dtype")
    if is_complex_dtype(ds_real.dtype):
        raise TypeError("Array cannot have complex dtype")

    if ds_real.shape[-1] / 2 != ds_real.shape[-1] // 2:
        raise ValueError("Last dimension must be even sized")
    half_point = ds_real.shape[-1] // 2

    if isinstance(ds_real, da.core.Array):
        lazy = True

    if lazy and not isinstance(ds_real, da.core.Array):
        ds_real = lazy_load_array(ds_real)

    return ds_real[..., :half_point] + 1j * ds_real[..., half_point:]


def validate_dtype(dtype):
    """
    Checks the provided object to ensure that it is a valid dtype that can be written to an HDF5 file.
    Raises a type error if invalid. Returns True if the object passed the tests

    Parameters
    ----------
    dtype : object
        Object that is hopefully a :class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    status : bool
        True if the object was a valid data-type
    """
    if isinstance(dtype, (h5py.Datatype, np.dtype)):
        pass
    elif isinstance(np.dtype(dtype), np.dtype):
        # This should catch all those instances when dtype is something familiar like - np.float32
        pass
    else:
        raise TypeError('dtype should either be a numpy or h5py dtype')
    return True


def validate_single_string_arg(value, name):
    """
    This function is to be used when validating a SINGLE string parameter for a function. Trims the provided value
    Errors in the string will result in Exceptions

    Parameters
    ----------
    value : str
        Value of the parameter
    name : str
        Name of the parameter

    Returns
    -------
    str
        Cleaned string value of the parameter
    """
    if not isinstance(value, (str, unicode)):
        raise TypeError(name + ' should be a string')
    value = value.strip()
    if len(value) <= 0:
        raise ValueError(name + ' should not be an empty string')
    return value


def validate_list_of_strings(str_list, parm_name='parameter'):
    """
    This function is to be used when validating and cleaning a list of strings. Trims the provided strings
    Errors in the strings will result in Exceptions

    Parameters
    ----------
    str_list : array-like
        list or tuple of strings
    parm_name : str, Optional. Default = 'parameter'
        Name of the parameter corresponding to this string list that will be reported in the raised Errors

    Returns
    -------
    array-like
        List of trimmed and validated strings when ALL objects within the list are found to be valid strings
    """

    if isinstance(str_list, (str, unicode)):
        return [validate_single_string_arg(str_list, parm_name)]

    if not isinstance(str_list, (list, tuple)):
        raise TypeError(parm_name + ' should be a string or list / tuple of strings')

    return [validate_single_string_arg(x, parm_name) for x in str_list]


def validate_string_args(arg_list, arg_names):
    """
    This function is to be used when validating string parameters for a function. Trims the provided strings.
    Errors in the strings will result in Exceptions

    Parameters
    ----------
    arg_list : array-like
        List of str objects that signify the value for a position argument in a function
    arg_names : array-like
        List of str objects with the names of the corresponding parameters in the function

    Returns
    -------
    array-like
        List of str objects that signify the value for a position argument in a function with spaces on ends removed
    """
    if isinstance(arg_list, (str, unicode)):
        arg_list = [arg_list]
    if isinstance(arg_names, (str, unicode)):
        arg_names = [arg_names]
    cleaned_args = []
    if not isinstance(arg_list, (tuple, list)):
        raise TypeError('arg_list should be a tuple or a list or a string')
    if not isinstance(arg_names, (tuple, list)):
        raise TypeError('arg_names should be a tuple or a list or a string')
    for arg, arg_name in zip(arg_list, arg_names):
        cleaned_args.append(validate_single_string_arg(arg, arg_name))
    return cleaned_args


def is_complex_dtype(dtype):
    """
    Checks if the provided dtype is a complex dtype

    Parameters
    ----------
    dtype : object
        Object that is a class:`h5py.Datatype`, or :class:`numpy.dtype` object

    Returns
    -------
    is_complex : bool
        True if the dtype was a complex dtype. Else returns False
    """
    validate_dtype(dtype)
    if dtype in [np.complex, np.complex64, np.complex128]:
        return True
    return False


def integers_to_slices(int_array):
    """
    Converts a sequence of iterables to a list of slice objects denoting sequences of consecutive numbers

    Parameters
    ----------
    int_array : :class:`collections.Iterable`
        iterable object like a :class:`list` or :class:`numpy.ndarray`

    Returns
    -------
    sequences : list
        List of :class:`slice` objects each denoting sequences of consecutive numbers
    """
    if not contains_integers(int_array):
        raise ValueError('Expected a list, tuple, or numpy array of integers')

    def integers_to_consecutive_sections(integer_array):
        """
        Converts a sequence of iterables to tuples with start and stop bounds

        @author: @juanchopanza and @luca from stackoverflow

        Parameters
        ----------
        integer_array : :class:`collections.Iterable`
            iterable object like a :class:`list`

        Returns
        -------
        iterable : :class:`generator`
            Cast to list or similar to use

        Note
        ----
        From https://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
        """
        integer_array = sorted(set(integer_array))
        for key, group in groupby(enumerate(integer_array),
                                  lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]

    sequences = [slice(item[0], item[1] + 1) for item in integers_to_consecutive_sections(int_array)]
    return sequences


def get_exponent(vector):
    """
    Gets the scale / exponent for a sequence of numbers. This is particularly useful when wanting to scale a vector
    for the purposes of plotting

    Parameters
    ----------
    vector : array-like
        Array of numbers

    Returns
    -------
    exponent : int
        Scale / exponent for the given vector
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError('vector should be of type numpy.ndarray. Provided object of type: {}'.format(type(vector)))
    if np.max(np.abs(vector)) == np.max(vector):
        exponent = np.log10(np.max(vector))
    else:
        # negative values
        exponent = np.log10(np.max(np.abs(vector)))
    return int(np.floor(exponent))