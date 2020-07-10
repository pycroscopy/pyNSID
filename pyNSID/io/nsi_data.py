from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt


from .hdf_utils import check_if_main, get_attr, create_results_group, link_as_main, write_main_dataset, copy_attributes
## taken out temporarily
# get_sort_order, get_unit_values,
from .dtype_utils import  contains_integers, get_exponent, is_complex_dtype, \
    validate_single_string_arg, validate_list_of_strings, lazy_load_array

from .write_utils import Dimension

## taken out temporarily
#flatten_to_real,
from ..viz.jupyter_utils import simple_ndim_visualizer
from ..viz.plot_utils import plot_map, get_plot_grid_size
from ..viz.plot_nsid import plot_stack, plot_spectrum_image, plot_curve, plot_image

if sys.version_info.major == 3:
    unicode = str



class NSIDataset(h5py.Dataset):
    """
     A class that simplifies slicing, visualization, reshaping, reduction etc. of USID datasets in HDF5 files.
    This class extends the :class:`h5py.Dataset`.
    """

    def __init__(self, h5_ref, ):
        """
        Parameters
        ----------
        h5_ref : :class:`h5py.Dataset`
            The dataset which is actually a USID Main dataset
            This dataset has dhdf5 dimensional scales

        Methods
        -------

        self.slice
        self.data_descriptor():
            returns the label of the dataset
        self.get_dimension_labels():
            returns the labels of the dimensions
        self.get_dimens_types()
            returns dictionary of dimension_types (keys) with the axis numbers as values
        self.visualize(slice):
            not tested
            basic visualization of dataset based on dimension_types and slice (optional)
            returns fig and axis


        Attributes
        ----------
        self.data_type: str
            The data_type (supported are:  'image', 'image_stack',  'spectrum', 'linescan' and 'spectrum_image' )
        self.quantity: str
            The physical quantity represented in the dataset
        self.units: str
            The units of the dataset
        self.axes_units: list of str
            The units for the dimensional axes.
        self.axes_quantities: list of str
            The quantities (physical property) for the dimensional axes.
        self.dimension_types: list of str
            The dimension_types (supported is 'spatial', 'spectral', 'reciprocal' and 'time') for the dimensional axes.
        self.axes_first_pixels: list of int
            A list of the sizes of first pixel of each  dimension.

    """

        super(NSIDataset, self).__init__(h5_ref.id)

        self.data_type = get_attr(self,'data_type')
        self.quantity = self.attrs['quantity']
        self.units = self.attrs['units']

        #self.axes_names = [dim.label for dim in h5_ref.dims]
        units = []
        quantities = []
        dimension_types = []
        pixel_sizes = []

        for dim in h5_ref.dims:
            units.append(get_attr(dim[0],'units'))
            quantities.append(get_attr(dim[0],'quantity'))
            dimension_types.append(get_attr(dim[0],'dimension_type'))
            pixel_sizes.append(abs(dim[0][1]-dim[0][0]))
        self.axes_units = units
        self.axes_quantities = quantities
        self.dimension_types = dimension_types
        self.axes_first_pixels = pixel_sizes

        self.data_descriptor = '{} ({})'.format(get_attr(self, 'quantity'), get_attr(self, 'units'))


    def get_dimension_labels(self):
        """
        Takes the labels and units attributes from NSID datasetand returns a list of strings
        formatted as 'quantity k [unit k]'

        Parameters
        ----------
        h5_dset : h5py.Dataset object
            dataset which has labels and units attributes

        Returns
        -------
        labels : list
            list of strings formatted as 'label k (unit k)'
        """

        axes_labels = []
        for dim, quantity in enumerate(self.axes_quantities):
            axes_labels.append(f"{quantity} [{self.axes_units[dim]}] ")
        return axes_labels

    def get_dimens_types(self):
        dim_type_dict  = {}
        spectral_dimensions = []
        for dim, dim_type in enumerate(self.dimension_types):
            if dim_type not in dim_type_dict:
                dim_type_dict[dim_type] = []
            dim_type_dict[dim_type].append(dim)
        return dim_type_dict

    def __repr__(self):
        h5_str = super(NSIDataset, self).__repr__()

        dim_type_dict = self.get_dimens_types()
        usid_str = ' \n'.join(['located at:',
                                'Data contains:', '\t' + self.data_descriptor,
                                'Data dimensions and original shape:',  '\t' +str(self.shape),
                                'Data type:', '\t' + self.data_type])
        if 'spatial' in dim_type_dict:
            usid_str = '\n'.join([usid_str,
                                  'Position Dimensions: ',  '\t' +str(dim_type_dict['spatial'])])
        if 'spectral' in dim_type_dict:
            usid_str = '\n'.join([usid_str,
                                  'Spectral Dimensions: ',  '\t' +str(dim_type_dict['spectral'])])

        if self.dtype.fields is not None:
            usid_str = '\n'.join([usid_str,
                                  'Data Fields:', '\t' + ', '.join([field for field in self.dtype.fields])])
        else:
            usid_str = '\n'.join([usid_str,
                                   'Numeric Type:', '\t' + self.dtype.name])

        if sys.version_info.major == 2:
            usid_str = usid_str.encode('utf8')

        return '\n'.join([h5_str, usid_str])

    def make_extent(self, ref_dims):
        x_axis = self.dims[ref_dims[0]][0]
        min_x = (x_axis[0] - abs(x_axis[0]-x_axis[1])/2)
        max_x = (x_axis[-1] + abs(x_axis[-1]-x_axis[-2])/2)
        y_axis = self.dims[ref_dims[1]][0]
        min_y = (y_axis[0] - abs(y_axis[0]-y_axis[1])/2)
        max_y = (y_axis[-1] + abs(y_axis[-1]-y_axis[-2])/2)
        extent = [min_x, max_x,max_y, min_y]
        return extent


    def visualize(self, slice_dict=None, verbose=False, **kwargs):
        """
        Interactive visualization of this dataset. **Only available on jupyter notebooks**

        Parameters
        ----------
        slice_dict : dictionary, optional
            Slicing instructions
        verbose : bool, optional
            Whether or not to print debugging statements. Default = Off

        Returns
        -------
        fig : :class:`matplotlib.figure` handle
            Handle for the figure object
        axis : :class:`matplotlib.Axes.axis` object
            Axis within which the data was plotted. Note - the interactive visualizer does not return this object
        """

        dim_type_dict = self.get_dimens_types()
        output_reference = None
        data_slice = self
        if 'spatial' in dim_type_dict:

            if len(dim_type_dict['spatial'])== 1:
                ### some kind of line
                if len(dim_type_dict) == 1:
                    ## simple profile
                    self.view = plot_curve(self, pos_dims)
                else:
                    print('visualization not implemented, yet')


            elif len(dim_type_dict['spatial'])== 2:
                ## some kind of image data
                if len(dim_type_dict) == 1:
                    ## simple image
                    self.view = plot_image(self, dim_type_dict)
                elif 'time' in dim_type_dict:
                    ## image stack
                    self.view = plot_stack(self, dim_type_dict)

                elif 'spectral' in dim_type_dict:
                    ### spectrum image data in dataset
                    if len(dim_type_dict['spectral'])== 1:
                        self.view = plot_spectrum_image(self,dim_type_dict)
                        return self.view.fig, self.view.axes
                else:
                    print('visualization not implemented, yet')
            else:
                print('visualization not implemented, yet')

        elif 'reciprocal' in dim_type_dict:
            if len(dim_type_dict['reciprocal'])== 2:
                ## some kind of image data
                if len(dim_type_dict) == 1:
                    ## simple diffraction pattern
                    self.view = plot_image(self, dim_type_dict)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            if 'spectral' in dim_type_dict:
                ### Only spectral data in dataset
                if len(dim_type_dict['spectral'])== 1:
                    print('spectr')
                    self.view = plot_curve(self, dim_type_dict['spectral'], figure = None)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError



def __validate_slice_dict(self, slice_dict):
    """
    Validates the slice dictionary

    Parameters
    ----------
    slice_dict : dict
        Dictionary of array-likes.

    Returns
    -------
    None
    """
    if not isinstance(slice_dict, dict):
        raise TypeError('slice_dict should be a dictionary of slice objects')
    for key, val in slice_dict.items():
        # Make sure the dimension is valid
        if key not in self.n_dim_labels:
            raise KeyError('Cannot slice on dimension {}.  '
                           'Valid dimensions are {}.'.format(key, self.n_dim_labels))
        if not isinstance(val, (slice, list, np.ndarray, tuple, int)):
            raise TypeError('The slices must be array-likes or slice objects.')
    return True

def __slice_n_dim_form(self, slice_dict, verbose=False, lazy=False):
    """
    Slices the N-dimensional form of the dataset based on the slice dictionary.
    Assumes that an N-dimensional form exists and is what was requested

    Parameters
    ----------
    slice_dict : dict
        Dictionary of array-likes. for any dimension one needs to slice
    verbose : bool, optional
        Whether or not to print debugging statements
    lazy : bool, optional. Default = False
        If set to false, data_slice will be a :class:`numpy.ndarray`
        Else returned object is :class:`dask.array.core.Array`

    Returns
    -------
    data_slice : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        Slice of the dataset.
    success : bool
        Always True
    """
    nd_slice = []

    for dim_name in self.n_dim_labels:
        nd_slice.append(slice_dict.get(dim_name, slice(None)))

    # Dask multidimensional slicing does not work if list is passed:
    nd_slice = tuple(nd_slice)
    if verbose:
        print(self.n_dim_labels)
        print(nd_slice)

    sliced_dset = self.__curr_ndim_form[nd_slice]
    if not lazy:
        sliced_dset = sliced_dset.compute()

    return sliced_dset, True

def slice(self, slice_dict, ndim_form=True, as_scalar=False, verbose=False, lazy=False):
    """
    Slice the dataset based on an input dictionary of 'str': slice pairs.
    Each string should correspond to a dimension label.  The slices can be
    array-likes or slice objects.

    Parameters
    ----------
    slice_dict : dict
        Dictionary of array-likes. for any dimension one needs to slice
    ndim_form : bool, optional
        Whether or not to return the slice in it's N-dimensional form. Default = True
    as_scalar : bool, optional
        Should the data be returned as scalar values only.
    verbose : bool, optional
        Whether or not to print debugging statements
    lazy : bool, optional. Default = False
        If set to false, data_slice will be a :class:`numpy.ndarray`
        Else returned object is :class:`dask.array.core.Array`

    Returns
    -------
    data_slice : :class:`numpy.ndarray`, or :class:`dask.array.core.Array`
        Slice of the dataset.  Dataset has been reshaped to N-dimensions if `success` is True, only
        by Position dimensions if `success` is 'Positions', or not reshape at all if `success`
        is False.
    success : str or bool
        Informs the user as to how the data_slice has been shaped.

    """
    # TODO: Accept sequences of integers and build a list of slice objects for each dimension
    if slice_dict is None:
        slice_dict = dict()
    else:
        self.__validate_slice_dict(slice_dict)

    if not isinstance(as_scalar, bool):
        raise TypeError('as_scalar should be a bool')
    if not isinstance(verbose, bool):
        raise TypeError('verbose should be a bool')

    if self.__n_dim_form_avail and ndim_form:
        return self.__slice_n_dim_form(slice_dict, verbose=verbose, lazy=lazy)

    # Convert the slice dictionary into lists of indices for each dimension
    pos_slice, spec_slice = self._get_pos_spec_slices(slice_dict)
    if verbose:
        print('Position slice: shape - {}'.format(pos_slice.shape))
        print(pos_slice)
        print('Spectroscopic slice: shape - {}'.format(spec_slice.shape))
        print(spec_slice)

    # Now that the slices are built, we just need to apply them to the data
    # This method is slow and memory intensive but shouldn't fail if multiple lists are given.
    if lazy:
        raw_2d = self.__lazy_2d
    else:
        raw_2d = self

    if verbose:
        print('Slicing to 2D based on dataset of shape: {} and type: {}'
              ''.format(raw_2d.shape, type(raw_2d)))

    if lazy:
        data_slice = raw_2d[pos_slice[:, 0], :][:, spec_slice[:, 0]]
    else:
        if len(pos_slice) <= len(spec_slice):
            # Fewer final positions than spectra
            data_slice = np.atleast_2d(raw_2d[pos_slice[:, 0], :])[:, spec_slice[:, 0]]
        else:
            # Fewer final spectral points compared to positions
            data_slice = np.atleast_2d(raw_2d[:, spec_slice[:, 0]])[pos_slice[:, 0], :]

    if verbose:
        print('data_slice of shape: {} and type: {} after slicing'
              ''.format(data_slice.shape, type(data_slice)))
    if not lazy:
        orig_shape = data_slice.shape
        data_slice = np.atleast_2d(np.squeeze(data_slice))
        if data_slice.shape[0] == orig_shape[1] and data_slice.shape[1] == orig_shape[0]:
            data_slice = data_slice.T
    if verbose:
        print('data_slice of shape: {} after squeezing'.format(data_slice.shape))

    pos_inds = self.h5_pos_inds[pos_slice, :]
    spec_inds = self.h5_spec_inds[:, spec_slice].reshape([self.h5_spec_inds.shape[0], -1])
    if verbose:
        print('Sliced position indices:')
        print(pos_inds)
        print('Spectroscopic Indices (transposed)')
        print(spec_inds.T)

    # At this point, the empty dimensions MUST be removed in order to avoid problems with dimension sort etc.
    def remove_singular_dims(anc_inds):
        new_inds = []
        for dim_values in anc_inds:
            if len(np.unique(dim_values)) > 1:
                new_inds.append(dim_values)
        # if all dimensions are removed?
        if len(new_inds) == 0:
            new_inds = np.arange(1)
        else:
            new_inds = np.array(new_inds)
        return new_inds

    pos_inds = np.atleast_2d(remove_singular_dims(pos_inds.T).T)
    spec_inds = np.atleast_2d(remove_singular_dims(spec_inds))

    if verbose:
        print('After removing any singular dimensions')
        print('Sliced position indices:')
        print(pos_inds)
        print('Spectroscopic Indices (transposed)')
        print(spec_inds.T)
        print('data slice of shape: {}. Position indices of shape: {}, Spectroscopic indices of shape: {}'
              '.'.format(data_slice.shape, pos_inds.shape, spec_inds.shape))

    success = True

    if ndim_form:
        # TODO: if data is already loaded into memory, try to avoid I/O and slice in memory!!!!
        data_slice, success = reshape_to_n_dims(data_slice, h5_pos=pos_inds, h5_spec=spec_inds, verbose=verbose, lazy=lazy)
        data_slice = data_slice.squeeze()

    if as_scalar:
        return flatten_to_real(data_slice), success
    else:
        return data_slice, success
