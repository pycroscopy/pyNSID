# -*- coding: utf-8 -*-
"""
Definition of NSID Dataset
a N-dimensional dataset format living in HDF5 files

Created on Thu August 20 2020

@author: Gerd Duscher, Suhas Somnath
"""
# TODO: Move the visualization logic to sidpy.Dataset
# TODO: Move the read from h5py.Dataset logic to NSIDReader
# TODO: After completing two todos above, delete this class

from __future__ import division, print_function, absolute_import, unicode_literals

import sys
from warnings import warn
import h5py
import numpy as np

from sidpy.viz.dataset_viz import ImageStackVisualizer, CurveVisualizer, \
    ImageVisualizer, SpectralImageVisualizer
from sidpy.hdf.hdf_utils import get_attr

if sys.version_info.major == 3:
    unicode = str


class NSIDataset(h5py.Dataset):
    """
     A class that simplifies slicing, visualization, reshaping, reduction etc. of
     NSID datasets in HDF5 files.

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

        warn('NSIDataset will likely be removed in a future version of pyNSID.'
             ' Please use a ScopeReader class to read a sidpy.Dataset object '
             'from an NSID HDF5 Dataset instead',
             DeprecationWarning)

        self.data_type = get_attr(self, 'data_type')
        self.quantity = self.attrs['quantity']
        self.units = self.attrs['units']
        self.view = None

        # self.axes_names = [dim.label for dim in h5_ref.dims]
        units = []
        quantities = []
        dimension_types = []
        pixel_sizes = []

        for dim in h5_ref.dims:
            units.append(get_attr(dim[0], 'units'))
            quantities.append(get_attr(dim[0], 'quantity'))
            dimension_types.append(get_attr(dim[0], 'dimension_type'))
            pixel_sizes.append(abs(dim[0][1]-dim[0][0]))
        self.axes_units = units
        self.axes_quantities = quantities
        self.dimension_types = dimension_types
        self.axes_first_pixels = pixel_sizes

        self.data_descriptor = '{} ({})'.format(get_attr(self, 'quantity'), get_attr(self, 'units'))

    def get_dimension_labels(self):
        """
        Takes the labels and units attributes from NSID dataset and returns a list of strings
        formatted as 'quantity k [unit k]'

        Parameters
        ----------

        Returns
        -------
        labels : list
            list of strings formatted as 'label k (unit k)'
        """

        axes_labels = []
        for dim, quantity in enumerate(self.axes_quantities):
            axes_labels.append('{} [{}]'.format(quantity, self.axes_units[dim]))
        return axes_labels

    def get_dimens_types(self):
        dim_type_dict = {}
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
                               'Data dimensions and original shape:', '\t' + str(self.shape),
                               'Data type:', '\t' + self.data_type])
        if 'spatial' in dim_type_dict:
            usid_str = '\n'.join([usid_str,
                                  'Position Dimensions: ',  '\t' + str(dim_type_dict['spatial'])])
        if 'spectral' in dim_type_dict:
            usid_str = '\n'.join([usid_str,
                                  'Spectral Dimensions: ',  '\t' + str(dim_type_dict['spectral'])])

        if self.dtype.fields is not None:
            usid_str = '\n'.join([usid_str,
                                  'Data Fields:', '\t' + ', '.join([field for field in self.dtype.fields])])
        else:
            usid_str = '\n'.join([usid_str, 'Numeric Type:', '\t' + self.dtype.name])

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
        extent = [min_x, max_x, max_y, min_y]
        return extent

    def visualize(self, **kwargs):
        """
        Interactive visualization of this dataset. **Only available on jupyter notebooks**

        Parameters
        ----------
        kwargs

        Returns
        -------
        fig : :class:`matplotlib.figure` handle
            Handle for the figure object
        axis : :class:`matplotlib.Axes.axis` object
            Axis within which the data was plotted. Note - the interactive visualizer does not return this object
        """

        dim_type_dict = self.get_dimens_types()

        if 'spatial' in dim_type_dict:

            if len(dim_type_dict['spatial']) == 1:
                # ## some kind of line
                if len(dim_type_dict) == 1:
                    # simple profile
                    self.view = CurveVisualizer(self, 0, kwargs)  # TODO: correct dimension needed
                else:
                    print('visualization not implemented, yet')

            elif len(dim_type_dict['spatial']) == 2:
                # some kind of image data
                if len(dim_type_dict) == 1:
                    # simple image
                    self.view = ImageVisualizer(self, dim_type_dict)
                elif 'time' in dim_type_dict:
                    # image stack
                    self.view = ImageStackVisualizer(self, dim_type_dict)

                elif 'spectral' in dim_type_dict:
                    # spectrum image data in dataset
                    if len(dim_type_dict['spectral']) == 1:
                        self.view = SpectralImageVisualizer(self, dim_type_dict)
                        return self.view.fig, self.view.axes
                else:
                    print('visualization not implemented, yet')
            else:
                print('visualization not implemented, yet')

        elif 'reciprocal' in dim_type_dict:
            if len(dim_type_dict['reciprocal']) == 2:
                # some kind of image data
                if len(dim_type_dict) == 1:
                    # simple diffraction pattern
                    self.view = ImageVisualizer(self, dim_type_dict)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            if 'spectral' in dim_type_dict:
                # Only spectral data in dataset
                if len(dim_type_dict['spectral']) == 1:
                    print('spectr')
                    self.view = CurveVisualizer(self, dim_type_dict['spectral'], figure=None)
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