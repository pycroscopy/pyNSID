from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt


from sidpy.hdf.hdf_utils import get_attr
## taken out temporarily
from sidpy.base.num_utils import contains_integers, get_exponent
from sidpy.base.string_utils import validate_single_string_arg, validate_list_of_strings
from sidpy.hdf.hdf_utils import lazy_load_array
from sidpy.hdf.dtype_utils import is_complex_dtype
from sidpy.sid import Dimension
## taken out temporarily
from sidpy.hdf.dtype_utils import flatten_to_real
from sidpy.viz.jupyter_utils import simple_ndim_visualizer
from sidpy.viz.plot_utils import plot_map, get_plot_grid_size

from .hdf_utils import check_if_main, create_results_group, link_as_main, write_main_dataset, copy_attributes
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
            axes_labels.append("{} [{}]".format(quantity, self.axes_units[dim]))
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

    def slice(self, slice_dict, verbose=False, lazy=False):
        """
        Slices the N-dimensional form of the dataset based on the slice dictionary.

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
        self.__validate_slice_dict(slice_dict)

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

        return sliced_dset

    """@classmethod
    def from_hdf5(cls, dset, chunks=None, name=None, lock=False):

        # determine chunks, guessing something reasonable if user does not
        # specify
        chunks = get_chunks(np.array(dset), chunks)

        # create vanilla dask array
        darr = da.from_array(np.array(dset), chunks=chunks, name=name, lock=lock)

        # view as sub-class
        cls = view_subclass(darr, cls)

        if 'title' in dset.attrs:
            cls.title = dset.attrs['title']
        else:
            cls.title = dset.name

        if 'units' in dset.attrs:
            cls.units = dset.attrs['units']
        else:
            cls.units = 'generic'

        if 'quantity' in dset.attrs:
            cls.quantity = dset.attrs['quantity']
        else:
            cls.quantity = 'generic'

        if 'data_type' in dset.attrs:
            cls.data_type = dset.attrs['data_type']
        else:
            cls.data_type = 'generic'

        #TODO: mdoality and source not yet properties
        if 'modality' in dset.attrs:
            cls.modality = dset.attrs['modality']
        else:
            cls.modality = 'generic'

        if 'source' in dset.attrs:
            cls.source = dset.attrs['source']
        else:
            cls.source = 'generic'

        cls.axes ={}

        for dim in range(np.array(dset).ndim):
            #print(dim, dset.dims[dim].label)
            #print(dset.dims[dim][0][0])
            dim_dict = dict(dset.parent[dset.dims[dim].label].attrs)
            #print(dset.dims[dim].label, np.array(dset.dims[dim][0]))
            #print(dset.parent[dset.dims[0].label][()])
            #print(dim_dict['quantity'], dim_dict['units'], dim_dict['dimension_type'])
            cls.set_dimension(dim, Dimension(dset.dims[dim].label, np.array(dset.parent[dset.dims[dim].label][()]),
                                                    dim_dict['quantity'], dim_dict['units'],
                                                    dim_dict['dimension_type']))
        cls.attrs = dict(dset.attrs)

        cls.original_metadata = {}
        if 'original_metadata' in dset.parent:
            cls.original_metadata = dict(dset.parent['original_metadata'].attrs)


        return cls

    def to_hdf5(self, h5_group):
        if  self.title.strip() == '':
            main_data_name = 'nDim_Data'
        else:
            main_data_name = self.title
        print(h5_group)
        print(h5_group.keys())

        print(main_data_name)

        dset = write_main_dataset(h5_group, np.array(self), main_data_name,
                                 self.quantity, self.units, self.data_type, self.modality,
                                 self.source, self.axes, verbose=False)
        print('d',dset)

        for key, item in self.attrs.items():
            #TODO: Check item to be simple
            dset.attrs[key] = item

        original_group = h5_group.create_group('original_metadata')
        for key, item in self.original_metadata.items():
            original_group.attrs[key] = item

        if hasattr(self, 'aberrations'):
            aberrations_group = h5_group.create_group('aberrations')
            for key, item in self.aberrations.items():
                aberrations_group.attrs[key] = item

        if hasattr(self, 'annotations'):
            annotations_group = h5_group.create_group('annotations')
            for key, item in self.annotations.items():
                annotations_group.attrs[key] = item
    """
