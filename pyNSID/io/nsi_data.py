from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
from warnings import warn
import h5py
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from .hdf_utils import check_if_main, get_attr, create_results_group, write_reduced_anc_dsets, link_as_main, \
    write_main_dataset,  \
    copy_attributes
## taken out temporarily
#get_dimensionality, get_sort_order, get_unit_values, reshape_to_n_dims,reshape_from_n_dims,
from .dtype_utils import  contains_integers, get_exponent, is_complex_dtype, \
    validate_single_string_arg, validate_list_of_strings, lazy_load_array
## taken out temporarily
#flatten_to_real, 
from .write_utils import Dimension
#from ..viz.jupyter_utils import simple_ndim_visualizer
#from ..viz.plot_utils import plot_map, get_plot_grid_size

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
        sort_dims : bool, Optional. Default=False
            If set to True - Dimensions will be sorted from slowest to fastest
            Else - Dimensions will be arranged as they appear in ancillary datasets

        Methods
        -------
        self.get_current_sorting
        self.toggle_sorting
        self.get_pos_values
        self.get_spec_values
        self.get_n_dim_form
        self.slice


        Attributes
        ----------
        self.h5_spec_vals : :class:`h5py.Dataset`
            Associated Spectroscopic Values dataset
        self.h5_spec_inds : :class:`h5py.Dataset`
            Associated Spectroscopic Indices dataset
        self.h5_pos_vals : :class:`h5py.Dataset`
            Associated Position Values dataset
        self.h5_pos_inds : :class:`h5py.Dataset`
            Associated Position Indices dataset
        self.pos_dim_labels : list of str
            The labels for the position dimensions.
        self.spec_dim_labels : list of str
            The labels for the spectroscopic dimensions.
        self.n_dim_labels : list of str
            The labels for the n-dimensional dataset.
        self.pos_dim_sizes : list of int
            A list of the sizes of each position dimension.
        self.spec_dim_sizes : list of int
            A list of the sizes of each spectroscopic dimension.
        self.n_dim_sizes : list of int
            A list of the sizes of each dimension.

        Notes
        -----
        The order of all labels and sizes attributes is determined by the current value of `sort_dims`.

        """

        #if not check_if_main(h5_ref):
        #    raise TypeError('Supply a h5py.Dataset object that is a USID main dataset')

        super(NSIDataset, self).__init__(h5_ref.id)

        # User accessible properties
        self.data_type = h5_ref.attrs['data_type'] # #ToDo: where is shape
        # The dimension labels as they appear in the ancillary datasets
        #self.__orig_pos_dim_labels = get_attr(self.h5_pos_inds, 'labels')
        #self.__orig_spec_dim_labels = get_attr(self.h5_spec_inds, 'labels')

        # Data descriptors
        #self.data_descriptor = '{} ({})'.format(get_attr(self, 'quantity'), get_attr(self, 'units'))
        #self.pos_dim_descriptors = self.__get_anc_labels(self.h5_pos_inds)
        #self.spec_dim_descriptors = self.__get_anc_labels(self.h5_spec_inds)

        # The size of each dimension
        #self.__orig_pos_dim_sizes = np.array(get_dimensionality(np.transpose(self.h5_pos_inds)))
        #self.__orig_spec_dim_sizes = np.array(get_dimensionality(np.atleast_2d(self.h5_spec_inds)))

        # Sorted dimension order
        #self.__pos_sort_order = get_sort_order(np.transpose(self.h5_pos_inds))
        #self.__spec_sort_order = get_sort_order(np.atleast_2d(self.h5_spec_inds))

        # internal book-keeping / we don't want users to mess with these?
        #self.__orig_n_dim_sizes = np.append(self.__orig_pos_dim_sizes, self.__orig_spec_dim_sizes)
        #self.__orig_n_dim_labs = np.append(self.__orig_pos_dim_labels, self.__orig_spec_dim_labels)
        #self.__n_dim_sort_order_orig_s2f = np.append(self.__pos_sort_order[::-1],
        #                                             self.__spec_sort_order[::-1] + len(self.__pos_sort_order))
        #self.__n_dim_sort_order_orig_f2s = np.append(self.__pos_sort_order,
        #                                             self.__spec_sort_order + len(self.__pos_sort_order))

        #self.__n_dim_data_orig = None
        #self.__n_dim_data_s2f = None
        #self.__curr_ndim_form = None
        #self.__n_dim_form_avail = False

        # Should the dimensions be sorted from slowest to fastest
        #self.__sort_dims = sort_dims

        # Declaring var names within init
        #self.pos_dim_labels = None
        #self.spec_dim_labels = None
        #self.pos_dim_sizes = None
        #self.spec_dim_sizes = None
        #self.n_dim_labels = None
        #self.n_dim_sizes = None

        self.__lazy_2d = lazy_load_array(self)

        self.__set_labels_and_sizes()

        try:
            pass
            #self.__n_dim_data_orig = self.get_n_dim_form(lazy=True)
            #self.__n_dim_form_avail = True
            # TODO: This line keeps failing. Fix it
            #self.__n_dim_data_s2f = self.__n_dim_data_orig.transpose(self.__n_dim_sort_order_orig_s2f)
        except ValueError:
            warn('This dataset does not have an N-dimensional form')

        #self.__set_n_dim_view()

    def __eq__(self, other):
        if isinstance(other, h5py.Dataset):
            return super(USIDataset, self).__eq__(other)

        return False

    def __repr__(self):
        h5_str = super(NSIDataset, self).__repr__()

        #pos_str = ' \n'.join(['\t{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
        #                      zip(self.__orig_pos_dim_labels, self.__orig_pos_dim_sizes)])
        #spec_str = ' \n'.join(['\t{} - size: {}'.format(dim_name, str(dim_size)) for dim_name, dim_size in
        #                       zip(self.__orig_spec_dim_labels, self.__orig_spec_dim_sizes)])

        #usid_str = ' \n'.join(['located at:',
        #                        '\t' + self.name,
        #                        'Data contains:', '\t' + self.data_descriptor,
        #                        'Data dimensions and original shape:',
        #                        'Position Dimensions:',
        #                        pos_str,
        #                        'Spectroscopic Dimensions:',
        #                        spec_str])

        if self.dtype.fields is not None:
            usid_str = '\n'.join([usid_str,
                                  'Data Fields:', '\t' + ', '.join([field for field in self.dtype.fields])])
        else:
            usid_str = '\n'.join([usid_str,
                                   'Data Type:', '\t' + self.dtype.name])

        if sys.version_info.major == 2:
            usid_str = usid_str.encode('utf8')

        return '\n'.join([h5_str, usid_str])

    def __set_labels_and_sizes(self):
        """
        Sets the labels and sizes attributes to the correct values based on
        the value of `self.__sort_dims`
        """
        #if self.__sort_dims:
        #    self.pos_dim_labels = self.__orig_pos_dim_labels[self.__pos_sort_order].tolist()
        #    self.spec_dim_labels = self.__orig_spec_dim_labels[self.__spec_sort_order].tolist()
        #    self.pos_dim_sizes = self.__orig_pos_dim_sizes[self.__pos_sort_order].tolist()
        #    self.spec_dim_sizes = self.__orig_spec_dim_sizes[self.__spec_sort_order].tolist()
        #    self.n_dim_labels = self.__orig_n_dim_labs[self.__n_dim_sort_order_orig_s2f].tolist()
        #    self.n_dim_sizes = self.__orig_n_dim_sizes[self.__n_dim_sort_order_orig_s2f].tolist()

        #else:
        #    self.pos_dim_labels = self.__orig_pos_dim_labels.tolist()
        #    self.spec_dim_labels = self.__orig_spec_dim_labels.tolist()
        #    self.pos_dim_sizes = self.__orig_pos_dim_sizes.tolist()
        #    self.spec_dim_sizes = self.__orig_spec_dim_sizes.tolist()
        #    self.n_dim_labels = self.__orig_n_dim_labs.tolist()
        #    self.n_dim_sizes = self.__orig_n_dim_sizes.tolist()

    def __set_n_dim_view(self):
        """
        Sets the current view of the N-dimensional form of the dataset
        """

        self.__curr_ndim_form = self.__n_dim_data_s2f if self.__sort_dims else self.__n_dim_data_orig

    @staticmethod
    def __get_anc_labels(h5_dset):
        """
        Takes any dataset which has the labels and units attributes and returns a list of strings
        formatted as 'label k (unit k)'

        Parameters
        ----------
        h5_dset : h5py.Dataset object
            dataset which has labels and units attributes

        Returns
        -------
        labels : list
            list of strings formatted as 'label k (unit k)'
        """
        labels = []
        for lab, unit in zip(get_attr(h5_dset, 'labels'), get_attr(h5_dset, 'units')):
            labels.append('{} ({})'.format(lab, unit))
        return labels

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

        if slice_dict is None:
            if len(self.pos_dim_labels) > 2 or len(self.spec_dim_labels) > 2:
                raise NotImplementedError('Unable to support visualization of more than 2 position / spectroscopic '
                                          'dimensions. Try slicing the dataset')
            data_slice = self.get_n_dim_form()
            spec_unit_values = get_unit_values(self.h5_spec_inds, self.h5_spec_vals)
            pos_unit_values = get_unit_values(self.h5_pos_inds, self.h5_pos_vals)

            pos_dims = []
            for name, units in zip(self.pos_dim_labels, get_attr(self.h5_pos_inds, 'units')):
                pos_dims.append(Dimension(name, units, pos_unit_values[name]))
            spec_dims = []
            for name, units in zip(self.spec_dim_labels, get_attr(self.h5_spec_inds, 'units')):
                spec_dims.append(Dimension(name, units, spec_unit_values[name]))

        else:
            pos_dims, spec_dims = self._get_dims_for_slice(slice_dict=slice_dict, verbose=verbose)

            # see if the total number of pos and spec keys are either 1 or 2
            if not (0 < len(pos_dims) < 3) or not (0 < len(spec_dims) < 3):
                raise ValueError('Number of position ({}) / spectroscopic dimensions ({}) more than 2'
                                 '. Try slicing again'.format(len(pos_dims), len(spec_dims)))

            # now should be safe to slice:
            data_slice, success = self.slice(slice_dict, ndim_form=True, lazy=False)
            if not success:
                raise ValueError('Something went wrong when slicing the dataset. slice message: {}'.format(success))
            # don't forget to remove singular dimensions via a squeeze
            data_slice = np.squeeze(data_slice)
            # Unlikely event that all dimensions were removed and we are left with a scalar:
            if data_slice.ndim == 0:
                # Nothing to visualize - just return a value
                return data_slice
            # There is a chance that the data dimensionality may have reduced to 1:
            elif data_slice.ndim == 1:
                if len(pos_dims) == 0:
                    data_slice = np.expand_dims(data_slice, axis=0)
                else:
                    data_slice = np.expand_dims(data_slice, axis=-1)

        if verbose:
            print('Position Dimensions:')
            for item in pos_dims:
                print('{}\n{}'.format(len(item.values), item))
            print('Spectroscopic Dimensions:')
            for item in spec_dims:
                print('{}\n{}'.format(len(item.values), item))
            print('N dimensional data sent to visualizer of shape: {}'.format(data_slice.shape))

        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        def plot_curve(ref_dims, curve):
            x_suffix = ''
            x_exp = get_exponent(ref_dims[0].values)
            if x_exp < -2 or x_exp > 3:
                ref_dims[0].values /= 10 ** x_exp
                x_suffix = ' x $10^{' + str(x_exp) + '}$'

            if is_complex_dtype(curve.dtype):
                # Plot real and image
                fig, axes = plt.subplots(nrows=2, **fig_args)
                for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                    axis.plot(ref_dims[0].values, ufunc(np.squeeze(curve)), **kwargs)
                    if comp_name is 'Magnitude':
                        axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                        axis.set_ylabel(self.data_descriptor)
                    else:
                        axis.set_title(comp_name, pad=15)
                        axis.set_ylabel('Phase (rad)')
                        axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)

                fig.tight_layout()
                return fig, axes
            elif len(curve.dtype) > 0:
                plot_grid = get_plot_grid_size(len(curve.dtype))
                fig, axes = plt.subplots(nrows=plot_grid[0], ncols=plot_grid[1], **fig_args)
                for axis, comp_name in zip(axes.flat, curve.dtype.fields):
                    axis.plot(ref_dims[0].values, np.squeeze(curve[comp_name]), **kwargs)
                    axis.set_title(comp_name, pad=15)
                    axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)
                    axis.set_ylabel(comp_name)
                # fig.suptitle(self.name)
                fig.tight_layout()
                return fig, axes
            else:
                y_exp = get_exponent(np.squeeze(curve))
                y_suffix = ''
                if y_exp < -2 or y_exp > 3:
                    curve = np.squeeze(curve) / 10 ** y_exp
                    y_suffix = ' x $10^{' + str(y_exp) + '}$'

                fig, axis = plt.subplots(**fig_args)
                axis.plot(ref_dims[0].values, np.squeeze(curve), **kwargs)
                axis.set_xlabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + x_suffix)
                axis.set_ylabel(self.data_descriptor + y_suffix)
                axis.set_title(self.name)

                return fig, axis

        def plot_image(ref_dims, img):
            exponents = [get_exponent(item.values) for item in ref_dims]
            suffix = []
            for item, scale in zip(ref_dims, exponents):
                curr_suff = ''
                if scale < -1 or scale > 3:
                    item.values /= 10 ** scale
                    curr_suff = ' x $10^{' + str(scale) + '}$'
                suffix.append(curr_suff)

            if is_complex_dtype(img.dtype):
                # Plot real and image
                fig, axes = plt.subplots(nrows=2, **fig_args)
                for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                    cbar_label = self.data_descriptor
                    if comp_name is 'Phase':
                        cbar_label = 'Phase (rad)'
                    plot_map(axis, ufunc(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                             cbar_label=cbar_label, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values,
                             **kwargs)
                    axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                    axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                    axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
                fig.tight_layout()
                return fig, axes
            elif len(img.dtype) > 0:
                # Compound
                # I would like to have used plot_map_stack by providing it the flattened (real) image cube
                # However, the order of the components in the cube and that provided by img.dtype.fields is not matching
                plot_grid = get_plot_grid_size(len(img.dtype))
                fig, axes = plt.subplots(nrows=plot_grid[0], ncols=plot_grid[1], **fig_args)
                for axis, comp_name in zip(axes.flat, img.dtype.fields):
                    plot_map(axis, np.squeeze(img[comp_name]), show_xy_ticks=True, show_cbar=True,
                             x_vec=ref_dims[1].values, y_vec=ref_dims[0].values, **kwargs)
                    axis.set_title(comp_name, pad=15)
                    axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                    axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])

                # delete empty axes
                for ax_ind in range(len(img.dtype), np.prod(plot_grid)):
                    fig.delaxes(axes.flatten()[ax_ind])

                # fig.suptitle(self.name)
                fig.tight_layout()
                return fig, axes
            else:
                fig, axis = plt.subplots(**fig_args)
                # Need to convert to float since image could be unsigned integers or low precision floats
                plot_map(axis, np.float32(np.squeeze(img)), show_xy_ticks=True, show_cbar=True,
                         cbar_label=self.data_descriptor, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values, **kwargs)
                try:
                    axis.set_title(self.name, pad=15)
                except AttributeError:
                    axis.set_title(self.name)

                axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
                axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
                fig.tight_layout()
                return fig, axis

        if np.prod([len(item.values) for item in spec_dims]) == 1:
            # No spectroscopic dimensions at all
            if len(pos_dims) == 2:
                # 2D spatial map
                # Check if we need to adjust the aspect ratio of the image (only if units are same):
                if pos_dims[0].units == pos_dims[1].units:
                    kwargs['infer_aspect'] = True
                return plot_image(pos_dims, data_slice)
            elif np.prod([len(item.values) for item in pos_dims]) > 1:
                # 1D position curve:
                return plot_curve(pos_dims, data_slice)

        elif np.prod([len(item.values) for item in pos_dims]) == 1:
            if len(spec_dims) == 2:
                # 2D spectrogram
                return plot_image(spec_dims, data_slice)
            elif np.prod([len(item.values) for item in pos_dims]) == 1 and \
                    np.prod([len(item.values) for item in spec_dims]) > 1:
                # 1D spectral curve:
                return plot_curve(spec_dims, data_slice)

        elif len(pos_dims) == 1 and len(spec_dims) == 1 and \
            np.prod([len(item.values) for item in pos_dims]) > 1 and \
            np.prod([len(item.values) for item in spec_dims]) > 1:
            # One spectroscopic and one position dimension
            return plot_image(pos_dims + spec_dims, data_slice)

        # If data has at least one dimension with 2 values in pos. AND spec., it can be visualized interactively:
        return simple_ndim_visualizer(data_slice, pos_dims, spec_dims, verbose=verbose, **kwargs)

    def reduce(self, dims, ufunc=da.mean, to_hdf5=False, dset_name=None, verbose=False):
        """

        Parameters
        ----------
        dims : str or list of str
            Names of the position and/or spectroscopic dimensions that need to be reduced
        ufunc : callable, optional. Default = dask.array.mean
            Reduction function such as dask.array.mean available in dask.array
        to_hdf5 : bool, optional. Default = False
            Whether or not to write the reduced data back to a new dataset
        dset_name : str (optional)
            Name of the new USID Main datset in the HDF5 file that will contain the sliced data.
            Default - the sliced dataset takes the same name as this source dataset
        verbose : bool, optional. Default = False
            Whether or not to print any debugging statements to stdout

        Returns
        -------
        reduced_nd : dask.array object
            Dask array object containing the reduced data.
            Call compute() on this object to get the equivalent numpy array
        h5_main_red : USIDataset
            USIDataset reference if to_hdf5 was set to True. Otherwise - None.
        """
        dims = validate_list_of_strings(dims, 'dims')

        for curr_dim in self.n_dim_labels:
            if curr_dim not in self.n_dim_labels:
                raise KeyError('{} not a dimension in this dataset'.format(curr_dim))

        if ufunc not in [da.all, da.any, da.max, da.mean, da.min, da.moment, da.prod, da.std, da.sum, da.var,
                         da.nanmax, da.nanmean, da.nanmin, da.nanprod, da.nanstd, da.nansum, da.nanvar]:
            raise NotImplementedError('ufunc must be a valid reduction function such as dask.array.mean')

        # At this point, dims are valid
        da_nd, status, labels = reshape_to_n_dims(self, get_labels=True, verbose=verbose, sort_dims=False,
                                                  lazy=True)

        # Translate the names of the dimensions to the indices:
        dim_inds = [np.where(labels == curr_dim)[0][0] for curr_dim in dims]

        # Now apply the reduction:
        reduced_nd = ufunc(da_nd, axis=dim_inds)

        if not to_hdf5:
            return reduced_nd, None

        if dset_name is None:
            dset_name = self.name.split('/')[-1]
        else:
            dset_name = validate_single_string_arg(dset_name, 'dset_name')

        # Create the group to hold the results:

        h5_group = create_results_group(self, 'Reduce')

        # check if a pos dimension was sliced:
        pos_sliced = False
        for dim_name in dims:
            if dim_name in self.pos_dim_labels:
                pos_sliced = True
                if verbose:
                    print('Position dimension: {} was reduced. Breaking...'.format(dim_name))
                break
        if not pos_sliced:
            h5_pos_inds = self.h5_pos_inds
            h5_pos_vals = self.h5_pos_vals
            if verbose:
                print('Reusing this main datasets position datasets')
        else:
            if verbose:
                print('Creating new Position dimensions:\n------------------------------------------')
            # First figure out the names of the position dimensions
            pos_dim_names = []
            for cur_dim in dims:
                if cur_dim in self.pos_dim_labels:
                    pos_dim_names.append(cur_dim)
            if verbose:
                print('Position dimensions reduced: {}'.format(pos_dim_names))

            # Now create the reduced position datasets
            h5_pos_inds, h5_pos_vals = write_reduced_anc_dsets(h5_group, self.h5_pos_inds, self.h5_pos_vals,
                                                               pos_dim_names, is_spec=False, verbose=verbose)

            if verbose:
                print('Position dataset created: {}. Labels: {}'.format(h5_pos_inds, get_attr(h5_pos_inds, 'labels')))

        spec_sliced = False
        for dim_name in dims:
            if dim_name in self.spec_dim_labels:
                spec_sliced = True
                if verbose:
                    print('Spectroscopic dimension: {} was reduced. Breaking...'.format(dim_name))
                break
        if not spec_sliced:
            h5_spec_inds = self.h5_spec_inds
            h5_spec_vals = self.h5_spec_vals
            if verbose:
                print('Reusing this main datasets spectroscopic datasets')
        else:
            if verbose:
                print('Creating new spectroscopic dimensions:\n------------------------------------------')

            # First figure out the names of the position dimensions
            spec_dim_names = []
            for cur_dim in dims:
                if cur_dim in self.spec_dim_labels:
                    spec_dim_names.append(cur_dim)
            if verbose:
                print('Spectroscopic dimensions reduced: {}'.format(spec_dim_names))

            # Now create the reduced position datasets
            h5_spec_inds, h5_spec_vals = write_reduced_anc_dsets(h5_group, self.h5_spec_inds, self.h5_spec_vals,
                                                                 spec_dim_names, is_spec=True, verbose=verbose)

            if verbose:
                print('Spectroscopic dataset created: {}. Labels: {}'.format(h5_spec_inds,
                                                                             get_attr(h5_spec_inds, 'labels')))

                # Now put the reduced N dimensional Dask array back to 2D form:
        reduced_2d, status = reshape_from_n_dims(reduced_nd, h5_pos=h5_pos_inds, h5_spec=h5_spec_inds, verbose=verbose)
        if status != True and verbose:
            print('Status from reshape_from_n_dims: {}'.format(status))
        if verbose:
            print('2D reduced dataset: {}'.format(reduced_2d))

        # Create a HDF5 dataset to hold this flattened 2D data:
        h5_red_main = h5_group.create_dataset(dset_name, shape=reduced_2d.shape,
                                              dtype=reduced_2d.dtype)  # , compression=self.compression)
        if verbose:
            print('Created an empty dataset to hold flattened dataset: {}. Chunks: {}'.format(h5_red_main,
                                                                                              h5_red_main.chunks))

        # Copy the mandatory attributes:
        copy_attributes(self, h5_red_main)

        # Now make this dataset a main dataset:
        link_as_main(h5_red_main, h5_pos_inds, h5_pos_vals, h5_spec_inds, h5_spec_vals)
        if verbose:
            print('{} is a main dataset?: {}'.format(h5_red_main, check_if_main(h5_red_main, verbose=verbose)))

        # Now write this data to the HDF5 dataset:
        if verbose:
            print('About to write dask array to this dataset at path: {}, in file: {}'.format(h5_red_main.name,
                                                                                              self.file.filename))
        reduced_2d.to_hdf5(self.file.filename, h5_red_main.name)

        return reduced_nd, USIDataset(h5_red_main)

    def to_csv(self, output_path=None, force=False):
        """
        Output this USIDataset and position + spectroscopic values to a csv file.
        This should ideally be limited to small datasets only

        Parameters
        ----------
        output_path : str, optional
            path that the output file should be written to.
            By default, the file will be written to the same directory as the HDF5 file
        force : bool, optional
            Whether or not to force large dataset to be written to CSV. Default = False

        Returns
        -------
        output_file: str

        Author - Daniel Streater, Suhas Somnath
        """
        if not isinstance(force, bool):
            raise TypeError('force should be a bool')

        if self.dtype.itemsize * self.size / (1024 ** 2) > 15:
            if force:
                print('Note - the CSV file can be (much) larger than 100 MB')
            else:
                print('CSV file will not be written since the CSV file could be several 100s of MB large.\n'
                      'If you still want the file to be written, add the keyword argument "force=True"\n'
                      'We recommend that you save the data as a .npy or .npz file using numpy.dump')
                return

        if output_path is not None:
            if not isinstance(output_path, str):
                raise TypeError('output_path should be a string with a valid path for the output file')
        else:
            parent_folder, file_name = os.path.split(self.file.filename)
            csv_name = file_name[:file_name.rfind('.')] + self.name.replace('/', '-') + '.csv'
            output_path = os.path.join(parent_folder, csv_name)

        if os.path.exists(output_path):
            if force:
                os.remove(output_path)
            else:
                raise FileExistsError('A file of the following name already exists. Set "force=True" to overwrite.\n'
                                      'File path: ' + output_path)

        header = ''
        for spec_vals_for_dim in self.h5_spec_vals:
            # create one line of the header for each of the spectroscopic dimensions
            header += ','.join(str(item) for item in spec_vals_for_dim) + '\n'
        # Add a dashed-line separating the spec vals from the data
        header += ','.join(
            '--------------------------------------------------------------' for _ in self.h5_spec_vals[0])

        # Write the contents to a temporary file
        np.savetxt('temp.csv', self, delimiter=',', header=header, comments='')

        """
        Create the spectral and position labels for the dataset in string form then
        create the position value array in string form, right-strip the last comma from the 
        string to deliver the correct number of values, append all of the labels and values together,
        save the data and header to a temporary csv output
        """
        # First few lines will have the spectroscopic dimension names + units
        spec_dim_labels = ''
        for dim_desc in self.spec_dim_descriptors:
            spec_dim_labels += ','.join('' for _ in self.pos_dim_labels) + str(dim_desc) + ',\n'

        # Next line will have the position dimension names
        pos_labels = ','.join(pos_dim for pos_dim in self.pos_dim_descriptors) + ',\n'

        # Finally, the remaining rows will have the position values themselves
        pos_values = ''
        for pos_vals_in_row in self.h5_pos_vals:
            pos_values += ','.join(str(item) for item in pos_vals_in_row) + ',\n'
        pos_values = pos_values.rstrip('\n')

        # Now put together all the rows for the first few columns:
        output = spec_dim_labels + pos_labels + pos_values

        left_dset = output.splitlines()

        with open('temp.csv', 'r+') as in_file, open(output_path, 'w') as out_file:
            for left_line, right_line in zip(left_dset, in_file):
                out_file.write(left_line + right_line)

        os.remove('temp.csv')
        print('Successfully wrote this dataset to: ' + output_path)

        return output_path
