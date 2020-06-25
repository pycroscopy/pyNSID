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


    def __init__(self, h5_ref):
        super(NSIDataset, self).__init__(h5_ref.id)

        self.data_type = get_attr(self,'data_type')
        self.quantity = self.attrs['quantity']
        self.units = self.attrs['units']
        
        self.axes_names = [dim.label for dim in h5_ref.dims]
        units = []
        quantities = []
        dimension_types = []
        pixel_sizes = []
        
        for label in self.axes_names:
            units.append(get_attr(self.parent[label],'units'))
            quantities.append(get_attr(self.parent[label],'quantity'))
            
            dimension_types.append(get_attr(self.parent[label],'dimension_type'))
            
            pixel_sizes.append(abs(h5_ref.parent[label][1]-h5_ref.parent[label][0]))
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
                    return plot_curve(pos_dims, data_slice)
                else:      
                    print('visualization not implemented, yet')
            else:      
                print('visualization not implemented, yet')
            
            if len(dim_type_dict['spatial'])== 2: 
                ## some kind of image data
                if len(dim_type_dict) == 1:
                    ## simple image
                    self.plot_image()
                elif 'time' in dim_type_dict:
                    ## image stack
                    output_reference = self.plot_stack()
                    
                elif 'spectral' in dim_type_dict:
                    ### spectrum image data in dataset
                    if len(dim_type_dict['spectral'])== 1:
                        output_reference = self.spectrum_image()
                else:
                    print('visualization not implemented, yet')
                
        elif 'reciprocal' in dim_type_dict:
            if len(dim_type_dict['reciprocal'])== 2: 
                ## some kind of image data
                if len(dim_type_dict) == 1:
                    ## simple diffraction pattern
                    output_reference = self.plot_image()
                else:      
                    print('visualization not implemented, yet')
            else:      
                print('visualization not implemented, yet')
        else:
            if 'spectral' in dim_type_dict:
                ### Only spectral data in dataset
                if len(dim_type_dict['spectral'])== 1:
                    fig, axis = self.plot_curve(dim_type_dict['spectral'],data_slice)
                else:      
                    print('visualization not implemented, yet')
            else:      
                print('visualization not implemented, yet')
                
        
            
    def plot_curve(self,ref_dims, curve, **kwargs):
        # Handle the simple cases first:
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp

        if len(ref_dims) > 1:
            for dim in ref-dims:
                plot_grid = get_plot_grid_size(len(ref_dims))
                fig, axes = plt.subplots(nrows=plot_grid[0], ncols=plot_grid[1])#, **fig_args)
                for i, dim in enumerate(ref_dims):
                    axis = axes[i]
                    axes[i].plot(self.dims[ref_dims[dim]][0], self, **kwargs)
                    axes[i].set_title(self.name, pad=15)
                    axes[i].set_xlabel(self.__get_anc_labels()[ref_dims[dim]])# + x_suffix)
                    axes[i].set_ylabel(self.data_descriptor)
                    axes[i].ticklabel_format(style='sci', scilimits=(-2, 3))
                
            fig.suptitle(self.name)
            
            fig.tight_layout()
            return fig, axes
        
        elif is_complex_dtype(curve):
            # Plot real and image
            fig, axes = plt.subplots(nrows=2, **fig_args)
            
            for axis, ufunc, comp_name in zip(axes.flat, [np.abs, np.angle], ['Magnitude', 'Phase']):
                axis.plot(self.dims[ref_dims][0], ufunc(np.squeeze(curve)), **kwargs)
                if comp_name is 'Magnitude':
                    axis.set_title(self.name + '\n(' + comp_name + ')', pad=15)
                    axis.set_xlabel(self.__get_anc_labels()[ref_dims[0]])# + x_suffix)
                    axis.set_ylabel(self.data_descriptor)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))
                else:
                    axis.set_title(comp_name, pad=15)
                    axis.set_ylabel('Phase (rad)')
                    axis.set_xlabel(self.__get_anc_labels()[ref_dims[0]])# + x_suffix)
                    axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            
            fig.tight_layout()
            return fig, axes
        
        else:
            fig, axis = plt.subplots()#**fig_args)
            axis.plot(self.dims[ref_dims[0]][0], curve, **kwargs)
            axis.set_title(self.name, pad=15)
            axis.set_xlabel(self.__get_anc_labels()[ref_dims[0]])# + x_suffix)
            axis.set_ylabel(self.data_descriptor)
            axis.ticklabel_format(style='sci', scilimits=(-2, 3))
            fig.tight_layout()
            return fig, axis
   

    def plot_image(ref_dims, img):
        fig_args = dict()
        temp = kwargs.pop('figsize', None)
        if temp is not None:
            fig_args['figsize'] = temp
        
        if is_complex_dtype(img):
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
        elif len(ref_dims) > 0:
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
            plot_map(axis, np.float32(np.squeeze(img).T), show_xy_ticks=True, show_cbar=True,
                     cbar_label=self.data_descriptor, x_vec=ref_dims[1].values, y_vec=ref_dims[0].values, **kwargs)
            try:
                axis.set_title(self.name, pad=15)
            except AttributeError:
                axis.set_title(self.name)

            axis.set_xlabel(ref_dims[1].name + ' (' + ref_dims[1].units + ')' + suffix[1])
            axis.set_ylabel(ref_dims[0].name + ' (' + ref_dims[0].units + ')' + suffix[0])
            fig.tight_layout()
            return fig, axis

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
